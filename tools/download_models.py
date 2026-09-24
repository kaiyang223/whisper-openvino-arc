#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从魔搭 (ModelScope) 下载 Whisper OpenVINO IR 模型。
HuggingFace 在国内不可达时使用本脚本。

用法:
    python tools/download_models.py                # 下载全部预设模型
    python tools/download_models.py turbo          # 只下 turbo
    python tools/download_models.py largev3        # 只下 large-v3
    python tools/download_models.py --list         # 列出预设
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

API_FILES = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
API_RAW = "https://www.modelscope.cn/api/v1/models/{repo}/repo"

PRESETS = {
    "turbo": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "largev3": "OpenVINO/whisper-large-v3-fp16-ov",
    "turbo-int8": "OpenVINO/whisper-large-v3-turbo-int8-ov",
    "largev3-int8": "OpenVINO/whisper-large-v3-int8-ov",
}

# VAD 是特殊项：不是魔搭仓库，而是从官方 silero-vad wheel 里提取
VAD_REPO = "silero-vad"
VAD_OUT = "silero_vad_16k_ov.onnx"
VAD_SRC_IN_WHEEL = "silero_vad/data/silero_vad_openvino_16k.onnx"

# 这些文件用不上，跳过
SKIP_SUFFIX = (".gitattributes",)
SKIP_NAMES = {"README.md", "configuration.json"}

CHUNK = 1 << 20  # 1 MiB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"


def list_files(repo: str) -> list[tuple[str, int]]:
    r = requests.get(
        API_FILES.format(repo=repo),
        params={"Revision": "master", "Recursive": "true"},
        timeout=40,
    )
    r.raise_for_status()
    files = r.json().get("Data", {}).get("Files", [])
    out = []
    for f in files:
        if f.get("Type") == "tree":
            continue
        p = f["Path"]
        if p.endswith(SKIP_SUFFIX) or os.path.basename(p) in SKIP_NAMES:
            continue
        out.append((p, int(f.get("Size", 0))))
    return out


def download_one(repo: str, path: str, size: int, dest: Path,
                 session: requests.Session) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0

    if size and done == size:
        print(f"      [跳过] {path}  (已完整)")
        return
    if done > size:
        dest.unlink()
        done = 0

    headers = {"Range": f"bytes={done}-"} if done else {}
    url = API_RAW.format(repo=repo)
    with session.get(url, params={"Revision": "master", "FilePath": path},
                     headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        if done and r.status_code != 206:  # 服务端不支持断点，重头来
            done = 0
            dest.unlink(missing_ok=True)
        mode = "ab" if done and r.status_code == 206 else "wb"
        t0 = time.time()
        got = done
        with open(dest, mode) as fp:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                fp.write(chunk)
                got += len(chunk)
                if size:
                    pct = got / size * 100
                    spd = (got - done) / max(time.time() - t0, 1e-6)
                    sys.stdout.write(
                        f"\r      {pct:5.1f}%  {human(got)}/{human(size)}"
                        f"  {human(spd)}/s      ")
                    sys.stdout.flush()
    if size and dest.stat().st_size != size:
        raise RuntimeError(f"大小不符: {path} 期望 {size} 实得 {dest.stat().st_size}")
    print(f"\r      [完成] {path}{' ' * 30}")


def fetch(key: str) -> None:
    repo = PRESETS[key]
    name = repo.split("/")[-1]
    target = MODELS_DIR / name
    print(f"\n=== {key}  ->  {name} ===")
    files = list_files(repo)
    total = sum(s for _, s in files)
    print(f"    {len(files)} 个文件, 共 {human(total)}")

    # 已存在则跳过整个仓库
    all_ok = all(
        (target / p).exists() and (target / p).stat().st_size == s
        for p, s in files
    )
    if all_ok:
        print("    已完整存在，跳过。")
        return

    session = requests.Session()
    for i, (p, s) in enumerate(files, 1):
        print(f"  ({i}/{len(files)}) {p}")
        download_one(repo, p, s, target / p, session)
    print(f"    -> {target}")


def fetch_vad() -> None:
    """
    下载 Silero VAD（用于抑制 Whisper 静音幻觉）。

    来源：官方 PyPI 包 silero-vad 的 wheel，里面自带一个
    `silero_vad_openvino_16k.onnx` —— 静态形状、无 sr 输入，
    OpenVINO 能直接吃。**不要**用 HuggingFace/master 分支那个
    `silero_vad.onnx`，它是动态量化版，带 If 分支和动态 shape，
    OpenVINO 的 ONNX 前端转换会失败。
    """
    import io
    import zipfile

    target = MODELS_DIR / VAD_REPO
    dst = target / VAD_OUT
    target.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 100_000:
        print(f"\n=== vad ===\n    {dst.name} 已存在 ({dst.stat().st_size/1048576:.2f} MB)，跳过。")
        return

    print(f"\n=== vad  ->  {VAD_OUT} ===")
    print("    查询 PyPI 上 silero-vad 的最新 wheel ...")
    meta = requests.get("https://pypi.org/pypi/silero-vad/json", timeout=40).json()
    ver = meta["info"]["version"]
    wheels = [f for f in meta["releases"][ver] if f["filename"].endswith(".whl")]
    if not wheels:
        raise RuntimeError("PyPI 上没有找到 silero-vad 的 wheel")
    url = wheels[0]["url"]
    print(f"    版本 {ver}")
    print(f"    下载 {wheels[0]['filename']} ...")
    data = requests.get(url, timeout=300).content

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if VAD_SRC_IN_WHEEL not in names:
            cand = [n for n in names if n.endswith(".onnx") and "openvino" in n]
            if not cand:
                raise RuntimeError(f"wheel 里找不到 OpenVINO 版 VAD 模型，现有 onnx: "
                                   f"{[n for n in names if n.endswith('.onnx')]}")
            src = cand[0]
        else:
            src = VAD_SRC_IN_WHEEL
        dst.write_bytes(z.read(src))
    print(f"    已提取 {src}")
    print(f"    -> {dst}  ({dst.stat().st_size/1048576:.2f} MB)")


def main() -> int:
    args = [a for a in sys.argv[1:]]
    if "--list" in args or not args:
        if "--list" in args:
            for k, v in PRESETS.items():
                print(f"  {k:<14} {v}")
            print(f"  {'vad':<14} silero-vad (OpenVINO 16k), 来自 PyPI wheel")
            return 0

    keys = args or ["turbo", "largev3", "vad"]
    unknown = [k for k in keys if k not in PRESETS and k != "vad"]
    if unknown:
        print(f"未知模型: {unknown}\n可用: {list(PRESETS) + ['vad']}")
        return 2

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for k in keys:
        if k == "vad":
            fetch_vad()
        else:
            fetch(k)
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
