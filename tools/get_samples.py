#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载一批测试音频，用来验证转写链路是否正常。

- 英文：whisper.cpp 官方 jfk.wav（经 jsDelivr CDN）
- 中文：魔搭 FunASR/Paraformer 模型仓库里的示例音频

用法:
    python tools/get_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"

MS = "https://www.modelscope.cn/api/v1/models/{repo}/repo"
JS = "https://cdn.jsdelivr.net/gh/{repo}@{ref}/{path}"

TARGETS = [
    # (输出文件名, 直链, 说明)
    ("jfk.wav", JS.format(repo="ggerganov/whisper.cpp", ref="master",
                          path="samples/jfk.wav"),
     "英文 · JFK 演讲 · 11 秒"),
    ("zh_long.wav", MS.format(
        repo="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
     + "?Revision=master&FilePath=example/asr_example.wav",
     "中文 · 长句 · 13 秒"),
    ("zh_short.wav", MS.format(
        repo="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
     + "?Revision=master&FilePath=example/asr_example.wav",
     "中文 · 短句 · 5 秒"),
    ("zh_hotword.wav", MS.format(
        repo="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
     + "?Revision=master&FilePath=asr_example_hotword.wav",
     "中文 · 含专名 · 8 秒"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for name, url, desc in TARGETS:
        dst = OUT / name
        if dst.exists() and dst.stat().st_size > 1024:
            print(f"  [跳过] {name:<18} 已存在 ({dst.stat().st_size / 1024:.0f} KB)")
            ok += 1
            continue
        try:
            r = requests.get(url, timeout=90, stream=True)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            print(f"  [完成] {name:<18} {dst.stat().st_size / 1024:>7.0f} KB   {desc}")
            ok += 1
        except Exception as e:
            print(f"  [失败] {name:<18} {type(e).__name__}: {e}")
            fail += 1

    print(f"\n{ok} 个就绪, {fail} 个失败  ->  {OUT}")
    if fail == 0:
        print("\n试一下:\n  venv\\Scripts\\python.exe transcribe.py samples/jfk.wav -l en --show-text\n"
              "  venv\\Scripts\\python.exe transcribe.py samples/zh_long.wav -l zh --show-text")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
