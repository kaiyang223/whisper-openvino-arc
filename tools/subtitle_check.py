#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕 / 歌词文件格式合规校验器。

按各格式的实际规范逐条检查，**不依赖任何第三方库**。
可以当命令行工具用来体检产物，也可以被 selftest 当断言库调用。

支持：
  .srt   SubRip           序号 / 时间戳 HH:MM:SS,mmm / --> 分隔 / 块间空行 / end>start / 不重叠
  .vtt   WebVTT           头部 WEBVTT / 时间戳 HH:MM:SS.mmm / cue 内禁止 --> / 实体转义 / 块间空行
  .lrc   LRC(歌词)        时间戳 [MM:SS.xx] / 秒<60 / 可选元数据标签

用法:
    venv\\Scripts\\python.exe tools\\subtitle_check.py 文件或目录 [...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- 正则
SRT_TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
VTT_TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")
LRC_TS = re.compile(r"\[(\d{1,3}):(\d{2})\.(\d{2,3})\]")
LRC_TAG = re.compile(r"^\[(ti|ar|al|by|offset|length|re|ve|au):(.*)\]$", re.I)
VTT_CUE_LINE = re.compile(r"^(\S+)\s+-->\s+(\S+)(?:\s+(.*))?$")

# WebVTT 允许的实体引用（&amp; &lt; &gt; &#123; &#x1F; …）
VTT_ENTITY = re.compile(r"&(?:amp|lt|gt|lrm|rlm|nbsp|#\d{1,7}|#[xX][0-9a-fA-F]{1,6});")
# WebVTT 允许的 cue 标签（<b> </i> <c.yellow> <00:00:01.000> 等）
VTT_TAG = re.compile(r"</?(?:b|i|u|c|v|lang|ruby|rt)(?:\.[^\s>]*)?(?:\s[^>]*)?/?>"
                     r"|<\d{2}:\d{2}:\d{2}\.\d{3}>")


def _vtt_unescaped(line: str) -> list[str]:
    """返回 cue 正文里**未转义**的特殊字符。已转义的实体和合法标签不算。"""
    bad = []
    # 先摘掉合法实体，剩下的 & 就是裸的
    rest = VTT_ENTITY.sub("", line)
    if "&" in rest:
        bad.append("&")
    # 再摘掉合法标签，剩下的 < > 就是裸的
    rest2 = VTT_TAG.sub("", rest)
    for ch, name in (("<", "&lt;"), (">", "&gt;")):
        if ch in rest2:
            bad.append(ch)
    return bad


def _ms(h: str, m: str, s: str, f: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(f)


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.warns: list[str] = []
        self.stats: dict = {}
        self.ff: bool | None = None
        self.ffmsg: str = ""

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------- 公共检查
def _check_common(r: Report, pairs: list[tuple[int, int]]) -> None:
    """pairs: [(start_ms, end_ms), ...]，检查时长与重叠。"""
    for i, (a, b) in enumerate(pairs):
        if b < a:
            r.err(f"第 {i+1} 条 end < start（{a} > {b}）")
        elif b == a:
            r.err(f"第 {i+1} 条零时长（start == end == {a}ms）")
    for i in range(len(pairs) - 1):
        if pairs[i][1] > pairs[i + 1][0]:
            r.err(f"第 {i+1}/{i+2} 条时间重叠（{pairs[i][1]} > {pairs[i+1][0]}）")


# ---------------------------------------------------------------- SRT
def check_srt(text: str, r: Report) -> None:
    if text.startswith("\ufeff"):
        r.warn("文件带 UTF-8 BOM（SRT 不要求，部分老播放器可能显示异常）")
        text = text[1:]
    if "\r\n" in text:
        r.warn("使用 CRLF 换行（可行，但多数工具输出 LF）")

    norm = text.replace("\r\n", "\n")
    if not norm.endswith("\n"):
        r.err("文件末尾缺少换行")
    blocks = re.split(r"\n\s*\n", norm.strip("\n"))
    if not blocks or blocks == [""]:
        r.err("文件中没有任何字幕块")
        return

    pairs = []
    for i, blk in enumerate(blocks, 1):
        lines = blk.split("\n")
        if len(lines) < 2:
            r.err(f"第 {i} 块行数不足（应为 序号/时间戳/文本 至少三行）")
            continue
        if not lines[0].strip().isdigit():
            r.err(f"第 {i} 块首行不是序号：{lines[0][:30]!r}")
        elif int(lines[0].strip()) != i:
            r.err(f"第 {i} 块序号不连续：写的是 {lines[0].strip()}")
        m = re.match(r"^(\S+)\s+-->\s+(\S+)\s*$", lines[1])
        if not m:
            r.err(f"第 {i} 块时间戳行格式错误：{lines[1][:50]!r}")
            continue
        ts = []
        for side in (m.group(1), m.group(2)):
            mm = SRT_TS.match(side)
            if not mm:
                r.err(f"第 {i} 块时间戳不合规：{side!r}（应为 HH:MM:SS,mmm）")
                ts = None
                break
            h, mi, s, f = mm.groups()
            if int(mi) > 59 or int(s) > 59:
                r.err(f"第 {i} 块时间戳越界：{side!r}（分/秒必须 <60）")
            ts.append(_ms(h, mi, s, f))
        if ts:
            pairs.append((ts[0], ts[1]))
        body = "\n".join(lines[2:])
        if not body.strip():
            r.err(f"第 {i} 块文本为空")
        for ln in lines[2:]:
            if "-->" in ln:
                r.warn(f"第 {i} 块正文含 '-->'，可能被解析为时间戳行")

    r.stats["blocks"] = len(blocks)
    r.stats["cues"] = len(pairs)
    _check_common(r, pairs)


# ---------------------------------------------------------------- VTT
def check_vtt(text: str, r: Report) -> None:
    if text.startswith("\ufeff"):
        r.warn("文件带 UTF-8 BOM（WebVTT 规范不推荐）")
        text = text[1:]
    if not text.startswith("WEBVTT"):
        r.err("首行不是 WEBVTT（WebVTT 必须以 WEBVTT 开头）")
        return
    first = text.split("\n", 1)[0]
    if first != "WEBVTT" and not re.match(r"^WEBVTT[ \t]", first):
        r.err(f"WEBVTT 头部格式错误：{first!r}（应为 'WEBVTT' 或 'WEBVTT 描述'）")

    norm = text.replace("\r\n", "\n")
    body = norm.split("\n", 1)[1] if "\n" in norm else ""
    if not body.startswith("\n"):
        r.err("WEBVTT 头部之后必须跟一个空行")
    if not norm.endswith("\n"):
        r.err("文件末尾缺少换行")

    blocks = [b for b in re.split(r"\n\s*\n", body.strip("\n")) if b.strip()]
    pairs = []
    for i, blk in enumerate(blocks, 1):
        lines = blk.split("\n")
        lines = [l for l in lines if l.strip()]
        if not lines:
            continue
        head = None
        for j, ln in enumerate(lines):
            if "-->" in ln:
                head, ti = ln, j
                break
        if head is None:
            r.err(f"第 {i} 块找不到时间戳行：{blk[:40]!r}")
            continue
        m = VTT_CUE_LINE.match(head)
        if not m:
            r.err(f"第 {i} 块时间戳行格式错误：{head[:60]!r}")
            continue
        ts = []
        for side in (m.group(1), m.group(2)):
            mm = VTT_TS.match(side)
            if not mm:
                r.err(f"第 {i} 块时间戳不合规：{side!r}（应为 HH:MM:SS.mmm）")
                ts = None
                break
            h, mi, s, f = mm.groups()
            if int(mi) > 59 or int(s) > 59:
                r.err(f"第 {i} 块时间戳越界：{side!r}（分/秒必须 <60）")
            ts.append(_ms(h, mi, s, f))
        if ts:
            pairs.append((ts[0], ts[1]))

        for ln in lines[ti + 1:]:
            if "-->" in ln:
                r.err(f"第 {i} 块 cue 正文含 '-->'（WebVTT 明令禁止）")
            for ch in _vtt_unescaped(ln):
                name = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}[ch]
                r.err(f"第 {i} 块 cue 正文含未转义的 '{ch}'（应写作 {name}）")

    r.stats["cues"] = len(pairs)
    _check_common(r, pairs)


# ---------------------------------------------------------------- LRC
def check_lrc(text: str, r: Report) -> None:
    if text.startswith("\ufeff"):
        r.warn("文件带 UTF-8 BOM（部分播放器需要，多数现代播放器可读）")
        text = text[1:]
    lines = [l for l in text.replace("\r\n", "\n").split("\n")]
    if not lines[-1] == "":
        r.warn("文件末尾没有换行（可行，但习惯上加一个）")

    cues, tags, bad = 0, 0, 0
    seen_after_tag = False
    for ln in lines:
        if not ln.strip():
            continue
        if LRC_TAG.match(ln):
            tags += 1
            continue
        stamps = list(LRC_TS.finditer(ln))
        if not stamps:
            if seen_after_tag:
                r.warn(f"非时间戳、非标签的行：{ln[:40]!r}")
            continue
        seen_after_tag = True
        first_end = stamps[0].end()
        if ln[:first_end].replace(" ", "") != "".join(
                m.group(0) for m in stamps if m.end() <= first_end):
            # 允许一行多个时间戳前缀
            pass
        rest = ln[stamps[-1].end():]
        if not rest.strip():
            r.warn(f"时间戳后没有文本：{ln[:30]!r}")
        for m in stamps:
            mi, s, f = m.groups()
            if int(s) > 59:
                r.err(f"秒数越界：[{mi}:{s}.{f}] 里秒必须 <60")
                bad += 1
            cues += 1
    r.stats["cues"] = cues
    r.stats["tags"] = tags
    if cues == 0:
        r.err("文件中没有任何歌词行")
    if bad == 0 and cues:
        pass


# ---------------------------------------------------------------- 入口
CHECKERS = {".srt": check_srt, ".vtt": check_vtt, ".lrc": check_lrc}


def check_with_ffmpeg(p: Path) -> tuple[bool | None, str]:
    """
    用 ffmpeg（PyAV）真实解析一遍 —— 这是比自写正则更硬的第三方验证。
    返回 (结果, 说明)；PyAV 不可用时返回 (None, 原因)。
    """
    if p.suffix.lower() == ".lrc":
        return None, "LRC 不是媒体容器，ffmpeg 不处理（属正常）"
    try:
        import av
    except ImportError:
        return None, "未安装 PyAV，跳过"
    try:
        c = av.open(str(p))
        st = next((s for s in c.streams if s.type == "subtitle"), None)
        if st is None:
            c.close()
            return False, "ffmpeg 没找到字幕流"
        tb = st.time_base
        evs = []
        for pkt in c.demux(st):
            for _fr in (pkt.decode() or []):
                a = float(pkt.pts * tb)
                b = float((pkt.pts + pkt.duration) * tb) if pkt.duration else a
                evs.append((a, b))
        codec = st.codec_context.name
        c.close()
        if not evs:
            return False, "ffmpeg 解析出 0 条字幕"
        if not all(b > a for a, b in evs):
            return False, "存在零时长或负时长"
        if any(evs[i][1] > evs[i + 1][0] + 1e-6 for i in range(len(evs) - 1)):
            return False, "存在时间重叠"
        return True, f"容器={codec} 条数={len(evs)} 跨度={evs[-1][1]-evs[0][0]:.2f}s"
    except Exception as e:
        return False, f"ffmpeg 解析异常：{type(e).__name__}: {str(e)[:70]}"


def check_file(p: Path) -> Report:
    r = Report(p)
    ext = p.suffix.lower()
    fn = CHECKERS.get(ext)
    if not fn:
        r.err(f"不支持的类型：{ext}")
        return r
    try:
        raw = p.read_bytes()
    except Exception as e:
        r.err(f"读取失败：{e}")
        return r
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        r.err("不是合法 UTF-8（字幕文件应为 UTF-8）")
        return r
    fn(text, r)
    # 正则检查通过后，再用 ffmpeg 真实解析一遍做交叉验证
    r.ff, r.ffmsg = check_with_ffmpeg(p)
    if r.ff is False:
        r.err(f"第三方解析失败：{r.ffmsg}")
    return r


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    targets: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            for ext in CHECKERS:
                targets.extend(sorted(p.rglob(f"*{ext}")))
        elif p.exists():
            targets.append(p)
        else:
            print(f"  路径不存在: {p}")

    if not targets:
        print("没有找到可校验的字幕/歌词文件。")
        return 2

    n_ok = n_bad = 0
    for p in targets:
        r = check_file(p)
        head = p.name
        mark = "[合规]" if r.ok else "[不合规]"
        if r.ok:
            n_ok += 1
        else:
            n_bad += 1
        ff = ("ffmpeg✔" if r.ff is True else
              ("ffmpeg✘" if r.ff is False else "ffmpeg-"))
        extra = "  ".join(f"{k}={v}" for k, v in r.stats.items())
        print(f"  {mark} {head:<26} {ff:<9} {extra}")
        if r.ff is True:
            print(f"           ffmpeg: {r.ffmsg}")
        if not r.ok:
            for e in r.errors[:12]:
                print(f"           ✗ {e}")
            if len(r.errors) > 12:
                print(f"           … 另有 {len(r.errors)-12} 条")
        for w in r.warns[:4]:
            print(f"           ⚠ {w}")

    print()
    print(f"  合规 {n_ok}   不合规 {n_bad}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
