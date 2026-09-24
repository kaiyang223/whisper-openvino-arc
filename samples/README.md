# 测试音频

这些是回归自检和精度评测用的样本，随仓库一起分发，**clone 下来即可直接跑测试**。
全部为 16kHz 单声道 WAV。

| 文件 | 时长 | 内容 | 来源 |
|---|---|---|---|
| `jfk.wav` | 11 s | 英文，肯尼迪就职演说片段 | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) 官方测试样本 |
| `bench_en_44s.wav` | 44 s | 英文，`jfk.wav` 重复 4 次拼接 | 本地合成，基准测试用 |
| `zh_long.wav` | 13 s | 中文，长句 | 魔搭 [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) 示例音频 |
| `zh_short.wav` | 5 s | 中文，短句 | 同上（另一模型仓库） |
| `zh_hotword.wav` | 8 s | 中文，含机构专名 | 魔搭 [SeACo-Paraformer](https://www.modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) 示例音频 |

## 用途

| 样本 | 用在 |
|---|---|
| `jfk.wav` | `selftest.py` 逐字比对（有确定原文，可算 WER）、`eval_accuracy.py` 的劣化基准 |
| `bench_en_44s.wav` | 速度基准（长音频才能摊薄固定开销） |
| `zh_long.wav` / `zh_hotword.wav` | 中文识别、VAD 行为、专名识别 |
| `zh_short.wav` | 短音频边界情况 |

## 关于许可

- `jfk.wav` 来自 whisper.cpp 项目（MIT），原始录音为美国政府公开演讲，属公有领域。
- 中文样本来自魔搭 ModelScope 上 FunASR / Paraformer 模型的示例音频
  （模型本身为 Apache-2.0）。**仅用于本项目测试**，如用于其它用途请自行确认授权。
- `bench_en_44s.wav` 由 `jfk.wav` 本地拼接生成，无独立版权。

## 重新获取

删掉后可以重新下载：

```bat
venv\Scripts\python.exe tools\get_samples.py
```

> 注意 `get_samples.py` 不会生成 `bench_en_44s.wav`（它是本地拼接的）。
> 需要的话用任意音频工具把 `jfk.wav` 连播 4 次即可。

## 生成物

转写结果（`.txt` / `.srt` / `.vtt` / `.lrc` / `.json`）不会入库，
已在 `.gitignore` 中排除。跑完测试后可以放心删除。
