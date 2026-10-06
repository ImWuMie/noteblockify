# 使用指南

[English README](README.md) | [中文说明](README_zh.md) | [Usage guide](USAGES.md)

## 安装

```bash
# 需要 Python 3.12+、uv；音频路径需要 NVIDIA GPU
uv sync
```

依赖：torch（pyproject 里钉了 CUDA 12.8 源）、muscriptor、mido、pynbs。
纯 MIDI 转换在 CPU 上就能跑；只有音频转录需要 GPU（medium 模型约 5 GB 显存）。

## 音频转 NBS

```bash
noteblockify --in song.mp3               # 输出 song.nbs
noteblockify --in song.mp3 --out out.nbs # 指定输出路径
noteblockify --in song.mp3 --model small # 更小的转录模型
```

执行步骤：

1. muscriptor `medium` 把音频转录成 MIDI（有缓存：.mid 已存在则直接复用）。
2. `noteblockify.song.arrange` 把 MIDI 转成 NBS 谱。
3. 结果保存在音频同目录、`.nbs` 后缀。

首次转录约每 4 分钟音频耗时 2 分钟。

## 转换 MIDI 文件

单个文件：

```bash
noteblockify --in song.mid
```

整个目录：

```bash
for f in data/midi/*.mid; do noteblockify --in "$f"; done
```

编程接口：

```python
from noteblockify.song import arrange

song = arrange("song.mid")     # 返回 pynbs.File
song.save("song.nbs")
```

## 转换器对你的音乐做了什么

| 情况 | 处理方式 |
|---|---|
| 音符超出 F♯3–F♯5 | 按八度折叠到最近的窗口内同名音 |
| 整个声部八度位置不佳 | 八度模型整体移动 ±1/±2 个八度 |
| 低音需要上折一整个八度以上才进窗 | 删除（否则会浑浊旋律音区） |
| GM 音色 | 按 OpenNBS 128 项表映射，通道级音色差异化 |
| 鼓通道（10） | 按 OpenNBS GM 鼓件表映射 |
| 同时发声数超过层数 | 自动加层——零丢音 |
| 歌曲超过 65535 tick | 按比例重排网格（相对时值保留） |
| 多个速度事件 | 第一个生效（OpenNBS 行为） |

## 给转换结果打分

```python
from noteblockify.hear import compare

score = compare("song.mid", "song.nbs")
print(score.f1, score.instrument, score.total)   # 各 0..1
```

- `f1` —— 按起始时间（±60 ms + 漂移容差）和音高（±1 半音）匹配的音符 F1
- `instrument` —— 匹配音符中携带正确原版乐器的比例
- `total` —— 加权 0.6·f1 + 0.4·instrument

规整 MIDI 的忠实转换约 0.99；最后一点差额是 tick 网格量化。

## 在 Minecraft 里播放（Meteor Client）

1. 把 `.nbs` 拷进 `.minecraft/meteor-client/notebot/`。
2. 在身边摆音符盒（NoteBot 扫触达范围内 6³ 区域），启用 Notebot 模块并加载歌曲。
3. 模块会先把音符盒调音到目标音高，然后靠攻击方块演奏。

游戏内限制（已核对 Meteor 解码器源码）：

- **力度被忽略** —— 解码器读掉 per-note 力度字节后丢弃，所有音全音量。
- **声像被忽略** —— 立体声来自方块的实际物理位置。
- **精确乐器模式** —— NoteBot 按乐器类型匹配音符盒；确保你有对应乐器类型的方块
  （harp/bass/drum/……由音符盒下方方块决定）。
- **时序抖动** —— Meteor 以 20 游戏刻/秒重算时间，亚 tick 放置会有 ±1 tick（约 50 ms）舍入。
  所有 NBS 文件都受此限制。

## 重训八度模型

```bash
uv run python -m noteblockify.octave        # 用 data/midi/*.mid 训练，存 octave.pt
uv run python score_model.py       # 训练集、held-out、转换层三项分数
```

训练好的模型随仓库附带（`octave.pt`），开箱即用；重训可适配你的曲库。
往 `data/midi/` 加 MIDI 可以让模型适配你的曲库。

## 疑难解答

- **`ValueError: every note was dropped`** —— 整首歌都是窗口以下的低音，
  原版音符盒无法表示。
- **muscriptor 模型下载失败** —— 在 `.env` 里设 `HF_TOKEN`，
  并先在 HuggingFace 接受 MuScriptor 模型许可。
- **CUDA 显存不足** —— 改用 MIDI 输入，或用 CPU 转录（慢很多）。
