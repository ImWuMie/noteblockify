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
# 第一阶段：输出裸转换基线
noteblockify --in song.mp3               # 输出 song.pre.nbs
noteblockify --in song.mp3 --out out.pre.nbs
noteblockify --in song.mp3 --model small # 更小的转录模型
# 第二阶段：读取裸文件并做 MC 约束优化
noteblockify --in song.pre.nbs --stage model # 输出 song.model.nbs
# 生成多个合法候选供人工 A/B
noteblockify --in song.pre.nbs --stage model --candidates --out candidates/
```

执行步骤：

1. muscriptor `medium` 把音频转录成 MIDI（有缓存：.mid 已存在则直接复用）。
2. `noteblockify.song.arrange_pre` 把 MIDI 转成裸 `.pre.nbs`。
3. `--stage model` 读取已经落盘的 `.pre.nbs`，用声部连续性解码器为每个音选择同音级的合法八度。

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
import pynbs
from noteblockify.mc_model import refine
from noteblockify.song import arrange_pre

pre = arrange_pre("song.mid")
pre.save("song.pre.nbs")
model = refine(pynbs.read("song.pre.nbs"))
model.save("song.model.nbs")
```

## 转换器对你的音乐做了什么

| 情况 | 处理方式 |
|---|---|
| 裸转换阶段 | 保留 OpenNBS 映射的原始键位，出窗音也不改 |
| MC 约束阶段 | 每个声部选择合法的同音级八度，窗口内的音也可以切换到另一合法八度 |
| 模型优化目标 | 最小化八度移动，保持声部局部旋律线，避免同拍同音 |
| 音符、tick、乐器、层、力度、声像 | 全部保留，不删除 |
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

## 模型训练

解码器先生成硬约束合法候选，再由 PyTorch 排序器在声部动态规划中逐音选择。
原始音域较低的声部会被分配到较低的合法八度，避免多轨歌曲的低音全部挤到中音区。
没有 `mc_ranker.pt` 时回退到确定性的 `balanced` 候选。GPU 可用时自动使用 CUDA：

```bash
noteblockify-train --midi-guided data/flitered \\
  --output mc_ranker.pt --width 192 --depth 4
noteblockify-feedback --data data/preferences.jsonl \\
  --pre song.pre.nbs \\
  --preferred candidates/song.high.nbs \\
  --rejected candidates/song.low.nbs \\
  --note "主旋律更清楚"
noteblockify-train --data data/preferences.jsonl \\
  --init mc_ranker.pt --output mc_ranker.pt
```

自动目标结合 MIDI 接近度、旋律线条和低音分离，不等于人工听感真值；有人工
A/B 标签时仍应继续训练。`.pre.nbs` 始终保留，作为不变的听感基线。

## 疑难解答

- **`ValueError: every note was dropped`** —— 整首歌都是窗口以下的低音，
  原版音符盒无法表示。
- **muscriptor 模型下载失败** —— 在 `.env` 里设 `HF_TOKEN`，
  并先在 HuggingFace 接受 MuScriptor 模型许可。
- **CUDA 显存不足** —— 改用 MIDI 输入，或用 CPU 转录（慢很多）。
