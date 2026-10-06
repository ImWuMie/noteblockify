# 详细使用指南

[README](README.md) · [中文说明](README_zh.md) · [English usage](USAGES.md)

## 1. 安装

要求：

- Python 3.12 或更新版本
- `uv`
- 音频转录推荐 NVIDIA GPU

```powershell
uv sync
```

Windows/Linux 会从 CUDA 12.8 源安装 PyTorch。MIDI 转换和音域模型可以在 CPU
运行；音频转录使用 CUDA 会快很多。MuScriptor 模型可能需要 Hugging Face token
并接受对应模型的许可。

检查安装：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run noteblockify --help
```

## 2. 一条命令转换

MIDI：

```powershell
uv run noteblockify `
  --in "song.mid" `
  --stage model `
  --ranker mc_ranker.auto.best.pt `
  --out "song.nbs"
```

音频：

```powershell
uv run noteblockify `
  --in "song.mp3" `
  --stage model `
  --ranker mc_ranker.auto.best.pt `
  --out "song.nbs"
```

音频路径：

```text
音频 -> MuScriptor 转录 MIDI -> OpenNBS 裸转换 -> 声部平滑 MC 约束 -> NBS
```

转录结果缓存在输入旁边的 `<name>.mid`。用 `--model small`、`--model medium`
或 `--model large` 选择 MuScriptor 转录模型。

`--ranker mc_ranker.auto.best.pt` 可省略。不传权重时使用确定性的声部平滑解码器。
仓库包含的权重是当前音域排序模型，不是增删音符编辑模型。

## 3. 两阶段转换

需要试听裸基线或检查中间结果时：

```powershell
uv run noteblockify --in "song.mid" --stage pre --out "song.pre.nbs"
uv run noteblockify --in "song.pre.nbs" --stage model --ranker mc_ranker.auto.best.pt --out "song.nbs"
```

`pre.nbs` 不保证所有音都能由原版 Minecraft 音符盒播放。它保留 OpenNBS 映射后的
原始 key，包括 `33–57` 以外的音，是检查 MIDI 转录和音色映射是否正确的基线。

## 4. 音域算法

Minecraft 阶段的硬约束：

```text
33 <= NBS key <= 57
```

对于每条原始声部，解码器会：

1. 生成全部合法的同音级八度候选。
2. 对完整声部序列进行动态规划。
3. 惩罚旋律轮廓误差和八度/音域切换。
4. 考虑其他声部的同拍同音冲突。
5. 多轮遍历声部，让通道间的选择互相协调。

这是声部级决策。一个低音无法放进 MC 窗口时，不会只把该音单独抬高，而是允许周围
乐句一起移动到同一合法八度，避免最近八度折叠造成的 `38 -> 43` 反向跳跃。

默认生产阶段只改 key，保留：

```text
tick、layer、instrument、velocity、panning、音符数量
```

输出 key 全部在 `33–57` 时，可用于 OpenNBS 和 Meteor。

## 5. 生成候选

传统合法八度候选：

```powershell
uv run noteblockify --in song.pre.nbs --stage model --candidates --out candidates/
```

会写出 `balanced`、`nearest`、`low`、`high`、`spread` 等确定性候选。

有限编辑候选：

```powershell
uv run noteblockify --in song.pre.nbs --stage model --edit-candidates --out edit-candidates/
```

候选包括：

```text
balanced、drop、replace、clamp、phrase、smooth、add
```

编辑候选用于试听和标注，音符数量可能不同。默认生产路径仍然只改 key，不会静默删除音符。

## 6. 训练

### 原有 key-only 排序模型

记录人工 A/B 选择：

```powershell
uv run noteblockify-feedback `
  --data data/preferences.jsonl `
  --pre song.pre.nbs `
  --preferred candidates/song.high.nbs `
  --rejected candidates/song.low.nbs `
  --note "主旋律更清楚"
```

根据人工偏好训练：

```powershell
uv run noteblockify-train `
  --data data/preferences.jsonl `
  --output mc_ranker.pt
```

key-only 偏好格式要求音符数量相同，只训练合法同音级八度候选的排序；正常
`model` 路径仍由声部平滑动态规划负责连续性。

### 可选编辑排序模型

编辑候选音符数量可不同，所以使用单独的训练入口：

```powershell
uv run noteblockify-feedback `
  --data data/edit_preferences.jsonl `
  --pre song.pre.nbs `
  --preferred edit-candidates/song.replace.nbs `
  --rejected edit-candidates/song.drop.nbs `
  --note "replace 更好听"

uv run noteblockify-train-edits `
  --edit-data data/edit_preferences.jsonl `
  --output mc_edit_ranker.pt
```

显式使用编辑模型：

```powershell
uv run noteblockify `
  --in song.pre.nbs `
  --stage model `
  --edit-ranker mc_edit_ranker.pt `
  --out song.edit.nbs
```

不要用自动规则标签代替听感标签。样本偏好错误时，排序器也会稳定地学到错误偏好。

## 7. NBS 与 Meteor 的行为

OpenNBS 格式保存 key、力度、声像和微调音高。Meteor 当前 NBS 解码器会解析文件，
但实体 NoteBot 演奏只保留乐器和 key：

- per-note 力度被丢弃；
- per-note 声像被丢弃；
- per-note 微调音高被丢弃；
- 立体声感来自方块实际位置；
- NoteBot 先调音，再按 20 游戏刻的时间网格敲击方块。

因此 `--pitch` 不是 Meteor 的解决方案，只对会读取 NBS 微调字段的播放器有用。

把最终文件放到：

```text
.minecraft/meteor-client/notebot/
```

Meteor 支持经典 NBS 和 OpenNBS v5。Exact Instruments 模式下，周围音符盒必须
提供所需原版乐器。实体布置缺少某些乐器/key 组合时，NoteBot 会报告 missing notes。

## 8. 评分

```python
from noteblockify.hear import compare

score = compare("song.mid", "song.nbs")
print(score.f1, score.instrument, score.total)
```

这是客观转换检查，不是音乐质量评判：

- `f1`：按时间、音级和事件存在性一对一匹配音符；
- `instrument`：原版映射乐器的一致率；
- `total`：`0.6 * f1 + 0.4 * instrument`。

它无法判断哪个合法八度更好听。这个问题要通过 OpenNBS/Meteor 实际试听决定。

## 9. 疑难解答

### 转换后出现反向八度跳跃

使用当前声部平滑 `model` 路径，不要继续播放启用新解码器之前生成的旧 `.model.nbs`。
从原始 MIDI 或 `.pre.nbs` 重新生成：

```powershell
uv run noteblockify --in song.mid --stage model --ranker mc_ranker.auto.best.pt --out song.nbs
```

### 音频转录失败

在 `.env` 设置 `HF_TOKEN`，接受 MuScriptor 模型许可后重试。CUDA 显存不足时，
改用 `--model small`，或在 CPU 上进行转录。

### NoteBot 无法加载输出

确认文件是有效 NBS，并且所有 key 位于 `33–57`：

```powershell
uv run python -c "import pynbs; s=pynbs.read('song.nbs'); assert all(33<=n.key<=57 for n in s.notes); print(len(s.notes),len(s.layers))"
```

### `add.nbs` 候选损坏

使用独立层修复之后重新生成的输出。多个新增音符绝不能在同一个 tick 共用同一个 NBS layer。

## 10. 验证

```powershell
uv run python -m py_compile noteblockify/*.py
uv run python -m pytest tests/ -q
```
