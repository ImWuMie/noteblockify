# noteblockify —— 音频/MIDI 转 Minecraft NBS

[English](README.md) · [详细使用指南](USAGES_zh.md) · [Detailed usage](USAGES.md)

`noteblockify` 把 MIDI 或音频转换为 Open Note Block Studio 和 Meteor Client
NoteBot 可使用的 `.nbs`。项目分为裸转换基线和 Minecraft 音域约束两个阶段。

## 转换流程

```text
音频 ── MuScriptor ──▶ MIDI ── OpenNBS 映射 ──▶ pre ──▶ 声部平滑 MC 约束 ──▶ NBS
MIDI ─────────────────▶ MIDI ── OpenNBS 映射 ──▶ pre ──▶ 声部平滑 MC 约束 ──▶ NBS
```

`model` 阶段可以直接接收 MIDI/音频，不需要先手动生成 `.pre.nbs`。

## 快速开始

```powershell
uv sync

# 一步完成 MIDI -> Minecraft NBS
uv run noteblockify `
  --in "song.mid" `
  --stage model `
  --ranker mc_ranker.auto.best.pt `
  --out "song.nbs"

# 一步完成音频 -> NBS；首次会下载/加载 MuScriptor 权重
uv run noteblockify `
  --in "song.mp3" `
  --stage model `
  --ranker mc_ranker.auto.best.pt `
  --out "song.nbs"
```

仓库已包含训练好的 `mc_ranker.auto.best.pt`。显式传入 `--ranker` 后，会使用
神经网络对合法候选评分。不传权重时，`model` 仍可运行，使用确定性的声部平滑解码器。

需要保留裸转换文件做试听对比时：

```powershell
uv run noteblockify --in "song.mid" --stage pre --out "song.pre.nbs"
uv run noteblockify --in "song.pre.nbs" --stage model --ranker mc_ranker.auto.best.pt --out "song.nbs"
```

## 音域处理

### 裸 `pre` 阶段

- 使用 OpenNBS 的 GM 音色和鼓件映射。
- 保留时间网格、通道层带、力度、声像和原始键位。
- 不强制把音符折叠进 Minecraft 的两八度范围。
- 用作直接试听的还原基线。

### Minecraft `model` 阶段

Minecraft 音符盒可表示的 NBS 键位是 `33–57`。解码器会：

- 为每个原始音生成所有合法的同音级八度；
- 对整条声部进行动态规划，不再孤立地逐音决策；
- 惩罚不必要的八度/音域切换；
- 保持局部旋律方向和声部音域；
- 分离密集声部，避免不必要的同拍同音；
- 保留音符数量、tick、乐器、层、力度和声像。

例如 `224264 - 室内系的TrackMaker` 的 Fantasia 乐句，不再把原本下行的
`38 → 31` 变成反向跳高的 `38 → 43`，而是把整句移到一致八度，形成
`50 → 43` 这样的连续音域路径。

默认生产模型**不会增删音符**。编辑候选属于单独的人工试听流程。

## 命令模式

```powershell
# 裸转换基线
uv run noteblockify --in song.mid --stage pre --out song.pre.nbs

# 确定性的声部平滑 MC 约束
uv run noteblockify --in song.mid --stage model --out song.nbs

# 声部平滑 + 仓库中的训练权重
uv run noteblockify --in song.mid --stage model --ranker mc_ranker.auto.best.pt --out song.nbs

# 生成传统合法八度候选，供 A/B 试听
uv run noteblockify --in song.pre.nbs --stage model --candidates --out candidates/

# 生成有限的 drop/replace/clamp/phrase/smooth/add 编辑候选
uv run noteblockify --in song.pre.nbs --stage model --edit-candidates --out edit-candidates/
```

支持的输入：

```text
.mid、.midi、.mp3、.wav、.flac、.ogg、.m4a、.pre.nbs
```

音频转录结果缓存在输入旁边的 `<name>.mid`，后续自动复用。

## 可选音符编辑流程

编辑候选可以有不同的音符数量。它们用于人工 A/B 听感标注，不把自动规则当作听感真值：

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

uv run noteblockify `
  --in song.pre.nbs `
  --stage model `
  --edit-ranker mc_edit_ranker.pt `
  --out song.edit.nbs
```

编辑阶段对已经生成的有限候选排序，不会无限制地生成新的 MIDI 事件。

## Minecraft 与 NBS 限制

现代 NBS 文件可以保存力度、声像和微调音高。Meteor 当前 NBS 解码器只使用
乐器和 key，会读取后丢弃 per-note 力度、声像和微调音高。因此：

- Meteor 中听不到 NBS 力度变化；
- 立体声来自音符盒的实际位置，不来自 NBS 声像；
- `--pitch` 适合 OpenNBS 兼容播放器，**不适合 Meteor 的实体音符盒演奏**；
- NoteBot 需要调音并敲击实体方块，所以目标 key 必须位于 `33–57`。

## 项目结构

| 路径 | 职责 |
|---|---|
| `noteblockify/song.py` | OpenNBS 映射、时间网格、层、tempo |
| `noteblockify/mc_model.py` | 声部级平滑 MC 音域约束 |
| `noteblockify/preference_model.py` | 合法八度候选排序器 |
| `noteblockify/edit_model.py` | 可选的不同长度编辑候选排序器 |
| `noteblockify/hear.py` | MIDI/NBS 客观事件评分 |
| `noteblockify/cli.py` | 主 CLI |
| `tests/test_song.py` | 转换与约束不变量测试 |
| `mc_ranker.auto.best.pt` | 仓库内的训练权重 |

## 验证

```powershell
uv run python -m pytest tests/ -q
uv run python -m py_compile noteblockify/*.py
```

## 许可

MIT。GM 映射复刻自 [OpenNBS/NoteBlockStudio](https://github.com/OpenNBS/NoteBlockStudio)。
MuScriptor 为第三方依赖，音频结果质量受其转录输出限制。
