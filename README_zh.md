[English](README.md) | [Usage guide](USAGES.md) | [使用指南](USAGES_zh.md)

# noteblockify —— 音频/MIDI 转 Minecraft NBS

把任意歌曲转换成 Minecraft 音符盒谱（.nbs），全部音符落在原版两个八度窗口（F♯3–F♯5）内，每个声部的八度归属由模型决定。

流程：

```
音频 (mp3/wav)  ──muscriptor 转录──▶  MIDI  ──noteblockify──▶  .nbs
MIDI 文件       ───────────────────────────────▶  .nbs
```

- **忠实复刻 OpenNBS 导入** —— 128 项 GM 音色映射、GM 鼓件映射、2x 时间精度网格、通道层带布局，全部照搬
  [OpenNBS/NoteBlockStudio](https://github.com/OpenNBS/NoteBlockStudio)（MIT），转换结果与 OpenNBS 手动导入听感一致。
- **硬窗口处理** —— 每个音按八度折叠进键位 33–57；原版音符盒在此之外发不出声。
- **模型决定八度归属** —— 小型加权 MLP（`octave.pt`）决定每个声部坐在哪个八度，标签从 MIDI 自身生成
  （通道中位数按整八度对齐）。必须上折一整个八度以上才能进窗的低音直接删除，避免浑浊。
- **零丢音** —— 通道层带按需增高，层冲突不再静默丢音。
- **超长歌重网格** —— 超过 65535 tick 上限的歌按比例重排，而不是报错。
- **立体声层声像** —— 层携带声场（音符跟随层），在立体声上拉开。
- **客观评分** —— `noteblockify.hear` 把 NBS 映射回 MIDI 事件，按音符 F1 + 乐器一致率打分，
  时间容差随歌长增长，吸收 NBS tempo 字段的量化漂移。

## 快速开始

```bash
uv sync

# 从音频转换（先用 muscriptor 转录；需要 ~5 GB 显存）
noteblockify --in a.mp3              # 输出 a.nbs 到输入旁
noteblockify --in a.mp3 --out b.nbs  # 指定输出路径
noteblockify --in song.mid           # MIDI 直接转换，无需 GPU
```

批量转换整个目录：

```bash
for f in data/midi/*.mid; do noteblockify --in "$f"; done
```

输出的 `.nbs` 可以用 [Open Note Block Studio](https://opennbs.org/) 打开，
也可以放进 Meteor Client 的 NoteBot（`.minecraft/meteor-client/notebot/`）在游戏内播放。
注意 Meteor 会丢弃音符力度和声像——哪些信息在游戏内保留，见 USAGES_zh.md。

## 重训八度模型

```bash
uv run python -m noteblockify.octave              # 加权 MLP，GPU 约 25 秒
uv run python score_model.py             # 完整评测报告
```

标签从 MIDI 自身派生——无需人工标注。往 `data/midi/` 加歌再重训即可定制。

## 目录结构

| 文件 | 职责 |
|---|---|
| `noteblockify/song.py` | 转换器：OpenNBS 映射、折叠、分层、tempo |
| `noteblockify/octave.py` | 八度模型、训练、特征 |
| `noteblockify/hear.py` | NBS→MIDI 事件评分器 |
| `noteblockify/cli.py` | `noteblockify` CLI 入口 |
| `score_model.py` | 模型评测脚本 |

`sounds/` 是 OpenNBS（MIT）的 16 个原版乐器 OGG，评分器和预览渲染使用。

## 许可

MIT —— 见 LICENSE。GM 音色/鼓件映射复刻自 OpenNBS/NoteBlockStudio（MIT）。
muscriptor 为第三方依赖，音频转录质量以其为上限。
