[English](README.md) | [Usage guide](USAGES.md) | [使用指南](USAGES_zh.md)

# noteblockify —— 音频/MIDI 转 Minecraft NBS

把任意歌曲转换成 Minecraft 音符盒谱（.nbs），并明确分成“裸转换”和“MC 约束优化”两个阶段。

流程：

```
音频 ──muscriptor──▶ MIDI ──pre──▶ 裸 `.pre.nbs` ──model──▶ `.model.nbs`
MIDI ───────────────────────────▶ 裸 `.pre.nbs`
```

- **忠实复刻 OpenNBS 导入** —— 128 项 GM 音色映射、GM 鼓件映射、2x 时间精度网格、通道层带布局，全部照搬
  [OpenNBS/NoteBlockStudio](https://github.com/OpenNBS/NoteBlockStudio)（MIT），转换结果与 OpenNBS 手动导入听感一致。
- **裸转换基线** —— 只做 OpenNBS 的音色映射、时间网格、层、力度和声像；键位原样保留，超出 33–57 的音也不改。这是最还原的听感基准。
- **MC 约束模型阶段** —— 读取已经落盘的 `.pre.nbs`，只改键位：保证全部进入 33–57；每个音都可以选择同音级的合法八度，动态规划让每个声部尽量保持稳定的八度路径，再保持原始音高距离和旋律线条，并按原始声部中位音高分层：低音声部固定在较低合法八度，避免多轨音乐全部折叠到中音区。
- **可选音符编辑阶段** —— `--edit-candidates` 会为低音异常音生成有限的 `drop`、`replace` 和基于原始声部的 `add` 候选，供人工 A/B 听感标注；默认 key-only 模型不会自动删音。
- **事件不可破坏** —— 不删除音符，不改 tick、乐器、层、力度或声像。
- **零丢音** —— 通道层带按需增高，层冲突不再静默丢音。
- **超长歌重网格** —— 超过 65535 tick 上限的歌按比例重排，而不是报错。
- **立体声层声像** —— 层携带声场（音符跟随层），在立体声上拉开。
- **客观评分** —— `noteblockify.hear` 把 NBS 映射回 MIDI 事件，按音符 F1 + 乐器一致率打分，
  时间容差随歌长增长，吸收 NBS tempo 字段的量化漂移。

## 快速开始

```bash
uv sync

# 第一步：输出裸转换基线
noteblockify --in a.mp3              # 输出 a.pre.nbs
noteblockify --in song.mid           # MIDI 直接转换，无需 GPU
# 第二步：读取已经生成的裸文件，做 MC 约束优化
noteblockify --in a.pre.nbs --stage model  # 输出 a.model.nbs
# 生成多种合法候选，供人工试听 A/B
noteblockify --in a.pre.nbs --stage model --candidates \
  --out candidates/
```

批量转换整个目录：

```bash
for f in data/midi/*.mid; do noteblockify --in "$f"; done
```

输出的 `.nbs` 可以用 [Open Note Block Studio](https://opennbs.org/) 打开，
也可以放进 Meteor Client 的 NoteBot（`.minecraft/meteor-client/notebot/`）在游戏内播放。
注意 Meteor 会丢弃音符力度和声像——哪些信息在游戏内保留，见 USAGES_zh.md。

当前 MC 解码器先用硬约束生成合法候选，再由 PyTorch 偏好排序器选择候选。没有 `mc_ranker.pt` 时回退到 `balanced`。记录人工评价：

```bash
noteblockify-feedback --data data/preferences.jsonl \
  --pre song.pre.nbs \
  --preferred candidates/song.high.nbs \
  --rejected candidates/song.low.nbs \
  --note "主旋律更清楚"
noteblockify-train --data data/preferences.jsonl
```

模型权重由人工 A/B 选择训练，不把自动规则伪装成听感真值。`pre` 文件始终保留，方便直接 A/B 评价。

编辑候选单独使用 JSONL 偏好文件训练，允许候选音符数量不同：

```bash
noteblockify-feedback --data data/edit_preferences.jsonl \\
  --pre song.pre.nbs \\
  --preferred edit-candidates/song.replace.nbs \\
  --rejected edit-candidates/song.drop.nbs
noteblockify-train-edits --edit-data data/edit_preferences.jsonl \\
  --output mc_edit_ranker.pt
```

## 目录结构

| 文件 | 职责 |
|---|---|
| `noteblockify/song.py` | 转换器：OpenNBS 映射、折叠、分层、tempo |
| `noteblockify/mc_model.py` | MC 音域约束优化器 |
| `noteblockify/hear.py` | NBS→MIDI 事件评分器 |
| `noteblockify/cli.py` | `noteblockify` CLI 入口 |
| `tests/test_song.py` | 转换器与 MC 约束不变量测试 |

`sounds/` 是 OpenNBS（MIT）的 16 个原版乐器 OGG，评分器和预览渲染使用。

## 许可

MIT —— 见 LICENSE。GM 音色/鼓件映射复刻自 OpenNBS/NoteBlockStudio（MIT）。
muscriptor 为第三方依赖，音频转录质量以其为上限。
