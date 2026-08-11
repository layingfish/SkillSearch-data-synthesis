# Experiment: Frozen teacher c1 hard-negative coverage audit
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

只读取冻结 SkillRouter teacher panel、`100,078` 条 source-query rows 与固定 LETTER depth=`4` Semantic IDs。对每个 gold skill mention 排除同 query 的全部 sibling GT，统计 teacher top-K 中与 gold 共享 c0、在 c1 首次分叉的 non-GT 覆盖率。

K 固定为 `{1,5,10,20,50}`。还要报告首次分叉位置=`{c0,c1,c2,c3}` 的覆盖、teacher rank 与 gold-minus-negative margin 分布。该审计不改变训练数据或 checkpoint；输出的 candidate manifest 只能作为后续 loss probe 的冻结输入。

## Commands

Analyze: `scripts/analyze_legacy_c1_hard_negative_coverage.py` 使用 source rows=`100,078`、teacher candidate limit=`50`、semantic depth=`4`、K=`{1,5,10,20,50}` 生成 summary 与逐 source-query candidate manifest。analyzer SHA-256=`dc6f4323528181f1999ecc9dc4a51f81ceecb0b84863a0b197d16e9e5982858e`。

## Outputs

Output: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_c1_hard_negative_coverage_20260811_v1/`

- `c1_hard_negative_coverage_summary.json` SHA-256=`68f717f752b4b00a833eb8d9aef92f63dac7af6037c7c8c78dd033a36ee020b4`
- `c1_hard_negative_candidates.jsonl` SHA-256=`1cf68e43a6f79702a718908e950b0405717f6e97a670eaf04e0305dfb0d27a1d`（`100,078` rows）

## Verification

- [x] source rows、teacher rows 按 JSON scalar `index` 一对一对齐
- [x] query text 与 gold set 跨两份输入对齐
- [x] teacher skill/score 长度、唯一性、有限性通过；排序单调性只检查前 `50` 个 retrieval candidates，其后是冻结 panel 追加的 missing-GT scores
- [x] 全部 `129,341` 个 gold mentions 与 teacher candidates 均存在于 LETTER IDs
- [x] 同 query 全部 sibling GT 不进入 negative candidate
- [x] summary 与逐 row candidate manifest 计数守恒：rows=`100,078`、gold mentions=`129,341`
- [x] Minimal local check：focused unittest=`7/7`
- [x] Reviewer required? yes；enhanced artifact 计数与语义有效

## Result / Notes

teacher top-K 中 c1 首分叉负例的 gold-mention 覆盖率如下。条件为 source queries=`100,078`、gold mentions=`129,341`、teacher retrieval domain=原始 rank 前 `K` 个候选后排除全部 sibling GT、LETTER depth=`4`。

| teacher K | c1-diverging covered mentions | coverage |
|---:|---:|---:|
| `1` | `4,354/129,341` | `3.3663%` |
| `5` | `27,361/129,341` | `21.1542%` |
| `10` | `41,250/129,341` | `31.8924%` |
| `20` | `59,932/129,341` | `46.3364%` |
| `50` | `89,661/129,341` | `69.3214%` |

top-50 覆盖在 single/multi rows 上分别为 `57,491/82,621=69.5840%` 和 `32,170/46,720=68.8570%`（上述同一 candidate 条件），没有明显的 multi-GT 覆盖偏差。LETTER catalog 中 `129,274/129,341` 个 gold mentions 结构上存在 same-c0/different-c1 skill，只有 `67/129,341` 个不存在；teacher top-50 的条件覆盖率为 `69.3573%`。

已选 c1 negative 的 teacher rank 中位数=`12`、p90=`39`（covered mentions=`89,661`、candidate limit=`50`）。gold-minus-negative teacher margin 中位数=`0.1722`；`18,762/89,661=20.9255%` 的 margin `<=0`，说明该候选集包含真正难例，不只是随机同 prefix skill。

结论：冻结 teacher top-50 足以支持第一轮限定 probe，且 single/multi 覆盖平衡。auxiliary loss 只作用于有 c1 negative 的 `89,661/129,341` 个 expanded gold rows；其余 rows 保留 CE，不用无 teacher relevance 证据的随机结构负例补齐。

该 artifact 的训练数据条件是 enhanced source queries=`100,078`、expanded rows=`129,341`。RTX strict-base probe 改用独立 base artifact（source queries=`94,912`、expanded rows=`124,175`、manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d`），避免把 mode3 augment=`5,166` source rows 混入 parity 结论。

首次完整产物在增加 single/multi 与 structural-availability 分层后删除并从同一冻结输入重新生成；旧的两个可重建审计文件不再保留。
