# Experiment: Legacy base teacher c1 hard-negative coverage audit
Status: done
Risk Tier: 2

## Goal / Hypothesis / Change Scope

为严格 RTX parity 训练冻结 base candidate manifest。输入固定为 base source queries=`94,912`、expanded gold mentions=`124,175`、LETTER semantic depth=`4`；teacher panel 只消费前 `94,912` 个逐行对齐记录，并审计尾部恰有 enhanced-only rows=`5,166`。

## Plan

1. 校验 base rows SHA-256=`ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5`。
2. 校验 base 与 enhanced 前 `94,912` 行逐 JSON object 相同。
3. 校验 teacher 前缀与 base 的 index、query、gold set 逐行相同。
4. 在 teacher top-50 排除全部 sibling GT 后选 same-c0/different-c1 negative。
5. 生成 `94,912` 行 candidate manifest，并核对 expanded gold mentions=`124,175`。
6. 运行 focused unittest、Ruff 和独立语义复核。

## Commands

Analyze: `scripts/analyze_legacy_c1_hard_negative_coverage.py --expected-rows 94912 --expected-teacher-trailing-rows 5166 --candidate-limit 50 --top-ks 1,5,10,20,50`。

## Outputs

Output: `/models_rw/SkillSearch-data-synthesis/experiments/legacy_base_c1_hard_negative_coverage_20260811_v1/`

## Verification

- [x] base/enhanced 前缀逐 JSON object 一致；teacher 前缀 index/query/gold set 一致
- [x] base source rows=`94,912`、expanded gold mentions=`124,175`
- [x] candidate manifest 与 summary 计数、hash、权限完整
- [x] Minimal local check：focused unittest=`9/9`；Ruff 通过
- [x] Reviewer required? yes；复算计数一致

## Result / Notes

输出 summary SHA-256=`72b0f525d6c3d9feabc7770c054ee70d8478f1b280d1a100793da8d5b6c6dd6d`；candidate manifest SHA-256=`8df15d909949e53c0fe8908ab889e164c70a2a2dc5e46a1b6ad505fb3ef17d1d`（base source queries=`94,912`、expanded gold mentions=`124,175`、文件权限=`0644`）。

teacher top-K 的 c1 coverage 为 K=`1/5/10/20/50` 时 `4,268/26,513/39,886/57,816/86,305` 个 covered mentions，对应 `3.4371%/21.3513%/32.1208%/46.5601%/69.5027%`（denominator=`124,175` gold mentions、teacher 原始 rank 前 K 后排除 sibling GT、LETTER depth=`4`）。

top-50 条件下，Single coverage=`54,135/77,455=69.8922%`，Multi coverage=`32,170/46,720=68.8570%`。selected negative 的 teacher rank median=`12`、p90=`39`；`18,665/86,305=21.6268%` 的 gold-minus-negative teacher margin `<=0`。这些条件固定为 base source queries=`94,912`、teacher candidate limit=`50`、same-c0/different-c1。
