# Placeholder tracker — Results / Discussion / Conclusion drafts

Checklist of everything still marked `[PLACEHOLDER: ...]` or `% TODO` across
[results_section_draft.tex](paper/results_section_draft.tex),
[discussion_section_draft.tex](paper/discussion_section_draft.tex), and
[conclusion_section_draft.tex](paper/conclusion_section_draft.tex), now under
`paper/` along with the new `method.tex`, `materials.tex`, and
`introduction.tex`. Grouped by what kind of input is needed, not by file, so
you can batch similar work (e.g. run all the missing benchmarks in one
sitting).

**NEW:** `paper/introduction.tex` now has a real Related Work subsection with
actual citations and numbers — `wangGaitSeamlessKnee2025` (6.83° overall /
2.93° walk, 15 IMUs), `shahGaitSpeedTask2025` (<6°, 7 IMUs),
`sung2021prediction` (R²>0.74, RMSE<7°, 1 IMU), `thudor2025knee` (MAE 1.19°,
3.7ms, 1 IMU+GONIO), `huang2026multi` (RMSE 5.77°, R²=0.889, TCN-MHA). The
rewritten Conclusion now cites these directly in its literature-positioning
paragraph. `paper/method.tex` also confirms real citation keys
`he2022mae` (MAE pre-training) and `vaswani2017attention` (positional
encoding) — Discussion's still-placeholder `\cite{PLACEHOLDER}` for the core
MAE citation (item in §4 below) can likely reuse `he2022mae` directly instead
of needing a new lookup.

## format TODO
- [] Rewrite limitations (less bulletpointy) — now prose paragraphs in `paper/conclusion_section_draft.tex`

## 1. Numbers you need to compute / look up

- [x] **Number of LOSO folds** — `10`, one per subject. Filled into `paper/results_section_draft.tex:10`.
- [ ] **TCN overall RMSE/MAE ± SD** (Table `tab:main_results`) — still missing. `paper/results_section_draft.tex:30`
- [x] **Transformer overall RMSE/MAE** — `14.31° / 10.13°` (fold AB156, from `eval_encoder_final.json`). Filled into `paper/results_section_draft.tex` (Table `tab:main_results`, prose). **SD across folds still pending** — only one fold evaluated so far.
- [ ] **Margin between Transformer and TCN** (RMSE/MAE, with sign of which is better) — blocked on TCN's overall numbers above.
- [ ] **Significance test result** (statistic, $p$-value, test used: paired t-test vs. Wilcoxon) — `paper/results_section_draft.tex` (overall comparison subsection)
- [ ] **TCN within-subject and LOSO RMSE/MAE ± SD** (Table `tab:split_strategy`) — still missing
- [x] **Transformer LOSO MAE** — `10.13°` (fold AB156). Filled into `tab:split_strategy`.
- [x] **Transformer within-subject MAE** — `4.32°` (from `eval_encoder_persubject.json`). Filled into `tab:split_strategy`. **All cross-fold SDs still pending** (both splits currently single-run point estimates).
- [x] **Ratio: LOSO RMSE / within-subject RMSE for Transformer** — `14.31/7.03 ≈ 2.0×`. Filled into `paper/results_section_draft.tex` and `paper/discussion_section_draft.tex`.
- [ ] **TCN's own LOSO-vs-within-subject gap**, and whether it's larger/smaller relative to the Transformer's ~2× gap — `paper/discussion_section_draft.tex` (generalisation subsection)
- [x] **TCN parameter count + inference latency benchmark** — found in `inference_benchmark_tcn.json`: \num{59081} params, \SI{1.00}{ms} @ batch 1, \num{17864} samples/s @ batch 64. Filled into `paper/discussion_section_draft.tex` (Table `tab:compute_comparison` + prose) and `paper/conclusion_section_draft.tex` summary paragraph.
- [ ] **Real-time control-loop latency requirement** (the threshold your deployment context needs, in ms) — `paper/discussion_section_draft.tex` (computational considerations subsection)
- [ ] **Clinically/functionally acceptable RMSE threshold** for prosthetic/exoskeleton control, from literature — `paper/discussion_section_draft.tex` (sources-of-error subsection)
- [x] **Residual bias (LOSO)** — mean `+4.53°` (`eval_encoder_final.json`), i.e. the encoder systematically over-predicts under LOSO. Filled into `paper/results_section_draft.tex` and `paper/discussion_section_draft.tex`. (Within-subject residual / per-activity gap breakdown were cut back out — the split-strategy subsection is meant to stay a brief ablation, not a second generalisation study; see note below.)

## 2. Figures to generate

- [x] **Per-step RMSE vs. forecast horizon (LOSO)** — generated from real data (`plot_rmse_over_horizon.py` → `rmse_over_horizon.png`), Transformer only (fold AB156). TCN curve still pending.
- [x] **Per-activity RMSE bar chart (LOSO)** — (`plot_activity_rmse.py` → `activity_rmse.png`), Transformer only. New `paper/results_section_draft.tex` subsection "Error by activity". TCN bars pending.
- [ ] **Per-subject RMSE bar chart** (LOSO, both models, error bars) — `paper/results_section_draft.tex` (`fig:per_subject_rmse`) — still a placeholder box; needs multi-fold LOSO results (only fold AB156 exists so far)
- [ ] **Training/validation loss curves** (MAE pre-training + both fine-tuning phases, phase boundary marked) — `paper/results_section_draft.tex` (`fig:loss_curves`)
- [ ] **Qualitative prediction traces** (good case + bad case, ground truth vs. prediction) — `paper/results_section_draft.tex` (`fig:qualitative_examples`)
- [ ] **NEW — decide whether to use `symposium_final_rmse_plot.png`** (already exists, built by `plot_results.py`) instead of the new `activity_rmse.png`. It plots Encoder vs. a model labelled "TCN without GONIO" — see item in §5 below before using its TCN numbers anywhere.

## 3. Interpretive sentences (need the numbers/figures above first)

- [ ] Best-/worst-case subject discussion, consistency of model ranking across subjects — `paper/results_section_draft.tex` (per-subject variation subsection) — blocked on multi-fold LOSO + TCN results
- [x] Whether per-step error grows monotonically or plateaus — answered: rises to a peak of ~16.95° around 1.0-1.1s, then plateaus/declines slightly. Filled into `paper/results_section_draft.tex` and `paper/discussion_section_draft.tex`.
- [ ] Training convergence behaviour (early stopping epoch, overfitting signs) — `paper/results_section_draft.tex` (training dynamics subsection)
- [ ] Systematic error pattern in qualitative figure (phase lag, amplitude underestimation) — `paper/results_section_draft.tex` (qualitative examples subsection) — note: the residual bias (+4.53°, over-prediction) is now known at the aggregate level; this item is about the *qualitative trace* picture specifically
- [x] Whether error concentrates at specific gait phases/activity transitions — answered via per-activity breakdown: stair ascent worst, ramps best. Filled into `paper/results_section_draft.tex` (new "Error by activity" subsection) and `paper/discussion_section_draft.tex`.
- [ ] Whether MAE pre-training narrows or widens the LOSO/within-subject gap relative to TCN, and what that implies about subject-invariant representations — `paper/discussion_section_draft.tex` (generalisation subsection) — blocked on TCN's split-strategy numbers
- [ ] Per-subject failure pattern: same subjects hard for both models (outlier subject) vs. architecture-specific — `paper/discussion_section_draft.tex` (generalisation subsection) — blocked on multi-fold results
- [ ] Whether Transformer's accuracy gain justifies a ~68× larger, ~17× slower model relative to TCN (the compute gap itself is known; only the accuracy side is missing) — `paper/discussion_section_draft.tex` (computational considerations), `paper/conclusion_section_draft.tex` (summary)
- [ ] Whether the achieved error is consistent with self-supervised pretraining literature on small biomedical datasets — `paper/discussion_section_draft.tex` (related work subsection)
- [ ] Qualitative comparison to prior IMU joint-angle literature (in spirit, not exact numbers) — `paper/discussion_section_draft.tex` (related work subsection)
- [ ] Closing synthesis paragraph tying generalisation / error / compute / related-work threads together — `paper/discussion_section_draft.tex` (summary)
- [x] Conclusion summary paragraph — rewritten in `paper/conclusion_section_draft.tex` grounded in real Transformer numbers, the literature comparison, and the compute gap. TCN-side sentences remain explicit placeholders (margin, significance, split-strategy gap).
- [ ] **NEW** — confirm whether the cited prior studies (`wangGaitSeamlessKnee2025`, `shahGaitSpeedTask2025`, `sung2021prediction`, `thudor2025knee`, `huang2026multi`) use a LOSO/subject-independent split or a within-subject/mixed split. This determines whether the Conclusion's claim — that the stricter LOSO protocol (not the model) explains most of the gap to literature RMSE values — is actually defensible, or needs softening. `paper/conclusion_section_draft.tex` (Summary, closing paragraph)

## 4. Citations to find (`\cite{PLACEHOLDER}`)

- [ ] MAE / BERT-style pretraining-for-transfer paper — `paper/discussion_section_draft.tex:45`
- [ ] Clinically acceptable knee-angle RMSE threshold paper — `paper/discussion_section_draft.tex:87`
- [ ] Real-time control-loop latency requirement source — `paper/discussion_section_draft.tex:130`
- [ ] Core MAE citation (e.g. He et al. 2022) — `paper/discussion_section_draft.tex:142`
- [ ] Time-series MAE / PatchTST-style pretraining citation — `paper/discussion_section_draft.tex:144`
- [ ] Prior IMU-to-joint-angle regression papers (ideally other ENABL3S studies) — `paper/discussion_section_draft.tex:154`
- [ ] TimesNet citation — `paper/discussion_section_draft.tex:161`
- [ ] TCN (dilated convolution) citation — `paper/discussion_section_draft.tex:162`
- [ ] Self-supervised scaling-with-data-volume citation — `paper/conclusion_section_draft.tex:65`, `paper/conclusion_section_draft.tex:97`

## 5. Decisions / confirmations only you can make

- [ ] Confirm whether Limitations should be its own section/chapter, or stay folded into Discussion/Conclusion as currently drafted — `paper/discussion_section_draft.tex:8`, `paper/discussion_section_draft.tex:179`
- [x] **TimesNet exclusion confirmed structurally** — `paper/method.tex` only describes the Transformer encoder and TCN baseline; TimesNet was never implemented. The Conclusion's Limitations subsection now states this directly as a scope reduction rather than asking you to confirm it. **Remaining action: decide whether to update the working thesis title**, which still says "...against TCN and TimesNet."
- [x] Ablation subsection was removed from Results (confirmed earlier in session) — Future Work item rewritten to reference the standalone ablation configs in `conf/experiment/performance_check/` instead of the dangling `\ref{sec:results-ablation}`
- [ ] State whether the significance test was pre-registered/powered, or note the limited power of only ~10 LOSO folds — `paper/conclusion_section_draft.tex:52`
- [ ] **NEW — clarify "TCN without GONIO" labelling.** `plot_results.py` already contains per-activity RMSE numbers for a model labelled `"TCN without GONIO"` (Walk 25.09°, Ramp Up 22.69°, Ramp Down 23.47°, Stair Up 25.03°, Stair Down 25.71°) and an existing rendered figure `symposium_final_rmse_plot.png`. This label doesn't match the TCN baseline described in Methods (no mention of excluding a "GONIO"/goniometer-derived feature there). Confirm whether this is the same TCN baseline used in `tab:main_results`/`tab:split_strategy` (in which case these numbers should be incorporated directly — they'd resolve several TCN placeholders above) or a different ablation variant that shouldn't be mixed into the main comparison.

## 6. Mechanical fixes (not content, just bookkeeping)

- [ ] Add `\usepackage{multirow}` to the preamble (used in `tab:split_strategy` and `tab:compute_comparison`)
- [ ] Swap each `\fbox{\parbox...}` figure placeholder for the real `\includegraphics` line once the PNG exists (commented-out line already sits right below each box)
- [ ] Fix `Section~\ref{sec:limitations-or-conclusion}` dangling cross-reference once limitations placement is decided — `paper/discussion_section_draft.tex:179`
