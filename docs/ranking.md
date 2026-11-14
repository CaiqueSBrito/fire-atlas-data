# Retrospective cell rankings

This is an executable exploratory ranking pipeline, not a calibrated fire-probability system or an operational forecast. It consumes the feature builder's fixed, future-independent eligible H3 candidates. A positive means a **new observed detection** under the feature builder's definition; a zero is an unlabeled control, not evidence of no fire. NOAA20-only, short-window labels and missing `available_at` preclude claims about deployment latency or scientific generalization.

```sh
uv run python -m fire_atlas_data.ranking train \
  --features output/features/features.parquet --output-dir output/model --k 10
uv run python -m fire_atlas_data.ranking publish \
  --features output/features/features.parquet --model-dir output/model \
  --output-dir output/ranking
```

The CLI default is K=100; use the explicitly exploratory K=10 for the small local bounding box to avoid trivially selecting almost every candidate. Neither bounding box nor K should be optimized against test outcomes.

## Consumer contract

Parquet input may be one file or a directory recursively containing Parquet files. Required columns:

- `cell_id` (H3 string), `issue_date` and `target_date` (DATE; target is issue + one day), `region` (string).
- Numerical `seasonal_modis`, `seasonal_viirs`, `doy_sin`, `doy_cos`, `neighbors_3d`, `neighbors_7d`, nullable `days_since_last`, `regional_7d`, `regional_30d`.
- `label`: 1 for observed target detection, 0 for unlabeled control; null allowed in `publish` only.
- `split`: `train`, `validation`, `test`, optionally `publish`.

Each issue-date/cell pair is unique. Training requires three nonempty, strictly chronological, nonoverlapping splits. Full candidate sets must be preserved in validation/test. The builder, not this consumer, establishes candidate geometry without future detections and excludes incompletely observed target days. The partial October 22 acquisition must not be evaluated as a complete target day.

Publication prefers `publish` rows if present and otherwise publishes `test`. A file containing only publish candidates with unknown labels is accepted; no next-day observation is needed to score. Publish rows never enter sampling, selection or evaluation. A historical publish date remains historical, even if its target labels are unknown.

## Sampling and model selection

All training positives are kept. At most ten controls per positive **per date-region** are selected using a deterministic SHA-256 ordering with seed 42 (configurable). Training groups without positives contribute no controls. Validation and test are never downsampled.

Two modest CPU XGBoost classifiers are saved: `seasonal_only.json` uses only the two seasonal feature columns; `full.json` uses the full listed numerical schema, in that exact order. Both use 120 trees, depth 3, learning rate 0.05, histogram trees and two CPU threads. Models are reloaded from JSON for scoring, with file hashes, schema order and feature count checked. Scores are uncalibrated ordering values even though the classifier internally exposes a logistic output.

Model selection maximizes mean daily validation recall@K over days with observed positives; ties select seasonal-only. A validation split without positives is an explicit error, not permission to use test for selection. The selected model is not refitted on validation. Test evaluation is independent and uses both models on exactly the same complete eligible candidates. Ties in rankings use ascending `cell_id`.

Daily evaluation and publication budgets are independent per `(region, issue_date)`; comparison rows include region and calendar month for disaggregation. Training and publication reject an already-existing output directory instead of overwriting artifacts.

`comparison.json` contains daily candidate count, observed positive count, hits, configured K, actual selected count, precision (hits/selected) and recall (hits/observed positives). No-positive days have null recall; precision remains defined when candidates exist. A direct seasonal-column-sum reference is reported on the same candidates. **No previous-years seasonal baseline is supported by the available local data.** All-zero seasonal features produce a deterministic tie ranking, not a meaningful historical baseline; neither this comparison nor a full-model advantage establishes a scientific win.

## Artifacts

- `manifest.json`: model choice, configuration, feature order, seed, package versions, input SHA-256 hashes, model hashes and a manifest-derived model version. Includes the feature builder's sibling manifest when present, its original configuration hash, and ranking/feature code hashes. Publication checks the feature method version, complete feature order, non-temporal configuration (including grid/region), and code hashes; source files and split/acquisition dates may differ for future scoring. Without a sibling manifest, configuration provenance is explicitly null; production consumers should retain the builder manifest.
- `comparison.json`: validation/test daily results and limitations.
- `backtest_predictions.parquet`: every held-out TEST candidate, not just top K: `cell_id`, `region`, `issue_date`, `target_date`, observed/unlabeled `label`, selected-model `score`, regional daily `rank`, `selected_top_k`, `model_version`, `model_name`. These rows preserve occurred outcomes for auditing the untouched test prediction; they never participate in fitting or model selection. Manifest input hashes identify the feature dataset and `backtest_output` records the export hash after model-version assignment, avoiding a circular self-hash.
- `rankings.parquet`: top K per region and issue date in the published partition, with `region`, `cell_id`, timezone-aware `issued_at_utc`, DATE `target_date_utc`, `score`, `rank`, `model_version`.
- `publication.json`: actual generation time, partition and historical-not-live flag.
- `map.html`: self-contained Portuguese SVG map with actual H3 polygons and embedded ranking rows, no external JavaScript, tile server, app or API key. It displays the latest target date only; the Parquet contains every scored issue date. Polygon hover shows cell, rank and score. Coordinates are geographic with north up, not a basemap or area-preserving projection.

`issued_at_utc` is the **nominal historical end of issue day**, not an assertion that the data/model existed then. The map separately displays real artifact generation time and marks all outputs retrospective/non-operational. Local data from October 2026 must not be presented as a current forecast in November.

## Checks

`tests/test_ranking.py` exercises no-positive sampling, deterministic controls and score ties, null recurrence encoding, schema-order rejection, temporal split rejection, real JSON training/reload, held-out comparison, unlabeled publication and self-contained H3 map output. Run it as part of the integrator's consolidated test pass.
