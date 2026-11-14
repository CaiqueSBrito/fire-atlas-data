# Retrospective fixed-geography features

Run from the repository root:

```sh
uv run python -m fire_atlas_data.features --config configs/prototype.json --output-dir output/prototype-features
```

The output directory must not exist. Input is normalized Parquet; normalization is not modified. DuckDB uses the configured memory limit, on-disk intermediate tables and a staging-directory spill area. Python retains only file metadata, the small declared H3 grid, calendar and neighbor edges; the H3 scalar UDF maps distinct coordinate pairs rather than all detections. Intermediate/spill capacity is required in addition to outputs.

The declared box is west/south/east/north `[-48,-17,-46,-15]`, resolution 5. Candidate cells have centers inside that fixed polygon and are generated without observations. These are **bbox candidates, not land-certified cells**. A two-degree read halo supplies neighboring cells. Geometry is never learned from future positives. `candidates.parquet` records the complete declared grid. `daily.parquet` aggregates exact semantic-deduplicated detections by H3 cell, acquisition date, sensor, platform and source product, including the one-ring halo. Dedup ignores file provenance/record IDs but retains acquisition timestamp, location, source product/version, sensor/platform, confidence, radiometry, scan/track and day/night.

`features.parquet` includes every eligible cell-day (no control sampling). Issue dates are October 1–10 training, 11–15 validation and 16–20 test, 2026, after September warmup. October 22 is a partial extraction day and is not used as a complete label target. Eligibility excludes cells with NOAA-20 VIIRS detections on D, D−1 or D−2. A detection on D−3 alone does not exclude a cell. Labels are observed NOAA-20 VIIRS detections on D+1. Zero labels are **unlabeled controls**, not verified absence of fire.

All temporal features use acquisition dates at or before D. `neighbors_3d`/`neighbors_7d` count distinct neighboring cells with supplied detections over D−2…D / D−6…D, excluding the focal cell. `regional_7d`/`regional_30d` count distinct active cells in the fixed candidate grid over D−6…D / D−29…D. Repeated pixels, platforms or days in the same window do not increase a cell's contribution. `days_since_last` measures the last supplied focal-cell detection; null means no earlier record, not infinite time. These observation features are not sensor-harmonized.

Seasonal MODIS and VIIRS columns count **new cell-days**, not pixels, in years strictly before the issue year, within a circular **30-day total** target-date window [−15,+14] on a 366-day calendar (February 29 is retained). Newness requires no detection in the three preceding acquisition days for that cell, sensor and platform, across supplied source products. Same-day platform overlap is then collapsed to one cell-day per sensor family; MODIS and VIIRS remain separate. `seasonal_support_years` counts years with matching new cell-days, not years of certified observation coverage. Missing prior input near a historical boundary can left-censor novelty. Without historical records, seasonal columns are numeric zero and support is zero; the manifest explicitly reports unsupported historical baseline. This is not a validated seasonal risk estimate. `doy_sin`/`doy_cos` encode target date using that same calendar.

The manifest includes configuration and source SHA256 hashes, per-split eligible/positive/support counts, sensor/platform counts, output hashes and limitations. Acquisition min/max coverage is required, but cannot certify daily satellite coverage. The present NOAA-20 data has unknown `available_at`: this is **retrospective acquisition-time analysis, not operational backtesting**. No historical scientific validation, calibrated probabilities, land mask or independent fire-truth labels are claimed.

Optional `publish_date` uses exactly the same eligibility and feature transformation with `split='publish'` and null label. The local configuration publishes issue October 21 / target October 22 without consuming October 22 as a label; source coverage is required through D only for publication. This is an unlabeled retrospective ranking, not a current operational forecast. Metrics must exclude publication rows.

## Any-activity D+1 target (method 1.3.0)

Human-approved `configs/activity.json` sets `target_mode="any_activity"`:
every fixed-geography cell is retained, including cells active on D, D−1 or
D−2. A label of one means any supplied NOAA-20 NRT detection in D+1,
including continuation; zero remains an unlabeled control. Base/daily history
in this mode uses only the target NOAA-20 product; SNPP archive covariates stay
separate. Seasonal and prior-novel-event **features** retain their documented
definitions; they do not turn the new target back into novelty.

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.features --config configs/activity.json --output-dir output/rf10-activity/features
.venv/bin/python .omp/evidence/RF10-activity/baselines.py
```

Outputs must not already exist. The manifest declares `target_mode`,
`target_definition` and `candidate_eligibility`. Omitted `target_mode` means
`novel_activity`, preserving reproduction of the earlier experiments.
The quality trainer rejects a frozen baseline trained for a different target:
old novelty metrics/models cannot establish improvement on this target.
Existing September/October periods remain reused diagnostics, not a new holdout.
Real build: 161 cells per issue, 4,186 training rows/259 positives,
805 validation rows/24 positives, 805 diagnostic test rows/44 positives,
161 publication rows with null labels. Subsequent fixed-profile training via
`PYTHONPATH=src .venv/bin/python .omp/evidence/RF10-activity/train.py`
saved a native JSON and predictions in `output/rf10-activity/model-fixed/`.
It captured 8/24 validation and 9/44 reused diagnostic TEST positives at top10/day,
without weather/vegetation inputs, search or promotion. Reloaded JSON validation
predictions matched the in-memory model exactly. These are not independent results.

A subsequent user-fixed weighting experiment changes only `scale_pos_weight`
to 3.9, keeping these same 40 features, all training rows and every other
parameter. Run `PYTHONPATH=src .venv/bin/python .omp/evidence/RF10-activity/train-weighted.py`
with a new output location configured in the script if the preserved output
already exists. Artifacts: `output/rf10-activity/model-weight-3_9/`.
Observed capture is unchanged: 8/24 validation and 9/44 diagnostic TEST,
with exact native-JSON reload verification and 161 null-label inference rows
per model. Validation macro recall ties, so the baseline remains selected.
No automatic promotion or independent gain claim.

The requested no-external-enrichment ablation removes all eight SNPP columns
from that weighted model, retaining 32 NOAA-20-derived/calendar/location
predictors and `scale_pos_weight=3.9`. It uses the identical rows and splits;
the two unavailable seasonal columns were verified to be zero. FIRMS lags
and FRP remain, but ERA5, GFS, vegetation and SNPP values are absent.
Run `PYTHONPATH=src .venv/bin/python .omp/evidence/RF10-activity/train-noaa20-only.py`.
Preserved artifacts and the projected dataset are in
`output/rf10-activity/model-noaa20-only-3_9/`.
Observed validation capture increased from 8/24 to 9/24 and diagnostic TEST
capture from 9/44 to 10/44 (recall22.73%,precision20%). Validation chose the
NOAA-20-only candidate before TEST; exact reload equality and 161 null-label
inference rows per model were exercised. This small reused-test improvement
is not independent generalization evidence or operational promotion.

A subsequent user-requested train-only undersampling experiment retains all
259 training positives and selects 259 of 3,927 controls without replacement
by a stable seed-42 hash. It retains the 32-feature model, weight3.9 and all
other parameters; validation and TEST retain all 805 rows each. Run
`PYTHONPATH=src .venv/bin/python .omp/evidence/RF10-activity/train-undersampled.py`.
Artifacts, sample and per-day counts:
`output/rf10-activity/model-undersampled-1_1/`.
Validation capture ties at 9/24; diagnostic TEST drops from 10/44 to 5/44
(recall11.36%,precision10%). The full-training reference is retained by the
validation tie policy. Exact native reload equality and 161 null-label
inference candidates per model were verified. No promotion, ratio search,
calibrated-probability claim or independent generalization claim.




Technical regression coverage includes continuing versus new activity, strict
NOAA-20 labels, future-label isolation, invalid targets and cross-target model
comparison rejection. Consolidated run: 99 tests passed. Protocol and evidence:
[RF10-atividade-d1](../.omp/tasks/RF10-atividade-d1.md).


## Opt-in recurrence experiment

```sh
uv run python -m fire_atlas_data.features --config configs/quality.json --output-dir output/quality-features
```

This separate configuration leaves `configs/prototype.json` and original outputs unchanged. It keeps the same bbox, H3 resolution 5, candidate IDs, NOAA-20 D/D−1/D−2 eligibility rule, next-day labels and publication date. Training issue dates extend to September 15–October 10, with September 1 as warmup. Validation remains October 11–15 and test remains October 16–20. The previously inspected test period is **diagnostic, not independent evidence**; model selection must use validation only. Top-10 ranking and target geography are not changed.

`enhanced_features` defaults to false. `minimum_history_days` defaults to 30 and must be an integer at least 7; the first issue date must be at least that many days after `warmup_start`. The enhanced configuration deliberately uses 14. `history_days=min(30, D−warmup_start+1)` records inclusive calendar exposure, not verified observation coverage. Thus September 15 has 15 history days and its 30-day features are explicitly left-censored. The manifest records the requested minimum and the number of rows with `history_days<30`. Elapsed times and all-history novel-event totals can remain left-censored even after this 30-day counter saturates.

The original base columns retain their definitions, including any supplied non-target platforms. Every **additional** recurrence/FRP column uses only `sensor=VIIRS`, `platform=NOAA-20`, `source_product=VIIRS_NOAA20_NRT`. All windows end on issue date D; only the label uses D+1. No weather, other source, synthetic prior-year observation or future FRP is introduced.

The module exports ordered `ENHANCED_FEATURES`: the existing full model columns (`seasonal_modis`, `seasonal_viirs`, `doy_sin`, `doy_cos`, `neighbors_3d`, `neighbors_7d`, `days_since_last`, `regional_7d`, `regional_30d`) followed by these additions:

| Columns | Definition |
| --- | --- |
| `cell_active_days_7d`, `cell_active_days_14d`, `cell_active_days_30d` | Distinct acquisition dates with target detections in the focal cell over the inclusive trailing window. |
| `cell_detections_7d`, `cell_detections_30d` | Counts of target detections after the existing exact semantic deduplication, not active-day counts. |
| `cell_prior_novel_events` | Supplied target active cell-days through D with no target detection on the preceding three dates. It is an all-supplied-history total, not a 30-day total; missing pre-source antecedents can censor novelty. |
| `ring1_active_cells_1d`, `ring1_active_cells_3d`, `ring1_active_cells_7d`, `ring1_active_cells_14d`, `ring1_active_cells_30d` | Distinct active cells at H3 distance one, excluding the focal cell; each cell contributes at most one regardless of days or pixels. |
| `ring2_active_cells_3d`, `ring2_active_cells_7d`, `ring2_active_cells_14d`, `ring2_active_cells_30d` | Distinct active cells in the H3 radius-two **disk excluding the focal cell**. This includes distance-one neighbors; it is not the distance-exactly-two annulus. |
| `days_since_neighbor` | Days since the last supplied target detection in that radius-two disk, through D. Null means none is supplied. |
| `nearby_frp_max_3d`, `nearby_frp_mean_3d`, `nearby_frp_max_7d`, `nearby_frp_mean_7d` | Maximum/arithmetic mean supplied non-null detection-level FRP in MW in the same radius-two disk. Null FRP is ignored; no observations or all-null FRP yields null, while observed zero FRP remains numeric zero. |
| `latitude`, `longitude` | Fixed H3 candidate-center coordinates for this known-region forecast, not a claim of out-of-region transfer. |
| `history_days` | Inclusive calendar days from configured warmup through D, capped at 30 as described above. |

The enhanced daily output also retains the radius-two halo, including neighbors outside the candidate polygon. Candidate geometry is unchanged, and base one-ring counts still use only one-ring edges. Count features are zero when no supplied activity exists; missing elapsed-time/FRP features remain null. Prior-year seasonal columns remain zero with zero support when no prior-year input exists, rather than inventing a seasonal baseline. The manifest adds `enhanced_features`, ordered `feature_columns`, enhanced units and a `history` record; its enhanced method version is `1.1.0`.

## Optional Suomi-NPP archive covariates (method 1.2.0)

`snpp_history: true` enables the enhanced logical mode plus `HISTORY_FEATURES`,
the unchanged ordered 32 `ENHANCED_FEATURES` followed by eight `DOUBLE` columns:

| Column | Definition over all supplied acquisitions on or before issue D |
| --- | --- |
| `snpp_cell_active_days` | Number of focal-cell active dates. |
| `snpp_cell_new_days` | Active dates without activity on the three preceding dates. |
| `snpp_cell_detections` | Semantically deduplicated focal-cell detection total. |
| `snpp_days_since_last` | Elapsed days since focal-cell activity; nullable. |
| `snpp_ring1_active_cell_days` | Neighbor/date pairs in disk one excluding the focal cell. |
| `snpp_ring2_active_cell_days` | Neighbor/date pairs in disk two, including disk one and excluding the focal cell. |
| `snpp_cell_frp_mean` | Mean non-null focal-cell detection FRP in MW; nullable. |
| `snpp_cell_frp_max` | Maximum non-null focal-cell detection FRP in MW; nullable. |

Only `sensor=VIIRS`, `platform=Suomi-NPP`, `source_product=VIIRS_SNPP_SP`
feeds these columns. All supplied thermal-anomaly types are retained without filtering.
Count zeros differ from unknown elapsed time and missing FRP. Archive history is not
truncated at the NOAA-20 September warmup. First observed events are left-censored.
In this mode existing daily/base recurrence uses only NOAA-20 NRT, preventing archive
additions from filling previously null `days_since_last` or changing other baseline
columns. NOAA-20 eligibility and D+1 labels are unchanged. Without this option,
existing baseline/enhanced behavior remains unchanged. Unknown `available_at` means
this is retrospective acquisition-time history, not an operational-availability claim.

