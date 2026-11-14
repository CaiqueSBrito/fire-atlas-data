# Public NOAA-20 standard-processing history

The historical CLI downloads the actual public [FIRMS country archive](https://firms.modaps.eosdis.nasa.gov/country/) Brazil files, selected by exact filename and byte size from the [yearly inventory](https://firms.modaps.eosdis.nasa.gov/data/country/yearly_summary_files.txt). It does not download global ZIP archives or require an API key. The archive [attribute README](https://firms.modaps.eosdis.nasa.gov/data/country/AttributesReadme.txt) is preserved with its receipt; it directs users to the source version field to distinguish standard processing from NRT.

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.history acquire \
  --output-dir output/rf10-environment/history/acquisition

PYTHONPATH=src .venv/bin/python -m fire_atlas_data.history prepare \
  --input-dir output/rf10-environment/history/acquisition \
  --output-dir output/rf10-environment/history/prepared \
  --memory-limit 512MB --threads 2
```

`acquire` defaults to all eight years 2018–2025. `--years 2020 2021` selects a subset; other years are refused. The default `--max-bytes 1000000000` bounds the sum of selected inventory sizes, and each streamed request is independently capped at its exact inventory size. Disk capacity must cover three times the selection plus 128 MiB before country downloads start. Metadata requests are capped at 2 MiB (inventory) and 64 KiB (README). No request searches the reserved future target interval.

The exact public paths are `https://firms.modaps.eosdis.nasa.gov/data/country/viirs-jpss1/YEAR/viirs-jpss1_YEAR_Brazil.csv`, with filenames first checked against the downloaded inventory. Complete CSV bytes are retained in `acquisition/raw/`, together with immutable-by-default, read-only receipt files containing source/final URLs, HTTP status and headers, actual receipt timestamps, byte counts and SHA256 hashes. Inventory and README bytes and receipts are also retained. Existing preserved files are never overwritten: a resumed acquisition must match their receipts and the preserved inventory. A different inventory snapshot or year selection belongs in a new acquisition directory. A failed partial request is not published as a raw source; interrupted `.part` files are not reused. An interrupted publication with only a raw file or only a receipt is refused, not silently repaired or backdated.

## Separate schema, not normalization 1.1

Prepared data uses **`historical-observations/1.0.0`**, not the project's normalization schema. `VIIRS_NOAA20_SP` and standard processing are explicitly declared provenance, not an assertion that SP observations have the NRT target's availability or processing vintage. This command neither invokes the NRT normalizer nor changes target definitions, features, labels or training datasets.

Preparation uses DuckDB with a configured memory limit, disk-backed annual staging and spill. Python reads only CSV headers and metadata; it does not accumulate observation rows. One annual Brazil source is parsed at a time with explicit VARCHAR source types and strict CSV parsing. Source-record ordinals are stable because the CSV scan is serial and insertion order is preserved. Geographic output includes both edges of longitude/latitude bounds **[-50, -19, -44, -13]**, the two-degree halo around core bounds [-48, -17, -46, -15] in EPSG:4326.

Each annual `observations.parquet` contains:

- Separate historical schema, declared product and source-file year.
- Source URL, raw hash, source-record ordinal and actual receipt time.
- Parsed coordinates, acquisition date, zero-padded source HHMM time and acquisition timestamp in UTC.
- Checked `N20` satellite source and derived `NOAA-20` platform; source instrument and source version only when available.
- Source categorical confidence (`l`, `n`, `h`), FRP in MW, scan/track in km and VIIRS I4/I5 brightness in K when present.
- Original day/night and detection-type fields when present, plus a `source_fields` struct preserving **every** original CSV field, including extra columns.

Invalid coordinates, acquisition datetimes or file-year mismatches, platform/instrument mismatches, explicit NRT versions, invalid categorical confidence and invalid numeric field bounds go to annual `rejected.parquet` files with original fields and a first-failure reason. Required observation fields are not fabricated. Optional absent metadata remains null and is explicitly marked absent in the manifest. No rows are silently confidence-filtered, deduplicated or restricted to vegetation-fire `type=0`; source `type` is preserved for a future explicit scientific decision. Geographically excluded valid rows remain recoverable from the complete raw Brazil CSVs.

`prepared/manifest.json` records per-year Brazil source, valid, rejected, regional and geographically excluded counts; exact reconciliation (`raw = rejected + region + outside`); Brazil/regional acquisition ranges; all source column names and explicit SQL types; output column names/types; available metadata fields; observed instrument/platform/version/type/confidence/daynight distributions; raw hashes and sizes; and each Parquet hash and size. Counts refer to **Brazil country files, not the world**. They cannot establish global sensor coverage.

## Scientific and provenance limits

SHA256 and HTTPS receipts establish local byte preservation and retrieval provenance, not NASA authenticity proof. Receipt timestamps and inventory `updated` text are not historical observation availability; `available_at` is deliberately not invented. Detection points contain no observability mask, and absence of detections is not a no-fire label. Annual spans do not establish uninterrupted satellite coverage; the initial 2018 year and any incomplete later year need explicit coverage treatment. Seasonal representation, instrument commissioning, source processing and SP/NRT differences remain scientific limitations. This archive must not be merged into the NOAA-20 NRT target or a fixed-profile experiment without an explicit scientifically compatible contract.

Offline behavior tests are in `tests/test_history.py`. They exercise schema separation, source preservation, real field semantics, platform/date/numeric bounds, rejection reconciliation, optional metadata and exact inventory selection. They are not evidence of live NASA access; live download receipts and prepared manifests provide that separate retrieval evidence.

## Preserved acquisition in this checkout

The actual public download selected **885,498,520 bytes**, below the 1 GB cap, with approximately 169 GB free before acquisition. All eight files matched the inventory byte sizes. Receipts retain the actual retrieval clock (2026-11-13 UTC), without backdating it to observation dates or the task date. No future labels were acquired.

The completed 512 MB/two-thread preparation produced **10,830,346 Brazil source records**, **504,665 valid halo observations**, **10,325,637 valid records outside the halo**, and **44 rejected records**. The rejections are 38 invalid FRP values and six invalid I4 brightness values; their original source fields remain in rejected Parquet and the complete raw files.

| Year | Brazil raw records | Valid halo records | Rejected | Brazil acquisition span |
| --- | ---: | ---: | ---: | --- |
| 2018 | 849,957 | 32,998 | 0 | April 1–December 31 |
| 2019 | 1,434,146 | 78,961 | 0 | January 1–December 31 |
| 2020 | 1,633,648 | 61,002 | 15 | January 1–December 31 |
| 2021 | 1,304,753 | 91,541 | 0 | January 1–December 31 |
| 2022 | 1,452,881 | 66,236 | 15 | January 1–December 31 |
| 2023 | 1,312,648 | 47,271 | 0 | January 1–December 31 |
| 2024 | 1,988,404 | 50,051 | 14 | January 1–December 31 |
| 2025 | 853,909 | 76,605 | 0 | January 1–December 31 |

All observed rows declare `satellite=N20`, `instrument=VIIRS` and `version=2`; source detection types are 0, 2 and 3, all retained. The 2025 source spans the full calendar year, whereas 2018 begins April 1; neither fact establishes uninterrupted daily coverage.

Full source hashes, exact byte sizes, HTTP headers and receipt times are in `output/rf10-environment/history/acquisition/acquisition.json` and adjacent individual receipts. Full Parquet hashes, per-year ranges, column types, distributions and count reconciliation are in `output/rf10-environment/history/prepared/manifest.json`. The CSV raw files are under `acquisition/raw/`; each prepared year's observations and rejections are under `prepared/year=YEAR/`. These are local data artifacts, not an assertion that ignored output files are version-controlled.

## Daily labeled grid and seven-day lags

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.historical_grid \
  --history output/rf10-environment/history/prepared \
  --candidates output/rf10-activity/features/candidates.parquet \
  --horizon-days 1 \
  --output-dir output/rf10-environment/history/grid
```

The executed grid retains the 161 fixed H3 resolution-5 cells on every issue
date from 2018-04-01 through 2025-12-30, with next-day targets through 2025-12-31:
455,791 cell-days and 15,279 positive cell-days. A one means at least one
supplied NOAA-20 SP detection row; zero means no supplied detection, with
physical fire status remaining unknown. It contains current-D detection-row
counts and trailing D−6…D cell/regional detection-row counts. Source rows are
not deduplicated or interpreted as independent ignitions. All labels and lags
stay in the separate SP contract; no automatic training split or NRT merge.

Initial lag windows are left-censored. The dataset is not yet enriched with
environmental covariates for all eight years. The paired 2026 experiment uses
only its original NRT population; see [environmental](environmental.md).

## Current objective: any detection in the next seven days

The human-approved target is now any supplied NOAA-20 SP detection in the
inclusive window D+1…D+7, including continuation. A detection only on D
does not count; D+7 counts and D+8 does not. Labels refer to observed thermal
anomalies, not confirmed physical fire absence or independent ignitions.

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.historical_grid \
  --history output/rf10-environment/history/prepared \
  --candidates output/rf10-activity/features/candidates.parquet \
  --horizon-days 7 \
  --output-dir output/rf10-weekly/grid
```

Both the API and CLI default to horizon7. Horizon1 remains explicit for
reproducing the earlier task; existing artifacts are never rewritten.
With no `--end-date`, the final issue is source calendar cutoff minus the
horizon: 2025-12-24 for seven days, or 2025-12-30 for one day.
An explicitly requested incomplete future window is rejected before output.
The source snapshot is fixed to the acquired 2018-04-01…2025-12-31 archive;
this does not certify uninterrupted satellite coverage.

New outputs use `historical-cell-grid/2.0.0`, with `target_start_date`,
`target_end_date` and `horizon_days` instead of an ambiguous `target_date`.
`target_detection_rows`, `target_active_days`, `label` and observation status
are **outcomes**, never predictors. The manifest explicitly lists predictor
and outcome columns; old D+1 training contracts must not be reused silently.
Current predictor counts end on D: detection rows on D and trailing-seven-day
cell/regional counts. Missing observability remains unknown.

Executed result: **454,825 cell-days, 68,706 positive windows and 386,119 zeros**,
161 cells over 2,825 issue dates. Six formerly valid D+1 issue dates
(966 cell-days) are excluded at the tail. These are overlapping positive
windows, not 68,706 separate fires. No new model has been trained.

Smoke: actual CLI completed in2.57s. An independent rolling sum of the preserved
D+1 labels and target counts matched every seven-day label, active-day count
and target detection count; predictor lags matched the old dataset exactly.
216 tests passed in16.47s, including inclusive boundaries, exclusion of D/D+8,
future-label isolation and incomplete-tail rejection.

Before fitting, freeze temporal folds with target-window purging so that
training outcomes do not cross validation/test boundaries. Do not randomly
split overlapping daily windows or directly compare D+1 capture percentages
with seven-day metrics. Utility thresholds, calibration and scientific
acceptance are not approved by this target change.
Protocol and evidence: [seven-day dossier](../.omp/tasks/RF10-atividade-7dias.md).


