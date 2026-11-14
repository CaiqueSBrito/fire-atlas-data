"""Daily NOAA-20 SP cell grid with an explicit future detection window."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import tempfile

import duckdb
import h3

from .normalization import literal, sha256
from .ranking import _json

SOURCE_START = date(2018, 4, 1)
SOURCE_END = date(2025, 12, 31)


def _coordinate_cell(latitude, longitude):
    return h3.latlng_to_cell(latitude, longitude, 5)

def _sources(directory):
    directory=Path(directory).resolve(strict=True)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema')!='historical-observations/1.0.0' or manifest.get('product_declared')!='VIIRS_NOAA20_SP':
        raise ValueError('Require separate scientific NOAA-20 history')
    paths=[]
    for item in manifest['outputs']:
        path=directory/item['path']
        if sha256(path)!=item['sha256']:
            raise ValueError('Historical output hash mismatch')
        if path.name=='observations.parquet':
            paths.append(path)
    if not paths:
        raise ValueError('No historical observations')
    return paths,manifest


def build(history,candidates,start_date,end_date,output_dir,horizon_days=7):
    """Create all cell-days; past lags end at D, labels use D+1..D+horizon."""
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days not in (1, 7):
        raise ValueError('horizon_days must be 1 or 7')
    latest_issue = SOURCE_END - timedelta(days=horizon_days)
    start = date.fromisoformat(start_date)
    stop = date.fromisoformat(end_date) if end_date is not None else latest_issue
    if stop < start or start < SOURCE_START or stop > latest_issue:
        raise ValueError(f'Historical issues must lie in {SOURCE_START}..{latest_issue} for horizon {horizon_days}')
    paths,provenance=_sources(history)
    candidates=Path(candidates).resolve(strict=True);target=Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=f'.{target.name}-',dir=target.parent))
    try:
        with duckdb.connect(str(stage/'work.duckdb')) as con:
            con.execute("SET memory_limit='512MB'");con.execute('SET threads=2')
            con.read_parquet(str(candidates)).create_view('candidates')
            cells=[x[0] for x in con.execute('SELECT cell_id FROM candidates').fetchall()]
            if not cells or len(set(cells))!=len(cells) or any(h3.get_resolution(c)!=5 for c in cells):
                raise ValueError('Require unique fixed H3 resolution5 candidate cells')
            con.read_parquet([str(p) for p in paths],union_by_name=True).create_view('source')
            con.create_function('coordinate_cell', _coordinate_cell, ['DOUBLE','DOUBLE'], 'VARCHAR')
            con.execute('''CREATE TABLE coordinates AS SELECT latitude,longitude,coordinate_cell(latitude,longitude) cell_id
                FROM (SELECT DISTINCT latitude,longitude FROM source)''')
            con.execute('''CREATE TABLE daily AS SELECT c.cell_id,s.acquisition_date,count(*)::BIGINT detections
                FROM source s JOIN coordinates c USING(latitude,longitude)
                JOIN candidates g USING(cell_id) GROUP BY c.cell_id,s.acquisition_date''')
            con.execute(f'''CREATE TABLE calendar AS SELECT issue_date::DATE issue_date,
                issue_date::DATE+1 target_start_date,
                issue_date::DATE+{horizon_days} target_end_date,{horizon_days}::INTEGER horizon_days
                FROM generate_series(DATE {literal(str(start))},DATE {literal(str(stop))},INTERVAL 1 DAY) dates(issue_date)''')
            # Duplicate records are preserved by the source contract: counts are supplied rows, not independent fires.
            con.execute('''CREATE TABLE panel AS SELECT c.cell_id,t.issue_date,t.target_start_date,t.target_end_date,t.horizon_days,
                CASE WHEN y.detections>0 THEN 1 ELSE 0 END::INTEGER label,
                coalesce(y.detections,0)::BIGINT target_detection_rows,
                y.active_days::INTEGER target_active_days,
                coalesce(today.detections,0)::BIGINT detection_rows_on_issue,
                coalesce(lag.n,0)::BIGINT cell_detection_rows_7d,
                coalesce(region.n,0)::BIGINT regional_detection_rows_7d,
                CASE WHEN y.detections>0 THEN 'detected' ELSE 'unknown' END fire_observation_status
                FROM candidates c CROSS JOIN calendar t
                LEFT JOIN LATERAL(SELECT sum(detections) detections,count(*) active_days FROM daily d
                    WHERE d.cell_id=c.cell_id AND d.acquisition_date BETWEEN t.target_start_date AND t.target_end_date) y ON true
                LEFT JOIN daily today ON today.cell_id=c.cell_id AND today.acquisition_date=t.issue_date
                LEFT JOIN LATERAL(SELECT sum(detections) n FROM daily d
                    WHERE d.cell_id=c.cell_id AND d.acquisition_date BETWEEN t.issue_date-6 AND t.issue_date) lag ON true
                LEFT JOIN LATERAL(SELECT sum(detections) n FROM daily d
                    WHERE d.acquisition_date BETWEEN t.issue_date-6 AND t.issue_date) region ON true''')
            count,positives=con.execute('SELECT count(*),sum(label) FROM panel').fetchall()[0]
            expected=len(cells)*((stop-start).days+1)
            if count!=expected:
                raise ValueError('Grid population reconciliation failed')
            annual=con.execute('SELECT year(target_start_date),count(*),sum(label),sum(target_detection_rows) FROM panel GROUP BY 1 ORDER BY 1').fetchall()
            con.execute(f"COPY (SELECT * FROM panel ORDER BY issue_date,cell_id) TO {literal(str(stage/'grid.parquet'))} (FORMAT PARQUET,COMPRESSION ZSTD)")
        (stage/'work.duckdb').unlink()
        manifest=dict(schema='historical-cell-grid/2.0.0',target_product='VIIRS_NOAA20_SP',
                      label_definition=f'1 any supplied NOAA20 SP detection on D+1..D+{horizon_days} inclusive;0 otherwise, not verified absence of fire',
                      target_mode='any_activity_window',horizon_days=horizon_days,timezone='UTC',
                      h3_resolution=5,candidates_sha256=sha256(candidates),
                      issue_dates=[str(start),str(stop)],
                      target_start_dates=[str(start+timedelta(days=1)),str(stop+timedelta(days=1))],
                      target_end_dates=[str(start+timedelta(days=horizon_days)),str(stop+timedelta(days=horizon_days))],
                      source_calendar_cutoff=str(SOURCE_END),
                      latest_complete_issue_date=str(latest_issue),
                      feature_cutoff='All predictor counts end at issue D; target counts/dates/label/status are outcomes, not predictors.',
                      predictor_columns=['detection_rows_on_issue','cell_detection_rows_7d','regional_detection_rows_7d'],
                      outcome_columns=['label','target_detection_rows','target_active_days','fire_observation_status'],
                      cells=len(cells),rows=count,positives=positives,
                      annual_counts=[dict(target_year=y,rows=n,positives=p,detection_rows=d) for y,n,p,d in annual],
                      history_manifest_sha256=sha256(Path(history)/'manifest.json'),
                      history_counts=provenance['counts'],code_sha256=sha256(Path(__file__)),
                      outputs=[dict(path='grid.parquet',sha256=sha256(stage/'grid.parquet'))],
                      limitations=['SP is separate from NRT; this grid is not merged into the 2026 paired experiment.',
                                   'No observability mask; all zeros retain unknown physical fire status.',
                                   'Initial 7day windows are left-censored; source span does not certify coverage.',
                                   'Counts preserve supplied rows; no deduplication or event/ignition interpretation.',
                                   'Daily target windows overlap; temporal splits must purge cross-boundary target windows before fitting.',
                                   'No training splits, independent evaluation or scientific acceptance claimed.'])
        _json(stage/'manifest.json',manifest);stage.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',required=True);parser.add_argument('--candidates',required=True)
    parser.add_argument('--start-date',default='2018-04-01');parser.add_argument('--end-date',default=None)
    parser.add_argument('--horizon-days',type=int,choices=(1,7),default=7)
    parser.add_argument('--output-dir',required=True)
    print(json.dumps(build(**vars(parser.parse_args(argv))),indent=2))


if __name__=='__main__':
    main()
