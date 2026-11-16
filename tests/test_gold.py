from datetime import date

import duckdb
import pyarrow as pa
import pytest
from deltalake import DeltaTable, write_deltalake

from fire_atlas_data.gold import (
    build_map_cells, optimize_map_cells, repartition_map_cells, verify_map_cells,
)


def _rows(target):
    with duckdb.connect() as connection:
        connection.register("cells", DeltaTable(str(target)).to_pyarrow_dataset())
        return connection.execute("SELECT * EXCLUDE (month) FROM cells ORDER BY ALL").fetchall()


def _silver(days):
    return pa.table({
        "acquisition_date": days, "acquisition_year": [d.year for d in days],
        "acquisition_month": [d.month for d in days], "sensor": ["MODIS"] * len(days),
        "platform": ["Terra"] * len(days), "frp_mw": [2.0] * len(days),
        "brightness_k": [310.0] * len(days),
        "brightness_i4_k": pa.array([None] * len(days), pa.float64()),
        "confidence_category": pa.array([None] * len(days), pa.string()),
        "confidence_percent": [80.0] * len(days),
        **{f"h3_r{r}": [r + 100] * len(days) for r in range(7)},
    })


def test_repartition_preserves_rows_and_previous_snapshot(tmp_path):
    target = str(tmp_path / "gold")
    rows = pa.table({"resolution": [6, 6, 0, 0], "year": [2024, 2024, 2023, 2024],
                     "day": [date(2024, 1, 1), date(2024, 2, 29), date(2023, 12, 31), date(2024, 1, 1)],
                     "detections": [1, 2, 3, 4], "frp_sum_mw": [None, 0.0, 3.5, 4.5]})
    write_deltalake(target, rows, partition_by=["resolution", "year"])
    before = sorted(rows.to_pylist(), key=lambda r: (r["resolution"], r["year"], r["day"]))
    report = repartition_map_cells(target, threads=1)
    table = DeltaTable(target)
    with duckdb.connect() as connection:
        connection.register("cells", table.to_pyarrow_dataset())
        actual = connection.sql("SELECT * EXCLUDE (month) FROM cells ORDER BY resolution, year, day").to_arrow_table().to_pylist()
        assert actual == before
        assert connection.execute("SELECT count(*) FROM cells WHERE month <> month(day)").fetchone()[0] == 0
    assert table.metadata().partition_columns == ["resolution", "year", "month"]
    assert (report["rows"], report["partitions"], report["fingerprint_mismatches"]) == (4, 4, 0)
    assert DeltaTable(target, version=0).metadata().partition_columns == ["resolution", "year"]
    assert repartition_map_cells(target)["already_partitioned"]
    assert DeltaTable(target).version() == 1


def test_year_replacement_and_zorder_preserve_other_years(tmp_path):
    silver, target = str(tmp_path / "silver"), str(tmp_path / "gold")
    days = [date(2023, 12, 31), date(2024, 1, 1), date(2024, 2, 29), date(2024, 2, 29)]
    write_deltalake(silver, _silver(days))
    build_map_cells(silver, target, threads=1)
    verify_map_cells(target, silver)
    baseline = _rows(target)
    optimize_map_cells(target)
    assert _rows(target) == baseline
    assert optimize_map_cells(target) == {}
    # Replacing a year must also remove a month no longer in that year's input.
    write_deltalake(silver, _silver(days[:2]), mode="overwrite")
    build_map_cells(silver, target, years=[2024], threads=1)
    verify_map_cells(target, silver)
    with duckdb.connect() as connection:
        connection.register("cells", DeltaTable(target).to_pyarrow_dataset())
        assert connection.execute("SELECT year, month, sum(detections) FROM cells GROUP BY 1, 2 ORDER BY 1, 2").fetchall() == [
            (2023, 12, 7), (2024, 1, 7)]
    assert set(optimize_map_cells(target)) == {f"r{r}/2024/01" for r in range(7)}


def test_verification_rejects_monthly_errors_even_when_year_total_matches(tmp_path):
    silver, target = str(tmp_path / "silver"), str(tmp_path / "gold")
    write_deltalake(silver, _silver([date(2024, 1, 1), date(2024, 2, 1)]))
    build_map_cells(silver, target, threads=1)
    table = DeltaTable(target)
    with duckdb.connect() as connection:
        connection.register("cells", table.to_pyarrow_dataset())
        corrupted = connection.sql("SELECT * REPLACE (CASE WHEN month=1 THEN 2 ELSE 0 END::BIGINT AS detections) FROM cells").to_arrow_table()
    write_deltalake(target, corrupted, mode="overwrite", partition_by=["resolution", "year", "month"])
    with pytest.raises(RuntimeError, match="mismatches"):
        verify_map_cells(target, silver)
