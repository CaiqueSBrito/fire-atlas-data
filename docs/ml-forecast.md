# Previsão diária com o modelo XGBoost (RF10/CA08)

`ml-forecast` aplica o `model.ubj` gerado por `ml-train` a todas as células H3 res6 do universo para uma data de emissão D e grava a probabilidade de pelo menos uma detecção FIRMS em `[D 09:00 UTC, D+7 09:00 UTC)`.

```bash
LAKE=s3://data-lake-fire-atlas-553607359608-us-east-2-an
PYTHONPATH=src python -m fire_atlas_data ml-forecast \
  --silver $LAKE/prata/silver \
  --model $LAKE/models/xgb-rf10/model.ubj --report $LAKE/models/xgb-rf10/report.json \
  --target $LAKE/gold/forecast --date 2026-11-13
```

- Features: `ml_dataset.features_sql`, o mesmo SQL das linhas de treino, só com detecções adquiridas antes do corte de D.
- Universo: células com detecção não offshore antes do corte. Célula fora do universo não tem previsão, o que não significa risco zero.
- Recusa: o comando falha se o registro mais novo da silver estiver mais de `--max-input-lag-hours` (padrão 6) antes do corte.
- `--report`: confere se as features e a versão do dataset são as do treino.
- Saída: Delta particionado por `year`/`month`, com `issue_date`, `h3_r6`, `h3_r1`, as 9 features, `active` e `prob_fire_7d`. Rodar de novo a mesma data substitui só as linhas dessa data.
- Metadados do commit: `model_sha256`, `silver_version`, `silver_last`, `cutoff_hour`, `input_lag_hours`.
- Execução diária agendada (busca FIRMS, append na silver e previsão) e container: [daily-ingest](daily-ingest.md).
- Continente, país e cidade de cada célula: join de `h3_r6` com `gold/h3_places` (`resolution = 6`), ver [gold-places](gold-places.md).

## Uso por um backend

Decisão do usuário em 14/11/2026: o backend recebe só o resultado da previsão (`gold/forecast` no S3), não o modelo. Exemplo com DuckDB, com credenciais AWS de leitura:

```python
import duckdb
from deltalake import DeltaTable

con = duckdb.connect()
con.register("forecast", DeltaTable("s3://data-lake-fire-atlas-553607359608-us-east-2-an/gold/forecast").to_pyarrow_dataset())
cells = con.execute("""
    SELECT h3_r6, prob_fire_7d FROM forecast
    WHERE issue_date = (SELECT max(issue_date) FROM forecast) AND active AND prob_fire_7d >= ?
""", [0.7]).fetchall()
```

O limiar (`0.7` no exemplo) não está aprovado: os limiares de aceite do CA08 continuam pendentes, e a previsão é experimental.

Smoke em 2026-11-14 (silver v632): D=2026-11-13 levou 22 s com pico de 3,3 GB de RAM e gerou 2.217.804 células. Para D=2026-10-01, as features e as probabilidades coincidiram com `gold/ml_dataset`.
