# Fire Atlas Data

Pipeline de dados do Fire Atlas para detecções NASA FIRMS. O fluxo é: normalização dos CSVs, lake Delta no S3 (bronze/prata/gold), harmonização MODIS/VIIRS para o calendário (RF07/CA07) e previsão XGBoost da probabilidade de alguma detecção por célula H3 res6 em `[D 09:00 UTC, D+7 09:00 UTC)` (RF10/CA08). Um job diário na AWS atualiza a silver, a previsão, a harmonização e a localização das células. Requisitos e processo ficam no [harness `.omp/`](.omp/README.md).

Nenhum resultado aqui tem aceite científico: os limiares de CA07 e CA08 não foram aprovados, as probabilidades são experimentais e ausência de detecção não prova ausência de fogo.

## Escala

A implementação é vetorizada com **DuckDB out-of-core**. O banco intermediário fica em disco e as saídas são Parquet/Delta particionadas. Nenhum CSV é materializado inteiro em pandas ou listas Python. Memória, threads e spill são configuráveis por comando.

`memory_limit` controla a memória gerenciada pelo DuckDB, não o RSS total. Capacidade para bilhões de registros depende de disco, memória e particionamento; os tempos e picos medidos estão em cada doc. A normalização publica por execução completa. Silver, `ml-dataset` e `harmonize` gravam um commit Delta por mês ou ano e podem ser retomados.

## Instalação

```bash
# uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync --locked --python 3.12
source .venv/bin/activate
```

Dependências do produto e das verificações ficam no lock da raiz. StrictDoc, OpenSpec e os papéis do harness ficam em `.omp/`.

## Rodar sem credenciais (avaliadores)

Nada abaixo exige chave da NASA, conta AWS ou arquivo `.env`.

```bash
uv sync --locked --python 3.12 && source .venv/bin/activate
python -m pytest -q          # 94 testes: normalização, silver, mapa, harmonização, previsão e job diário em dados sintéticos

# Focos reais das últimas 24 h (NOAA-20, arquivo público do FIRMS), num lake local
curl -sSfL -o j1.csv https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv
export PYTHONPATH=src
python -m fire_atlas_data normalize j1.csv --output-dir demo/norm --source-product VIIRS_NOAA20_NRT
python -m fire_atlas_data silver demo/norm --target demo/lake/silver
python -m fire_atlas_data gold-map --silver demo/lake/silver --target demo/lake/map_grid
```

- Saída: tabelas Delta em `demo/lake/`, com os focos por hexágono H3 de res0 a res6.
- O histórico completo (660 M focos), a série harmonizada e a previsão ficam no bucket S3 do projeto. Para avaliadores, existe uma credencial **somente leitura** (`gold/`, `prata/silver/`, `models/`), enviada por canal privado e nunca pelo repositório. Com ela num arquivo `.env`, a série harmonizada é lida assim:

```bash
set -a; . ./.env; set +a
python -c "
import duckdb, os; from deltalake import DeltaTable
con = duckdb.connect()
con.register('h', DeltaTable(os.environ['LAKE'] + '/gold/harmonized/cells').to_pyarrow_dataset())
print(con.execute('SELECT year(day), round(sum(harmonized)) FROM h WHERE resolution = 0 GROUP BY 1 ORDER BY 1').fetchall())"
```
- O job diário precisa de uma `FIRMS_MAP_KEY` própria, gratuita, obtida em <https://firms.modaps.eosdis.nasa.gov/api/map_key/>. Use `.env.example` como modelo.

## Lake no S3

Bucket `s3://data-lake-fire-atlas-553607359608-us-east-2-an` (us-east-2). O S3 é a fonte de verdade: as cópias locais da silver e da previsão foram apagadas em 14/11/2026.

| Prefixo | Conteúdo | Comando |
|---|---|---|
| `bronze/raw/`, `bronze/daily-ingest/` | CSVs FIRMS intactos com `validation.json`; execuções do job diário | carga manual (`raw`); `daily-ingest --archive-uri` |
| `prata/silver` | Observações normalizadas (contrato 1.3.0) com `h3_r0..h3_r6`, sem filtro de confiança | `normalize` → `silver` |
| `coverage/granules`, `coverage/cells` | Footprints CMR das passagens e passagens por célula res4 × janela de 24 h | `coverage-granules`, `coverage-cells` |
| `gold/map_grid` | Detecções diárias por célula res0–6 e sensor, para o mapa | `gold-map` |
| `gold/h3_places` | Continente, país, cidade e fonte fixa por célula | `gold-places` |
| `gold/harmonized/` | Série diária harmonizada (escala Aqua) por célula res0–6 | `harmonize` |
| `gold/ml_dataset` | Célula res6 × dia D: features até o corte e alvo de 7 dias | `ml-dataset` |
| `gold/forecast` | Probabilidade de detecção em 7 dias por célula res6 e emissão D | `ml-forecast` |
| `models/xgb-rf10/`, `models/harmonization/` | Modelo XGBoost e modelo de harmonização | `ml-train`, `harmonize-fit` |

Todos os comandos rodam com `PYTHONPATH=src python -m fire_atlas_data <comando> --help`. As decisões, execuções e contagens de cada camada estão no [dossiê do lake](.omp/tasks/lake-camadas-prata-gold.md).

## Normalizar

```bash
PYTHONPATH=src python -m fire_atlas_data normalize \
  data/fire_nrt_J1V-C2_829091.csv \
  --output-dir output/viirs-noaa20 \
  --source-product VIIRS_NOAA20_NRT \
  --memory-limit 256MB --threads 1
```

O contrato atual é 1.3.0 e o método é 1.6.0. Inválidos são registrados, não descartados silenciosamente. FRP negativo e brilho ≤ 0 K viram NULL sem descartar a detecção. `hotspot_type` vem da coluna `type` do arquivo histórico e fica NULL no NRT. Os produtos `_SP` têm rótulo próprio, que registra a proveniência mas não harmoniza os streams. `--source-url` é metadado declarado pelo operador, não prova de download. Detalhes: [normalização](docs/normalization.md), [schema lógico](contracts/normalized-observation.schema.json), [aquisição e recibos](docs/acquisition.md).

## Silver e gold do mapa

```bash
LAKE=s3://data-lake-fire-atlas-553607359608-us-east-2-an
PYTHONPATH=src python -m fire_atlas_data silver output/firms-backfill/*/*/*/ --target $LAKE/prata/silver --optimize
PYTHONPATH=src python -m fire_atlas_data gold-map --silver $LAKE/prata/silver --target $LAKE/gold/map_grid --optimize
```

A silver guarda todas as observações aceitas com uma célula H3 por resolução de 0 a 6, particionadas por ano e mês de aquisição. Os níveis são aninhados exatamente. `gold-map` agrega por (resolução, célula, dia UTC, sensor) e não soma sensores. Detalhes do mapa: [normalização § agregação para mapa](docs/normalization.md#agregação-para-mapa). Para localização das células: [gold-places](docs/gold-places.md).

## Harmonização para o calendário (RF07)

`harmonize-fit`, `harmonize-eval`, `harmonize-eras` e `harmonize` põem Terra, Aqua, Suomi-NPP, NOAA-20 e NOAA-21 na escala do Aqua. O resultado é uma série diária por célula de 2000 até hoje. Em produção está o modelo `models/harmonization/v0.3.0`: um modelo de detecção XGBoost para os VIIRS, a curva para o Terra e fatores regionais com encolhimento (κ = 30).

Erro contra o Aqua real, por região res2 × mês, em anos fora do ajuste:
- VIIRS: ~10%;
- Terra: ~23%;
- soma dos satélites sem harmonizar: ~800%;
- piso entre dois VIIRS: ~7%.

O método é uma proposta com teste retrospectivo, sem limiar de aceite aprovado. Ver [harmonização](docs/harmonization.md) e o [dossiê RF07](.omp/tasks/RF07-harmonizacao-hackathon.md). Para a apresentação: [roteiro do pitch](docs/pitch.md) e [indicadores e alertas do app (proposta)](docs/app-indicators.md).

## Previsão de 7 dias (RF10)

1. `coverage-granules` e `coverage-cells` registram as passagens dos satélites. Um alvo 0 exige passagem registrada em todas as 7 janelas; sem passagem, o alvo é desconhecido e fica fora do treino.
2. `ml-dataset` monta `gold/ml_dataset`. O universo são as células res6 com alguma detecção não offshore antes de D. As features (`det_24h`, `det_3d`, `det_7d`, `frp_mean_7d`, brilho MODIS/VIIRS de 7 dias e vizinhança em anéis 1–2) usam só detecções anteriores a D 09:00 UTC.
3. `ml-train` faz uma busca aleatória de 20 configs XGBoost com treino 2012-02-01..2020-12-17, validação 2021-01-01..2022-12-17, embargo de 14 dias e teste único a partir de 2023-01-01.
4. `ml-forecast` grava em `gold/forecast` as probabilidades de uma emissão D e recusa silver defasada. Ver [previsão](docs/ml-forecast.md).

Resultado retrospectivo do modelo atual no teste 2023-01-01..2026-11-06 (873.756.119 linhas, taxa base 0,062): AP 0,568, contra 0,496 da persistência e 0,363 da climatologia. Brier 0,0375. É uma medida reportada, não aprovada. O modelo não prevê células sem histórico. O experimento `ml-coldstart`/`ml-coldstart-eval` com células frias teve AP 0,083 e só alcança cerca de 28% das primeiras detecções. Ver o [dossiê de covariáveis](.omp/tasks/RF10-covariaveis-pontos-novos.md).

## Job diário e AWS

`daily-ingest` busca o FIRMS Area API (NRT mundial), normaliza e faz append sem duplicar na silver. Em seguida grava a previsão de D e, opcionalmente, reconstrói `gold/h3_places` e regrava os anos recentes de `gold/harmonized`. Ver [job diário](docs/daily-ingest.md).

- Local: `Dockerfile` e `compose.yaml`; copiar `.env.example` para `.env`. Não rodar junto com o job da AWS, pois não há lock entre hosts.
- AWS: `deploy/aws/daily-ingest.yaml` (ECS Fargate, agendado às 10:00 e 14:00 UTC) e `deploy/aws/deploy.sh`. A chave FIRMS fica no SSM `/fire-atlas/firms-map-key`. `deploy/aws/experiment-vm.yaml` é a VM do experimento cold-start.
- O backend consome só `s3://…/gold/forecast` e as tabelas gold; o modelo não é entregue a ele.

## Experimentos res5 removidos

Os experimentos com H3 res5 (protótipo D+1 no recorte `[-48,-17,-46,-15]`, grade ambiental ERA5/GFS/NDVI, histórico NOAA-20 SP e pipeline global semanal) e o comando `map-grid` foram removidos em 14/11/2026, por decisão do usuário. O código está no commit `cb3e35c`. Protocolos, resultados e limites continuam nos dossiês `.omp/tasks/RF10-*.md` e nas evidências `.omp/evidence/`. As saídas locais desses experimentos já tinham sido apagadas.

## Verificação

```bash
python -m pytest -q
python .omp/scripts/validate_specs.py
(cd .omp && ../.venv/bin/strictdoc export requirements --formats=html --output-dir build/strictdoc)
(cd .omp && npm exec --yes --package=node@22.22.2 -- npm run spec:validate)
```

Pact não é executado porque não existe API real de backend consumidor. Os contratos exercitados são o schema Parquet normalizado e as invariantes conferidas por `verify` em cada tabela Delta. Dados brutos e gerados ficam fora do Git.
