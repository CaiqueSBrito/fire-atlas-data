# Grade célula-dia enriquecida — RF10/CA08

## Unidade, rótulo e população

A unidade permanece H3 resolução 5 por dia UTC: **161 células** cujos centros ficam em `[-48,-17,-46,-15]`. A grade é fixa e independente das detecções. **Não é uma grade de 1 km²**: as áreas locais são aproximadamente 280–288 km². Não declara máscara terrestre nem recorte exato das bordas do polígono. `cell_id` identifica a geometria H3; `candidates.parquet` guarda os centros. A resolução foi mantida para comparar entradas sem alterar simultaneamente geografia e alvo.

Para emissão D, `target_date=D+1`: rótulo 1 se há detecção VIIRS NOAA-20 NRT na célula/dia-alvo, 0 nas demais células, sem balanceamento ou descarte de controles. Todas as células entram diariamente. A partição `publish` tem rótulos nulos. Zero significa ausência de detecção fornecida, não ausência confirmada de incêndio. `fire_observation_status` registra `detected` ou `unknown`; não fabrica uma máscara de nuvens. QA da vegetação não mede observabilidade do alvo fogo.

## Entradas e corte temporal

- **Lags FIRMS:** contagens de detecção em 7/30 dias, dias ativos em 7/14/30, tempo desde última detecção, vizinhança e contexto regional. Terminam em D; `regional_7d` conta células ativas, não pixels ou ignições. A janela de sete dias é inclusiva, D−6…D.
- **[ERA5 e GFS](weather.md)** são separados: ERA5 fornece condições históricas em D e chuva D−6…D; GFS prevê D+1 usando inicialização D00. Não entra ERA5 observado em D+1. Reanálise final não é a versão comprovadamente disponível em D; inicialização GFS não é horário de publicação. Este é um experimento retrospectivo, não operação causal comprovada.
- **[Vegetação](vegetation.md):** NDVI/EVI com QA, idade e fração válida de um patch no centro H3. Só entram composições encerradas e processadas até D quando informado. Não representa toda a célula nem mede diretamente umidade do combustível. Nulos não são imputados.
- **[Histórico científico plurianual](history.md):** adquirido separadamente, sem misturar SP 2018–2025 com o NRT original. A grade diária histórica foi materializada com `historical_grid.py`; ver comando abaixo. Ainda não tem as covariáveis ambientais de todos esses anos e não foi usada no treino pareado de 2026.

Cada aquisição preserva brutos, URL, hash, horário de download, unidades e configuração. O enriquecimento verifica SHA256, chaves únicas e população idêntica à base. Chaves são VARCHAR/DATE; covariáveis são DOUBLE finitas ou nulas. Duplicatas, chaves deslocadas/faltantes/extras e hashes alterados falham antes da publicação. Rótulos, splits e features antigas são preservados.

## Comandos

Executar na raiz, sempre com saídas novas:

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.weather --features output/rf10-activity/features/features.parquet --output-dir output/rf10-environment/weather
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.weather --features output/rf10-activity/features/features.parquet --output-dir output/rf10-environment/era5 --historical-only
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.vegetation --features output/rf10-activity/features/features.parquet --output-dir output/rf10-environment/vegetation
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.environmental enrich --features output/rf10-activity/features/features.parquet --weather output/rf10-environment/weather --era5 output/rf10-environment/era5 --vegetation output/rf10-environment/vegetation --output-dir output/rf10-environment/features
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.environmental train --features output/rf10-environment/features/features.parquet --baseline-model-dir output/rf10-activity/model-fixed --output-dir output/rf10-environment/model-contract
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.environmental score --features output/rf10-environment/features/features.parquet --model-dir output/rf10-environment/model-contract --output-dir output/rf10-environment/inference-contract
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.historical_grid --history output/rf10-environment/history/prepared --candidates output/rf10-activity/features/candidates.parquet --horizon-days 1 --output-dir output/rf10-environment/history/grid
```

A grade SP histórica tem **455.791 células-dia e 15.279 positivos**, com emissões 01/04/2018–30/12/2025 e alvos no dia seguinte. Inclui contagens de linhas de detecção na célula e região nos sete dias anteriores, sem deduplicação adicional ou interpretação como eventos independentes. Zeros mantêm estado físico desconhecido. O começo das janelas é censurado pela disponibilidade do arquivo.

## Comparação sem retuning

Perfil fixo: 120 árvores, profundidade 3, taxa 0,05, `min_child_weight=1`, lambda 1, CPU `hist`, duas threads, seed 42, subsample 0,8 e colsample 1. Mesmas linhas: treino 4.186/259 positivos, validação 805/24 e TEST 805/44. Ablações predeclaradas: baseline de 40 features congelado, +GFS, +ERA5, +vegetação e +todas (56 features). Não há busca de hiperparâmetros ou ajuste pelo TEST.

Seleção estrita por recall macro diário@10 somente na validação; empates preservam baseline/ordem listada. A decisão é gravada antes de prever TEST. Todos os perfis são reportados como diagnósticos no TEST já conhecido, sem promoção automática. AP e AUC agrupadas são informações complementares, não probabilidades calibradas ou prova independente.

### Resultado executado: manter baseline, sem promoção

| Entradas | Capturas na validação | Capturas no TEST diagnóstico | Recall micro TEST |
|---|---:|---:|---:|
| Baseline congelado | 8/24 | 9/44 | 20,45% |
| + GFS | 5/24 | 6/44 | 13,64% |
| + ERA5 | 6/24 | 5/44 | 11,36% |
| + vegetação | 7/24 | 4/44 | 9,09% |
| + todas | 5/24 | 5/44 | 11,36% |

Adicionar essas entradas **não melhorou este experimento**. O baseline venceu pela validação; não escolher outro perfil depois de observar o TEST. O resultado não prova que meteorologia é inútil: há somente 26 dias de treino, amostragem meteorológica em 75 nós e vegetação em pequenos patches. Não isola causas nem valida uso operacional.

## Evidência e limites

- GFS e ERA5: 5.957 linhas cada, 75 nós meteorológicos deduplicados em lattice de 0,25°, 161 células e 37 emissões. Todos os agregados completos, sem imputação. GFS bruto 6.092.825 bytes; ERA5 bruto 2.894.576 bytes.
- Vegetação: 5.957 linhas, 644 composições, 163 requisições; 24 linhas com índices nulos por QA. Coleção não é inventada quando a API omite a versão.
- Grade enriquecida preserva as 5.957 linhas e os rótulos da base; contém 56 features, mais identificadores e estado de observação.
- JSONs recarregados produziram escores de validação exatamente iguais aos modelos em memória. TEST usa os JSONs recarregados. Inferência histórica sem rótulos futuros executada em 161 células.
- **209 testes passaram em 18,11 s**, incluindo lags sem futuro, rejeição de unidades incompatíveis, integridade, população, QA e inferência. A falha inicial de resultado SQL não consumido foi corrigida consumindo os resultados DuckDB antes de abrir novas relações. O teste que comparava representação textual de fuso foi substituído por comparação do instante real.
- Saídas definitivas: `model-contract/` e `inference-contract/`. A primeira execução `model/` e `inference/` permanece preservada; foi repetida com os mesmos perfis para incluir `column_units` no contrato, sem retuning.
- Máscara NOAA-20 diária: metadados CMR acessíveis, mas o GET limitado do arquivo retornou 401 sem Earthdata. Não foi substituída por MODIS de oito dias nem por QA da vegetação. Detecção física ausente permanece desconhecida.
- Emissões reservadas 13/nov–12/dez não foram consultadas ou avaliadas. Horários dos recibos refletem o relógio real da execução, inclusive a passagem para 13/nov; não foram retrodatados.

Artefatos: modelos JSON, plano/seleção/comparação, manifestos encadeados, versões/hashes e CSV/Parquet de predições. A inferência valida nomes, ordem, unidades, hash e contrato. Salva escores históricos, não um mapa operacional. Autorizações e evidências no [dossiê ambiental](../.omp/tasks/RF10-entradas-ambientais.md).
