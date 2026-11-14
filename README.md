# Fire Atlas Data

Normalização FIRMS e experimentos retrospectivos de previsão de detecções. O novo alvo aprovado é **alguma detecção em uma célula entre D+1 e D+7**, inclusive; os experimentos XGBoost D+1 anteriores ficam preservados. Requisitos e processo: [harness `.omp/`](.omp/README.md), RF11/CA09 e RF10/CA08. Sem declaração de harmonização científica, probabilidade calibrada de incêndio ou aceite operacional.

## Escala

Implementação vetorizada com **DuckDB out-of-core**, banco intermediário em disco e **Parquet particionado por ano/mês**. Não materializa o CSV inteiro em pandas/listas Python nem gera JSON por registro. Memória, threads e spill são configuráveis. Leitores consultam colunas e partições necessárias, sem converter todo o dataset para DataFrame.

Esse desenho permite processar dados maiores que RAM disponível em uma máquina; capacidade/throughput para bilhões de registros depende de disco, memória, particionamento e infraestrutura e requer benchmark representativo. `memory_limit` controla memória gerenciada pelo DuckDB, não estabelece teto rígido para RSS total. Arquivo raw, staging, banco intermediário, spill e saída precisam caber em armazenamento local. Não há execução distribuída ou retomada de carga parcialmente concluída nesta versão; a publicação é por execução completa. Processar fontes em datasets independentes evita arquivo monolítico e permite consulta via glob de vários datasets.

## Instalação

```bash
# uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync --locked --python 3.12
source .venv/bin/activate
```

Dependências do produto e verificações ficam no lock da raiz. As ferramentas de StrictDoc/OpenSpec e papéis do harness continuam em `.omp/`; não há CLI de orquestração LangGraph.

## Normalizar

```bash
PYTHONPATH=src python -m fire_atlas_data normalize \
  data/fire_nrt_J1V-C2_829091.csv \
  --output-dir output/viirs-noaa20 \
  --source-product VIIRS_NOAA20_NRT \
  --memory-limit 256MB --threads 1
```

O comando é do processamento de dados, não um workflow de agentes. `--source-url`, se fornecida, é metadata declarada pelo operador, não prova de download; não incluir URLs com chaves. O arquivo disponível não veio acompanhado de URL exata/data de download, portanto isso não será inventado.

Contrato e limites: [docs/normalization.md](docs/normalization.md), [schema lógico](contracts/normalized-observation.schema.json). Entrada original não é alterada. Inválidos são registrados, não descartados silenciosamente. Disponibilidade histórica desconhecida fica ausente, nunca inferida da aquisição ou hora da execução. Confiança por sensor não é probabilidade de fogo; FRP não é área queimada. Não deduplicar observações como eventos de incêndio nesta etapa.

## Consultar para backend/ML

Parquet tem tipos nativos e permite predicate/projection pushdown. Exemplo com DuckDB:

```python
import duckdb

connection = duckdb.connect()
connection.execute("SET memory_limit='256MB'")
# Exemplo de agregado pequeno; não coletar bilhões de registros com fetchall().
result = connection.execute("""
    SELECT acquisition_date, sensor, count(*) AS detections, sum(frp_mw) AS observed_frp_mw
    FROM read_parquet('output/viirs-noaa20/accepted/**/*.parquet', hive_partitioning=true)
    WHERE acquisition_year = 2026 AND acquisition_month = 9
    GROUP BY acquisition_date, sensor
    ORDER BY acquisition_date, sensor
""").fetchall()
```

Essa soma é descritiva das observações, não harmonização de sensores nem estimativa de área de fogo. A normalização preserva apenas observações; features, rótulos de ocorrência observada e ranking pertencem à etapa separada abaixo.

## Agregar para o mapa

```bash
PYTHONPATH=src python -m fire_atlas_data map-grid output/firms-backfill/*/*/*/ \
  --output-dir output/map-h3-daily \
  --resolutions 2,3,4,5,6,7,8 --group-by platform --memory-limit 4GB --threads 4
```

Renderizar uma feição por detecção é inviável nessa escala: no acervo local de 660.284.137 registros, a célula de 1° mais densa acumula 942.406 detecções. `map-grid` conta detecções por célula **H3, dia UTC e sensor/plataforma** em várias resoluções, uma por faixa de zoom (decisão do usuário: res2–8, diário). Qualquer intervalo de datas é somado na consulta. ML usa somente H3 res5, com grade própria; as resoluções do mapa não definem features nem rótulos. MODIS e VIIRS ficam em linhas separadas, sem harmonização nem deduplicação entre plataformas; célula ausente é desconhecida, não ausência de fogo. Detalhes e saída: [docs/normalization.md](docs/normalization.md#agregação-para-mapa).

## Experimento XGBoost com os dados locais

Configuração reproduzível: [`configs/prototype.json`](configs/prototype.json). Recorte fixo `[-48,-17,-46,-15]`, H3 resolução 5, 161 células candidatas sem máscara terrestre certificada. Ausência de detecção é desconhecida, não ausência confirmada de fogo.

```bash
PYTHONPATH=src python -m fire_atlas_data.features \
  --config configs/prototype.json --output-dir output/rf10-prototype/features
PYTHONPATH=src python -m fire_atlas_data.ranking train \
  --features output/rf10-prototype/features/features.parquet \
  --output-dir output/rf10-prototype/model-heldout --k 10
PYTHONPATH=src python -m fire_atlas_data.ranking publish \
  --features output/rf10-prototype/features/features.parquet \
  --model-dir output/rf10-prototype/model-heldout --output-dir output/rf10-prototype/publication-heldout
```

Saídas existentes não são sobrescritas: escolher novos diretórios para repetir. A transformação usa informações adquiridas até D e observa D+1 somente como alvo; a partição de publicação não exige conhecer D+1. Como `available_at` histórico é desconhecido, isso é **backtest por aquisição**, não disponibilidade operacional comprovada.

Treino: emissões 01–10/10/2026; validação: 11–15/10; teste reservado: 16–20/10, alvos 17–21/10. O dia 22/10 está parcial na fonte e foi excluído dos alvos de avaliação. Somente a validação seleciona o modelo; o teste não escolhe hiperparâmetros. Controles são amostrados apenas no treino; avaliação considera todos os candidatos elegíveis. Publicação local: emissão nominal retrospectiva 21/10, alvo 22/10 sem rótulo.

Execução real: modelo completo selecionado na validação; capturou 10 de 36 novos registros célula-dia no teste (27,8%), com dez destaques por dia e 10/50 destaques com registro observado (20%). Apenas cinco dias e um recorte: não comprova generalização. Sem anos anteriores, a referência sazonal é constante e **não suportada**; não alegar superioridade sobre uma referência sazonal válida.

Artefatos: `output/rf10-prototype/model-heldout/comparison.json`, modelos JSON e `backtest_predictions.parquet` com os 731 candidatos de teste, score, posição e resultado observado por célula-dia. Publicação: ranking Parquet e [mapa HTML local](output/rf10-prototype/publication-heldout/map.html), sem backend/app integrado. Protocolo, hashes, evidências e limitações no [dossiê RF10](.omp/tasks/RF10-ranking-prototipo.md).

Detalhes: [aquisição e recibos](docs/acquisition.md), [célula-dia e features](docs/features.md), [treino, teste e publicação](docs/ranking.md). Aquisição histórica exige URL autorizada; Area API exige chave em ambiente. Nenhum histórico autenticado foi baixado nesta execução. Recibo de importação local não recupera disponibilidade histórica. A normalização aceita os streams NRT e SP (padrão/científico); os produtos `_SP` passaram a ter rótulo próprio no contrato 1.2.0, mas isso é rotulagem de proveniência, não harmonização científica entre streams. O contrato atual é 1.3.0 (método 1.5.0): acrescenta `hotspot_type` a partir da coluna `type` do FIRMS e, desde o método 1.5.0, anula FRP negativo e brilho ≤ 0 K sem descartar a detecção; datasets 1.2.0 precisam ser renormalizados para o pipeline global.

## Melhoria de qualidade: experimento e limite observado

O experimento [`configs/quality.json`](configs/quality.json) amplia o treino para 15/09–10/10 e acrescenta recorrência própria, duas faixas de vizinhança, FRP histórico e coordenadas do centro da célula. Treino: 3.625 candidatos/155 positivos, contra 1.433/52 antes. As janelas iniciais de 30 dias são censuradas, com suporte de calendário explícito.

```bash
PYTHONPATH=src python -m fire_atlas_data.features \
  --config configs/quality.json --output-dir output/rf10-quality/features
PYTHONPATH=src python -m fire_atlas_data.quality train \
  --features output/rf10-quality/features/features.parquet \
  --baseline-model-dir output/rf10-prototype/model-heldout \
  --output-dir output/rf10-quality/model --k 10
```

Busca predefinida de 11 candidatos mais o campeão original congelado, selecionada apenas na validação. O vencedor passou de 1/12 para 4/12 capturados na validação, mas **regrediu de 10/36 para 4/36 no teste diagnóstico de outubro**. Não houve aumento comprovado de qualidade e essa versão **não substitui o modelo anterior**; sua publicação é bloqueada.

Foi também baixado e normalizado um CSV público oficial NOAA-20 de sete dias: 65.247 observações reais, sem MAP_KEY. Avaliação congelada em emissões 08–10/11, alvos09–11/11: 461 candidatos e zero positivos observados, portanto recall indefinido e evidência insuficiente para medir ganho. Existe lacuna de entrada 23/10–04/11; não apresentar esse lote como histórico contínuo ou comprovação operacional.

Dados/artefatos: `output/rf10-quality/`; resultados em `model/comparison.json` e `independent-evaluation/comparison.json`. [Protocolo, fontes e evidências](.omp/tasks/RF10-qualidade-preditiva.md), [comandos e contrato](docs/quality.md). A avaliação nova não retreina, não seleciona e não desbloqueia automaticamente promoção. Diretórios existentes não são sobrescritos.

## Experimento D+1 preservado

`configs/activity.json` mantém região/H3 e inclui todas as161células por dia,
sem excluir atividade recente. Rótulos continuam específicos a NOAA-20 NRT;
zero não comprova ausência de fogo. Build real:259positivos no treino,
24na validação e44no TEST diagnóstico já conhecido. Perfil XGBoost fixo treinado
com dados locais:8/24capturas na validação e9/44no diagnóstico,sem meteorologia
ou vegetação e sem promoção. Não comparar com modelos antigos de novidade.
Contrato/comando em [features](docs/features.md); autorização e evidências no
[dossiê de atividade](.omp/tasks/RF10-atividade-d1.md).

## Grade diária enriquecida e histórico plurianual

Implementados cruzamentos reais com ERA5 histórico até D, previsão GFS por
rodada e NDVI/EVI com qualidade e idade. A grade mantém 161 células H3 por dia
(aproximadamente 280–288 km² cada, não 1 km²), rótulos 1/0 de detecção observada
e lags de focos. Dataset enriquecido: `output/rf10-environment/features/`.

O experimento pareado não melhorou: baseline 9/44 capturas no diagnóstico,
contra 5/44 com todas as novas entradas. Baseline mantido pela validação,
sem promoção. Modelos e inferência definitivos: `model-contract/` e
`inference-contract/` sob `output/rf10-environment/`. Suíte: 209 testes passaram.

Histórico público NOAA-20 SP 2018–2025 adquirido sem credenciais e materializado
separadamente em 455.791 células-dia, com 15.279 positivos e lags de sete dias.
Não foi misturado silenciosamente ao alvo NRT nem usado no treino curto de 2026.
Comandos, fontes, resultado e limites: [grade ambiental](docs/environmental.md),
[histórico](docs/history.md) e [dossiê](.omp/tasks/RF10-entradas-ambientais.md).

## Objetivo atual: alguma detecção nos próximos sete dias

Para emissão no fim de D, rótulo 1 se houver detecção NOAA-20 SP na célula
entre D+1 e D+7, inclusive; 0 se não houver detecção fornecida nessa janela.
Continuidade e novas ocorrências contam. Features usam somente D e o passado.
Não prever ignição exata nem tratar zero como ausência física confirmada.

Grade real preparada em `output/rf10-weekly/grid/`: **454.825 células-dia,
68.706 janelas positivas e 386.119 zeros**, nas mesmas 161 células H3.
Emissões 01/04/2018–24/12/2025; somente janelas futuras completas.
O primeiro modelo semanal e a busca de melhoria v2 foram treinados com folds
temporais expurgados. A v2 usa 32 preditores e selecionou XGBoost `rank:ndcg`.
No **diagnóstico reutilizado de 2025**, precisão@10 passou de **34,92% para
44,05%**, com recall macro de 17,03% para 24,79%; em 2024, precisão passou
de 33,06% para 40,22%. Não são resultados de teste independente.

A meta de 80% não foi atingida. Com dez destaques obrigatórios todos os dias,
o teto matemático no agregado 2024–2025 é 77,87%, mesmo com um oráculo.
Artefatos completos: `output/rf10-weekly/model-v2-run2/`; modelo anterior
preservado, sem promoção operacional. [Método predeclarado](docs/weekly-improvement.md),
[comandos e resultados](docs/quality.md#melhoria-semanal-v2-precisão-sem-alterar-o-orçamento)
e [evidências da melhoria](.omp/tasks/RF10-melhoria-7dias.md).

Contrato/comando em [histórico](docs/history.md) e autorização/evidências no
[dossiê de sete dias](.omp/tasks/RF10-atividade-7dias.md). Verificação consolidada:
216 testes passaram; rótulos futuros reconciliados com os dados D+1 preservados,
sem alteração dos lags.




## Pipeline global semanal por recorte

O novo caminho usa H3 res5 e alvo multissensor de detecção não estática em
D+1…D+7. Não substitui nem reinterpreta os experimentos históricos acima.
O recorte escolhe células com histórico conhecido até a emissão; não muda
as features da mesma célula. Ausência de previsão fora do domínio não é zero.

- [Protocolo proposto e limites científicos](docs/global-weekly-protocol.md).
- [Máscara fina e atividade multissensor](docs/global-activity.md).
- [Features causais, amostragem e contratos](docs/global-features.md).
- [Inferência por bbox e artefatos](docs/global-forecast.md).
- Configuração de treino: `configs/global-weekly.json`; módulo
  `PYTHONPATH=src python -m fire_atlas_data.global_training --help`.

Normalização 1.3.0 preserva `hotspot_type` anulável. Máscara e parâmetros
científicos são explícitos; inferência não instala extensões nem baixa fontes.
Backfill completo e treinamento real continuam não autorizados. Testes e
modelos sintéticos verificam engenharia, não precisão mundial nem aceite CA08.

## Verificação

```bash
python -m pytest -q
python .omp/scripts/validate_specs.py
(cd .omp && ../.venv/bin/strictdoc export requirements --formats=html --output-dir build/strictdoc)
(cd .omp && npm exec --yes --package=node@22.22.2 -- npm run spec:validate)
```

Pact não é executado para inventar uma integração: ainda não existe API real do backend consumidor. Nesta etapa, o contrato exercitado é o schema do dataset Parquet e suas invariantes. Dados brutos/gerados ficam fora do Git; somente código, contratos e evidências pequenas devem ser versionados.
