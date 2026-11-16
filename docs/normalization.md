# Normalização de observações FIRMS — RF11 / CA09

Esta implementação entrega normalização de engenharia para consumo em backend/ML, em apoio a RF07/RF10. Não entrega harmonização científica MODIS/VIIRS, grade, rótulos, escalonamento, treinamento XGBoost ou API. A ausência de observações **não significa ausência de fogo**; detecções podem incluir outras anomalias térmicas. O bruto permanece imutável.

## Executar

Com o pacote e sua dependência DuckDB instalados pelo ambiente raiz:

```sh
PYTHONPATH=src python -m fire_atlas_data normalize data/fire_nrt_J1V-C2_829091.csv \
  --output-dir output/normalized \
  --source-product VIIRS_NOAA20_NRT \
  --memory-limit 1GB --threads 2
```

`--source-url URL` é metadado opcional declarado pelo operador, nunca obtido ou verificado pela execução. Omitir mantém URL desconhecida (`null`), em vez de inventar origem exata de download. Não fornecer URL com chave/segredo. `--temp-dir DIRETORIO` seleciona o volume de spill; por padrão fica no staging. Produtos aceitos: `VIIRS_NOAA20_NRT`/`VIIRS_NOAA20_SP` (N20), `VIIRS_NOAA21_NRT`/`VIIRS_NOAA21_SP` (N21), `VIIRS_SNPP_NRT` (N), `VIIRS_SNPP_SP` (SNPP) e `MODIS_NRT`/`MODIS_SP` (T/Terra ou A/Aqua). O sufixo `_SP` declara o stream padrão/científico (`fire_archive_*.csv` do FIRMS); `_NRT`, o stream near-real-time (`fire_nrt_*.csv`). Sensor/plataforma incompatíveis com o produto são rejeitados. O identificador é uma declaração do operador, não prova da autenticidade NASA do arquivo.

O contrato 1.2.0 / método 1.3.0 acrescenta `VIIRS_NOAA20_SP`, `VIIRS_NOAA21_SP` e `MODIS_SP`, com as mesmas regras de sensor/plataforma/confiança dos respectivos NRT: só muda o rótulo de proveniência, para que dados archive não sejam declarados como NRT. Nesses arquivos `instrument` vem literalmente como `VIIRS`/`MODIS` e `satellite` como `N20`, `N21`, `Terra` ou `Aqua`; versões `2`, `6.03` e `61.03` ficam literais em `source_version`, sem equivalência inferida. O contrato 1.1.0 / método 1.2.0 já distinguia o arquivo científico Suomi-NPP de NRT. Para `VIIRS_SNPP_SP`, `satellite=SNPP` é normalizado para `Suomi-NPP`; `instrument=SNPP` é um alias de arquivo aceito para VIIRS apenas nesse produto/plataforma (o valor explícito `VIIRS` também é aceito). Esse alias não é aceito nos demais produtos, cujas regras N/N20/N21 permanecem separadas. `sensor_origin=source_instrument_or_snpp_archive_alias_checked_against_product_and_platform` registra a verificação do instrumento/alias, sem afirmar que o bruto contém literalmente `VIIRS`. A versão numérica textual `2` é preservada em `source_version`; não se exige sufixo NRT.

O contrato 1.3.0 / método 1.4.0 acrescenta a coluna anulável `hotspot_type SMALLINT`, logo após `observation_type`. A origem é a coluna CSV `type`, presente apenas nos arquivos archive/SP do FIRMS: `0` = fogo de vegetação presumido, `1` = vulcão ativo, `2` = outra fonte estática em terra, `3` = offshore. É uma classe inferida pelo FIRMS, não verdade de campo. Coluna ausente (streams NRT) ou valor vazio viram NULL, que significa **desconhecido**, não fogo de vegetação. Valor presente, não vazio e diferente de exatamente `0`, `1`, `2` ou `3` após aparar espaços (por exemplo `4`, `-1`, `x`, `0.0`, `1.5`) rejeita o registro com o motivo `invalid_type`. Esse motivo é o último da cadeia: um registro que já era rejeitado por outro motivo mantém o motivo anterior. O texto bruto de `type` continua em `source_fields`, e o manifesto ganha `hotspot_type_counts` (contagem dos aceitos por classe, nulo por último). Datasets gravados com o contrato 1.2.0 não têm a coluna e precisam ser renormalizados para tê-la.

O método 1.5.0 (contrato 1.3.0 inalterado) deixa de rejeitar a detecção quando apenas a medida é fisicamente impossível: FRP negativo e brilho ≤ 0 K (`brightness`, `bright_t31`, `bright_ti4`, `bright_ti5`) viram NULL na coluna tipada, e o registro é aceito com coordenadas, horário e sensor preservados. O valor bruto continua em `source_fields` e o manifesto contabiliza cada anulação em `nullified_value_counts`. FRP igual a 0 continua 0. Valores não numéricos ou não finitos (`abc`, `nan`, `inf`) continuam rejeitados com `invalid_<campo>`, assim como scan/track ≤ 0. NULL nessas colunas significa medida ausente ou inválida, nunca zero.

O método 1.6.0 (contrato 1.3.0 inalterado) também aceita o par `instrument=SNPP`/`satellite=SNPP` em `VIIRS_SNPP_NRT`, como no arquivo NRT de 2026 (`fire_nrt_SV-C2_832954.csv`), que o método 1.5.0 rejeitava por inteiro com `sensor_product_mismatch`. Em `VIIRS_SNPP_NRT`, `instrument=SNPP` com outro satélite continua rejeitado; NOAA-20/21 e MODIS seguem sem o alias.

O comando para o arquivo científico local é:

```sh
PYTHONPATH=src python -m fire_atlas_data normalize data/fire_archive_SV-C2_832954.csv \
  --output-dir output/viirs-snpp-sp \
  --source-product VIIRS_SNPP_SP \
  --memory-limit 1GB --threads 2
```

Esse exemplo é uma instrução de uso, não evidência de execução do arquivo. `source_fields` conserva todas as colunas brutas, inclusive `type` e seus rótulos científicos, sem filtrar tipos ou reinterpretá-los como alvo. `bright_t31` continua sem equivalência inferida para I-5. `available_at` permanece desconhecido: o arquivo é covariável retrospectiva, não autorização de disponibilidade histórica ou mudança de alvo/split.

O método também aceita o CSV público de sete dias do FIRMS: quando `instrument` está ausente, o sensor deriva do produto declarado e a plataforma continua sendo validada; `sensor_origin` no manifesto explicita essa origem. Se `instrument` vier no arquivo, precisa corresponder ao produto ou ao alias científico restrito acima. Nenhum campo ausente é inventado em `source_fields`.

## Arquivos e contrato

- `accepted/acquisition_year=YYYY/acquisition_month=M/*.parquet`: observações aceitas, compressão ZSTD, alvo de arquivo 128 MB e grupos de 16.384 linhas. O alvo não é limite rígido. Ler como dataset Hive para recuperar as colunas de partição. Dataset vazio possui `accepted/empty.parquet` com schema completo.
- `rejected/*.parquet`: ordinal do registro CSV, hash da fonte, primeiro motivo de rejeição e `source_fields`; arquivos com alvo 128 MB e grupos de 16.384 linhas. Dataset sem rejeições possui `rejected/empty.parquet`.
- `manifest.json`: versões de contrato/método, versão DuckDB, configuração de recursos, caminho/tamanho/SHA256 da fonte, produto/URL declarados, contagens, intervalo de aquisição UTC, sensores/plataformas, contagens de motivos, hashes/tamanhos das saídas e limitações.

[`normalized-observation.schema.json`](../contracts/normalized-observation.schema.json) descreve a representação lógica de uma observação Parquet, convertendo DATE/TIMESTAMPTZ para ISO 8601 e incluindo partições Hive. Não é um contrato Pact nem prova de integração com consumidor não implementado. Campos numéricos são DOUBLE, confiança MODIS SMALLINT, ordinal BIGINT, datas DATE e aquisição/disponibilidade TIMESTAMPTZ. `available_at` é sempre nulo: aquisição não prova disponibilidade histórica e não autoriza uso sem vazamento em previsão D+1.

`record_id = SHA256_DO_ARQUIVO:ORDINAL`, com ordinal começando em 1 após cabeçalho; registros CSV multilinha não equivalem a números de linha física. `source_fields` é STRUCT de todas as colunas originais VARCHAR, inclusive campos extras e strings vazias. Preserva conteúdo de campos, não bytes originais de aspas/separadores; o bruto e seu hash preservam esses bytes. Cabeçalhos duplicados, vazios, sem campos obrigatórios ou colidindo com nomes internos (`source_record_number`, `source_fields`, `parsed_time`, `reason`, prefixo `n_`) falham antes de publicar.

Não há deduplicação, descarte por confiança, imputação ou conversão de faltante para zero. Faltantes numéricos opcionais viram NULL; FRP zero permanece zero. Cada registro estruturalmente legível é aceito ou rejeitado; total = aceitos + rejeitados. Um registro com múltiplos erros recebe o primeiro motivo determinístico. Erro estrutural CSV, falha de I/O ou memória aborta a execução inteira, sem dataset de sucesso parcial. O destino deve não existir (inclusive diretório vazio é recusado). Arquivos são preparados em diretório irmão e publicados por rename no mesmo filesystem após conclusão; falhas anteriores removem o staging. Não usar operações concorrentes que modifiquem o bruto/destino.

## Regras e semântica de sensores

Coordenadas finitas em latitude [-90,90], longitude [-180,180], graus EPSG:4326. Data `YYYY-MM-DD` de calendário real, ano 0001–9999, hora UTC `HHMM`, aceitando representação sem zeros iniciais (7 → 0007); horas 00–23/minutos 00–59. Scan/track opcionais, se presentes finitos e positivos em km; FRP opcional finito não negativo em MW. Temperaturas opcionais, se presentes finitas e positivas em K. `daynight` D/N; versão da fonte não vazia. Strings canônicas são aparadas; campos originais não.

VIIRS conserva confiança l/n/h (aceitando caixa diferente e normalizando para minúscula), sem transformá-la em probabilidade. `bright_ti4` e o alias oficial `brightness` mapeiam para `brightness_i4_k`; se ambos estão presentes com números diferentes, rejeitar a ambiguidade. `bright_ti5` mapeia para `brightness_i5_k`. A documentação oficial confirma **Brightness_2**, não **bright_t31**, como alias de I-5: por isso o campo `bright_t31` do arquivo local é conservado em `bright_t31_k`, sem atribuir banda VIIRS. `brightness_k` mantém o valor original de brightness também.

As categorias públicas `low`, `nominal` e `high` também são normalizadas para `l`, `n` e `h`, preservando os nomes originais em `source_fields`. Compatibilidade exercitada em 65.247 observações reais do [CSV NOAA-20 América do Sul de sete dias](https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_America_7d.csv), link obtido na [página oficial](https://firms.modaps.eosdis.nasa.gov/active_fire/) em 2026-11-12. O endpoint é móvel: hashes e recibos da consulta estão no [dossiê de qualidade](../.omp/tasks/RF10-qualidade-preditiva.md), não há garantia de obter os mesmos bytes em outro dia.

MODIS mantém `brightness_k` como temperatura de brilho de canal21/22 e `bright_t31_k` como canal31; campos I-4/I-5 ficam nulos. Confiança original inteira 0–100 é preservada em `confidence_percent`, não calibrada como probabilidade comparável a VIIRS. Suporte de código/fixture MODIS não equivale a evidência de execução sobre uma fonte real MODIS.

## Escala e custos

A transformação é SQL vetorizada DuckDB sobre CSV com colunas VARCHAR explícitas e parser estrito. `preserve_insertion_order=true` mantém a ordem da leitura durante a captura do ordinal; depois a exportação permite reordenação para reduzir memória. A leitura paralela respeita o limite de threads. Não há iteração Python sobre observações, pandas, JSONL ou carga integral em RAM. A tabela intermediária fica em banco DuckDB em disco; limites de memória configuráveis e spill permitem execução out-of-core. Escrita usa no máximo dois arquivos de partição abertos e flush a cada 16.384 registros por estado de thread para conter buffers; partições mensais evitam arquivos diários pequenos. Parquet habilita projeção de colunas, pushdown de filtros e poda por ano/mês pelos consumidores.

Dimensionar disco para banco intermediário (inclui strings originais), Parquet, spill e margem operacional; `memory_limit` limita o buffer manager DuckDB, não promete teto absoluto de RSS do processo. Dataset maior que recursos disponíveis pode falhar com erro explícito. SHA256 lê a fonte uma vez adicional antes do processamento; hashing de arquivos Parquet acrescenta I/O, não memória proporcional ao número de observações. Metadados de arquivos são proporcionais ao número de arquivos, não às linhas. Nenhum benchmark com bilhões de linhas é alegado apenas pela arquitetura. Evidência executada e tamanho real devem constar no dossiê da integração.

## Agregação para mapa

Um mapa não pode desenhar uma feição por detecção nesta escala: na varredura dos 62 CSVs do acervo local (660.284.137 registros brutos), a célula de 1° mais densa acumula 942.406 detecções. A agregação diária para o mapa é feita na camada gold, a partir da silver, pelo comando `gold-map` (`gold.py`). O antigo `map-grid`, que lia o Parquet normalizado em res2–8, foi removido em 14/11/2026.

```sh
LAKE=s3://data-lake-fire-atlas-553607359608-us-east-2-an
PYTHONPATH=src python -m fire_atlas_data gold-map --silver $LAKE/prata/silver \
  --target $LAKE/gold/map_grid --optimize --temp-dir .tmp/gold-spill \
  --memory-limit 2GB --threads 2
```

Cada linha é (`resolution` 0–6, `h3_cell`, `day` UTC, `sensor`) com `detections`, `platforms`, FRP (soma, média e máximo), `brightness_max_k`, contagem VIIRS por confiança `l`/`n`/`h` e média de confiança MODIS. Partição `resolution`/`year`/`month`, com mês derivado de `day` UTC. O build substitui cada (`resolution`, `year`) em um commit Delta, incluindo todas as suas partições mensais; Z-order em (`day`, `h3_cell`) com `--optimize` é executado por mês. Ao final, o `verify` confere as sete resoluções e a soma de detecções contra a silver em cada mês, além da coerência entre data e partição. Não há filtro de confiança. Sensores ficam separados, sem harmonização nem deduplicação. Célula ausente é desconhecida, não ausência confirmada de fogo. No lake: `s3://data-lake-fire-atlas-553607359608-us-east-2-an/gold/map_grid`. Continente, país e cidade: [gold-places](gold-places.md).

Para migrar uma tabela existente particionada apenas por `resolution/year`:

```sh
PYTHONPATH=src python -m fire_atlas_data gold-map-repartition \
  --target $LAKE/gold/map_grid --memory-limit 32GB --threads 16 \
  --temp-dir /workspace/spill
```

A migração reescreve os agregados existentes, acrescentando somente `month`, e publica o novo layout em **um commit Delta atômico no mesmo caminho**. Não recalcula a silver nem atualiza a data dos dados. Valida contagem de linhas, soma de detecções e soma de hashes de todas as colunas originais por resolução/ano/mês antes/depois; hashes não constituem prova sem colisões. Não executar escritores concorrentes nem vacuum durante a migração: os arquivos antigos permanecem disponíveis para leitura da versão anterior. Leitores devem renovar seu snapshot Delta para enxergar o novo layout; leitura direta de todos os Parquets mistura arquivos antigos e novos e não é suportada. Uma tabela já mensal não é reescrita. Dimensionar RAM também para os buffers do escritor Delta, não apenas para o limite DuckDB.

## Fontes primárias

Consultadas em 2026-11-12:

- [NASA FIRMS VIIRS attributes](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html): aliases Bright_ti4/Brightness e Bright_ti5/Brightness_2, confiança categórica, UTC, FRP e plataformas.
- [NASA FIRMS MODIS attributes](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_MODIS_Firehotspots.html): bandas21/22 e31, confiança0–100 e plataformas Terra/Aqua.
- [DuckDB partitioned writes](https://duckdb.org/docs/2.0/data/partitioning/partitioned_writes.html): partições Hive e limite de arquivos abertos.
- [DuckDB COPY](https://duckdb.org/docs/2.0/sql/statements/copy.html): exportação Parquet, opções de row group e alvo de tamanho.

As fontes definem semântica de campos; não validam cientificamente harmonização ou predição.

## Evidência executada

CSV real completo: 3.210.758 registros, dos quais 3.210.756 aceitos e 2 rejeitados por FRP negativo, sem descarte silencioso. Execução local: 23,61 s, limite DuckDB 256 MB/uma thread; RSS máximo 398.240 KiB. Parquets gerados em `output/viirs-noaa20-real`, hashes e contagens conferidos. Suite: 15 testes passaram. Evidências e detalhes no [dossiê RF11](../.omp/tasks/RF11-normalizacao-backend-ml.md).

Isso demonstra processamento maior que o limite de buffer configurado, não benchmark bilionário nem teto rígido de RSS. A escolha dos buffers foi ajustada após uma falha real de memória da escrita inicial. Parquet TIMESTAMPTZ preserva o instante; leitores podem exibi-lo no fuso da sessão. Configurar `SET TimeZone='UTC'` para serialização UTC. Pact/provider de backend não foram executados porque essa integração ainda não existe.
