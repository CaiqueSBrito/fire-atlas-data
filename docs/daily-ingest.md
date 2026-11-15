# Job diário: FIRMS → silver → previsão de 7 dias (RF11, RF10/CA08)

`daily-ingest` executa uma emissão D depois do corte (`--cutoff-hour`, padrão 09:00 UTC):

1. **Início da busca por produto:** 1 dia (`OVERLAP_DAYS`) antes da data do `acquired_at` mais novo na silver dentro de `--lookback-days` (padrão 10). Sem registro nesse período, a busca começa no início da janela. A sobreposição existe porque o NRT publica passagens fora de ordem: em 14/11/2026, 1.226 detecções NOAA-20 das 09:56–10:15 UTC chegaram depois de uma passagem das 10:42 já ingerida.
2. **Busca:** Area API `world`, em janelas de até 5 dias, desse início até a data UTC atual. A chave vem do ambiente (`--key-env`, padrão `FIRMS_MAP_KEY`) e nunca é gravada.
3. **Normalização** de cada CSV não vazio.
4. **Append na silver** (`silver.append`): só entram observações que a silver ainda não tem, comparando a identidade (`source_product`, `acquired_at`, `latitude`, `longitude`). Os meses são gravados do mais antigo para o mais novo. Repetir ou retomar uma execução não duplica linhas. `--optimize` aplica Z-order nos meses gravados.
5. **Previsão:** `ml-forecast` substitui as linhas de D em `--forecast-target`.
6. **Localização (opcional):** com `--places-target`, `--places-countries` e `--places-cities` (os três juntos), `gold-places` reconstrói a tabela de continente/país/cidade a partir da nova versão da silver e a confere com `verify`. As referências podem estar no S3; ver [gold-places](gold-places.md). Se esse passo falhar, a execução falha e é repetida, e a previsão é regravada sem duplicar.
7. **Harmonização (opcional):** com `--harmonize-model`, `--harmonize-target` e `--harmonize-availability` (os três juntos), `harmonize` regrava os anos do dia local que os dados novos podem alterar: do início da busca menos 30 dias até hoje. Na prática é o ano corrente, e também o anterior em janeiro. Em seguida, `verify` confere esses anos. As tabelas já precisam existir; a execução diária não cria uma tabela só com anos recentes. Ver [harmonização](harmonization.md). Se esse passo falhar, a execução falha e é repetida.

Produtos padrão: `MODIS_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT`. O S-NPP parou de gerar dados novos em 02/11/2026; para incluí-lo, use `--product`. Cada execução grava CSVs, recibos, Parquets normalizados e `report.json` em `<work-dir>/<D>/run-<hora UTC>/`, sempre local. Um lock de arquivo no `work-dir` garante um único escritor local.

`--silver`, `--model`, `--report` e `--forecast-target` aceitam caminho local ou URI `s3://`. As credenciais vêm da cadeia padrão da AWS (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` ou papel da instância), com `AWS_REGION=us-east-2`. O delta-rs 1.6 grava commits no S3 sem provedor de lock, mas isso não substitui ter um único escritor.

```bash
LAKE=s3://data-lake-fire-atlas-553607359608-us-east-2-an
PYTHONPATH=src python -m fire_atlas_data daily-ingest \
  --silver $LAKE/prata/silver \
  --model $LAKE/models/xgb-rf10/model.ubj --report $LAKE/models/xgb-rf10/report.json \
  --forecast-target $LAKE/gold/forecast \
  --places-target $LAKE/gold/h3_places \
  --places-countries $LAKE/reference/ne_10m_admin_0_countries.zip --places-cities $LAKE/reference/cities1000.zip \
  --harmonize-model $LAKE/models/harmonization/v0.3.0 \
  --harmonize-target $LAKE/gold/harmonized/cells --harmonize-availability $LAKE/gold/harmonized/availability \
  --work-dir data/daily --date 2026-11-14
```

Com `--daily-at HH:MM` (UTC), o processo continua rodando: processa cada dia a partir desse horário e, se falhar, tenta de novo a cada `--retry-minutes`.

## Produção na AWS (us-east-2)

A stack `deploy/aws/daily-ingest.yaml` (`fire-atlas-daily-ingest`) cria:
- um repositório ECR com tags imutáveis;
- um cluster ECS e uma task Fargate de 4 vCPU, 16 GB e 30 GB de disco efêmero;
- os papéis de execução, de task e do scheduler, com escopo mínimo;
- um security group só de saída;
- o log group `/ecs/fire-atlas-daily-ingest`, com retenção de 90 dias;
- um agendamento no EventBridge Scheduler, `cron(0 10,14 * * ? *)` em UTC.

A execução das 14:00 repete o dia de forma idempotente, cobrindo falha às 10:00 ou atraso do FIRMS. A task:
- lê e grava no S3 (`prata/silver`, `gold/forecast`, `gold/h3_places`, `gold/harmonized`) e lê `models/xgb-rf10/`, `models/harmonization/` e `reference/`;
- arquiva cada execução em `bronze/daily-ingest/<D>/<run>/`;
- recebe `FIRMS_MAP_KEY` do SSM SecureString `/fire-atlas/firms-map-key`, criado fora da stack.

```bash
aws ssm put-parameter --region us-east-2 --name /fire-atlas/firms-map-key --type SecureString --value "$FIRMS_MAP_KEY"  # uma vez
AWS_REGION=us-east-2 deploy/aws/deploy.sh   # build, deploy da stack com tag = hash da imagem, push no ECR
```

Para executar fora do horário, use `aws ecs run-task` com o mesmo cluster, task definition e configuração de rede das saídas da stack. Para pausar, faça o deploy com `ScheduleState=DISABLED`.

## Container local

```bash
cp .env.example .env   # preencher FIRMS_MAP_KEY e as credenciais AWS; .env é ignorado pelo git
docker compose up -d --build
docker compose logs -f daily-ingest
```

O `compose.yaml` lê e grava no S3 (`$LAKE`, padrão o bucket do projeto). Só `./data` é montado, para os artefatos de cada execução. O container executa às 10:00 UTC com `--optimize`. A imagem já traz a extensão H3 do DuckDB. O compose exige `FIRMS_MAP_KEY` no shell ou no `.env`, inclusive para `docker compose build`. Credenciais temporárias (`aws configure export-credentials`) expiram em minutos ou horas. Não rode o container local junto com o agendamento da AWS: o lock é por host.

## Limitações

- Registros publicados mais de 1 dia depois da data da aquisição mais nova já ingerida não entram.
- Linhas idênticas dentro do próprio arquivo FIRMS são mantidas, como em toda a silver. Exemplo: um par NOAA-21 de 14/11 05:35 UTC, registros 38253/38254.
- Download vazio significa cobertura desconhecida, não ausência de fogo.
- O NRT não traz `type`, então fontes estáticas (flares, vulcões) continuam no universo da previsão.
- O lock é local e não impede dois hosts de gravarem na mesma tabela.
- Cada previsão lê do S3 as colunas da silver inteira para montar o universo de células. Fora da AWS, isso gera tráfego de saída a cada execução.
- As probabilidades são experimentais: os limiares de CA08 não foram aprovados.
- Registros que o FIRMS retira ou substitui depois de publicar continuam na silver, porque ela só recebe inclusões. Em 14/11/2026, a contagem de NOAA-21 de 13–14/11 caiu de 75.118 para 75.116 entre duas consultas.

Smoke em 14/11/2026:
- **Container local contra o S3** (`prata/silver` v634 → v636): 2.355 observações novas (1.226 NOAA-20 tardias, mais 1.129 do fim de passagem de 13/11 que faltavam). A previsão para D=2026-11-14, com 2.217.846 células, foi gravada em `gold/forecast` em 635 s, quase todo o tempo lendo a silver de fora da AWS.
- **Fargate em us-east-2** (task definition rev. 4, imagem `d0091d03ea693d74`): saída 0 em cerca de 4 min. Entraram 4.844 observações novas, a previsão de D=2026-11-14 foi regravada e a execução foi arquivada em `bronze/daily-ingest/2026-11-14/run-20261114T154414Z/` (16 objetos).
- **Rodadas anteriores no Fargate:** falharam no arquivamento (HeadBucket e marcador `bronze/` sem permissão) depois de gravar silver e previsão. A execução seguinte não duplicou nada.
