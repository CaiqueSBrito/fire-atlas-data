# Meteorologia GFS e ERA5 retrospectiva — RF10/CA08

O modo padrão integra previsões NOAA GFS distribuídas por Open-Meteo Single Runs: não usa ERA5, observações do dia-alvo, série costurada ou `best_match`. O modo adicional `--historical-only` integra exclusivamente reanálise ERA5 até D, conforme contrato abaixo. Nenhum modo usa credenciais ou infraestrutura própria. Acesso gratuito somente não comercial, sem SLA; atribuir Open-Meteo, NOAA GFS e Copernicus/ECMWF ERA5 conforme a fonte (dados meteorológicos CC BY 4.0).

## Execução

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.weather \
  --features output/rf10-activity/features/features.parquet \
  --output-dir output/rf10-environment/weather \
  --batch-size 32 --concurrency 3
```

A saída deve ser inédita: arquivos existentes não são substituídos. O módulo lê somente `DISTINCT cell_id, issue_date` do Parquet, não rótulos, nem splits. Obtém os centros H3 dos IDs. As datas precisam ter tipo DATE, estar desde 2026-04-02 (início documentado do arquivo para a maioria dos modelos) e ser anteriores ao dia UTC da execução. A janela reservada futura 2026-11-13..2026-12-12 não integra esta aquisição.

### Amostragem espacial fixada antes da medição

Amostra a saída do modelo no lattice de 0,25°: `round(latitude*4)/4`, `round(longitude*4)/4` (empates: regra Python para par). Deduplica esses nós por emissão, consulta cada nó uma vez e propaga seus valores à célula H3 associada. Isso reduz requisições e deixa explícitos os valores compartilhados; não transforma 161 células em 161 observações meteorológicas independentes. **0,25° é o espaçamento desta amostragem de saída**, não uma afirmação de resolução nativa ou efetiva do `gfs_global`; não se faz interpolação espacial adicional.

Mantém a seleção de célula `land` e o tratamento de elevação padrão da API. O manifesto registra centro H3, nó solicitado e coordenadas/elevação retornadas para cada lote. O ponto retornado pode diferir do solicitado. Essa representatividade espacial limita o uso para ignição localizada; nenhum resultado desta integração valida precisão científica.

## Contrato temporal e campos

Endpoint fixo: `https://single-runs-api.open-meteo.com/v1/forecast`. Toda requisição fixa `models=gfs_global`, `run=D T00:00`, `timezone=UTC`, `forecast_days=2`, temperatura Celsius, precipitação mm e `wind_speed_unit=ms`.

Seleciona exclusivamente os 24 timestamps de **D+1 00:00 até D+1 23:00 UTC**, previsão do ciclo inicializado em D00, antecedências 24..47 horas. Não seleciona o primeiro dia nem desloca o ciclo para um modelo mais recente. A inicialização não é publicação: a documentação informa atraso típico de 4–6 horas para modelos globais, mas isso não comprova disponibilidade de nenhum ciclo histórico específico. `available_at` permanece `null`; o instante real de download fica separado. A API não expõe uma versão imutável do executável GFS: o identificador de modelo e o ciclo são fixos, a versão upstream continua desconhecida.

| Coluna DOUBLE | Derivação sobre os 24 timestamps selecionados | Unidade |
| --- | --- | --- |
| `wx_tmax_c` | máximo `temperature_2m` | °C |
| `wx_rhmin_pct` | mínimo `relative_humidity_2m` | % |
| `wx_precip_mm` | soma `precipitation` | mm |
| `wx_windmax_ms` | máximo `wind_speed_10m` | m/s |
| `wx_valid_fraction` | quantidade de valores não nulos nos quatro campos / 96 | fração |

Cada agregado exige os **24 valores do seu campo**; se faltar um timestamp ou houver nulo, esse agregado é `null`, sem soma parcial ou imputação por zero. Outros campos completos continuam válidos. A fração usa denominador fixo 96, inclusive quando um timestamp inteiro falta. Dados não numéricos, não finitos, arrays malformados, timestamps duplicados/desordenados/não horários, unidades incompatíveis, chuva/vento negativos e umidade fora de 0..100 fazem a aquisição falhar visivelmente.

A precipitação da API é a soma da **hora precedente** de cada timestamp. Portanto o contrato solicitado de D+1 00..23 soma os intervalos que terminam nesses horários: inclui D23..D+1 00 e exclui D+1 23..D+2 00. Não a apresentar como total exato do calendário D+1 00..24. O primeiro nulo de chuva em D00 é preservado no bruto e, por estar fora do alvo, não invalida D+1.

## Limites, evidência e publicação

Lotes têm 1..64 localizações (`--batch-size`, padrão 32), URL até 6.000 bytes, resposta até 4 MiB e timeout HTTPS de 120 segundos. Concorrência aceita de 1..3; não há chamadas sequenciais por hora. O total de nós-datas é limitado a 4.800 por execução. O controle conservador conta cada localização como uma chamada e limita 480/minuto, 4.800/hora e 9.500/dia, abaixo dos limites públicos de 600/minuto, 5.000/hora e 10.000/dia. Outros consumidores do mesmo IP podem compartilhar a quota; o código não conhece esse consumo externo. Não há repetição automática nem troca silenciosa de fonte/modelo. Em falha, inspecionar os recibos e usar `--batch-size 1` para um diagnóstico real explicitamente menor, se necessário; não inventar valores.

Saída bem-sucedida:

- `covariates.parquet`: chaves únicas `cell_id VARCHAR, issue_date DATE`, exatamente o conjunto distinto das chaves de entrada, mais os cinco DOUBLE acima, nulos reais permitidos;
- `raw/YYYY-MM-DD-NNN.json`: bytes brutos recebidos por lote;
- `raw/YYYY-MM-DD-NNN.receipt.json`: URL, HTTP/headers, início/download real, tamanho/SHA-256, modelo/ciclo/lead, unidades no bruto, pontos solicitados e coordenadas retornadas;
- `manifest.json`: versão do método, configuração, `columns` como lista ordenada de nomes (também exportada como `WEATHER_FEATURES`), `column_units`, hash do Parquet de entrada/saída, mapa espacial, contagens, hashes dos recibos/brutos e limitações de disponibilidade.

Constrói tudo em diretório temporário irmão. Publicação reserva o destino sem substituição e cria links atômicos para arquivos completos; `manifest.json` é o marcador de commit e fica visível por último. Um consumidor não deve ler saída sem manifesto. Em erro, preserva o staging `.weather-acquisition-*` e `failure.json`, sem apagar o bruto ou declarar sucesso. Corrigir erro de interpretação não autoriza apagar os recibos anteriores.

`tests/test_weather.py` contém fixtures sintéticos de engenharia: limites de datas (incluindo ano bissexto), seleção temporal exata, nulos/horas faltantes, invariantes de arrays/unidades, identidade dos lotes, amostragem espacial, recibos e proibição de sobrescrever. Esses testes não demonstram disponibilidade real nem publicação histórica; a execução consolidada pertence ao integrador.

## Reuso explícito do bruto após falha

`--reuse-raw-dir DIRETORIO` (repetível) aceita um diretório de aquisição anterior, inclusive staging preservado de falha. Somente respostas HTTP 200 originalmente validadas entram no índice; URL completa, tamanho e SHA-256 devem coincidir. O novo processamento valida novamente todos os valores, copia os mesmos bytes, preserva o instante original de download e registra caminho/hash do recibo anterior e instante de revalidação em `reused_from`. Não confunde revalidação com nova aquisição nem modifica evidência anterior. Respostas com falha de transporte ou erro de dados não são reutilizadas.

`counts.http_requests` informa somente downloads novos; `response_batches` inclui lotes reutilizados e `reused_batches` os discrimina. Esse mecanismo não altera ciclo/modelo/variáveis, não imputa dados e não repete chamadas automaticamente.

## ERA5 histórico até a emissão D

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.weather \
  --historical-only \
  --features output/rf10-activity/features/features.parquet \
  --output-dir output/rf10-environment/era5 \
  --batch-size 32 --concurrency 3
```

Este modo adicional atende à comparação retrospectiva com meteorologia histórica: fixa `models=era5` no endpoint `https://archive-api.open-meteo.com/v1/archive`. Não usa `auto`, `era5_seamless`, ERA5-Land ou IFS. Requer chaves DATE passadas (desde 1940-01-07, para comportar a janela inicial de sete dias), não lê rótulos e usa o mesmo lattice espacial predefinido. Consulta os nós em lotes para **um intervalo comum `[min(issue_date)-6, max(issue_date)]`**, não uma requisição por dia ou hora. O limite final do pedido é D máximo, nunca D+1 observado.

`ERA5_FEATURES` exporta a ordem abaixo; `manifest.columns` contém essa mesma lista:

| Coluna DOUBLE | Derivação | Unidade |
| --- | --- | --- |
| `era5_tmax_c` | máximo dos 24 timestamps D 00..23 UTC | °C |
| `era5_rhmin_pct` | mínimo dos 24 timestamps D 00..23 UTC | % |
| `era5_precip_mm` | soma dos 24 valores horários de D | mm |
| `era5_windmax_ms` | máximo dos 24 timestamps D 00..23 UTC | m/s |
| `era5_precip_7d_mm` | soma dos **168** timestamps D−6 00..D 23 UTC | mm |
| `era5_valid_fraction` | valores não nulos nos quatro campos de D /96 | fração |

Um campo diário incompleto permanece nulo. A precipitação de sete dias exige todas as 168 horas, inclusive a fronteira D−6; um único nulo ou hora ausente produz nulo, nunca soma parcial. A fração mede apenas os quatro campos de D; completude dos sete dias é indicada pela própria nulidade de `era5_precip_7d_mm`. Aplica a mesma validação estrita de tipos, timestamps, unidades e domínios do modo GFS. A convenção de precipitação da hora precedente também se aplica: o primeiro timestamp D−6 00 representa a hora que termina naquele instante, não um total de calendário exato D−6 00..D+1 00.

A documentação do arquivo informa ERA5 desde 1940, com atraso típico de cinco dias e revisões. Portanto estes campos são **somente retrospectivos**: consultar a reanálise agora não prova sua disponibilidade na emissão D. Datas de análise, produção, download e disponibilidade são conceitos separados. O recibo identifica `product_type=reanalysis`; `produced_at`, versão upstream e `available_at` continuam desconhecidos, e o manifesto declara explicitamente a impossibilidade de afirmar disponibilidade histórica no corte. Não promover este experimento a previsão operacional sem outra fonte/latência demonstrada.

Preserva `covariates.parquet`, `manifest.json`, respostas brutas e recibos/hash no diretório próprio `era5`, sem misturar o produto com GFS. As quotas são conservadoramente ponderadas por localização e `ceil(dias/14)`; o intervalo de 43 dias custa no máximo quatro unidades por nó. Mesmos limites de lote, resposta, URL e concorrência; os dois modos devem ser executados sequencialmente para manter concorrência global até três. Os testes sintéticos incluem fronteiras de ano/bissexto, missing no início da janela, nulidade em D, exclusão de D+1 observado, modelo explícito e preservação da proveniência em reuso.

## Aquisição real observada

Artefatos produzidos em `output/rf10-environment/weather` e `output/rf10-environment/era5`, a partir do Parquet de entrada SHA-256 `fb27d36b0ada02ee473824d4dc7c9d6d004fad39b258641574bd11b32515ade5`. Os dois têm **161 células, 37 emissões e 5.957 linhas**, amostradas em **75 nós** distintos do lattice. Todas as linhas apresentaram os campos completos; nenhum nulo de agregado foi fabricado/preenchido. ERA5 consultou 2026-09-09..2026-10-21, sem meteorologia observada posterior à última emissão.

| Evidência medida | GFS | ERA5 |
| --- | ---: | ---: |
| Lotes no artefato final | 111 | 3 |
| Novos pedidos HTTP da execução concluída | 79 | 3 |
| Lotes brutos reaproveitados com URL/hash idênticos | 32 | 0 |
| Nós × emissões (GFS), orçamento conservador ponderado (ERA5) | 2.775 | 300 |
| Bytes de respostas brutas no artefato final | 6.092.825 | 2.894.576 |
| Bytes do Parquet | 26.040 | 32.919 |

SHA-256 dos Parquets:

- GFS: `8c4b7428aeb92f52e241ade347b8c8601aa5aeeb2874ce3826cb8eab2348dffc`.
- ERA5: `64d11b3b0bcc16905af934440e51bda904cf2267bad08bdbd445e258012cf9b2`.

SHA-256 dos manifestos:

- GFS: `2c2e14a91306034595e5e46f943f7090b91cd62b29428e304485d628cf5adec6`.
- ERA5: `7a3c91511654d74e2794094018494605f89a2c2d6d0ea7168784134282364b58`.

Ocorrências preservadas, não escondidas: o primeiro processamento recusou seis respostas porque Open-Meteo omite `location_id=0` no primeiro objeto do lote; o parser passou a aceitar essa representação, exigindo ainda a ordem dos demais IDs. Um segundo processamento teve timeout TLS real em um lote, mantendo 32 respostas completas reutilizáveis. A requisição diagnóstica real de um ponto retornou HTTP 200, 2.179 bytes e valores completos, antes da conclusão por reuso explícito. Esses recibos permanecem, respectivamente, em `.weather-acquisition-uqewnslz`, `.weather-acquisition-dnn2deny` e `weather-transport-diagnostic`, sob `output/rf10-environment`. Não são saídas finais nem evidência de disponibilidade histórica.

O módulo foi efetivamente executado para adquirir os dois produtos. Não foram executados testes, treino, linters ou formatadores nesta entrega; a suíte consolidada é responsabilidade do integrador.

## Fontes primárias

Consultadas em 2026-11-12:

- [Single Runs API](https://open-meteo.com/en/docs/single-runs-api): ciclo individual por inicialização, arquivo desde abril/2026, inicialização ≠ publicação.
- [GFS API](https://open-meteo.com/en/docs/gfs-api): campos, unidades, múltiplas coordenadas, seleção espacial e precipitação da hora precedente.
- [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api): ERA5 explícito, atraso, campos e fronteiras horárias; não confundir reanálise com previsão emitida.
- [Model updates](https://open-meteo.com/en/docs/model-updates): atualização/disponibilização não equivalem a inicialização; metadados atuais não provam disponibilidade histórica.
- [Pricing e limites](https://open-meteo.com/en/pricing): acesso não comercial, quotas e contabilização por parâmetros/localizações.

Não usar o sucesso do download para declarar disponibilidade operacional no corte D ou prontidão do produto. A evidência de execução efetiva e seus hashes ficam no manifesto do artefato produzido.
