# Máscara estática e atividade global H3

Os módulos `fire_atlas_data.static_mask` e `fire_atlas_data.global_activity`
implementam as etapas 2–3 do protocolo global multissensor. Não baixam catálogos,
não treinam modelos e não executam backfill automaticamente. Consomem diretórios
de normalização **1.3.0**, incluindo `hotspot_type` anulável.

## Máscara congelada

API:

```python
from datetime import date
from fire_atlas_data.static_mask import build

build(datasets, external_points, output_dir,
      reference_end=date(2012, 12, 31), min_recurrent_years=2,
      mask_resolution=8, buffer_rings=0, memory_limit="2GB", threads=2)
```

Os valores científicos do exemplo são ilustrativos, não recomendações nem
padrões validados. Todos os quatro parâmetros científicos são obrigatórios:

- `reference_end`: corte inclusivo pela data UTC de aquisição;
- `min_recurrent_years`: número positivo de **anos distintos** com `hotspot_type=2`
  na mesma célula fina, até o corte; várias detecções no mesmo ano não aumentam
  o número de anos;
- `mask_resolution`: resolução H3 inteira entre 6 e 15, mais fina que res5; o protocolo usa res8;
- `buffer_rings`: expansão explícita em anéis H3, podendo ser zero.

`external_points` aceita `None` ou CSV com `latitude,longitude`; `lat` e `lon`
são aliases aceitos individualmente. Coordenadas inválidas abortam a publicação.
A validade temporal e a procedência científica dos pontos externos são
responsabilidade do operador: um CSV atual não pode ser considerado histórico
apenas por receber um `reference_end` antigo. Sem dados normalizados e sem CSV,
a máscara vazia é válida e fica documentada no manifesto.

Ao fornecer pontos externos, o operador declara que eles eram conhecidos até
`reference_end`; essa declaração é persistida em
`sources[].declared_knowledge_cutoff`, junto do SHA256 e do estado
`temporal_validity: caller_declared_not_verified`. Não é possível inferir
disponibilidade histórica pelo ano no nome ou em uma coluna do arquivo.
Em particular, o catálogo ORNL de flares 2012–2016 resolveu posições usando o
conjunto completo 2012–2016: suas linhas rotuladas 2012 não constituem uma fonte
causal disponível em 2012. Valores sentinela como `-9999` são coordenadas
inválidas e são recusados. Nenhum catálogo externo foi baixado por estes módulos;
a licença e as condições de acesso devem ser verificadas pelo operador.

A união dos pontos externos e das células recorrentes recebe o buffer. A origem
por célula tem prioridade `external`, depois `recurrent_type`, depois `buffer`.
O arquivo `mask.parquet` contém somente `cell_id VARCHAR, origin VARCHAR`, com
uma linha por célula. O manifesto usa `schema: static-source-mask/1.0.0`,
`parameters`, `sources`, `counts`, `outputs` e `limitations`.

CLI equivalente, sem parâmetros científicos implícitos:

```sh
python -m fire_atlas_data.static_mask \
  --datasets /dados/normalizado-historico \
  --external-points /dados/pontos-estaticos.csv \
  --output-dir /dados/mascara-congelada \
  --reference-end 2012-12-31 --min-recurrent-years 2 \
  --mask-resolution 8 --buffer-rings 0 --memory-limit 2GB --threads 2
```

É possível omitir `--external-points`, `--datasets`, ou ambos. Para uma avaliação
causal, o corte da máscara deve ser anterior ou igual a **todas** as datas de
emissão que a utilizam. Uma máscara calculada até 2022 não pode ser reutilizada
silenciosamente em exemplos de treino de 2013–2022; isso introduziria informação
futura. Para o painel iniciado em janeiro de 2013, use uma máscara congelada
antes da primeira emissão e fontes externas temporalmente compatíveis.

## Atividade multissensor

```python
from fire_atlas_data.global_activity import build

build(datasets, mask_dir, output_dir, memory_limit="2GB", threads=2)
```

```sh
python -m fire_atlas_data.global_activity \
  --datasets /dados/viirs /dados/modis \
  --mask-dir /dados/mascara-congelada --output-dir /dados/atividade \
  --memory-limit 2GB --threads 2
```

Uma detecção participa da atividade não estática somente quando:

1. `hotspot_type IS NULL OR hotspot_type = 0`; e
2. a célula H3 **fina das coordenadas da detecção** não pertence à máscara.

`NULL` significa tipo desconhecido, não comprovação de fogo em vegetação.
Tipos 1, 2 e 3 são excluídos mesmo fora da máscara. A máscara nunca elimina uma
célula res5 inteira quando apenas uma subcélula está marcada. Todas as detecções
aceitas, inclusive as estáticas, estabelecem o domínio de células, a primeira
data de detecção e os dias de reporte por plataforma.

Saídas:

| Arquivo | Conteúdo |
| --- | --- |
| `daily/year=YYYY/*.parquet` | Uma linha por célula res5/data UTC com pelo menos uma detecção não estática. Colunas `cell_id`, `acquisition_date`, `viirs_detections`, `modis_detections`, `viirs_frp_sum_mw`, `viirs_frp_rows`, `modis_frp_sum_mw`, `modis_frp_rows`; `year` é a partição Hive. |
| `cells.parquet` | Todas as células aceitas: `cell_id`, centro `latitude,longitude`, `first_detection_date`, `parent4_id`, `parent3_id`. |
| `neighbors.parquet` | `cell_id,neighbor_id`, anel 1 sem a própria célula; vizinhos podem estar fora do domínio observado; pentágonos têm cinco vizinhos. |
| `platform_days.parquet` | `sensor,platform,acquisition_date,detections`, incluindo detecções estáticas. |
| `manifest.json` | `schema: global-cell-activity/1.0.0`, `h3_resolution: 5`, intervalo de aquisição, hash/parâmetros da máscara, fontes, contagens, artefatos e limitações. |

Contagens e contagens FRP são `BIGINT`; somas FRP são `DOUBLE` em MW. Soma FRP
zero com `*_frp_rows=0` representa ausência de medidas, **não** FRP físico zero.
Um valor medido zero tem contagem FRP positiva. As famílias MODIS e VIIRS são
mantidas separadas, sem harmonização nem remoção de sobreposição entre
plataformas diferentes.

Se todas as detecções forem estáticas, `daily/empty.parquet` conserva o esquema
vazio com `year INTEGER`; células e reporte permanecem presentes. Use leitura
recursiva de `daily/**/*.parquet` com Hive. Uma coleção inteiramente sem
observações aceitas é rejeitada por não fornecer intervalo de aquisição.

## Integridade, recursos e interpretação

- SHA256 do CSV original identifica cada fonte: duas entradas com o mesmo hash
  são recusadas, mesmo que os diretórios sejam diferentes.
- Os artefatos Parquet aceitos são lidos a partir da lista do manifesto e têm
  tamanho/SHA256 verificados; versão e hash de origem das linhas também são
  conferidos. A máscara recebe a mesma verificação de artefato e unicidade.
- Intervalos fechados `[min(acquired_at), max(acquired_at)]` são calculados por
  diretório, sensor e plataforma. Interseções entre diretórios da mesma
  plataforma são recusadas, inclusive entre SP e NRT; lacunas internas não
  autorizam combinar intervalos sobrepostos. Intervalos de plataformas
  diferentes são permitidos. Essa política conservadora não deduplica eventos.
- Os manifestos registram hashes dos manifestos de entrada, hashes dos arquivos
  aceitos e hashes/tamanhos dos arquivos publicados. A proveniência da máscara
  e seu corte acompanham o manifesto de atividade.
- DuckDB usa banco intermediário e spill no diretório de staging, limite de
  memória e número de threads explícitos. Transformações/agregações são SQL;
  funções Python H3 convertem coordenadas/células sem carregar a tabela de
  observações em listas Python. Memória de processos, metadados e código Python
  não são uma garantia de RSS limitada pelo ajuste do DuckDB.
- O diretório final deve estar ausente. Staging irmão, manifesto completo e
  rename no mesmo filesystem publicam o conjunto atomicamente; erros removem
  staging sem sobrescrever resultados existentes. É necessário espaço para
  intermediários, saída e spill. Não há alegação de benchmark de escala global.
- Dias de reporte não comprovam cobertura completa de uma célula nem operação
  futura de uma constelação. Ausência de linha não é negativo físico. Hora de
  aquisição não é disponibilidade operacional (`available_at` desconhecido).
- Para features e alvo de sete dias, a primeira detecção deve ser anterior ou
  igual à emissão; não use o domínio observado no futuro para criar candidatos
  no passado. Janelas sem suporte não devem ser preenchidas com zero.

Os testes de contrato em `tests/test_global_activity.py` cobrem filtro fino,
classes FIRMS, corte/recorrência anual, prioridade/buffer, FRP ausente,
sobreposição, proveniência, saídas vazias e falhas sem publicação parcial.
