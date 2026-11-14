# Qualidade do ranking RF10/CA08 — experimento retrospectivo

Este experimento **não altera** `ranking.py`, os modelos originais nem os artefatos de `output/rf10-prototype`. Escores ordenam detecções observadas; não são probabilidades calibradas de incêndio. Não há meteorologia, credenciais externas ou dados inventados.

## Hipótese e protocolo congelado

Mais exemplos de treinamento e recorrência espacial podem melhorar recall diário de validação. Permanecem fixos: bbox `[-48,-17,-46,-15]`, H3 resolução 5, top 10, rótulo de nova detecção D+1 e elegibilidade sem detecções em D, D−1 ou D−2. Treino ampliado: emissões 15/set–10/out/2026; validação: 11–15/out; TEST: 16–20/out. NOAA20 local de 1/set a 22/out (último dia parcial) é a fonte disponível.

Warmup mínimo: 14 dias, começando em 1/set. As janelas de 30 dias no início do treino são **censuradas à esquerda**, não histórias completas. `history_days` informa idade de calendário desde o início declarado, limitada a 30 dias; não comprova cobertura observacional. Fontes, datas de partição e início do histórico podem variar na publicação; geometria, método, nomes/ordem/tipos das features e demais parâmetros não podem.

O TEST original já foi observado (10/36 capturados no top 10: 27,8%). Portanto **todo resultado neste TEST é diagnóstico reutilizado, nunca prova independente de generalização**. Uma melhora na validação pode regredir no TEST: o relatório explicita esse caso e o vencedor não é trocado depois de observar o diagnóstico.

## Busca finita, sem ajuste pelo TEST

`quality.PROFILES` e `_profiles()` predeclaram 11 candidatos treinados, além do original congelado (12 alternativas no total):

1. Modelo original `full`, carregado e verificado pelo hash do manifesto original, nas colunas `ranking.FULL` sem reconstruir seu treinamento.
2. Mesmos parâmetros originais (120 árvores, profundidade 3, taxa 0,05) retreinados nas features originais com treino ampliado e política 10.
3. Três perfis enhanced: `(profundidade, árvores, taxa, min_child_weight, lambda)` = `(2,80,0.03,5,10)`, `(2,160,0.08,1,1)`, `(3,80,0.08,5,10)`; cada qual com políticas 10 e ALL.
4. Dois perfis apenas espaciais/recorrência: `(3,160,0.03,1,10)`, `(4,160,0.08,5,10)`; cada qual com políticas 10 e ALL. Excluem sazonalidade, dia do ano e agregados regionais; mantêm vizinhança, tempo desde última detecção e todas as features enhanced de célula/anel, FRP, coordenadas e cobertura histórica.

Todos são árvores CPU `hist`, objetivo `binary:logistic`, seed 42, duas threads, subsample 0,8, colsample 1. O objetivo de ajuste não transforma seus escores em probabilidades calibradas. Modelos novos usam `xgboost.train`/`DMatrix` com nomes nativos exatos; JSONs originais sem nomes têm verificação explícita da ordem `FULL` e contagem.

**Política 10:** por região/data de treino, preserva todos os positivos e até dez controles por positivo, escolhidos por hash determinístico. Grupos sem positivos não contribuem controles. **ALL:** preserva todos os candidatos de treino, inclusive controles em grupos região/data sem positivos. A política não altera nenhuma linha de validação ou TEST. Controles são ausência de detecção observada, não ausência confirmada de fogo.

A seleção usa exclusivamente média diária (região/data) de recall@10 na validação. Dias sem positivos têm recall nulo, são preservados no relatório, mas não entram nessa média. Validação inteira sem positivos interrompe o experimento. Empates mantêm o original; entre candidatos que superam o original, favorecem o primeiro perfil listado. Nenhum candidato que empate ou perca na validação substitui o original.

A decisão e todas as métricas de validação são gravadas em `selection.json` **antes** de qualquer predição de TEST. Somente original e vencedor congelado são avaliados nesse TEST, nas mesmas linhas. As referências simples `neighbors_7d` e `cell_active_days_30d` são reportadas em validação/TEST, fora da seleção. Uma referência sazonal constante zero não é apresentada como baseline útil.

## Execução

Depois de gerar features com `configs/quality.json` (opções `enhanced_features=true`, `minimum_history_days=14`):

```sh
PYTHONPATH=src python -m fire_atlas_data.features --config configs/quality.json --output-dir output/rf10-quality/features
PYTHONPATH=src python -m fire_atlas_data.quality train --features output/rf10-quality/features/features.parquet --output-dir output/rf10-quality/model --baseline-model-dir output/rf10-prototype/model-heldout --k 10
PYTHONPATH=src python -m fire_atlas_data.quality publish --features output/rf10-quality/features/features.parquet --model-dir output/rf10-quality/model --output-dir output/rf10-quality/published
```

Diretórios de saída existentes nunca são sobrescritos. O manifesto de features é obrigatório. Para nova publicação, forneça uma partição `publish` com rótulos nulos: a inferência não exige nem consulta rótulos D+1 e não recorre silenciosamente ao TEST. A data nominal de emissão é o fim do dia histórico em UTC; o instante de geração do artefato é separado. O mapa local reutiliza o renderer original e não usa mapa-base remoto.

## Artefatos e auditoria

- `manifest.json`: vencedor e versão, ordem de features de cada modelo, tipos e contrato enhanced completo, perfis finitos, versões de bibliotecas, configuração/método, hashes do código, entradas, manifesto original e artefatos. Conserva o manifesto de proveniência das features, incluindo fontes e censura histórica.
- JSON nativo de cada modelo, incluindo cópia verificada do original. Cada inferência recarrega o JSON e verifica seu hash; no treino a predição recarregada de validação precisa ser exatamente igual à do modelo em memória.
- `selection.json`: todas as métricas de validação e contagens de positivos/controles amostrados por candidato, com decisão congelada antes do TEST. As contagens do treino original não são inferidas: consulte seu manifesto original preservado.
- `comparison.json`: métricas diárias e agregadas micro/macro, validação de todas as alternativas, TEST do original/vencedor, referências de recorrência, contagens do treino ampliado, explicação das políticas, indicador e delta de regressão diagnóstica.
- `test_predictions.parquet` e `.csv`: todas as linhas de TEST para cada modelo comparado, rótulo, escore, rank, seleção top 10, nome/versão. Se o original vence, é exportado uma vez, sem duplicar artificialmente a comparação.
- Publicação: `rankings.parquet`, `map.html`, `publication.json`, com região/célula, emissão nominal UTC, alvo, escore, posição e versão. A publicação verifica contrato, hashes de modelos/código e hashes dos artefatos de avaliação.

`tests/test_quality.py` cobre políticas por grupo, dias sem positivos, desempates conservadores, nomes/ordem, contrato com datas variáveis, treinamento/reload/publicação, integridade de artefatos, congelamento anterior ao TEST e invariância da seleção ao trocar exclusivamente seus rótulos. Suíte consolidada executada: **85 testes passaram em 11,67 s**; evidência em `.omp/evidence/RF10-quality/tests.log`.

## Novo lote: avaliação congelada e promoção bloqueada

```sh
PYTHONPATH=src python -m fire_atlas_data.quality evaluate --features output/rf10-quality/public-evaluation-features/features.parquet --model-dir output/rf10-quality/model --output-dir output/rf10-quality/independent-evaluation
```

`evaluate()` carrega exclusivamente original e vencedor já congelado, nunca chama ajuste, busca ou seleção. Exige linhas rotuladas `test` com datas posteriores ao diagnóstico anterior, novo lote de features, contrato idêntico e hashes dos artefatos/modelos/seleção íntegros. Exporta todas as linhas com rótulos, escores, posições e top 10 em CSV/Parquet; `comparison.json` contém métricas diárias/micro/macro, fontes e hashes. `independent_evaluation.json` encadeia os hashes dos resultados.

A ferramenta foi adicionada **depois** do treino congelado: não reescreve seu manifesto nem refaz seleção para atualizar hash. Registra explicitamente o hash histórico de `quality.py`, o hash do avaliador atual e se divergem. Mantém a verificação do código de scoring compartilhado, do contrato de features e dos JSONs nativos.

Para o novo lote público NOAA-20 de 5–12/nov/2026, a regra de datas foi definida sem consultar escores: emissões 8–10/nov, alvos 9–11/nov, excluindo bordas parciais e preservando os três dias de elegibilidade. O histórico fornecido combina setembro/outubro e novembro, com **lacuna conhecida de 23/out–4/nov**. O relatório inspeciona datas presentes nas fontes NOAA-20 e registra intervalos sem detecções fornecidas; isso não certifica cobertura satelital contínua. `history_days` é idade de calendário, não disponibilidade dos dados. Essa lacuna e a nova fonte/data podem alterar recorrência e tempo desde detecção: há covariate shift/censura e não uma comparação limpa com 14 dias contínuos de observação. Poucos dias independentes e emissão nominal histórica não demonstram disponibilidade operacional.

Uma regressão no TEST diagnóstico original bloqueia automaticamente `quality.publish` com erro explícito. A avaliação nova sempre declara `promotion_unlocked=false`: bons resultados posteriores não mudam o vencedor, não autorizam retuning e não desbloqueiam promoção automaticamente. Os testes adicionais proíbem chamadas a fit/search/select, conferem métricas contra as linhas exportadas, preservação dos arquivos congelados, bloqueio de publicação e rejeição de datas antigas, contrato alterado e modelo corrompido.

O build real desse novo lote, informado pelo integrador, contém **461 candidatos e zero positivos observados** nas emissões fixas 8–10/nov. As datas e a região não são deslocadas para procurar positivos. `evaluate()` aceita esse caso, conserva recall diário/micro/macro como `null` e declara `gain_measurement_supported=false`: é evidência insuficiente para medir ganho de captura, não confirmação de ausência de fogo. Esse lote não resolve a falta de prova independente de melhora neste recorte.

## Resultado observado: não promover

O treino real cresceu para 3.625 candidatos/155 positivos. O perfil `enhanced_2_ALL` (80 árvores, profundidade3, taxa0,08, `min_child_weight=5`, lambda10, todos os controles) venceu somente pela validação: capturou 4/12 contra 1/12 do original. No TEST diagnóstico congelado, capturou 4/36 (11,1%) contra 10/36 (27,8%) do original, mantendo 50 posições destacadas no total. A publicação dessa versão foi exercitada e recusada, sem criar diretório de ranking.

O lote público novo foi de fato baixado, normalizado e avaliado sem ajuste dos modelos: 65.247 observações aceitas; 461 candidatos e nenhum positivo no recorte/período fixados. Recall permanece indefinido, sem evidência de ganho. Não alterar recorte, datas ou orçamento para fabricar melhora. O modelo anterior e suas limitações continuam preservados; **não houve aumento comprovado da qualidade preditiva**. Protocolo, hashes, logs e próximo responsável no [dossiê](../.omp/tasks/RF10-qualidade-preditiva.md).

## Experimento opcional com histórico Suomi-NPP

`configs/quality-snpp-history.json` combina o NOAA-20 original com
`output/snpp-scientific-real/accepted/**/*.parquet`, mantendo bbox, resolução5,
top10, warmup NOAA-20, mínimo14 dias e exatamente os mesmos splits e publicação
do `quality.json`. O arquivo científico Suomi-NPP é covariável retrospectiva de
anomalias térmicas, não MODIS, novo alvo ou novo período de validação.

O contrato ativo é `HISTORY_FEATURES` (32 colunas existentes + oito descritas em
[features](features.md)), com ordem/tipos exatos também no treino, publicação e
avaliação congelada. `ENHANCED_FEATURES` continua sendo um modo ativo de32 colunas.
Os mesmos11 perfis e parâmetros finitos são usados: original expandido de9
colunas, seis perfis enhanced+history e quatro spatial+history, com os mesmos
critérios de validação e desempate. O campeão original congelado continua lendo
somente suas9 colunas; dados SNPP não alteram os valores dessas colunas.

Use diretórios novos para features e modelos; nenhuma saída anterior é sobrescrita.
TEST continua diagnóstico reutilizado, nunca entrada de seleção/retuning, e uma
regressão continua bloqueando publicação. A inclusão do histórico não demonstra
ganho por si só: apenas comparação medida, sem promoção automática, pode quantificar
o resultado. `available_at` desconhecido impede alegação operacional.

### Resultado executado com histórico

Em `output/rf10-snpp-history/`, o perfil `spatial_history_4_10` venceu pela
validação (recall macro diário 37,5% contra 8,33%; captura total 3/12 contra
1/12). No TEST diagnóstico reutilizado, capturou 5/36 contra 10/36 do original:
recall micro 13,89% contra 27,78%, macro 11,67% contra 26,33%.
Não há melhora comprovada de generalização; a publicação foi exercitada e bloqueada.
Os 11 candidatos tiveram inferência dos JSONs recarregados comparada exatamente
à dos modelos em memória. Suíte consolidada: 96 testes passaram.

Os comandos de treino/publicação devem apontar para `features/features.parquet`,
não para o diretório que também contém `daily.parquet` e `candidates.parquet`.
Manifestos, seleção congelada e predições CSV/Parquet foram preservados.
Detalhes e limites no [dossiê SNPP](../.omp/tasks/RF10-historico-snpp.md).

## Experimento semanal RF10/CA08

O alvo separado de alguma detecção em D+1…D+7 segue o [protocolo semanal v1](weekly-protocol.md), preservado como especificação anterior ao fit. O usuário autorizou o treinamento em 12/11/2026. Comando executado:

```sh
.venv/bin/python .omp/evidence/RF10-weekly/train.py
```

Saída completa: `output/rf10-weekly/model-v1-run2/`. O comando recusa sobrescrita. `model-v1/` contém tentativa parcial que falhou na exportação Arrow, antes de seleção ou avaliação final; não é entrega concluída.

Três folds temporais selecionaram XGBoost contra a referência de recorrência: recall macro@10 médio de 15,8725% contra 15,6347%, com deltas positivos nos três folds. Seleção congelada antes de avaliar 2024–2025, sem retuning.

| Recall macro diário@10 | 2024 | 2025 |
|---|---:|---:|
| XGBoost | 18,4694% | 17,0325% |
| Recorrência de sete dias, referência congelada | 18,1994% | 16,9938% |
| Climatologia célula/mês | 21,1578% | 21,2027% |

O ganho contra a referência congelada é pequeno; climatologia supera XGBoost nos dois anos finais. Não há superioridade geral, significância inferencial, calibração probabilística ou promoção operacional demonstrada. O período final agora observado não pode governar retuning com alegação posterior de independência.

Modelos nativos JSON dos folds e refit final, configuração efetiva, versões, hashes, seleção, métricas mensais/anuais e previsões Parquet foram preservados. Recarga exatamente igual à inferência em memória; TP e recall macro foram recalculados independentemente dos Parquets, com dez destaques e 161 células únicas por emissão/modelo. Evidência em `.omp/evidence/RF10-weekly/training-verification.json`; decisões e limites no [dossiê semanal](../.omp/tasks/RF10-atividade-7dias.md). A obrigação de produto D+1 permanece intacta.


## Melhoria semanal v2: precisão sem alterar o orçamento

O pedido de elevar os aproximadamente 35% refere-se à **precisão@10**, não
acurácia geral: dos dez destaques diários, quantos têm detecção fornecida
em D+1…D+7. O [protocolo v2](weekly-improvement.md) foi registrado antes
dos fits, com fontes primárias consultadas em 13/11/2026. A seleção v2
alinha-se explicitamente a essa métrica e impede regressão da precisão em
qualquer fold ou do recall macro médio em relação ao v1.

Implementação: `weekly_features.py` acrescenta histórico de 3–90 dias,
recência, sazonalidade, localização e vizinhança, sem desfechos futuros;
`weekly_metrics.py` conserva todos os dias no orçamento e informa o teto
do oráculo; `weekly_training.py` executa dez novos perfis e três referências,
além do modelo original, nos mesmos folds 2021–2023. Nenhuma linha foi
amostrada ou retirada da avaliação. As 454.825 linhas e 12 colunas originais
foram preservadas; são 32 preditores numéricos ordenados. O painel Arrow
e as matrizes ficam em memória; o limite DuckDB de 512MB não limita o RSS
total e esta etapa não é uma implementação bilionária out-of-core.

### Reprodução

Comandos reais; para repetir, escolher diretórios novos, pois saídas existentes
são recusadas:

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.weekly_features --grid output/rf10-weekly/grid/grid.parquet --output-dir output/rf10-weekly/features-v2
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.weekly_training train --features output/rf10-weekly/features-v2/features.parquet --assignments .omp/evidence/RF10-weekly/protocol-v1/assignments.parquet --protocol docs/weekly-improvement.md --output-dir output/rf10-weekly/model-v2-run2
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.weekly_training score --features output/rf10-weekly/inference-input-v2/features.parquet --model-dir output/rf10-weekly/model-v2-run2 --output-dir output/rf10-weekly/inference-v2-run2 --issue-date 2025-12-24
```

A inferência projeta apenas identificadores/limites do alvo e preditores;
`label` não é exigido nem consultado. O smoke real usou
`output/rf10-weekly/inference-input-v2/features.parquet`, uma projeção
documentada que fisicamente não contém rótulos/desfechos, e publicou
`output/rf10-weekly/inference-v2-run2/`: 161 escores e dez destaques.
É emissão histórica, não previsão operacional atual.

### Resultado observado

Selecionado `ndcg_1`: `rank:ndcg`, 240 árvores, profundidade3, eta0,05,
min_child_weight10, lambda10, pares topk12, seed42 e duas threads.
Precisão média dos folds: 45,1210% contra 38,5847% do original;
recall macro médio: 21,2923% contra 15,8725%. Seleção congelada antes
dos novos scores de 2024–2025; nenhum ajuste posterior pelos diagnósticos.

| Diagnóstico | Precisão@10 v1 | Precisão@10 v2 | Recall macro v1 | Recall macro v2 |
|---|---:|---:|---:|---:|
| 2024 | 33,0601% | 40,2186% | 18,4694% | 25,7484% |
| 2025 | 34,9162% | 44,0503% | 17,0325% | 24,7886% |
| Agregado | 33,9779% | 42,1133% | 17,7457% | 25,2650% |

Em2025, TP1577, FP2003, FN6278, com3580destaques e7855janelas positivas.
São327acertos adicionais no mesmo orçamento, não327incêndios independentes.
Todos os24meses foram reportados; não houve regressão mensal de precisão
contra v1 nesse diagnóstico. Isso não é teste de significância. Climatologia
continua melhor na AP global dos dois anos (2024:0,4147 contra0,4069;
2025:0,4279 contra0,4165); não alegar superioridade em todas as métricas.

**80% não foi atingido.** A precisão máxima possível no orçamento fixo é
`sum(min(10, positivos_do_dia)) / (10 * dias)`:73,0601% em2024,
82,7933% em2025 e77,8729% no agregado. Logo80% no agregado/ambos os anos
é impossível neste dataset sem mudar o contrato. Em2025 a distância
observada para a meta é35,9497pontos percentuais. Não reduzir destaques,
retirar meses, trocar rótulos ou reportar acurácia majoritária como solução.

### Evidência e estado

- 259 testes passaram; teste do fixture NaT corrigido também passou isoladamente.
- Construção,33fits de folds,refit do original/vencedor e recarga JSON reais.
- Auditoria independente via SQL:92hashes de artefatos, seleção reproduzida,
  população161/dia/modelo, dez destaques, ranks/rótulos e métricas anuais/mensais;
  zero divergências e v1 byte a byte preservado.
- Primeira execução `model-v2/` foi preservada. Um aviso NumPy na checagem de
  datas da inferência foi resolvido com `timedelta64` em dias explícitos.
  Reexecução idêntica em `model-v2-run2/` não alterou perfis, seleção, modelo
  vencedor ou previsões; igualdade byte a byte registrada em `replication.json`.
- A auditoria inicialmente usou soma inteira sobre TIMESTAMP; corrigida
  apenas a consulta para intervalos tipados, sem alterar dados/resultados.

2024–2025 já foram observados emv1; são diagnósticos reutilizados, **não
validação independente**. O próximo requisito de evidência é um novo período
com desfechos completos, sem seleção por seus resultados. Meteorologia e
vegetação plurianuais, observabilidade e disponibilidade histórica continuam
lacunas; não foram fabricadas. Não há promoção ou aceite operacional.

Relatórios completos, todos os candidatos e Parquets:
`output/rf10-weekly/model-v2-run2/`. Logs e auditorias:
`.omp/evidence/RF10-weekly-improvement/`. Aprovação, hashes e handoff no
[dossiê de melhoria](../.omp/tasks/RF10-melhoria-7dias.md).

