# RF10/CA08 — protocolo semanal v1

Data: 12/11/2026. Estado: proposta científica especificada antes de fit; não constitui aceite humano de desempenho ou autorização operacional. Pedido humano «faça» autoriza definir e conferir este protocolo, não treinar. D+1 permanece requisito de produto.

## Pergunta e domínio

Na mesma grade de 161 células H3res5 da bbox [-48,-17,-46,-15], um ranking retrospectivo consegue priorizar células com alguma detecção NOAA-20 SP em D+1…D+7 melhor que referências simples? Inclui continuidade; não estima ignição, incêndios independentes, área queimada ou ausência física de fogo. Emissão nominal: fim de D UTC. Generalização pretendida: novo período na mesma região, não nova geografia.

Entrada imutável: `output/rf10-weekly/grid/grid.parquet`, schema `historical-cell-grid/2.0.0`, SHA256 `fa93dc8bbed57263784a47d3ea787f5310d7162a5482006ca9b955402efe819e`. Manifesto e linhagem no dossiê. Rótulo binário fornecido; zeros significam ausência de detecção fornecida e observabilidade desconhecida. Não filtrar células sem atividade em D.

## Partições predefinidas

Todas as 161 células de uma emissão recebem a mesma partição. Excluir emissões 01–06/04/2018 por janela histórica inicial incompleta; começar em 07/04/2018. Treino expansivo, validações anuais completas em sazonalidade:

| Fold | Treino: emissões inclusivas | Validação: emissões inclusivas |
|---|---|---|
| 2021 | 07/04/2018–24/12/2020 | 01/01/2021–24/12/2021 |
| 2022 | 07/04/2018–24/12/2021 | 01/01/2022–24/12/2022 |
| 2023 | 07/04/2018–24/12/2022 | 01/01/2023–24/12/2023 |

Em cada fronteira B=01/jan, o último treino é B−8 dias: seu alvo termina em B−1, estritamente antes da primeira emissão de validação. Expurgar 25–31/dez entre períodos; não são negativos. Validação termina 24/dez para que seu alvo termine 31/dez. Disponibilidade dos desfechos de treino não usa o próprio dia B. Reutilizar anos anteriores como treino de folds seguintes é walk-forward, não independência entre folds.

Após seleção e congelamento por 2021–2023, refit único em 07/04/2018–24/12/2023; avaliação final designada em 01/01/2024–24/12/2025, com janelas completas. Relatar 2024 e 2025 separadamente. Não usar seus rótulos, scores, métricas ou prevalência para seleção. Os totais anuais já constam do manifesto previamente consultado: portanto este holdout é reservado ao novo ajuste, não um teste totalmente cego ou evidência independente incontaminada. A reserva independente de outros experimentos não será acessada.

A conferência deste protocolo lê somente metadados temporais/chaves, nunca rótulos ou covariáveis do holdout. Persistir atribuição por célula/data/fold, inclusive exclusões e expurgos. Verificar unicidade, 161 células por emissão, continuidade diária, horizonte=7 e alvo D+1…D+7. Não fazer split aleatório ou por linha. Sobreposição entre janelas de features históricas é permitida por refletir o uso temporal, mas não remove autocorrelação. Não alegar isolamento de eventos que persistem entre anos. Sem bloqueio espacial, não há alegação de transferência geográfica.

## Entradas e referências

Whitelist ordenada: `detection_rows_on_issue`, `cell_detection_rows_7d`, `regional_detection_rows_7d`. Sem novas features ambientais, sensores, coordenadas, rótulos ou normalização aprendida nesta etapa. Proibidos como features: alvo, datas futuras, contagens futuras e status de observação. Não amostrar controles, não fazer undersampling nem ponderar positivos neste primeiro experimento.

Referências avaliadas nas mesmas linhas, com desempate por `cell_id` ascendente:

1. Persistência em D: score = `detection_rows_on_issue`.
2. Recorrência de sete dias: score = `cell_detection_rows_7d`.
3. Climatologia célula/mês: frequência de `label=1` por célula e mês da emissão, estimada só no treino expurgado de cada fold. Sem grupo observado, usar frequência global do treino. Mesma regra no refit final; não usar estatística do ano avaliado.

O modelo candidato será XGBoost `binary:logistic`, CPU hist, 120 árvores, profundidade 3, eta 0,05, min_child_weight 1, gamma 0, lambda 1, alpha 0, subsample 1, colsample_bytree 1, scale_pos_weight 1, seed 42, nthread 2. Um candidato, sem busca ou early stopping, para evitar adaptar o orçamento ao resultado. Estes são parâmetros propostos de referência, não ótimo demonstrado. Registrar versão exata de XGBoost e configuração efetiva antes de fit. Score serve para ranking, não probabilidade calibrada.

## Métricas e decisão predeclarada

Orçamento proposto: top 10 das 161 células por emissão, mantido para comparabilidade de esforço com os experimentos anteriores, não capacidade operacional aprovada. Seleção determinística por score decrescente e cell_id ascendente.

Primária: recall@10 macro diário, média apenas em emissões com positivos. Relatar explicitamente quantos dias sem positivos foram excluídos dessa média, mantendo-os no restante da avaliação. Por fold e ano: TP, FP, FN, positivos, recall micro, precisão@10 sobre todas as emissões, AP global, prevalência, lift de precisão@10 sobre prevalência e métricas mensais. Recall/AP sem positivos são indefinidos, não zero; interromper seleção se algum fold não suportar a métrica primária. Não escolher limiar binário ou usar acurácia como sucesso.

Baseline de comparação selecionado pela maior média não ponderada da primária nos três folds; empates preservam a ordem da lista. Aceite experimental proposto exige delta estritamente positivo do XGBoost contra esse baseline em cada fold e na média. Empate mantém baseline. Congelar decisão, hashes, configuração, ordem das features e modelos antes de avaliar o período final. Se não superar, registrar resultado negativo; não retunar pelo holdout.

No período final, reportar vencedor congelado e as três referências; ganho experimental só é sustentado se o delta da primária contra a referência congelada for positivo em 2024 e 2025. Isso não concede CA08 nem promoção. Não há limiar de utilidade operacional aprovado; permanece decisão humana baseada em custos de omissão e falso destaque.

Incerteza: reportar deltas por mês e ano; não tratar células-dia ou janelas positivas como ensaios independentes. Não calcular IC binomial por linha, p-valor ingênuo ou transformar desvio entre folds em IC. Este primeiro protocolo não faz alegação inferencial de significância. Brier, log-loss e calibração não fazem parte de um aceite probabilístico; uma futura proposta probabilística exige partição de calibração separada e protocolo próprio, sem reutilizar o holdout para ajustar.

## Limites e entrega

SP reprocessado e disponibilidade histórica `available_at` desconhecida impedem demonstrar operação causal em D. O backtest é retrospectivo mesmo com corte por aquisição correto. Não há máscara de observação, identificação independente de eventos, cobertura certificada, dados meteorológicos plurianuais nem validação de harmonização MODIS/VIIRS. Contagens preservam linhas fornecidas e podem incluir duplicidades da origem. Dataset semanal não valida ciência por ter schema/testes corretos.

MLE recebe este protocolo e, antes de qualquer fit autorizado, deve validar hashes, persistir partições e versões, bloquear colunas fora da whitelist e produzir previsões diárias reprodutíveis dos modelos recarregados. DS revisa métricas e limitações; humano decide utilidade e aceite. Nenhum treinamento é executado nesta tarefa.

## Referências

Fontes primárias registradas em [.omp/docs/specialist-sources.md](../.omp/docs/specialist-sources.md), consulta registrada em 12/11/2026; não houve nova consulta web nesta tarefa: [scikit-learn, validação cruzada](https://scikit-learn.org/stable/modules/cross_validation.html), [leakage](https://scikit-learn.org/stable/common_pitfalls.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [Roberts et al., 2017](https://doi.org/10.1111/ecog.02881), [calibração](https://scikit-learn.org/stable/modules/calibration.html). Fundamentam cautelas metodológicas; não endossam datas, orçamento ou qualidade deste produto.
