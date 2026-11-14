# RF10/CA08 — melhoria semanal v2

## Autorização, estimando e limites

Consulta e protocolo: 13/11/2026, antes dos novos fits. Usuário solicitou buscar metodologias e executar melhorias de aproximadamente 35% para 80%, explicitamente sem esconder ou falsificar dados. Autoriza experimentação; não constitui aprovação de desempenho, mudança de alvo ou promoção operacional. Preservar v1, seus artefatos e D+1 como requisito de produto.

Os 35% correspondem à precisão@10 de 2025 do XGBoost v1 (1250/3580 = 34,9162%), não à acurácia geral nem ao recall. Alvo e população permanecem: alguma detecção NOAA-20 SP em D+1…D+7, emissão fim de D UTC, todas as 161 células H3res5 da bbox [-48,-17,-46,-15], dez destaques por dia com desempate por cell_id. Zeros não comprovam ausência física de fogo; scores não são probabilidades calibradas.

A meta solicitada é precisão@10 >= 0,80, com recall reportado e protegido durante seleção; a primária v1 (recall macro diário@10) continua publicada separadamente. A mudança explícita da função de seleção para precisão alinha o experimento à métrica dos 35%, não permite renomear outra métrica como assertividade. Não alterar k, retirar dias, amostrar avaliação ou mudar rótulos para atingir a meta.

**Factibilidade observada antes do fit:** mesmo um oráculo só captura min(10, positivos do dia). Nos diagnósticos já observados de 2024 e 2025, os tetos da precisão@10 são respectivamente 2674/3660 = 73,0601% e 2964/3580 = 82,7933%; combinados, 5638/7240 = 77,8729%. Portanto 80% em ambos os anos ou no agregado é matematicamente incompatível com este contrato/dataset. Em 2025 isoladamente não está excluído pelo teto, mas exigiria capturar quase todos os acertos possíveis. O oráculo é limite descritivo, nunca feature nem candidato implantável.

## Separação temporal e transparência

Reutilizar exatamente as partições v1: desenvolvimento por folds expansivos 2021/2022/2023, treino inicial em 07/04/2018, último treino 24/dez do ano anterior, validação 01/jan–24/dez. Expurgo de sete emissões na fronteira; alvo de treino termina antes da primeira emissão avaliada. Refit até 24/12/2023.

2024–2025 já foram examinados em v1 e são **diagnóstico reutilizado**, não validação independente. Seus resultados anteriores (incluindo vantagem da climatologia) motivaram hipóteses; o viés adaptativo é explicitado. Novos scores desses anos só serão produzidos depois de congelar a seleção em 2021–2023. Não retunar após esse diagnóstico. A reserva futura de outro experimento não será acessada; faltam novos desfechos completos e nunca usados para comprovar generalização. Não declarar a meta validada independentemente.

Persistir todos os candidatos de desenvolvimento, previsões, partições, métricas anuais/mensais/diárias, versões, parâmetros, hashes e resultado negativo. Nenhuma busca indefinida até encontrar um número favorável.

## Hipóteses e features causais por aquisição

H1: as três contagens v1 não representam sazonalidade, heterogeneidade espacial nem persistência em escalas distintas. Acrescentar, sem dados inventados:

- latitude/longitude derivadas de H3; seno/cosseno do dia do ano;
- contagens e dias ativos da célula em 3/7/14/30/60/90 dias, terminando em D;
- tempo desde a última detecção conhecida (nulo se nunca vista), histórico disponível em dias e indicação de janela inicial incompleta;
- detecções e células-dia ativas no primeiro anel H3, somente vizinhos da grade, em 3/7/30 dias; número de vizinhos conhecidos explicita bordas;
- contexto regional em 30 dias;
- frequência histórica de dias com detecção da célula/mês, apenas em anos estritamente anteriores ao ano de emissão. Não usar label semanal como target encoding; não consultar o futuro do ano corrente. Sem histórico: nulo e suporte zero.

Somente `detection_rows_on_issue`, datas e geometria geram novas covariáveis; nunca label, contagens futuras ou estado de observação. Preservar as três colunas v1, todas as linhas e rótulos. Janelas maiores que o histórico disponível ficam nulas, com suporte explícito. Não excluir o começo do treino de comparação. Ausência de registros em dias da grade não certifica observação satelital. SP reprocessado e available_at desconhecido mantêm a análise retrospectiva, não causalidade operacional demonstrada.

H2: otimizar ordenação por emissão pode capturar mais positivos no mesmo orçamento. Comparar binary:logistic com LambdaMART rank:ndcg e rank:pairwise, agrupando as 161 células por dia. Não usar desbalanceamento sintético, undersampling ou alteração da avaliação.

## Busca finita predeclarada

Referências: XGBoost v1 retreinado exatamente nas três features e parâmetros v1; persistência; recorrência de sete dias; climatologia célula/mês calculada exclusivamente no treino expurgado de cada fold.

Features ampliadas: quatro perfis logistic (profundidade, árvores, eta, min_child_weight, lambda): (2,240,0.05,10,10), (3,240,0.05,10,10), (5,400,0.03,10,10), (3,600,0.03,10,10). Dois perfis para cada objetivo rank:ndcg e rank:pairwise: (3,240,0.05,10,10), (5,400,0.03,10,10). Rank:ndcg usa pares topk=12; rank:pairwise usa mean=4. Adicionalmente, dois logistic com as features ampliadas e pesos de linha inversamente proporcionais ao número de positivos da emissão, constantes dentro do dia, normalizados para média um, nos perfis (3,240,0.05,10,10) e (5,400,0.03,10,10). Dias sem positivos têm peso não nulo (denominador mínimo um). A ponderação é hipótese de alinhamento com recall macro, não remoção de controles.

Todos: CPU hist, seed42, nthread2, subsample1, colsample1, gamma0, alpha0, scale_pos_weight1, sem early stopping. Dez novos candidatos treinados, além de v1 e referências; nenhuma expansão após consultar diagnóstico. Todas as linhas de treino e avaliação são mantidas.

Selecionar por maior média não ponderada de precisão@10 nos três folds entre alternativas cuja precisão não regrida frente ao v1 em nenhum fold e cujo recall macro médio não regrida. Ganho de precisão média deve ser estrito; empate mantém v1, depois a ordem declarada (referências antes dos novos modelos). Se ninguém satisfizer, manter v1. Preservar todas as métricas mesmo em candidatos rejeitados. Refit único do selecionado antes de abrir os novos scores de 2024–2025, usando a mesma seleção, parâmetros e contrato. Comparar selecionado, v1 e referências no diagnóstico, sem substituição posterior.

## Verificação

Testar invariância das features anteriores a D ao alterar observações após D; sazonalidade sem ano corrente; fronteiras das janelas; ausência de alvo em features; métricas com empates, dias sem positivos, tetos do oráculo e preservação da população. Executar construção, treino real e recarga JSON, conferir igualdade da inferência e recalcular contagens diretamente dos Parquets exportados. Inferência separada sem rótulos deve usar o modelo congelado e o mesmo contrato de features.

Relatórios incluem TP/FP/FN, precisão@10, recall micro e macro, AP, prevalência, lift, dias sem positivos, teto do oráculo e distância para 80%. Reportar todos os meses e anos, sem IC binomial ingênuo sobre janelas sobrepostas. Sem alegação inferencial de significância ou promoção automática.

## Metodologias pesquisadas — fontes primárias

Consultadas em 13/11/2026:

- [XGBoost, Learning to Rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html): grupos por consulta, objetivos pairwise/NDCG e pares top-k; fundamenta H2, não garante ganho nem 80%.
- [scikit-learn, Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): validação separada do teste, transformações ajustadas apenas no treino; fundamenta expurgo/seleção separada, sem validar este produto.
- [Xu et al., Spatio-Temporal Wildfire Prediction using Multi-Modal Data, 2022](https://arxiv.org/abs/2207.13250): dependência espaço-temporal e entradas multimodais; H1 é adaptação experimental, não reprodução de resultados publicados.
- [Huot et al., Next Day Wildfire Spread, 2021](https://arxiv.org/abs/2112.02447): meteorologia, vegetação, topografia e seca como variáveis explicativas. Seu alvo é propagação D+1 nos EUA, diferente deste alvo semanal. Não transferir suas métricas como previsão de ganho local.
- [NASA FIRMS FAQ](https://www.earthdata.nasa.gov/data/tools/firms/faq): produto, sensores e cautelas de detecção; não confundir detecção térmica com verdade completa de incêndio.

Meteorologia/vegetação plurianuais e máscara de observação continuam lacunas reais: as entradas locais de 2026 não cobrem 2018–2025. Não preencher anos ausentes com dados artificiais nem usar meteorologia observada em D+1…D+7 como previsão disponível em D. Novas fontes exigem proveniência e corte de disponibilidade antes de novo protocolo.
