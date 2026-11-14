# RF10/CA08 — protocolo global semanal, proposta v1

Consulta e redação: 13/11/2026. Estado: proposta implementável para revisão humana; não aprova método, treinamento real, desempenho nem operação. O pedido «comece», seguido de «continue», autoriza protocolo e código. Backfill completo permanece explicitamente não autorizado. Experimentos anteriores e a obrigação D+1 ficam preservados.

## Pergunta, população e estimando

Para uma célula H3 resolução 5 com histórico de detecção conhecido na emissão D, estimar a probabilidade de ao menos uma **detecção fornecida não estática** em D+1…D+7. Domínio mundial, aquisição e emissão nominal no fim de D em UTC. O usuário seleciona uma bbox; isso seleciona as células de saída, não redefine preditores nem reentreina o modelo. Uma célula mantém o mesmo escore em recortes diferentes.

A área média da célula (~253 km²) não é constante nem resolução nominal do sensor. Saída não representa coordenada de ignição, área queimada, incêndio confirmado ou certeza de fogo. Célula fora do domínio recebe ausência de previsão, nunca probabilidade zero. Domínio é composto pelas células com primeira detecção em ou antes de D, inclusive história MODIS desde 2000; construir elegibilidade usando todas as ocorrências até 2026 para avaliar 2013 seria vazamento.

Cobertura de células com histórico não mede cobertura de observação: percentuais anteriormente calculados dividindo células detectadas por uma fração estimada de terra **não são cobertura terrestre validada**. Falta máscara terra/mar e denominador de oportunidades de observação. Células sem detecção não comprovam falta de observação nem ausência de fogo.

## Sensores, rótulo e faltantes

MODIS Terra/Aqua e VIIRS SNPP/NOAA-20/NOAA-21 contribuem ao alvo binário. Múltiplos sensores não multiplicam o rótulo positivo. Contagens/FRP dos preditores são separadas por família; não são contagens de incêndios independentes. FIRMS `type` vira `hotspot_type` inteiro anulável no contrato 1.3.0; bruto é preservado. Valores 1–3 ficam fora do alvo; 0 significa vegetação presumida, não verdade de campo.

NRT sem `type` permanece desconhecido: fora da máscara estática pode contribuir como detecção não estática presumida, não como type 0 certificado. Comparar SP e NRT separadamente. Zeros significam nenhuma detecção elegível fornecida; observabilidade permanece desconhecida. Janelas de desfecho incompletas são recusadas, não rotuladas zero.

Quantidade de plataformas com registros nos últimos sete dias é covariável de disponibilidade observada, não contagem comprovada de satélites operacionais ou cobertura local. Uma covariável não harmoniza sensores nem elimina mudança de distribuição. A NASA confirmou fim da entrega de novos dados SNPP em 02/11/2026 às 15:40 UTC [S3]; não assumir constelação atual fixa ou SNPP futuro.

## Máscara estática e causalidade

Decisão humana: fontes externas de flares/vulcões mais recorrência type 2 do SP. Aplicar máscara em células H3 **mais finas que a previsão**, evitando excluir toda uma célula res5 por conter uma indústria. Resolução, número de anos de recorrência e buffer são parâmetros explícitos; não há limiar científico aprovado.

Perfil descritivo executado em `.omp/evidence/RF10-global-weekly/static_profile.py`, resultado `static-profile.json`: referência 2012–2022, diagnóstico 2023, res8. São contagens de detecções multissensor, não fontes independentes nem verdade de campo:

| Anos mínimos type 2 | Buffer | Células mascaradas | Type 2 de 2023 na máscara | Type 0 de 2023 na máscara |
|---|---:|---:|---:|---:|
| 2 | 0 | 95.822 | 93,75% | 0,62% |
| 3 | 0 | 60.710 | 90,59% | 0,52% |
| 2 | 1 anel | 374.272 | 95,01% | 0,95% |
| 3 | 1 anel | 232.269 | 92,34% | 0,74% |

Esses números mostram o custo de exclusão potencial de vegetação; não escolhem automaticamente um limiar. Como 2023 já foi examinado neste diagnóstico, não chamá-lo teste científico intocado.

**Uma máscara construída até 2022 não pode alimentar linhas de treino de 2013 como se conhecida na época.** O caminho executável inicial usa máscara congelada com referência anterior à primeira emissão. Para treino proposto desde 20/01/2013, cutoff da máscara deve ser no máximo 19/01/2013. Alternativa futura de máscaras por época exige protocolo e contrato próprios; não simular isso usando uma máscara final para todo o histórico. Catálogos retrospectivos publicados depois do cutoff não provam disponibilidade histórica, mesmo contendo ano de atividade anterior. Registrar esta limitação e não alegar backtest operacional causal.

O estudo de máscara 2012–2022 é apenas evidência descritiva, não artefato aprovado para treino. Catálogos não foram baixados; parâmetros finais e proveniência externa são pendências humanas.

## Features e reprodução

`global_features.py` define 39 preditores ordenados. Famílias: dias ativos e contagens MODIS/VIIRS em 1/7/30/90/365 dias; FRP médio por família em 7/30 dias; tempo desde primeira/última detecção; atividade no anel imediato e nos pais H3 res4/res3 em 7/30 dias; climatologia de mesmo mês somente em anos anteriores e suporte correspondente; anos MODIS ativos/suportados; seno/cosseno do dia do ano; latitude/longitude do centro; plataformas reportantes em sete dias.

Todas as janelas de features terminam em D. Contexto espacial vem do acervo completo, não apenas das células solicitadas. Janelas sem suporte suficiente são NULL; contagem zero em janela suportada não prova ausência física. FRP ausente não vira média zero. Labels são computados em consulta separada; inferência omite colunas de desfecho. Horário de aquisição não equivale a disponibilidade: `available_at` histórico desconhecido limita validade operacional.

Amostragem uniforme determinística de célula-dia por hash (célula, D, semente), com taxa explícita e chaves persistidas, evita materializar bilhões de linhas de features. Não amostrar apenas negativos e chamar o resultado probabilidade calibrada sem correção. Taxa de amostragem e limite de memória são registrados; insuficiência de suporte deve falhar, nunca truncar silenciosamente. Dados ambientais não estão incorporados; não declarar clima/vegetação já disponíveis.

## Partições temporais propostas

| Uso | Treino inclusivo | Avaliação inclusiva |
|---|---|---|
| Fold 2020 | 20/01/2013–24/12/2019 | 01/01/2020–24/12/2020 |
| Fold 2021 | 20/01/2013–24/12/2020 | 01/01/2021–24/12/2021 |
| Fold 2022 | 20/01/2013–24/12/2021 | 01/01/2022–24/12/2022 |
| Refit e calibração | 20/01/2013–24/12/2022 | Calibração: 01/01/2023–24/12/2023 |
| Teste retrospectivo | Modelo/calibrador congelados | 2024 e 2025: 01/jan–24/dez, separados |
| Diagnóstico SP | Sem ajuste | 01/01/2026–24/07/2026 |
| Diagnóstico NRT | Sem ajuste | 01/08/2026–06/11/2026 |

Expurgar emissões entre treino e validação/calibração para que o desfecho de sete dias termine antes da próxima emissão. Persistir papel de cada chave, incluindo expurgadas/não utilizadas. Configuração com sobreposição ou gap insuficiente é recusada. Períodos diagnósticos não selecionam hiperparâmetros. Limites de 2026 dependem da completude real de cada plataforma, não somente máximo global do arquivo.

2024–2025 já foram examinados em outros experimentos e descritivos: teste retrospectivo reutilizado, **não avaliação independente incontaminada**. Estas partições demonstram generalização temporal dentro do domínio, não transferência para geografia inédita. Reportar latitude/mês ajuda a revelar heterogeneidade, mas **não substitui validação espacial bloqueada**. Desenho de blocos espaciais com buffers e período realmente independente fica pendente antes de qualquer alegação científica mundial.

## Modelo, calibração e critérios

Proposta de candidato único: XGBoost `binary:logistic`, CPU hist, profundidade 6, eta 0,1, min_child_weight 20, lambda 10, 300 árvores, seed 42; sem busca nem early stopping. Configuração efetiva versionada. Parâmetros reduzidos em fixtures servem exclusivamente à verificação de engenharia.

Baselines: persistência (atividade em D), recorrência (dias ativos nos últimos sete dias) e climatologia VIIRS no mês em anos anteriores, fallback MODIS quando sem suporte. Escolher referência pela maior AP média dos folds, empate preserva ordem. Sustentação experimental exige AP estritamente superior à referência em todos os folds; caso contrário registrar insucesso. Escrever seleção antes de ler outcomes de calibração/teste.

Refit seguido de calibração isotônica em conjunto temporal separado. Calibrar baselines no mesmo período para comparação probabilística. Salvar modelo XGBoost nativo JSON, limiares isotônicos JSON e manifestos; verificar igualdade de inferência após recarga. Score logístico/calibrado não ganha validade probabilística operacional por existir um calibrador.

Reportar AP, prevalência, Brier, log-loss, Brier skill contra referência, confiabilidade por dez faixas, estratos de latitude/mês e contagens. AP sem classes suficientes é indefinida, não zero. Janelas sobrepostas e correlação espacial impedem intervalos binomiais ingênuos por linha. Não comparar precisão de sete dias com D+1 como se tarefa idêntica. Nenhuma meta de 80% aprovada ou garantida. Critérios de utilidade, tolerância de falsos positivos/negativos e aceite humano permanecem abertos.

## Fontes externas verificadas e limites

Consulta primária em 13/11/2026; pesquisa não valida ciência do produto:

- [S1: ORNL DAAC, DOI 10.3334/ORNLDAAC/1874](https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-methane-flaring-sites-viirs-1874-1): catálogo global anual de flares 2012–2019, localização/temperatura/volume; 24 arquivos, publicado em 29/10/2021. Página declara “openly shared, without restriction” sob orientação EOSDIS. Não cobre toda indústria; autenticação/formato/tamanho de downloads não conferidos.
- [S2: EOG Nightfire](https://eogdata.mines.edu/products/vnf/): CSV gzip/KMZ; desde 10/01/2025 acesso sujeito à licença VIIRS Nightfire e aplicação de licença. Não tratar como download irrestrito. EOG pede crédito e citação Elvidge et al. (2013). Nenhum acordo aceito, credencial criada ou dado baixado.
- [S3: NASA, cessação SNPP](https://www.earthdata.nasa.gov/data/alerts-outages/reminder-new-suomi-npp-observations-cease-november-2-2026): atualização de 02/11/2026 confirma fim de novos dados às 15:40 UTC; recomenda NOAA-20/21.
- [Smithsonian GVP](https://volcano.si.edu/volcanolist_holocene.cfm): consulta retornou HTTP403; licença/formato/conteúdo não verificados nesta rodada. Não aprovar download por inferência.

## Handoff e bloqueios

Código é verificável com fixtures e smoke de engenharia, sem treino científico real. Backfill, catálogos, máscara final, taxa amostral de produção, protocolo espacial, aceitação científica e treino real não foram aprovados. O dossiê `.omp/tasks/RF10-multissensor-7d-bbox.md` registra decisões/evidência; humano revisa este protocolo antes de execução científica. DS revisa causalidade, MLE verifica contratos, humano decide utilidade. Nenhum resultado sintético é apresentado como acurácia real.

### Proposta adicional de bloqueio espacial para revisão

Antes de alegar transferência geográfica, atribuir cada célula ao pai H3 res2; definir cinco folds por `MD5(parent2_id || ':42') mod 5`, sem consultar rótulos. Em cada fold, retirar do treino todos os descendentes dos blocos retidos e dos seus vizinhos res2 de um anel; avaliar somente os blocos retidos (buffer não entra como negativo). Combinar isso com os mesmos cortes temporais e expurgo semanal, não com split aleatório de linhas. Persistir bloco/fold/buffer por célula e relatar suporte, AP e Brier por bloco.

Res2 e um anel são propostas de separação conservadora diante do contexto res3, não distância validada de independência de eventos; comparar sensibilidade de buffer antes de aprovar. A execução de treinamento entregue nesta rodada é temporal: não implementa nem declara executado este experimento espacial. Ele é um critério explícito de aprovação científica, não uma capacidade atribuída indevidamente ao programa.

Atualização da pesquisa de fontes: o [guia ORNL](https://data.ornldaac.earthdata.nasa.gov/public/cms/Methane_Flaring_Sites_VIIRS/comp/Methane_Flaring_Sites_VIIRS.pdf) confirma CSV com latitude/longitude EPSG:4326, faltantes -9999 e cobertura desde abril2012. As posições2012–2016 foram resolvidas conjuntamente com todos esses anos; um catálogo rotulado2012 contém informação posterior. O [README MYC25Q4](https://zenodo.org/records/23091822/files/README.md) declara CC BY-NC4.0 e permissão EOG necessária para uso comercial. Consulta13/11/2026, sem aquisição dos datasets; detalhes em `.omp/evidence/RF10-global-weekly/static-catalogs.md`.

O treinador exige suporte finito dos baselines de persistência e recorrência em todas as linhas selecionadas; períodos sem reporte suficiente causam erro explícito. Não elimina essas linhas automaticamente para aumentar AP nem imputa ausência de reporte como zero. Reavaliar a política de amostragem/suporte exige revisão do protocolo, com contagens dos casos excluídos.

## Comando de treino e artefatos (somente após autorização)

```bash
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.global_training \
  --features-dir output/global/features \
  --config configs/global-weekly.json \
  --output-dir output/global/model
```

O comando não foi executado em dados reais. O diretório de features deve conter manifesto, `features.parquet`, outcomes completos e os39preditores na ordem/tipos declarados. `max_matrix_rows=2000000` limita cada partição carregada; exceder gera erro com tamanho mínimo estimado, não truncamento. Amostragem do painel precisa ser dimensionada previamente. Parâmetros de recursos não equivalem a teto rígido de RSS.

Saídas: `model.json`, `calibrator.json`, `selection.json`, `metrics.json`, `assignments.parquet`, `predictions.parquet` e `manifest.json`. O manifesto inclui versões, parâmetros efetivos, hash de inputs/config/código quando declarado, cutoff incluindo calibração e versão de transformação das features. Probabilidades dos baselines também são calibradas no período reservado; comparações não selecionam pelo teste. Publicação atômica, destino existente recusado.

Evidência de engenharia desta rodada: `tests-final.log` e `smoke-synthetic-summary.json` em `.omp/evidence/RF10-global-weekly/`. Smoke usa32linhas sintéticas e duas árvores para provar integração; esses números não medem capacidade, precisão ou calibração científica.
