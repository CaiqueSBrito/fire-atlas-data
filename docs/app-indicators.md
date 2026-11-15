# Indicadores, alertas e explicações para o app (proposta)

Proposta para o backend e a interface, usando `gold/harmonized` (calendário), `gold/forecast` (previsão de 7 dias) e `gold/h3_places` (lugar e fontes fixas). Os limiares marcados como **proposta** não foram aprovados; o desafio também não os define. Revise antes de publicar.

## 1. Calendário harmonizado (`gold/harmonized/cells`)

Base de todos os indicadores: para a área escolhida, filtre `resolution` pelo zoom e as células da área, e some `harmonized` por mês e ano. A unidade é a **atividade de fogo harmonizada (células ativas equivalentes ao MODIS Aqua)**.

| # | Indicador | Cálculo | Como mostrar |
|---|---|---|---|
| 1 | Calendário mês × ano | Soma de `harmonized` por mês e ano | Grade 2000–hoje, rampa de cor acessível (ex.: viridis/cividis). Célula vazia = sem dado, não zero. |
| 2 | Anomalia do mês | Valor do mês ÷ mediana do mesmo mês nos anos de referência (2002-07 em diante, sem o ano corrente) | Faixas propostas: < 50% «bem abaixo», 50–80% «abaixo», 80–125% «normal», 125–200% «acima», > 200% «bem acima». |
| 3 | Posição histórica | Percentil do mês atual entre os mesmos meses de 2002-07 em diante | «Este agosto é o 3º mais ativo desde 2002.» |
| 4 | Acumulado do ano | Soma de jan até o mês atual ÷ mediana do mesmo período nos anos de referência | Barra «ano atual x normal». |
| 5 | Temporada típica | Meses com maior mediana histórica | «Temporada de fogo: jul–out (pico em set).» |
| 6 | Ano mais ativo / menos ativo | Ranking dos totais anuais | Destaque no calendário. |
| 7 | Antes x depois (transparência) | `detections_<p>` / `units_<p>` por satélite contra `harmonized` | Gráfico de linhas: as séries brutas dão degraus, a harmonizada não. |
| 8 | Download | Linhas filtradas da área e do período, com `basis`, `low_confidence` e `contributing` | CSV com o rótulo da unidade no cabeçalho. |

**Avisos de qualidade, sempre que se aplicarem:**

| Situação | Como detectar | Texto sugerido |
|---|---|---|
| Período só com o satélite Terra | `low_confidence = true` (2000-11 a 2002-06 e raros dias de falha do Aqua) | «Menos confiável: só o satélite Terra; erro cerca de 2× maior.» |
| Dia sem satélite | `contributing = 0` ou `basis` NULL | «Sem dado neste dia» (nunca mostrar como zero). |
| Área pequena | Área selecionada < ~50 mil km², ou período < 1 mês | «Área pequena: o número pode variar ±30% ou mais.» |
| Pouquíssimo fogo | Soma do período baixa (proposta: < 30 unidades) | «Pouca atividade: use como ordem de grandeza.» |
| Mês corrente | Mês ainda aberto | «Parcial até hoje»; a anomalia compara só os dias decorridos ou fica marcada como provisória. |
| Dados recentes | Últimos ~3 meses | «Dados quase em tempo real; podem ser revisados pela NASA.» |
| Depois de 2019 | `basis = 'viirs'` | Nota fixa: «Série baseada no VIIRS convertido para a régua do Aqua.» |

Erro medido para colocar na tela de ajuda (por região de ~87 mil km² e mês, contra o Aqua real):

| Período | Erro típico |
|---|---|
| 2000–2002 (Terra) | ~23% |
| 2002–2019 | é o próprio Aqua |
| 2020 em diante (VIIRS, modelo de detecção v0.3) | ~10% |

O total geral fica a menos de 1% do Aqua.

## 2. Previsão de 7 dias (`gold/forecast`)

Probabilidade de pelo menos um foco na célula H3 res6 entre D 09:00 UTC e D+7 09:00 UTC. É calibrada: no teste 2023–2026, quando o modelo disse ~45%, aconteceu ~44%.

| # | Indicador | Cálculo | Como mostrar |
|---|---|---|---|
| 9 | Mapa de risco 7 dias | Probabilidade por célula | Faixas propostas: < 10% baixa, 10–30% moderada, 30–60% alta, > 60% muito alta. |
| 10 | Células esperadas com fogo na área | Soma das probabilidades das células da área | «Esperadas ~120 células com fogo nesta semana.» |
| 11 | Semana atípica | Indicador 10 ÷ média de `harmonized` na mesma semana do ano em anos anteriores | **Ver aviso abaixo:** as escalas não são as mesmas. |
| 12 | Ranking de células | As N células de maior probabilidade na área | Lista com país e cidade (`gold/h3_places`). No teste, quase todas as 100 primeiras do dia tiveram foco. |
| 13 | Tendência da previsão | Indicador 10 de hoje contra o de ontem | Seta ↑/↓. |

**Aviso sobre o indicador 11:** a previsão conta células com qualquer detecção FIRMS de qualquer satélite. O calendário usa a régua do Aqua. Para comparar, use uma referência construída com o mesmo alvo da previsão (por exemplo, a taxa base histórica do `gold/ml_dataset` para a região e a semana do ano) ou mostre os dois números separados. Essa proposta ainda não foi implementada.

**Avisos obrigatórios da previsão:**
- «Probabilidade de detecção por satélite, não certeza de incêndio nem local exato de ignição.»
- «Só cobre locais que já tiveram foco. Fogo em área nova não é previsto.» Células fora do universo devem aparecer sem cor, não como risco zero.
- Fontes fixas (flares, vulcões): esconder por padrão as células com `gold/h3_places.static_source = true` (join por `h3_cell` com `resolution = 6`), com opção de mostrar.
- «Modelo experimental: sem validação oficial (CA08 não aprovado).»

## 3. Alertas combinados (proposta)

| Alerta | Regra proposta | Uso |
|---|---|---|
| Mês acima do normal | Anomalia do mês > 200%, com pelo menos 30 unidades no mês | Destaque no calendário |
| Risco alto na semana | Indicador 10 acima do percentil 90 histórico da mesma semana, com base no mesmo alvo da previsão | Banner na área |
| Atenção | As duas regras acima ao mesmo tempo | Selo «atenção» na área |
| Fonte fixa | `static_source = true` | Ícone de flare/vulcão; fora dos totais quando o filtro estiver ligado |

Os limiares desta tabela são proposta. O produto também não define alerta automático, push ou e-mail como obrigação; tratá-los como opcionais.

## 4. Textos de explicação (tela «Como ler»)

- **O que é o número:** «Quantas células de ~36 km² com fogo o satélite MODIS Aqua teria registrado. Usamos uma régua só para que 2005 e 2025 sejam comparáveis, mesmo com satélites diferentes.»
- **Por que não somar os satélites:** «Satélites novos veem mais fogo e repetem os mesmos focos. Somar criaria aumentos falsos.»
- **De onde vem cada ano:** 2000–2002 Terra; 2002–2019 Aqua; 2020 em diante satélites VIIRS convertidos (ver `basis`).
- **Margem de erro:** a tabela do item 1, com a nota de que em áreas pequenas ou com pouco fogo o erro cresce.
- **O que a previsão significa:** a frase obrigatória do item 2 e a calibração («quando diz 30%, acontece em cerca de 3 de cada 10 casos»).
- **Ausência de foco:** «Nenhum foco detectado não prova que não houve fogo: nuvem, horário e tamanho do fogo afetam a detecção.»
