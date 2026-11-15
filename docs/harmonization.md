# Harmonização MODIS/VIIRS para o calendário de fogo (RF07)

`harmonize-fit`, `harmonize-eval` e `harmonize` colocam as cinco plataformas da silver (Terra, Aqua, Suomi-NPP, NOAA-20, NOAA-21) na mesma escala, a do Aqua, e gravam uma série diária por célula H3 de 2000 até hoje, em todos os níveis de 0 a 6. O método é uma proposta com teste retrospectivo; não há limiar de aceite aprovado para CA07.

No lake: `s3://data-lake-fire-atlas-553607359608-us-east-2-an/gold/harmonized/cells` e `…/gold/harmonized/availability`; modelo em `…/models/harmonization/v0.2.0/`. O [daily-ingest](daily-ingest.md) regrava os anos recentes a cada execução.

```bash
S=output/lake/silver            # ou s3://…/prata/silver
PYTHONPATH=src python -m fire_atlas_data harmonize-fit  --silver $S --model-dir output/harmonization/model
PYTHONPATH=src python -m fire_atlas_data harmonize-eval --silver $S --model-dir output/harmonization/model \
  --output output/harmonization/final-test.json
PYTHONPATH=src python -m fire_atlas_data harmonize --silver $S --model-dir output/harmonization/model \
  --target output/lake/gold/harmonized/cells --availability output/lake/gold/harmonized/availability
PYTHONPATH=src python -m fire_atlas_data harmonize-eras --silver $S --model-dir output/harmonization/model \
  --output output/harmonization/eras.json
# --model-dir aceita s3://; --year (repetível) regrava só esses anos
```

- `harmonize-fit` recusa gravar sobre um modelo existente.
- `harmonize-eval` recusa gravar sobre um relatório existente: o teste final roda uma vez.
- `harmonize` regrava um commit Delta por ano (`--year` repetível para atualizar só alguns) e confere cada ano com a silver.
- `harmonize-eras` mede, com o modelo fixo, o erro contra o Aqua da série composta em cada época de plataformas, nos anos de teste. Também mede o Terra sozinho em 2003–2004, anos usados no ajuste (por isso o resultado é otimista), e a concordância VIIRS/Aqua em 2020–2025, que é concordância com um Aqua em deriva, não erro. É uma segunda leitura dos anos de teste, portanto não é independente. Recusa gravar sobre um relatório existente.

## O que a série representa

`harmonized` estima **quantas unidades o MODIS Aqua, na órbita nominal de 2003–2019, teria registrado** na área e no dia. Escolha do Data Scientist, aprovada pelo usuário em 14/11/2026: é o único alvo que dá para conferir com uma observação real. "Atividade real de fogo" não tem referência observável.

## Unidade

Uma **unidade** é uma célula H3 res6 (~36 km²) ativa em um dia local de fogo, de dia ou de noite, para uma plataforma. O dia local de fogo vai das 06:00 às 06:00 no horário solar local (`acquired_at` deslocado pela longitude), de modo que as passagens das ~10:30/13:30 e ~22:30/01:30 ficam longe da virada do dia. As plataformas nunca são somadas antes da conversão.

## Método

Cada unidade de uma plataforma vira "unidades equivalentes do Aqua": quantas unidades o Aqua reportaria para a mesma queima.

1. **Curva de detecção:** probabilidade de o Aqua também estar ativo na mesma célula, dia e período, em função do FRP somado da unidade, do número de detecções, de dia/noite e da faixa de latitude. O Suomi-NPP e o Terra têm curvas próprias, ajustadas contra o Aqua. NOAA-20 e NOAA-21 usam a curva do Suomi-NPP (mesmo tipo de sensor).
2. **Fator regional:** total da referência dividido pelo total da curva, por região H3 res2 × mês do ano. Regiões com menos de 500 unidades nos anos de ajuste usam o fator res1 × mês e, depois, o global.
3. **Cadeia:** a referência do Terra e do Suomi-NPP é o próprio Aqua. A do NOAA-20 é a estimativa do Suomi-NPP e a do NOAA-21 é a estimativa do NOAA-20.
4. **Série harmonizada** (regra `aqua-first-0.2`): em cada dia, até 2019-12-31, se o Aqua funcionou, a série é o próprio Aqua. Senão, é a média das estimativas dos VIIRS que funcionaram; sem VIIRS, é a estimativa do Terra (só até 2019-12-31). De 2020 em diante, só VIIRS. Pane = unidades da plataforma no mundo abaixo de 50% da mediana de ±30 dias dela mesma. Até a versão 0.1.1 publicada antes, a série era a média de todas as plataformas que funcionavam, incluindo o Aqua.

| Plataforma | Curva | Referência do fator | Anos de ajuste |
|---|---|---|---|
| Terra | própria | Aqua | 2003–2017 |
| Suomi-NPP | própria | Aqua | 2013–2017 |
| NOAA-20 | Suomi-NPP | estimativa Suomi-NPP | 2020–2021 |
| NOAA-21 | Suomi-NPP | estimativa NOAA-20 | 2023–2024 |

2018–2019 ficam reservados: o ajuste nunca lê esses anos.

## Tabelas

`cells`, particionada por `resolution` e `year` (ano do dia local). Uma linha por célula, dia e nível com alguma unidade:

| Coluna | Conteúdo |
|---|---|
| `resolution`, `h3_cell`, `day` | Chave: nível H3 (0 a 6), célula naquele nível e dia local de fogo. |
| `units_<p>` | Células r6 ativas da plataforma dentro da célula, contando dia e noite separadamente. |
| `detections_<p>` | Detecções originais da plataforma, sem ajuste. |
| `aqua_eq_<p>` | Estimativa da plataforma em unidades equivalentes do Aqua; NULL sem unidade. |
| `harmonized` | Valor do calendário: valor da série na célula e no dia, somado nas células r6 filhas nos níveis 0 a 5. |
| `contributing` | Quantas plataformas entraram na série naquele dia; 0 = sem dado. |
| `basis` | Base da série no dia: `aqua`, `viirs` ou `terra`; NULL = sem dado. |
| `low_confidence` | `true` quando a base é o Terra: período 2000-11 a 2002-06 e dias de pane do Aqua sem VIIRS. O app mostra o aviso «Menos confiável: só o satélite Terra; erro cerca de 2× maior». |

`<p>` é `terra`, `aqua`, `snpp`, `noaa20` ou `noaa21`. Para o calendário de uma área, escolha o nível pelo zoom, filtre as células da área e some `harmonized` por mês e ano. A mediana histórica usada nas anomalias deve considerar só meses a partir de 2002-07 (decisão do Data Scientist): os meses só-Terra aparecem com aviso, mas não entram na referência. O `verify` confere, por ano e nível, que as detecções batem com a silver, que há uma linha por célula e dia e que o total de `harmonized` é o mesmo em todos os níveis.

**Rótulo da unidade** (gravado em cada commit como `unit_label_pt`/`unit_label_en`):
- PT: "Atividade de fogo harmonizada (células ativas equivalentes ao MODIS Aqua)"
- EN: "Harmonized fire activity (MODIS Aqua-equivalent active cells)"

Não chamar de "focos" nem de "incêndios". O original de cada satélite fica em `detections_<p>`.

`availability`, particionada por `year`: por plataforma e dia, `units` no mundo, `typical_units` (mediana de ±30 dias), `available`, `contributes` (entrou na série) e `basis`. Um dia sem nenhuma plataforma contribuindo não tem dado; não é dia sem fogo.

## Execução de 14/11/2026 (silver v644)

Executado em uma VM de 32 vCPU com `--memory-limit 80GB --threads 16`. O ajuste levou 26 s, com pico de 37 GB de RAM; o padrão de 16 GB não foi testado. A tabela dos 27 anos, com os sete níveis, levou 2 min 18 s, com pico de 19 GB. `cells` tem 10,1 GB e cobre de 2000-10-31 a 2026-11-14:

| Nível | r0 | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linhas (M) | 0,60 | 2,27 | 7,61 | 22,18 | 54,50 | 111,72 | 180,67 |

O `verify` passou em todos os anos e níveis; o total de `harmonized` é 47.776.951 em cada nível.

Teste final em 2018–2019, estimativa de cada plataforma contra as unidades reais do Aqua. Erro = Σ|estimativa − Aqua| / Σ Aqua por região e mês; "bruto" é o mesmo erro sem harmonizar.

| Série | Total estimado / Aqua | Erro res2 × mês | Erro res1 × mês | Erro res4 × mês | Erro bruto res2 × mês |
|---|---:|---:|---:|---:|---:|
| Suomi-NPP | 1,003 | 10,6% | 7,2% | 27,1% | 347% |
| NOAA-20 | 1,006 | 11,5% | 7,9% | 27,5% | 349% |
| Terra | 1,028 | 24,1% | 16,9% | 58,9% | 52% |
| Média das três, sem o Aqua | 1,011 | 12,8% | 8,9% | 32,7% | 802% |

- A ordem dos meses dentro de cada região res2 se mantém: correlação de postos mediana 0,97 (percentil 10: 0,91) na média das três.
- Concordância entre plataformas fora dos anos de ajuste: NOAA-20 / Suomi-NPP em 2023 = 1,008 (erro 7,3% em res2 × mês); NOAA-21 / NOAA-20 em 2025 = 1,004 (erro 6,9%).
- Série anual harmonizada de 2003 a 2025: entre 1,57 e 2,05 milhões de unidades. A soma das unidades brutas das plataformas vai de 3,1 milhões (2003) a 27,6 milhões (2024).
- As mudanças de ano para ano acompanham as do Aqua sozinho, que não entra na média depois de 2019: de 2020 a 2025 a série fica entre 3% abaixo e 5% acima das unidades do Aqua.

Modelo, relatório do teste e contagens por ano: `.omp/evidence/RF07-harmonization/`. Tabelas e modelo no lake: `s3://data-lake-fire-atlas-553607359608-us-east-2-an/experiments/rf07-harmonization/v0.1.0/`.

## Erro por época (14/11/2026, silver v650, modelo 0.1.0 fixo)

`harmonize-eras` recompõe cada época em 2018–2019 e compara com o Aqua real, só nos dias em que o Aqua funcionou. É a 2ª leitura desses anos: não serve como teste independente de uma v0.2. Relatório: `.omp/evidence/RF07-harmonization/eras.json`.

| Época | Composição medida (regra antiga, média) | Erro res2 × mês | res1 × mês | res4 × mês | Total / Aqua | Spearman meses (mediana) |
|---|---|---:|---:|---:|---:|---:|
| E0 2000-11..2002-06 | Terra | 24,1% | 16,9% | 58,9% | 1,028 | 0,935 |
| E1 2002-07..2011 | Terra + Aqua | 12,1% | 8,5% | 29,4% | 1,014 | 0,980 |
| E2 2012–2017 | Terra + Aqua + S-NPP | 9,9% | 6,9% | 25,2% | 1,010 | 0,979 |
| E3 2018–2019 | Terra + Aqua + S-NPP + NOAA-20 | 9,5% | 6,6% | 24,2% | 1,008 | 0,979 |
| E4 2020+ | S-NPP + NOAA-20 | 10,5% | 7,2% | 26,5% | 1,003 | 0,969 |

- **Com a regra `aqua-first-0.2`:** E1 a E3 passam a ser o próprio Aqua nos dias em que ele funcionou, então o erro contra o Aqua é zero por construção. O erro relevante fica em E0 (Terra, 24,1%) e E4 (VIIRS, 10,5%).
- **Terra em 2003–2004** (dentro dos anos de ajuste, limite otimista para E0): 21,5% em res2 × mês, total/Aqua 0,968. O erro da extrapolação para 2000–2002 não foi medido.
- **Regiões com pouco fogo:** no quartil de regiões res2 com menos unidades do Aqua, o erro passa de 85% em todas as épocas (E4: 381%). No quartil com mais fogo fica entre 8% e 22%.
- **VIIRS contra Aqua depois de 2019** (concordância com um Aqua em deriva, não erro), em res2 × mês:

| Ano | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|
| Diferença | 11,1% | 10,9% | 11,3% | 11,8% | 13,9% | 17,2% |
| VIIRS / Aqua | 0,996 | 1,002 | 0,998 | 0,986 | 0,973 | 0,967 |

## Reconstrução com a regra `aqua-first-0.2` (silver v650)

As tabelas `gold/harmonized/cells` e `availability` foram apagadas e regravadas a partir do zero. `cells` está na v26. A execução levou 9 min 47 s, com a silver lida do S3, pico de 21 GB e `verify` ok.

| Base do dia | Dias | Soma de `harmonized` (r0) |
|---|---:|---:|
| `aqua` | 6.356 | 32,7 M |
| `viirs` | 2.519 | 11,9 M |
| `terra` (`low_confidence`) | 605 | 3,4 M |
| sem dado | 6 | — |

## Versão 0.2: fatores regionais com encolhimento (15/11/2026)

Na v0.1, uma região res2 × mês com menos de 500 unidades nos anos de ajuste herdava o fator da região res1, e esta o global. Na v0.2 todo fator é encolhido para o do nível acima: `fator = (Aqua + κ · fator_pai) / (soma da curva + κ)`, com κ = 30. Com muitos dados, o fator é o da própria região; com poucos, fica perto do fator da região maior. A curva de detecção não mudou. Modelo: `models/harmonization/v0.2.0` (`harmonize-fit --kappa 30`).

**Validação** (`harmonize-cv`, protocolo do Data Scientist, relatório `.omp/evidence/RF07-harmonization/cv.json`):
- **Validação cruzada:** 3 dobras de anos dentro dos anos de ajuste, com 1 ano de intervalo de cada lado. Terra deixa de fora 2003–2005, 2009–2011 e 2015–2017; Suomi-NPP deixa de fora 2013, 2015 e 2017. NOAA-20/21 não são reajustados na validação.
- **Escolha de κ:** entre 30, 100, 300, 1000 e 3000, pelo menor erro no quartil de regiões com menos fogo, nas mesmas dobras (otimista). κ = 30 ficou na borda da grade: valores menores podem ser melhores e não foram testados.
- **Teste único de extrapolação:** Terra ajustado em 2006–2017 e testado em 2003–2004, rodado uma vez.

| | Terra res2 × mês | Terra res1 × mês | Terra, regiões com pouco fogo | S-NPP res2 × mês | S-NPP, regiões com pouco fogo |
|---|---:|---:|---:|---:|---:|
| Validação cruzada, v0.1 (corte) | 24,6% | 17,7% | 181% | 11,4% | 196% |
| Validação cruzada, v0.2 (κ = 30) | 22,7% | 16,3% | 157% | 11,1% | 188% |
| Extrapolação 2003–2004, v0.1 | 24,7% | 17,9% | 160% | — | — |
| Extrapolação 2003–2004, v0.2 | 22,9% | 16,7% | 132% | — | — |

- **Critério de aceite** (proposto pelo Data Scientist, adotado pelo usuário por delegação): não-inferioridade (erro até +0,5 p.p. em res1/res2 e +1,0 p.p. em res4), viés entre 0,97 e 1,03, correlação dos meses no máximo 0,01 menor, e ganho nas regiões com pouco fogo em pelo menos 2 de 3 dobras. **Passou** para Terra e Suomi-NPP, e na extrapolação (viés 0,975 nas duas versões).
- **Erro por época com o modelo v0.2** (2ª leitura de 2018–2019, `eras-v02.json`), res2 × mês: Terra 22,8% (v0.1: 24,1%); só VIIRS 10,3% (v0.1: 10,5%). Nas regiões com pouco fogo, o Terra cai de 169% para 145%.

O erro nas regiões com pouco fogo continua acima de 100%. Nelas, o número do calendário serve só para ordem de grandeza, e o app deve avisar.

## Versão 0.3: modelo de detecção para o VIIRS (15/11/2026)

Para Suomi-NPP, NOAA-20 e NOAA-21, a curva é substituída por um XGBoost que estima, por unidade, a chance de o Aqua também ter registrado a mesma célula na mesma passagem. O modelo é treinado no Suomi-NPP de 2013–2017 (alvo: Aqua ativo na mesma célula res6, dia e dia/noite, nos dias em que os dois estavam ok). Os fatores regionais com κ = 30 são aplicados por cima. O Terra continua na curva, porque o modelo piorou o erro dele.

As 14 entradas são calculadas a partir da silver no momento da aplicação:
- FRP somado e máximo e nº de focos;
- tamanho do pixel (scan/track);
- hora local e dia/noite;
- vizinhos ativos (anel 1);
- atividade no dia anterior e no seguinte;
- latitude, longitude e mês.

A entrada «passagem do Aqua» ficou de fora: a órbita do Aqua deriva depois de 2019, e a cobertura não é atualizada todo dia. Também testamos sem ganho: confiança, brilho, FRP dos vizinhos, anel 2 e a mistura com a curva.

Modelo: `models/harmonization/v0.3.0` (`harmonize-fit --kappa 30 --detection`, arquivo `detection.ubj` com sha256 em `model.json`).

| res2 × mês | v0.2 | v0.3 |
|---|---:|---:|
| Validação cruzada S-NPP | 11,1% | 10,3% |
| Validação cruzada S-NPP, pouco fogo | 188% | 139% |
| Suomi-NPP 2012 (2ª leitura) | 10,4% | 10,0% |
| Época 2020+ (2ª leitura de 2018–2019) | 10,3% | 10,3% |
| Época 2020+, res4 | 26,4% | 25,5% |
| Época 2020+, pouco fogo | 343% | 215% |

Relatórios: `detect-noaqua.json`, `detect-A.json`, `detect-B.json` e `eras-v03.json`. Passou na regra de aceite C para o Suomi-NPP.

## Limitações

- A unidade é célula ativa × meio dia na escala do Aqua. Não é número de incêndios, área queimada nem contagem de focos de um sensor.
- A estimativa é estatística: serve para somas por região e mês. Em área pequena (res4 × mês) o erro fica perto de 30%, e para uma célula em um dia ela é ruidosa.
- A estimativa do Terra é bem pior que a do VIIRS (passagem de manhã). Ela sustenta sozinha o período de 2000-11 a 2002-06 e entra com o mesmo peso das outras na média até 2019.
- Queima que o Aqua nunca detecta (fogos pequenos) fica fora da escala; a parte antiga da série não a recupera.
- Depois de 2019 não há referência para reconferir os fatores.
- A pane é detectada por plataforma e dia no mundo inteiro. Falha regional ou de parte do dia ainda reduz o valor daquele dia.
- Todos os tipos de foco entram (o NRT não traz o tipo), então fontes de calor estáticas contam como atividade.
- Os meses recentes são dados NRT e podem mudar quando o produto padrão os substituir. O dia corrente fica sem dado até completar.
- O `daily-ingest` regrava o ano corrente (e o anterior em janeiro). Uma mudança na silver mais antiga que isso, como reprocessamento do produto padrão, exige `harmonize --year <ano>`.
