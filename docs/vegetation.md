# Vegetação amostrada MOD13Q1 — RF10/CA08

## Fonte e execução

O módulo usa somente HTTP GET público do [ORNL DAAC TESViS](https://modis.ornl.gov/data/modis_webservice.html), sem credencial, Google Earth Engine ou tiles globais. A API oficial fornece inventário `/MOD13Q1/dates`, atributos `/MOD13Q1/bands` e `/MOD13Q1/subset`; cada subset aceita no máximo dez datas. Consulta de referências: 2026-11-12.

```sh
PYTHONPATH=src .venv/bin/python -m fire_atlas_data.vegetation \
  --features output/rf10-activity/features/features.parquet \
  --output-dir output/rf10-environment/vegetation \
  --concurrency 3
```

O destino deve ser novo. A aquisição projeta somente `DISTINCT cell_id, issue_date` do Parquet de entrada e calcula os centroides H3; não lê rótulos. Usa aquecimento desde o primeiro dia do mês anterior à primeira emissão, e solicita apenas composições cujo início é no máximo `max(issue_date)-15 dias`. Para setembro–outubro de 2026, inclui agosto. Não solicita nem pontua emissão a partir de 2026-11-13, preservando a janela futura reservada. Limites adicionais: 500 centroides, intervalo de emissão até 120 dias, 4 MiB por resposta e concorrência configurável de 1–3.

Por centro, a requisição pede **todas as bandas** com `kmAboveBelow=1`, `kmLeftRight=1`, agrupando até dez datas. A amostra é um pequeno patch próximo ao centro H3, **não uma agregação da célula inteira**. A resposta define o recorte efetivo: resolução sinusoidal esperada de aproximadamente 231,656 m (produto nominal de 250 m), dimensões, origem e limites projetados são preservados em `patches.json`. NDVI, EVI, pixel reliability, VI Quality, dia observado, reflectâncias e demais bandas ficam integralmente nos JSON brutos; nenhuma banda bruta é substituída pelo agregado.

## Corte temporal e significado retrospectivo

O produto é científico de 16 dias Terra/MODIS, não NOAA20 NRT. Para emissão D, seleciona-se a composição mais recente cujo `start+15 dias <= D` **e**, quando fornecido, `proc_date <= fim de D UTC`. O timestamp `YYYYDDDHHMMSS` é interpretado como produção, conforme a [convenção oficial NASA](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061). Produção posterior impede uso naquela emissão; pode selecionar composição anterior. Nenhuma data de produção é transformada em `available_at`.

**A disponibilidade histórica efetiva permanece desconhecida. Este é um experimento retrospectivo, não evidência de disponibilidade operacional.** Produção anterior a D é condição necessária quando observável, não comprovação de entrega antes de D. Produção ausente é preservada e contabilizada, sem timestamp inventado. Se nenhum valor real sobreviver aos cortes/QA, a aquisição falha e conserva a evidência, sem publicar um Parquet vazio de valores fictícios.

O shortname `MOD13Q1` não prova coleção `.061`: os exemplos antigos da API exibem `version`, mas respostas atuais podem omiti-lo. Quando retornada, a versão é preservada por composição; quando omitida, permanece explicitamente desconhecida. O catálogo NASA `.061` documenta o produto e suas variáveis, **não confirma por si só a coleção dos bytes ORNL adquiridos**. As identidades de produção, tile, versão e data devem concordar entre bandas da mesma composição; inconsistência é erro, não mistura silenciosa. Os brutos e seus hashes congelam os valores realmente consumidos.

## QA e contrato de colunas

Política estrita baseada na [especificação oficial de pixel reliability](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD13Q1): somente rank **0 — Good data**; rejeitar 1 (marginal), 2 (neve/gelo), 3 (nuvem) e -1 (fill). Exige-se ainda VI Quality não-fill, NDVI e EVI dentro do intervalo bruto `[-2000,10000]`, e dia observado pertencente à composição. O VI Quality bruto é preservado, mas seus bits não impõem um segundo limiar oculto. Os dois índices usam a mesma máscara; escala multiplicativa `0.0001`, offset zero, fill `-3000`, atributos validados contra `/bands`. Valores fora do intervalo/fill são ausentes, nunca truncados nem imputados. Rank/QA impossíveis, arrays desalinhados, banda/data requerida ausente e metadados diferentes falham visivelmente.

`covariates.parquet` tem chave única `(cell_id VARCHAR, issue_date DATE)` igual ao conjunto distinto da base. Todas as covariáveis são `DOUBLE`, na ordem de `manifest.columns` e da constante `VEGETATION_FEATURES`:

| Coluna | Unidade e cálculo |
| --- | --- |
| `veg_ndvi` | Média NDVI dos pixels conjuntamente válidos, razão adimensional |
| `veg_evi` | Média EVI da mesma amostra válida, razão adimensional |
| `veg_valid_fraction` | Pixels conjuntamente válidos / pixels solicitados do patch |
| `veg_age_days` | D menos fim nominal da composição (`start+15 dias`) |
| `veg_source_age_days` | D menos média dos dias de observação dos pixels válidos |

A fração válida descreve somente a amostra MODIS. **Não é cobertura de nuvens/observabilidade NOAA20 nem cobertura da célula H3.** Sem pixels válidos, NDVI/EVI e idade dos pixels são nulos; fração válida zero é uma medida real de QA, não imputação. Sem composição temporalmente elegível, todas as cinco colunas ficam nulas. Escolhe-se a composição antes de avaliar QA, sem backfill oportunista por composição mais antiga/verde. Greenness é proxy de condição da vegetação, não medida direta de umidade, carga de combustível ou ausência de fogo.

## Artefatos e falhas

- `raw/*.json`: bytes completos de cada resposta de bandas, inventário e subset.
- `receipts/*.json`: URL exata, URL final, instante real de download, cabeçalhos, status, tamanho e SHA-256; `available_at=null`.
- `patches.json`: centro, limites/resolução/dimensões e versão retornada ou ausente por célula.
- `composites.json`: agregados por célula/composição, produção, tile, versão, pixels válidos e data média observada.
- `covariates.parquet`: covariáveis numéricas, sem rótulos.
- `manifest.json`: versão do método/schema, configuração, ordem/unidades de colunas, hashes da entrada/código/todas as saídas, metadados de bandas, recibos, contagens e limitações.

Publicação exige destino novo e usa arquivos completos; `manifest.json` é o marcador de commit, publicado por último. Em falha, o diretório temporário informado na exceção conserva brutos/recibos e `failure.json`, sem os representar como execução bem-sucedida. Não há fallback de fonte, retry implícito, zero artificial ou alegação de operacionalidade.

Os testes em `tests/test_vegetation.py` usam fixtures sintéticos offline para datas, cortes de produção, QA, nulos, metadados, chave/schema/hashes, não sobrescrita, proteção do futuro e conservação de evidência. Esses testes não demonstram validação científica ou disponibilidade NASA. A execução dos testes pertence ao integrador; o worker somente executa a aquisição real autorizada.

## Aquisição real exercitada

O comando acima terminou com sucesso para a grade completa: **161 células × 37 emissões = 5.957 linhas**, 644 composições por célula e 163 requisições (bandas + inventário + uma por centro), concorrência 3. Foram solicitadas quatro datas: `A2026225`, `A2026241`, `A2026257`, `A2026273`. Os cortes selecionaram início 2026-08-29 em 3.059 linhas, 2026-09-14 em 1.932 e 2026-09-30 em 966. Não houve seleção temporal ausente nem produção desconhecida selecionada. NDVI/EVI e idade do pixel ficaram nulos em 24 linhas por QA; fração válida e idade da composição não ficaram nulas. Nenhum teste ou treino foi executado pelo worker.

Saídas imutáveis em `output/rf10-environment/vegetation`:

| Artefato | Bytes | SHA-256 |
| --- | ---: | --- |
| `covariates.parquet` | 20.043 | `4e872a6cf286cbb55f1ed877f138aec854895e37cc6926b10d220ddcb4f7fb5a` |
| `composites.json` | 282.228 | `9cbb665aeeffffd27370fb47ec1d7615f3816658632c67cd2d24d97f4680d2fd` |
| `patches.json` | 96.888 | `aa82a6663c9a60ddbbe2d43ada37b5fb83632901fdef59f71ec7fe274ccb1fab` |
| `manifest.json` | — | `fdae06681bc707b39a6bae953d3c7c2a528e11cc19f3b053070e1db82d899da4` |

O manifesto contém hashes individuais dos 163 brutos e 163 recibos, sem auto-hash circular. O hash de código nele corresponde ao arquivo efetivamente executado na aquisição; a exportação do nome público da constante e o endurecimento da limpeza de publicação foram aplicados depois, sem reescrever artefatos adquiridos.
