# Localização das células do mapa: continente, país e cidade

`gold-places` grava `gold/h3_places`, uma tabela de consulta com uma linha por (`resolution`, `h3_cell`) para cada célula H3 res0..6 que tem detecção na silver. O mapa faz join pela coluna `h3_cell` com `gold/map_grid` ou `gold/forecast` (`h3_r6`). As 448 M linhas diárias não são reescritas.

```bash
# fontes (baixadas uma vez; /data é ignorado pelo git)
curl -LO --output-dir data/reference https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip
curl -LO --output-dir data/reference https://download.geonames.org/export/dump/cities1000.zip

LAKE=s3://data-lake-fire-atlas-553607359608-us-east-2-an
PYTHONPATH=src python -m fire_atlas_data gold-places \
  --silver $LAKE/prata/silver \
  --countries $LAKE/reference/ne_10m_admin_0_countries.zip --cities $LAKE/reference/cities1000.zip \
  --target $LAKE/gold/h3_places --memory-limit 6GB --threads 8 --temp-dir .tmp/places-spill
```

Cada execução reconstrói a tabela inteira a partir da versão atual da silver, num único commit Delta, e depois a confere com `verify`. `--countries` e `--cities` aceitam caminho local ou URI `s3://`. No lake, as fontes ficam em `reference/` e a tabela em `gold/h3_places`. O [daily-ingest](daily-ingest.md) reconstrói a tabela em toda execução.

## Colunas

| Coluna | Conteúdo |
|---|---|
| `resolution`, `h3_cell` | Chave. A tabela é particionada por `resolution`. |
| `latitude`, `longitude` | Centro da célula H3. |
| `continent` | `CONTINENT` do Natural Earth. Ilhas oceânicas podem vir como `Seven seas (open ocean)`. |
| `country_code`, `country_iso_a2`, `country_name`, `country_name_pt` | `ADM0_A3`, `ISO_A2_EH`, `NAME` e `NAME_PT` do Natural Earth. |
| `country_share` | Fração das detecções da célula que estão no país atribuído. |
| `country_method` | `centroid`, `detection_mean` ou `nearest` em res6; `predominant` em res0..5. |
| `detections` | Total de detecções da célula na versão de silver usada. |
| `city_name`, `city_geonameid`, `city_country_code`, `city_distance_km` | Localidade GeoNames mais próxima do centro da célula. Só preenchidas em res ≥ 4. |
| `static_share` | Só em res6: fração das detecções classificadas pelo FIRMS que são vulcão, outra fonte fixa em terra ou offshore (`type` 1, 2 ou 3 do arquivo histórico). NULL com menos de 10 detecções classificadas. |
| `static_source` | Só em res6: `true` quando `static_share` ≥ 0,5. É o filtro para esconder flares e vulcões no mapa de previsão (`gold/forecast.h3_r6 = h3_cell` com `resolution = 6`). NULL = desconhecido. |

## Método

- **País em res6:** é o polígono que contém o centro da célula. Se não houver, usa o polígono que contém a posição média das detecções da célula. Se ainda não houver, usa o polígono mais próximo, até 0,05° de distância. Fora disso, o país fica NULL; em geral são detecções no mar, como plataformas de petróleo.
- **País em res0..5:** é o país predominante pelas detecções entre as células res6 filhas. As detecções sem país entram no denominador de `country_share`.
- **Continente:** é sempre o continente do país atribuído.
- **Cidade:** é a localidade com 1000 habitantes ou mais mais próxima do centro da célula (GeoNames `cities1000`, licença CC-BY 4.0). Não é o município que contém o foco e pode estar em outro país (`city_country_code`). A busca não tem distância máxima; use `city_distance_km` para filtrar.
- **Fonte fixa:** usa a classificação `type` que o FIRMS publica só no arquivo histórico. Em 14/11/2026 (silver v648), a fração por célula res6 se dividia em dois grupos: 2,14 M células com menos de 10% de focos fixos e 22,4 mil com 90% ou mais. Apenas ~14 mil ficavam entre 10% e 90% (0,6%); o limiar de 50% fica nesse vale. Exemplos marcados: flares no Iraque, Irã e Venezuela, e os vulcões Kilauea e Nyiragongo. Os incêndios de turfa de Sumatra e Bornéu (type 0) não são marcados. Medição: `.omp/evidence/RF10-static/`.

## Execução de 14/11/2026 (silver v634)

Levou 4 min 30 s, com pico de 3,9 GB de RAM. O `verify` passou: uma linha por célula da silver em cada resolução, e a soma de `detections` igual às 660.374.978 detecções em todas as resoluções.

| Resolução | Células | Detecções sem país |
|---|---:|---:|
| r0 | 122 | 2.400 |
| r2 | 3.680 | 298.408 |
| r4 | 79.689 | 2.189.147 |
| r6 | 2.234.290 | 2.572.161 (0,39%) |

Em res6, 654,8 M detecções tiveram o país atribuído pelo centro da célula, 2,0 M pela posição média e 1,0 M pela fronteira mais próxima. A mediana de `city_distance_km` em res6 é 24,7 km e o percentil 95 é 180,8 km.

## Limitações

- Células novas só aparecem depois da próxima reconstrução, que o `daily-ingest` faz a cada execução.
- Uma célula res6 (~36 km²) recebe um único país, então focos perto da fronteira podem ficar com o país vizinho.
- As fronteiras seguem a visão padrão do Natural Earth (versão 5.1.1, arquivo de 2022); áreas disputadas não são arbitradas.
- `static_source` depende da classificação histórica do FIRMS: células vistas só no NRT ficam NULL, e um incêndio dentro da mesma célula res6 de um flare some junto com ele no filtro.
