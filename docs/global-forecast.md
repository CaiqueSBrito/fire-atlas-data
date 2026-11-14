# Previsão global semanal experimental

`fire_atlas_data.global_forecast.predict(activity_dir, model_dir, output_dir, *, bbox, issue_date, memory_limit="2GB", threads=2)` publica probabilidades por célula H3 resolução 5, sem treinar, recalibrar ou selecionar modelos. O alvo é ao menos uma detecção multissensor aceita e não excluída como estática nas datas UTC D+1…D+7; não é ignição nem ocorrência física comprovada de incêndio de vegetação. Tipos desconhecidos fora da máscara podem conter fontes estáticas.

```sh
python -m fire_atlas_data.global_forecast \
  --activity artifacts/global-activity --model artifacts/global-model \
  --output artifacts/global-forecast \
  --bbox -75 -35 -30 6 --issue-date 2025-01-01
```

A bbox usa oeste, sul, leste, norte em graus EPSG:4326. Oeste maior que leste cruza o antimeridiano. Os limites incluem o centro da célula. Coordenadas não finitas, latitude invertida, longitude fora de ±180°, extensão longitudinal nula e domínio vazio são recusados. A bbox global é `-180 -90 180 90`.

## Domínio e contexto

Entram somente centros dentro da bbox cuja `first_detection_date <= D`, inclusive células historicamente observadas apenas em fontes posteriormente excluídas como estáticas. Células não emitidas estão **fora do domínio**, não têm probabilidade zero. Não existe extrapolação para células nunca observadas.

A seleção espacial produz apenas as chaves. O construtor de atributos recebe o diretório completo de atividade, com `labels=False`: anéis, pais e histórico não são cortados pela bbox. Portanto a mesma célula/data/modelo mantém a mesma previsão em bboxes diferentes. A disponibilidade de sensores representa registros, não cobertura garantida; ausência de suporte não deve ser interpretada como ausência de fogo. A construção de atributos também exige máscara causal, congelada antes das datas de emissão.

## Contratos e calibração

São conferidos schema e resolução da atividade, identidades da máscara, schema do modelo, horizonte de sete dias e ordem exata de `FEATURE_NAMES`. Os artefatos declarados têm tamanho e SHA-256 verificados. O modelo nativo XGBoost JSON deve usar `binary:logistic` e os mesmos nomes de atributos. Quando presentes, `data_cutoff_date`, `training_cutoff_date` e `model_cutoff_date` não podem ser posteriores à emissão. A ausência de corte no manifesto não comprova causalidade retrospectiva.

`calibrator.json` contém `method: isotonic`, vetores finitos `x` estritamente crescente e `y` não decrescente, ambos em [0,1]. A saída conserva `raw_probability` e calcula `probability = numpy.interp(raw_probability, x, y)`, usando os valores extremos fora do intervalo de nós. Não há ajuste durante a inferência.

Mesmo quando a avaliação não demonstra melhoria sustentada, previsões experimentais são permitidas. Nenhuma saída deste módulo constitui validação ou capacidade operacional.

## Publicação

O destino não pode existir. A execução prepara um diretório irmão temporário e só o renomeia depois de produzir todos os resultados; falhas removem a preparação, sem publicar diretório parcial.

- `keys.parquet`: pares únicos `cell_id`, `issue_date` elegíveis.
- `predictions.parquet`: `cell_id`, `issue_date`, `target_start_date`, `target_end_date`, `horizon_days`, `raw_probability`, `probability`.
- `features-manifest.json`: cópia inspecionável do manifesto de atributos, preservada para auditoria da linhagem; seu Parquet intermediário não é publicado.
- `manifest.json`: schema `global-weekly-forecast/1.0.0`, status experimental, bbox, domínio, contagens, recursos, limitações e hashes dos manifestos da atividade, modelo e atributos intermediários, além dos hashes/tamanhos dos resultados.

Os atributos intermediários e o banco de trabalho não são publicados. Testes técnicos sintéticos cobrem invariância espacial, antimeridiano, elegibilidade temporal, calibração, corrupção de contrato e publicação recusada; não demonstram habilidade científica.
