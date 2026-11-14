# Features globais semanais

`fire_atlas_data.global_features` implementa o contrato `global-weekly-features/1.0.0`, método `1.0.0`, para células H3 resolução 5 e horizonte de sete dias. A entrada é o diretório `global-cell-activity/1.0.0`; não há download, treino ou preenchimento retroativo automático.

## APIs e execução

```python
sample(activity_dir, output_dir, *, start, end, rate, seed="42",
       memory_limit="2GB", threads=2)
build(activity_dir, keys, output_dir, *, labels,
      memory_limit="2GB", threads=2)
```

`start` e `end` são `datetime.date`. `keys` é um Parquet com `cell_id VARCHAR` e `issue_date DATE`. Cada par deve ser único, não nulo e pertencer ao domínio causal: `first_detection_date <= issue_date`. Chaves vazias, desconhecidas, duplicadas ou anteriores à primeira detecção são rejeitadas. As datas de emissão devem estar no intervalo de aquisição declarado. A máscara deve ter `mask.parameters.reference_end` anterior ou igual à menor data de emissão; uma máscara retrospectiva não pode ser reutilizada silenciosamente em emissões anteriores.

```sh
python -m fire_atlas_data.global_features sample \
  --activity-dir data/global-activity --output-dir data/global-keys \
  --start 2013-01-19 --end 2013-12-24 --rate 0.01 --seed 42

python -m fire_atlas_data.global_features build \
  --activity-dir data/global-activity --keys data/global-keys/keys.parquet \
  --output-dir data/global-features --labels
```

Para inferência, use `--no-labels`. Ambos os comandos aceitam `--memory-limit` e `--threads`. Os diretórios de saída não podem existir. O exemplo pressupõe máscara congelada até a primeira emissão e suporte disponível para o horizonte final; não executa backfill nem treinamento.

## Amostragem

A amostragem Bernoulli uniforme considera todos os pares célula/data elegíveis no intervalo inclusivo solicitado, independentemente de haver detecção naquele dia. A decisão usa MD5 de `cell_id|issue_date|seed`, convertido em inteiro sem sinal de 64 bits, comparado à taxa solicitada. Não há balanceamento por rótulo, bbox ou atividade futura. Mesmas entradas, taxa e semente produzem as mesmas chaves ordenadas; `rate=1` seleciona todo o domínio elegível. A taxa deve estar em `(0,1]`. Uma amostra pode ser vazia; `build` exige ao menos uma chave.

## Ordem e significado das 39 entradas

A ordem efetiva é exportada por `FEATURE_NAMES` e repetida no manifesto:

1. `cell_active_days_{1,7,30,90,365}d`;
2. `cell_viirs_detections_{1,7,30,90,365}d`;
3. `cell_modis_detections_{1,7,30,90,365}d`;
4. `cell_viirs_frp_mean_7d_mw`, `cell_viirs_frp_mean_30d_mw`, `cell_modis_frp_mean_7d_mw`, `cell_modis_frp_mean_30d_mw`;
5. `days_since_last_detection`, `days_since_first_detection`;
6. `ring1_active_cell_days_{7,30}d`, `parent4_active_cell_days_{7,30}d`, `parent3_active_cell_days_{7,30}d`;
7. `modis_prior_month_active_fraction`, `modis_prior_month_supported_days`, `viirs_prior_month_active_fraction`, `viirs_prior_month_supported_days`;
8. `modis_prior_active_years`, `modis_prior_supported_years`;
9. `doy_sin`, `doy_cos`, `latitude`, `longitude`;
10. `viirs_platforms_7d`, `modis_platforms_7d`.

Todas são `DOUBLE`, com `NULL` preservado. As expansões percorrem as janelas na ordem indicada dentro de cada família.

As janelas incluem D: uma janela de sete dias cobre `[D-6,D]`, nunca D+1. Atividade significa pelo menos uma detecção não estática na célula/data; contagens de sensores somam pixels, não eventos. A média FRP é a soma de FRP dividida pelo número de linhas com FRP, não uma média não ponderada de médias diárias. Sem linhas com FRP, permanece `NULL`.

O suporte vem de `platform_days`, considerando apenas linhas com `detections > 0`. Contagens e médias por família exigem relato daquela família em **todos** os dias da janela; atividade geral e contexto exigem relato de pelo menos uma família em todos os dias. Janelas incompletas, inclusive prefixos do histórico, são `NULL`. Com suporte completo, ausência de atividade produz contagem zero. Esse suporte significa dias de relato global, não cobertura orbital local nem observação local sem nuvens.

`days_since_last_detection` usa a última atividade não estática até D e é nulo se não existe histórico dessa atividade. `days_since_first_detection` usa a primeira detecção aceita do domínio, inclusive estática, conforme `cells.parquet`. O anel 1 exclui a célula focal; os pais res4/res3 a incluem. Todas as agregações espaciais usam o contexto completo do diretório de atividade, nunca apenas as chaves solicitadas: mudar o bbox ou subconjunto de saída não muda as features de uma chave mantida.

A climatologia considera o mesmo mês de **todos os anos civis estritamente anteriores** ao ano de D; não inclui dias anteriores do ano corrente. O denominador conta datas distintas de relato da família e o numerador conta dias ativos da célula nessa família. Sem suporte, o denominador é zero e a fração é `NULL`. Anos MODIS suportados são anos anteriores com algum relato; anos ativos são anos anteriores com alguma atividade MODIS da célula. Sem nenhum ano suportado, anos ativos é `NULL`. Isso não exige anos completos e não corrige diferenças entre sensores.

A sazonalidade usa `sin/cos(2π × (dia_do_ano−1)/365,25)`; latitude/longitude são centros H3. Plataformas são identificadores distintos com relato nos últimos sete dias, por família, e exigem suporte diário da união das famílias. Assim, uma família sem relato pode ter zero plataformas quando a outra sustenta toda a janela. Esses valores não garantem continuidade de uma constelação nem cobertura local.

## Rótulos e causalidade

As features usam somente aquisições até D. Quando `labels=True`, uma consulta separada calcula atividade em `[D+1,D+7]`: `target_active_days INTEGER` e `label INTEGER`, equivalente a `target_active_days > 0`. Exige-se que o intervalo declarado chegue a D+7 e que cada dia do horizonte tenha relato de alguma família. Um horizonte incompleto é rejeitado, nunca tratado como negativo. Quando `labels=False`, as duas colunas de resultado não existem.

A saída `features.parquet` é ordenada por `issue_date,cell_id`: primeiro `cell_id`, `issue_date`, `target_start_date`, `target_end_date`, `horizon_days`, depois as 39 features e, somente quando solicitados, os dois resultados. Alterações em atividade posterior a D podem mudar os resultados, mas não as features.

## Recursos e proveniência

O processamento usa SQL DuckDB sobre Parquet e tabelas intermediárias em disco, com limite de memória, número de threads e diretório de spill explícitos; não materializa o painel global em pandas/NumPy. O diretório temporário é irmão do destino. Somente após concluir o Parquet e o manifesto ocorre a publicação por renomeação. Falhas removem o estágio; o banco intermediário e spill não fazem parte do artefato publicado.

O manifesto registra esquema, `feature_method_version`, nomes ordenados, presença de rótulos, linhas, positivos (nulo em inferência), intervalo de emissão, SHA-256 do manifesto de atividade e da máscara, intervalo de aquisição, SHA-256 das chaves e caminho/tamanho/SHA-256 da saída. A amostragem publica `keys.parquet` e manifesto `global-weekly-keys/1.0.0` com parâmetros e proveniência.

Antes de consultar a atividade, ambos os comandos verificam os caminhos relativos únicos, tamanhos e SHA-256 de todos os arquivos declarados em `outputs`. As tabelas obrigatórias devem estar declaradas; Parquet extra não declarado é rejeitado. A consulta diária usa exclusivamente a lista verificada de arquivos, não um glob aberto. Bytes modificados ou arquivos adicionados sem atualizar a proveniência impedem a publicação.

A ausência de detecção não comprova ausência de fogo. Datas de aquisição não comprovam disponibilidade operacional; tipo FIRMS desconhecido fora da máscara pode conter fontes estáticas não identificadas. Sobreposição de plataformas, diferença entre sensores e cobertura variável continuam limitações do alvo observado.
