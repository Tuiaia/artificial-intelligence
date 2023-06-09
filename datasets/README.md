# Datasets

## Sumário

- [Descrição](#descrição)
- [Porcentagem de rótulos por idioma](#porcentagem-de-rótulos-por-idioma)
- [Nuvens de palavras do dataset](#nuvens-de-palavras-do-dataset)

## Descrição

Neste diretório estão armazenados os datasets completos, parciais e auxiliares utilizados para alimentar o modelo de classificação, os detalhes referentes à coleta e construção estão registrados no [README](../jupyter-notebooks/dataset-notebook/README.md) do diretório dos Jupyter Notebooks.

Este documento apresenta uma análise do dataset construído. Aqui, exploramos a distribuição dos sentimentos nos textos e as palavras mais frequentes em cada idioma e sentimento. Essa análise auxilia na compreensão das características dos dados e na identificação de possíveis diferenças de sentimento entre os idiomas.

## Porcentagem de rótulos por idioma

O gráfico abaixo mostra a porcentagem de rótulos para cada idioma, destacando a distribuição dos rótulos dentro de cada grupo de idioma. Ele pode ser usado para identificar padrões nos dados e entender se o sentimento dos textos é consistente entre os idiomas ou se há diferenças significativas entre eles.

![Porcentagem de rótulos por idioma](../readme-assets/rotulos-por-idioma.png)

## Nuvens de palavras do dataset

Uma nuvem de palavras é uma representação visual das palavras mais frequentes em um texto ou conjunto de textos. As palavras são exibidas em uma nuvem com tamanhos diferentes, onde as palavras mais frequentes são exibidas em tamanho maior e as menos frequentes em tamanho menor.

As nuvens de palavras abaixo representam as palavras mais comuns em nosso dataset em relação ao idioma:

![Nuvem de palavras - Português](../readme-assets/nuvem-portugues.png)
![Nuvem de palavras - Inglês](../readme-assets/nuvem-ingles.png)

Abaixo estão as seis nuvens de palavras que representam as palavras mais comuns em nosso dataset em relação à idioma e sentimento:

![Nuvem de palavras - Português - Negativo](../readme-assets/nuvem-portugues-negativo.png)
![Nuvem de palavras - Português - Neutro](../readme-assets/nuvem-portugues-neutro.png)
![Nuvem de palavras - Português - Positivo](../readme-assets/nuvem-portugues-positivo.png)
![Nuvem de palavras - Inglês - Negativo](../readme-assets/nuvem-ingles-negativo.png)
![Nuvem de palavras - Inglês - Neutro](../readme-assets/nuvem-ingles-neutro.png)
![Nuvem de palavras - Inglês - Positivo](../readme-assets/nuvem-ingles-positivo.png)
