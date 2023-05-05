# Análise do dataset

## Sumário

- [Análise do dataset](#análise-do-dataset)
  - [Sumário](#sumário)
  - [Porcentagem de rótulos por idioma](#porcentagem-de-rótulos-por-idioma)
  - [Contagem de tokens](#contagem-de-tokens)
  - [Nuvens de palavras do dataset](#nuvens-de-palavras-do-dataset)

## Porcentagem de rótulos por idioma

O gráfico abaixo mostra a porcentagem de rótulos para cada idioma, destacando a distribuição dos rótulos dentro de cada grupo de idioma. Ele pode ser usado para identificar padrões nos dados e entender se o sentimento dos textos é consistente entre os idiomas ou se há diferenças significativas entre eles.

![Porcentagem de rótulos por idioma](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/rotulos-por-idioma.png)

## Contagem de tokens

Os gráficos gerados são do tipo scatter plot, que é um gráfico de dispersão utilizado para mostrar a relação entre duas variáveis. No eixo x, temos um índice numérico para cada uma das amostras do dataset. No eixo y, temos a contagem de tokens do texto.

O primeiro gráfico mostra a contagem de tokens do dataset com textos brutos em português e inglês. As cores dos pontos indicam o idioma em que o texto está escrito. As linhas horizontais pontilhadas representam valores estatísticos, como a média, mediana, mínimo e máximo da contagem de tokens em todo o dataset.

![Contagem de tokens - Textos brutos](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/contagem-tokens-bruto.png)

O segundo gráfico mostra a contagem de tokens do dataset após a remoção de stopwords, também em português e inglês. As cores dos pontos indicam o idioma em que o texto está escrito. As linhas horizontais pontilhadas representam os mesmos valores estatísticos do primeiro gráfico.

![Contagem de tokens - Textos limpos](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/contagem-tokens-limpos.png)

Esses gráficos são úteis para visualizar a distribuição dos dados de contagem de tokens no dataset, e podem ajudar a identificar tendências ou diferenças entre os textos em diferentes idiomas. As linhas horizontais pontilhadas fornecem uma referência visual para os valores estatísticos e podem ajudar a identificar anomalias ou padrões nos dados.

## Nuvens de palavras do dataset

Uma nuvem de palavras é uma representação visual das palavras mais frequentes em um texto ou conjunto de textos. As palavras são exibidas em uma nuvem com tamanhos diferentes, onde as palavras mais frequentes são exibidas em tamanho maior e as menos frequentes em tamanho menor.

As nuvens de palavras abaixo representam as palavras mais comuns em nosso dataset em relação ao idioma:

![Nuvem de palavras - Português](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-portugues.png)
![Nuvem de palavras - Inglês](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-ingles.png)

Abaixo estão as seis nuvens de palavras que representam as palavras mais comuns em nosso dataset em relação à idioma e sentimento:

![Nuvem de palavras - Português - Negativo](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-portugues-negativo.png)
![Nuvem de palavras - Português - Neutro](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-portugues-neutro.png)
![Nuvem de palavras - Português - Positivo](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-portugues-positivo.png)
![Nuvem de palavras - Inglês - Negativo](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-ingles-negativo.png)
![Nuvem de palavras - Inglês - Neutro](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-ingles-neutro.png)
![Nuvem de palavras - Inglês - Positivo](https://github.com/Tuiaia/artificial-intelligence/blob/main/readme-assets/nuvem-ingles-positivo.png)
