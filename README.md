# Inteligência Artificial

## Sumário

- [Descrição](#descrição)
- [Organização do Repositório](#organização-do-repositório)
- [Fluxo de trabalho](#fluxo-de-trabalho)
- [Como reproduzir o fluxo de trabalho](#como-reproduzir-o-fluxo-de-trabalho)
- [Contrução da base de dados](#contrução-da-base-de-dados)
- [Tecnologias e ferramentas utilizadas](#tecnologias-e-ferramentas-utilizadas)
- [Autor](#autor)

## Descrição

Este repositório tem como objetivo armazenar e controlar as versões de artefatos relacionados à análise de bases de textos noticiários e códigos de Machine Learning.

## Organização do Repositório

O repositório está organizado em:

- datasets: Este diretório contém as bases de dados utilizadas no projeto. Seu objetivo é disponibilizar os dados de maneira organizada e padronizada para que possam ser utilizados em diferentes etapas do projeto, como análise exploratória, pré-processamento e treinamento de modelos. O diretório conta com análises a respeito das notícias coletadas, essas estão descritas em seu respectivo [README](https://github.com/Tuiaia/artificial-intelligence/blob/main/datasets/README.md).
- jupyter-notebooks: Este diretório contém os arquivos Jupyter Notebook usados no projeto. Esses arquivos incluem a manipulação de dados e do modelo de classificação utilizados no projeto.
- trainings: Esse diretório é utilizado para armazenar os modelos de Machine Learning e suas respectivas configurações, para que possam ser utilizados em outras aplicações ou comparados com outros modelos. Como esses modelos ocupam bastante memória, o [README](https://github.com/Tuiaia/artificial-intelligence/blob/main/trainings/README.md) contido nesse diretório contém o endereço do nosso projeto no Hugging Face, que é usada especificamente para o armazenamento de modelos e conjuntos de dados de Machine Learning.

```text
artificial-intelligence
├── datasets
|  ├── b3
|  ├── financial-phrase-bank
|  ├── google-news
|  └── infomoney
├── jupyter-notebooks
|  └── dataset-notebook
└── trainings
|  ├── ...
```

## Fluxo de trabalho

Este projeto foi desenvolvido utilizando as seguintes etapas:

- Coleta de Dados:
As notícias utilizadas no treinamento foram coletadas de diversas fontes em português e inglês, incluindo:
  - B3 - Bora Investir
  - Google News (mais de 500 fontes distintas)
  - InfoMoney
  - Financial PhraseBank

  Os dados foram coletados por meio de "web scraping" utilizando a biblioteca Twint, Requests e BeautifulSoup em Python. Foram coletados apenas os textos das notícias, sem incluir imagens ou vídeos.
- Análise exploratória: A análise exploratória dos dados foi realizada a fim de entender a natureza dos dados e identificar padrões e tendências relevantes. Esta etapa incluiu a visualização dos dados, a identificação de valores ausentes, a análise da distribuição das variáveis, a frequência das palavras e a identificação de correlações. Algumas das análises feitas estão representadas por gráficos no [README](https://github.com/Tuiaia/artificial-intelligence/tree/main/datasets/README.md) do diretório dos datasets.
- Pré-processamento: O pré-processamento dos dados incluiu o tratamento de dados ausentes e a remoção de URLs e "palavras vazias", também chamadas de stopwords, estas são palavras comuns que geralmente não carregam informações significativas em análises de texto e são frequentemente removidas durante o pré-processamento de dados para reduzir o ruído e acelerar o processamento.
- Divisão dos dados: A divisão dos dados em treino, validação e teste foi feita utilizando o método holdout, com 70% dos dados para treino, 15% para validação e 15% para teste. Essa divisão foi realizada de forma estratificada em relação aos rótulos, idiomas e fontes das notícias para garantir que as proporções das classes (ou grupos) sejam mantidas nos subconjuntos de treinamento, validação e teste.
- Treinamento do Modelo: O treinamento foi feito a partir do ajuste fino do modelo [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) com a base de dados desenvolvida. Os parâmetros e as métricas de execução utilizados nos treinamentos estão disponíveis no [projeto](https://wandb.ai/tiagosanti/Tuiaia?workspace=user-tiagosanti) da plataforma Weights & Biases.
- Avaliação de desempenho do Modelo: A performance do algoritmo foi analisada com base na base de dados de teste através das métricas de:
  - Precisão
  - Recall
  - F1-score
  - Acurácia
- Análise de importância de variáveis e interpretabilidade: Foram realizadas análises de importância de variáveis e interpretabilidade para entender quais variáveis têm maior influência na classificação dos sentimentos. O procedimento para realizar essa análise está contido no Jupyter Notebook [insight-extraction.ipynb](https://github.com/Tuiaia/artificial-intelligence/blob/main/jupyter-notebooks/insight-extraction.ipynb).

## Como reproduzir o fluxo de trabalho

Sugerimos o uso do Google Colaboratory para reproduzir o fluxo de trabalho deste projeto. Essa plataforma possui várias bibliotecas pré-instaladas necessárias para a execução dos Jupyter Notebooks e requer apenas acesso ao Google Drive para utilizar a ferramenta.

Certifique-se de que os arquivos do repositório estejam organizados de acordo com a estrutura original e que o caminho armazenado pela variável `REPO_DIR` (se presente no código) seja o caminho raiz do repositório. Veja o exemplo abaixo:

```python
REPO_DIR = '/content/drive/MyDrive/pantanal.dev/artificial-intelligence'
```

Seguindo essas orientações, os procedimentos que utilizarem a navegação relativa ao `REPO_DIR` serão reprodutíveis em sua execução.

## Contrução da base de dados

A construção da base de dados está descrita detalhadamente no [README](https://github.com/Tuiaia/artificial-intelligence/blob/main/jupyter-notebooks/dataset-notebook/README.md) da pasta de Jupyter Notebooks relacionados à manipulação de datasets.

## Tecnologias e ferramentas utilizadas

Para o desenvolvimento da nossa base de dados e modelo de classificação, utilizamos as seguintes tecnologias e ferramentas:

- Python: Utilizado como linguagem de programação principal para o desenvolvimento do projeto.
- Google Colaboratory: Ambiente de desenvolvimento integrado (IDE) baseado em nuvem utilizado para escrever e executar o código Python, permitindo compartilhar o notebook com outros colaboradores de forma simples e rápida.
- PyTorch: Uma biblioteca popular de deep learning baseada em Python e amplamente utilizada para treinar e implementar modelos de aprendizado profundo, como redes neurais.
- Transformers: Biblioteca desenvolvida pela Hugging Face para trabalhar com modelos de aprendizado profundo voltados para processamento de linguagem natural (NLP), incluindo BERT e outros modelos state-of-the-art.
- Pandas: Biblioteca Python amplamente utilizada para manipulação e análise de dados, incluindo a leitura e escrita de arquivos CSV, bem como a manipulação de DataFrames.
- Numpy: Biblioteca Python para realizar operações matemáticas e manipulação de arrays multidimensionais, essencial para trabalhar com dados e modelos de aprendizado de máquina.
- Scikit-learn: Biblioteca Python utilizada para implementar algoritmos de aprendizado de máquina, pré-processamento de dados e avaliação de modelos.
- NLTK e spaCy: Bibliotecas Python para processamento de linguagem natural (NLP), como tokenização, lematização, remoção de stopwords e análise de frequência de palavras.
- Lime: Biblioteca para explicação de modelos de aprendizado de máquina, permitindo a interpretação de previsões e identificação de variáveis importantes.
- Matplotlib: Biblioteca Python para criação de visualizações de dados estáticas, animadas e interativas.
- W&B (Weights & Biases): Plataforma para rastreamento e visualização de experimentos de aprendizado de máquina, permitindo monitorar métricas, hiperparâmetros e artefatos do modelo.
- Datasets: Biblioteca da Hugging Face para carregar e manipular conjuntos de dados em formatos comuns, facilitando o pré-processamento e a transformação dos dados.
- WordCloud: Biblioteca Python para geração de nuvens de palavras a partir de um conjunto de texto, permitindo a visualização das palavras mais frequentes de maneira intuitiva e atraente. A nuvem de palavras é útil para análise exploratória e identificação de temas comuns em um conjunto de dados de texto.
- Requests e BeautifulSoup: Bibliotecas Python utilizadas para coletar dados da web (web scraping), permitindo a extração de informações de páginas da web de forma eficiente e rápida.

Todas as tecnologias utilizadas são de código aberto e disponíveis gratuitamente.

## Autor

Tiago Clarintino Santi - Graduando em Engenharia de Software na Universidade Federal de Mato Grosso do Sul
