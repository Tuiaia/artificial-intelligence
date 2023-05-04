# Inteligência Artificial

## Sumário

- [Inteligência Artificial](#inteligência-artificial)
  - [Sumário](#sumário)
  - [Descrição](#descrição)
  - [Funcionalidades](#funcionalidades)
  - [Organização do Repositório](#organização-do-repositório)
  - [Fluxo de trabalho](#fluxo-de-trabalho)
  - [Como reproduzir o fluxo de trabalho](#como-reproduzir-o-fluxo-de-trabalho)
  - [Contrução da base de dados](#contrução-da-base-de-dados)
    - [Infomoney](#infomoney)
    - [Financial PhraseBank](#financial-phrasebank)
    - [B3 - Bora Investir](#b3---bora-investir)
    - [Google News](#google-news)
  - [Treinamento do modelo](#treinamento-do-modelo)
  - [Avaliação e teste do modelo](#avaliação-e-teste-do-modelo)
  - [Tecnologias e ferramentas utilizadas](#tecnologias-e-ferramentas-utilizadas)
  - [Autor](#autor)

## Descrição

Este repositório tem como objetivo armazenar e controlar as versões de artefatos relacionados à análise de bases de textos noticiários e códigos de Machine Learning. Este contém o treinamento de um modelo de classificação de sentimentos de notícias em português e inglês, utilizando técnicas de Processamento de Linguagem Natural (NLP) e Deep Learning. O objetivo é identificar se uma notícia é positiva, negativa ou neutra com base no seu conteúdo.

Este projeto utiliza uma técnica de aprendizado semi-supervisionado para melhorar o desempenho do modelo. Inicialmente, o modelo é treinado com uma quantidade limitada de dados rotulados. Em seguida, o modelo é usado para rotular mais dados, que são adicionados ao conjunto de treinamento e usados para refinar o modelo posteriormente. Essa abordagem permite melhorar o desempenho do modelo com um esforço menor de rotulação manual de dados.

## Funcionalidades

- Classificador de sentimento de notícias
- Extração de Insights das classificações das notícias

## Organização do Repositório

O repositório está organizado em:

- datasets: Este repositório contém as bases de dados utilizadas no projeto. Seu objetivo é disponibilizar os dados de maneira organizada e padronizada para que possam ser utilizados em diferentes etapas do projeto, como análise exploratória, pré-processamento e treinamento de modelos.
- jupyter-notebooks: Este repositório contém os arquivos Jupyter Notebook utilizados no projeto, os quais contemplam desde a análise exploratória dos dados até o treinamento do modelo de classificação. Os notebooks também servem como uma forma de facilitar a reprodução dos procedimentos e resultados.
- trainings: Este repositório contém os modelos de Machine Learning treinados no projeto. Seu objetivo é armazenar os modelos e seus respectivos parâmetros para que possam ser utilizados posteriormente em outras aplicações ou para fins de comparação com outros modelos. Além disso, o repositório também contém informações sobre as métricas de avaliação dos modelos, que permitem avaliar a qualidade do modelo em relação aos dados utilizados no treinamento.

## Fluxo de trabalho

Este projeto foi desenvolvido utilizando as seguintes etapas:

- Coleta de Dados:
As notícias utilizadas no treinamento foram coletadas de diversas fontes em português e inglês, incluindo:
  - B3 - Bora Investir
  - Google News (mais de 300 fontes diferentes)
  - InfoMoney
  - Financial PhraseBank

Os dados foram coletados por meio de "web scraping" utilizando a biblioteca Twint, Requests e BeautifulSoup em Python. Foram coletados apenas os textos das notícias, sem incluir imagens ou vídeos.

Neste repositório, constam os seguintes processos:

- Análise exploratória: A análise exploratória dos dados foi realizada a fim de entender a natureza dos dados e identificar padrões e tendências relevantes. Esta etapa incluiu a visualização dos dados, a identificação de valores ausentes, a análise da distribuição das variáveis, a frequência das palavras e a identificação de correlações.
- Pré-processamento: O pré-processamento dos dados incluiu o tratamento de dados ausentes e a remoção de URLs e "palavras vazias", também chamadas de stopwords, estas são palavras comuns que geralmente não carregam informações significativas em análises de texto e são frequentemente removidas durante o pré-processamento de dados para reduzir o ruído e acelerar o processamento.
- Divisão dos dados: A divisão dos dados em treino, validação e teste foi feita utilizando o método holdout, com 70% dos dados para treino, 15% para validação e 15% para teste. Essa divisão foi realizada de forma estratificada em relação aos rótulos, idiomas e fontes das notícias para garantir que as proporções das classes (ou grupos) sejam mantidas nos subconjuntos de treinamento, validação e teste.
- Treinamento do Modelo: O treinamento foi feito a partir do ajuste fino do modelo [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) com a base de dados desenvolvida. Os parâmetros e as métricas de execução utilizados nos treinamentos estão disponíveis no [projeto](https://wandb.ai/tiagosanti/Tuiaia?workspace=user-tiagosanti) da plataforma Weights & Biases.
- Avaliação de desempenho do Modelo: A performance do algoritmo foi analisada com base na base de dados de teste através das métricas de:
  - Precisão
  - Recall
  - F1-score
  - Acurácia
- Análise de importância de variáveis e interpretabilidade: Foram realizadas análises de importância de variáveis e interpretabilidade para entender quais variáveis têm maior influência na classificação dos sentimentos.

## Como reproduzir o fluxo de trabalho

Recomendamos o uso do **Google Colaboratory** para reproduzir o fluxo de trabalho do projeto, uma vez que a plataforma possui diversas bibliotecas pré-instaladas necessárias para a execução dos Jupyter Notebooks.

Certifique-se de que os arquivos do repositório estejam organizados de acordo com a estrutura original e que o caminho armazenado pela variável `REPO_DIR` (se presente no código) seja o caminho raiz do repositório. Veja o exemplo abaixo:

```python
REPO_DIR = '/content/drive/MyDrive/pantanal.dev/artificial-intelligence'
```

Seguindo essas orientações, as funções que utilizarem a navegação relativa ao `REPO_DIR` serão reprodutíveis em sua execução.

## Contrução da base de dados

### Infomoney

InfoMoney é uma das principais fontes de notícias financeiras e de negócios no Brasil. O site fornece informações atualizadas sobre os mercados financeiros, economia, investimentos e outros tópicos relevantes. Para coletar e tratar os dados da InfoMoney, foram realizados os seguintes passos:

1. Clonar o repositório Twint e instalar as dependências: Clonar o repositório Twint e instalar as dependências necessárias para utilizar a biblioteca Twint na coleta de dados do Twitter.

```python
!git clone --depth=1 https://github.com/twintproject/twint.git
!pip3 install -r twint/requirements.txt
!pip install --upgrade aiohttp && pip install --force-reinstall aiohttp-socks

import nest_asyncio
nest_asyncio.apply()
```

1. Busca no Twitter com Twint: Configurar a busca no Twitter usando a biblioteca Twint para coletar tweets do perfil da InfoMoney.

```python
def run_scrap(date=None):
    # Configuração do objeto twint
    config = twint.Config()
    config.Profile_full = True  # Coleta perfil completo
    config.Pandas = True  # Habilita o armazenamento dos resultados em um DataFrame
    config.Username = "infomoney"  # Define o usuário-alvo para a busca de tweets

    if date:
        config.Until = date  # Define a data limite para coletar tweets, caso seja fornecida
    
    # Executa a busca de tweets com a configuração definida
    twint.run.Search(config)

    # Retorna o DataFrame com os tweets coletados
    return twint.storage.panda.Tweets_df
```

3. Coletar tweets: Executar o processo de busca e coleta de tweets até obter um total de 20.000 tweets, tratando o DataFrame a cada iteração.

```python
while tweets_df.shape[0] <= 20000:
    tweets = run_scrap(older_tweet_date)

    # Concatenando raspagem ao df total
    tweets_df = pd.concat([tweets_df, tweets], axis=0, ignore_index=True)

    # Removendo possíveis registros duplicados
    tweets_df.drop_duplicates(subset='id', inplace=True)
    tweets_df.drop_duplicates(subset='tweet', inplace=True)

    print(f'Tweets_df length: {tweets_df.shape[0]}')

    # Ordenando df pela data
    tweets_df.sort_values('date', ignore_index=True, inplace=True, ascending=False)

    # Resetando index
    tweets_df.reset_index()

    # Armazenando data do último tweet
    older_tweet_date = tweets_df.loc[len(tweets_df)-1, 'date'].split(' ')[0]
```

4. Tratamento do DataFrame: Tratamento da coluna de 'urls', remoção dos urls dos tweets e remoção de registros duplicados.

```python
tweets_df['urls'] = tweets_df['urls'].apply(lambda x: x.replace("['", "").replace("']", ""))
```

```python
import re

def remove_links(text):
    # Padrão para encontrar URLs (http, https, www)
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)

tweets_df['tweet'] = tweets_df['tweet'].apply(remove_links)
```

```python
tweets_df.drop_duplicates('tweet', inplace=True)
tweets_df.drop_duplicates('urls', inplace=True)
```

5. Coleta dos corpos das notícias: Coleta dos corpos das notícias através da url presente em cada tweet utilizando Requests e BeautifulSoup.

```python
def get_news_corpus(url):
    # Realiza uma requisição GET para a URL fornecida
    response = requests.get(url)
    # Verifica se a resposta foi bem-sucedida
    response.raise_for_status()
    # Cria um objeto BeautifulSoup para analisar o conteúdo HTML da resposta
    soup = BeautifulSoup(response.text, 'html.parser')

    # Cria um objeto BeautifulSoup para analisar o conteúdo HTML da resposta
    title = soup.find('h1', class_='typography__display--2').text
    subtitle = soup.find('div', class_='single__excerpt typography__body--2 spacing--mb4').text
    corpus = soup.find('div', class_='element-border--bottom spacing--pb4')
    corpus_text = []

    # Itera sobre os parágrafos do corpo e adiciona à lista 'corpus_text'
    for p in corpus:
      corpus_text.append(p.text)
    
    # Junta os parágrafos extraídos em uma única string
    corpus_text = "".join(corpus_text)
    # Combina o título, subtítulo e corpo do texto em uma única string
    text = f'{title}. {subtitle}. {corpus_text}'
    # Formata o texto (função 'format_text' não fornecida)
    text = format_text(text)  

    return text
```

6. Processamento de todos os tweets: coleta e armazenamento de todos os corpos de todos os urls presentes no DataFrame.

```python
import time

def process_dataset(dataset, output_file='datasets/infomoney_news.csv', save_interval=100):
    # Inicializa a coluna 'text' no dataset
    dataset['text'] = None
    total_records = len(dataset)
    processed_records = 0

    # Itera sobre as linhas do dataset
    for index, row in dataset.iterrows():
      # Verifica se a coluna 'text' está vazia
      if pd.isnull(row['text']):
        try:
            # Obtém o texto da notícia usando a função 'get_news_corpus'
            news_text = get_news_corpus(row['urls'])
            # Armazena o texto obtido no dataset
            dataset.loc[index, 'text'] = news_text
            print(f"Record {index} processed successfully")
        except ConnectionError as e:
            print(f"Connection error processing record {index}: {e}")
        except Exception as e:
            print(f"Error processing record {index}: {e}")

        # Atualiza o contador de registros processados
        processed_records += 1
        
        # Salva o dataset a cada 'save_interval' registros processados
        if processed_records % save_interval == 0:
            dataset.to_csv(output_file, index=False)
            print(f"Progress: {processed_records}/{total_records}")
            time.sleep(2)

    # Salva o dataset finalizado
    dataset.to_csv(output_file, index=False)
    print(f"Processing completed: {processed_records}/{total_records}")

process_dataset(tweets_df)
```

7. Tratamento dataset após rotulagem: formatação dos rótulos e remoção de valores nulos.

```python
# Remoção de registros não rotulados
infomoney_news_labelled = infomoney_news_labelled.dropna()

# Formatação de rótulos em letrar minúsculas
infomoney_news_labelled['label'] = infomoney_news_labelled['label'].apply(lambda x: x.lower())

# Remoção de registros com erros de rotulação
infomoney_news_labelled['label'] = infomoney_news_labelled['label'].apply(lambda s: s if s in ['positiva', 'negativa', 'neutra'] else None)
infomoney_news_labelled = infomoney_news_labelled.dropna()

# Reinício dos índices
infomoney_news_labelled.reset_index(drop=True, inplace=True)

# Remoção das colunas de 'tweet' e 'urls'
infomoney_news_labelled.drop(columns=['tweet', 'urls'], inplace=True)
```

Estes foram os procedimentos fundamentais para a construção do dataset das notícias do InfoMoney. No respectivo Jupyter Notebook, há passos adicionais para teste e visualização de alguns procedimentos.

### Financial PhraseBank

1. Download do dataset: Download do dataset em português através do [Kaggle](https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation)
2. Conversão dos rótulos: Conversão dos rótulos para rótulos numéricos.

```python
def encode_classes(class_name):
    # Converte o nome da classe em um valor numérico
    if class_name == 'positive':
        return 2
    if class_name == 'neutral':
        return 1
    if class_name == 'negative':
        return 0

# Aplica a função 'encode_classes' para cada valor na coluna 'label' do DataFrame 'fpb_ptbr_df'
fpb_ptbr_df['label'] = fpb_ptbr_df['label'].apply(encode_classes)
```

3. Tratamento do dataset: remoção de registros duplicados:

```python
# Remove as linhas duplicadas no DataFrame 'fpb_ptbr_df' com base na coluna 'text'
fpb_ptbr_df = fpb_ptbr_df.drop_duplicates(subset='text')
```

4. Download do dataset: Download do dataset em inglês através do [Research Gate](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) (Malo, Pekka & Sinha, Ankur & Takala, Pyry & Korhonen, Pekka & Wallenius, Jyrki. (2013). FinancialPhraseBank-v1.0.)
5. Formatação inicial do dataset: Carregamento do dataset e divisão dos textos e rótulos.

```python
with open('datasets/financial-phrase-bank/financial-phrase-bank-eng.txt', 'r', encoding='latin-1') as file:
    lines = file.readlines()

data = []

# Itera sobre cada linha na lista 'lines'
for line in lines:
    # Separa o texto e o rótulo usando '@' como delimitador
    text, label = line.split('@')
    # Remove a quebra de linha ('\n') do rótulo, se houver
    label = label.replace('\n', '')
    # Adiciona um dicionário contendo o texto e o rótulo à lista 'data'
    data.append({'text': text, 'label': label})

# Converte a lista 'data' em um DataFrame do Pandas
fpb_eng_df = pd.DataFrame(data)
```

6. Conversão de rótulos: Conversão de rótulos para rótulos numéricos.

```python
label_encode = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}

# Aplica a função lambda para cada valor na coluna 'label' do DataFrame 'fpb_eng_df'
# Convertendo o nome da classe em um valor numérico usando o dicionário 'label_encode'
fpb_eng_df['label'] = fpb_eng_df['label'].apply(lambda label: label_encode[label])
```

7. Tratamento do dataset: Remoção de registros duplicados.

```python
fpb_eng_df.drop_duplicates(subset='text', ignore_index=True, inplace=True)
```

8. União e formatação dos datasets: União e formatação das colunas e informações do dataset.

```python
# Concatena os DataFrames 'fpb_ptbr_df' e 'fpb_eng_df' em um único DataFrame 'fpb_df'
fpb_df = pd.concat([fpb_ptbr_df, fpb_eng_df], ignore_index=True)

# Adiciona uma coluna 'font' ao DataFrame 'fpb_df' e atribui o valor 'financial-phrase-bank' a todas as linhas
fpb_df['font'] = 'financial-phrase-bank'
```

### B3 - Bora Investir

O procedimento de raspagem das notícias da plataforma Bora Investir da B3 estão presentes no repositório [backend](https://github.com/Tuiaia/backend) da aplicação.

1. Carregamento do dataset: Carregamento do dataset após o scrap.

```python
raw_b3_df = pd.read_csv('b3/bora-investir.csv', sep='|')
```

2. Tratamento pré-rotulação: Formatação de texto, remoção de colunas não utilizadas e concatenação de título com corpo da notícia.

```python
import re

# Compila um padrão regex para identificar URLs
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def format_text(text):
    # Remove espaços e quebras de linha excessivos
    text = re.sub(r'\s+', ' ', text).strip()

    # Adiciona um espaço após a pontuação, se necessário
    text = re.sub(r'([.,;?!])([^\s])', r'\1 \2', text)

    # Remove strings específicas
    text = text.replace('CONTINUA DEPOIS DA PUBLICIDADE', '').replace('Relacionados', '')

    # Remove URLs do texto
    text = url_pattern.sub('', text)

    return text

# Cria uma cópia do DataFrame 'raw_b3_df'
b3_df = raw_b3_df.copy()

# Remove colunas desnecessárias
b3_df.drop(columns=['url', 'data', 'author'], inplace=True)

# Aplica a função 'format_text' para formatar a coluna 'text'
b3_df['text'] = b3_df['text'].apply(format_text)

# Remove quebras de linha da coluna 'text'
b3_df['text'] = b3_df['text'].apply(lambda text: text.replace('\n', ''))

# Combina as colunas 'title' e 'text'
b3_df['text'] = b3_df['title'] + '. ' + b3_df['text']

# Salva o DataFrame formatado em um arquivo CSV
b3_df.to_csv('b3/b3.csv', sep='|', index=False)
```

3. Processamento pós-rotulação: Conversão de rótulos e adição de informações a respeito do dataset.

```python
# Converte o texto na coluna 'label' para letras minúsculas
b3_df['label'] = b3_df['label'].apply(lambda label: label.lower())

# Define o dicionário 'label_dict' para mapear as strings das classes aos valores numéricos correspondentes
label_dict = {
    'positivo': 2,
    'neutro': 1,
    'negativo': 0
}

# Aplica a função lambda para converter as strings das classes em valores numéricos usando o dicionário 'label_dict'
b3_df['label'] = b3_df['label'].apply(lambda label: label_dict[label] if label in label_dict.keys() else None)

# Remove as linhas com valores ausentes (NaN) no DataFrame
b3_df.dropna(inplace=True)

# Converte a coluna 'label' para o tipo de dado inteiro
b3_df['label'] = b3_df['label'].astype(int)

# Adiciona uma coluna 'lang' com o valor 'ptbr' para todas as linhas
b3_df['lang'] = 'ptbr'

# Adiciona uma coluna 'font' com o valor 'b3' para todas as linhas
b3_df['font'] = 'b3'

# Renomeia as colunas do DataFrame 'b3_df' para 'text', 'label', 'lang' e 'font'
b3_df.columns = ['text', 'label', 'lang', 'font']
```

### Google News

Google News é um serviço de notícias agregado e personalizado, desenvolvido pela Google. Ele apresenta uma seleção contínua e atualizada de artigos de notícias de diversas fontes de informação, como jornais, revistas, sites de notícias e agências de notícias. Por esse motivo, o escolhemos como uma das fontes de coleta.

A coleta foi feita a partir do uso do Requests e BeautifulSoup, definindo uma rotina que seria executada a cada 30 minutos para a coleta das notícias em português e inglês. O script utilizado está localizado em [scripts/google-news.py](https://github.com/Tuiaia/artificial-intelligence/blob/main/jupyter-notebooks/insight-extraction.ipynb)

1. Definição das URLs das notícias em português e inglês.

```python
# URLs para notícias do Google News em português e inglês
pt_url = 'https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx6TVdZU0JYQjBMVUpTR2dKQ1VpZ0FQAQ?hl=pt-BR&gl=BR&ceid=BR%3Apt-419'
eng_url = 'https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen '
```

2. Definição da função responsável pela raspagem das notícias.

```python
def get_news(url, lang):
    """
    Raspa artigos de notícias do Google News no idioma especificado e salva os dados em um arquivo CSV.

    Args:
        url (str): URL da página de notícias.
        lang (str): Idioma da notícia ('ptbr' para português, 'eng' para inglês).
    """

    # Envia uma solicitação para a URL e verifica a resposta
    response = requests.get(url)
    response.raise_for_status()

    registry = []

    # Analisa o conteúdo HTML da página
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article', {'class': 'UwIKyb'})
    for article in articles:
        url = article.find('a', {'class': 'WwrzSb'})['href']
        font_icon = article.find('img', {'class': 'qEdqNd'})['src']
        font = article.find('span', {'class': 'vr1PYe'}).text
        text = article.find('h4', {'class': 'gPFEn'}).text
        datetime = article.find('time', {'class': 'hvbAAd'})['datetime']

        registry.append({
            'url': url,
            'font_icon': font_icon,
            'font': font,
            'text': text,
            'datetime': datetime,
            'lang': lang
        })

    registries_df = pd.DataFrame(registry)

    # Verifica se o arquivo CSV já existe e atualiza com os novos dados
    if os.path.isfile('google-news.csv'):
        df = pd.read_csv('google-news.csv', sep='|')
        print('Loaded google-news.csv | Length:', len(df))
        concat_df = pd.concat([df, registries_df], ignore_index=True)
        concat_df.drop_duplicates(subset=['text'], inplace=True, ignore_index=True)
        new_registries_count = np.absolute(len(concat_df) - len(df))
    else:
        concat_df = pd.DataFrame(registry)
        new_registries_count = len(concat_df)

    # Salva os dados atualizados no arquivo CSV
    if new_registries_count > 0:
        filename = 'google-news.csv'
        concat_df.to_csv(filename, index=False, sep='|')

    timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
    print(
        f'{timestamp} - Saved new {new_registries_count} {lang} registries to DataFrame | DF length: {len(concat_df)}')
```

3. Definir a rotina para a raspagem das notícias.

```python
# Agenda o raspador para executar a cada 30 minutos para cada idioma
schedule.every(30).minutes.do(get_news, pt_url, 'ptbr')
schedule.every(30).minutes.do(get_news, eng_url, 'eng')
```

4. Execução da coleta.

```python
# Executa o raspador em um loop infinito
while True:
    # Executa as tarefas agendadas
    schedule.run_pending()

    # Calcula o tempo restante até a próxima tarefa
    remaining = int(schedule.idle_seconds())
    minutes, seconds = divmod(remaining, 60)

    # Exibe uma barra de progresso com o tempo restante
    for i in tqdm(range(remaining), desc='Time remaining', unit='s', leave=False):
        time.sleep(1)
```

5. Tratamento do dataset.

```python

```

6. Formatação do dataset após a rotulação.

```python

```

## Treinamento do modelo

O treinamento do modelo de classificação de sentimentos foi realizado utilizando o modelo BERT pré-treinado "bert-base-multilingual-cased" no notebook [model-build.ipynb](https://github.com/Tuiaia/artificial-intelligence/blob/docs-readme/jupyter-notebooks/model-build.ipynb). Os passos a seguir descrevem o processo de treinamento do modelo, incluindo técnicas, parâmetros e configurações utilizadas:

1. Preparação dos dados: Os conjuntos de dados de treinamento, validação e teste foram carregados a partir de arquivos CSV e convertidos para objetos Dataset.

```python
# Lê os arquivos CSV dos conjuntos de dados de treinamento, validação e teste,
# usando o separador '|' para delimitar as colunas.
train_df = pd.read_csv('datasets/train_df.csv', sep='|')
val_df = pd.read_csv('datasets/val_df.csv', sep='|')
test_df = pd.read_csv('datasets/test_df.csv', sep='|')

# Converte os DataFrames do pandas para objetos Dataset da biblioteca HuggingFace,
# que são adequados para treinamento e avaliação de modelos de aprendizado de máquina.
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

```

2. Modelo e Tokenizador: Carregamos o modelo pré-treinado e seu tokenizador correspondente utilizando a biblioteca Transformers.

```python
# Carrega o modelo BERT pré-treinado para classificação de sequência e o tokenizador correspondente
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
```

3. Data Collator: Criamos um DataCollator com Padding para lidar com o preenchimento de sequências durante o treinamento.

```python
# Cria um coletor de dados com preenchimento (padding) para lidar com sequências de comprimentos variáveis
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

4. Tokenização dos conjuntos de dados: Tokenizamos os conjuntos de dados de treinamento, validação e teste.

```python
# Tokeniza os conjuntos de dados de treinamento, validação e teste
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
```

5. Definição dos argumentos de treinamento: Configuramos os argumentos de treinamento com os parâmetros e estratégias desejadas.

```python
# Define os argumentos de treinamento para o treinador (Trainer)
training_args = TrainingArguments(
    output_dir=f'./trainings/{repo_name}/',
    seed=seed,
    auto_find_batch_size=True,
    num_train_epochs=12,
    learning_rate=5e-6,
    weight_decay=0.01,
    eval_steps=100,
    logging_steps=100,
    save_steps=1000,
    save_strategy="steps",
    evaluation_strategy="steps",
    report_to="wandb",
)
```

6. Criação do objeto Trainer: Criamos um objeto Trainer para gerenciar o treinamento e a avaliação do modelo, fornecendo os componentes necessários, como o modelo, argumentos de treinamento, conjuntos de dados, DataCollator, tokenizador e métricas de avaliação.

```python
# Cria o objeto treinador (Trainer) com o modelo, argumentos de treinamento, conjuntos de dados e métricas
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

7. Treinamento e salvamento do modelo: Realizamos o treinamento do modelo e, em seguida, salvamos o modelo treinado.

```python
# Treina o modelo e salva-o no diretório especificado
trainer.train()
trainer.save_model(f'./trainings/{repo_name}/')
```

8. Finalização do registro no Wandb: Encerramos a execução do registro do experimento no Wandb.

```python
# Encerra o registro do experimento no Weights & Biases (WandB)
wandb.finish()
```

## Avaliação e teste do modelo

Nesta seção, descrevemos o processo de avaliação e teste do modelo treinado, incluindo as etapas e análises realizadas para medir o desempenho do modelo e entender como ele se comporta em diferentes fontes e idiomas.

1. Carregar o modelo treinado: Carregamos o modelo treinado e o tokenizer do diretório onde foram salvos.

```python
# Carrega o modelo BERT para classificação de sequência e o tokenizador a partir do diretório onde foi salvo
model = BertForSequenceClassification.from_pretrained(f'./trainings/{repo_name}/')
tokenizer = BertTokenizer.from_pretrained(f'./trainings/{repo_name}/')
```

2. Configurar o modelo para avaliação: Colocamos o modelo no modo de avaliação e medimos o tempo necessário para prever o sentimento de uma amostra de texto.

```python
# Coloca o modelo em modo de avaliação (evaluation mode)
model.eval()
# Mede o tempo necessário para prever o sentimento do primeiro exemplo do conjunto de teste
%timeit predict_sentiment(test_df.loc[0, 'text'])
```

3. Mover o modelo para o dispositivo: Transferimos o modelo para o dispositivo (GPU ou CPU) apropriado para acelerar as previsões.

```python
# Move o modelo para o dispositivo de processamento (GPU ou CPU)
model.to(device)
```

4. Prever o sentimento: Aplicamos a função predict_sentiment para cada texto no conjunto de testes e armazenamos as previsões em uma nova coluna chamada 'pred'.

```python
# Aplica a função predict_sentiment a cada exemplo do conjunto de teste e armazena os resultados na coluna 'pred'
test_df['pred'] = test_df['text'].apply(predict_sentiment)
```

5. Calcular métricas de desempenho: Calculamos a precisão, revocação, F1 e acurácia do modelo.

```python
# Calcula as métricas de avaliação (precisão, recall, f1 e acurácia)
precision, recall, f1, _ = precision_recall_fscore_support(test_df['label'], test_df['pred'], average='weighted')
acc = accuracy_score(test_df['label'], test_df['pred'])
```

6. Analisar a concordância e discordância das previsões: Calculamos a quantidade e a porcentagem de concordância, discordância parcial e discordância entre as previsões e os rótulos verdadeiros.

```python
# Calcula a diferença absoluta entre os rótulos verdadeiros e as previsões
test_df['diff'] = np.abs(test_df['label']-test_df['pred'])

# Conta a quantidade de exemplos em que o modelo concorda, discorda parcialmente e discorda totalmente
agree_count = test_df[test_df["diff"]==0].shape[0]
partial_disagree_count = test_df[test_df["diff"]==1].shape[0]
disagree_count = test_df[test_df["diff"]==2].shape[0]
```

7. Analisar as fontes e idiomas das discordâncias: Analisamos as proporções de discordâncias e concordâncias em relação às diferentes fontes e idiomas. Essas análises podem ajudar a identificar áreas de melhoria, ajustar o processo de treinamento ou ajustar os hiperparâmetros para melhorar ainda mais o desempenho do modelo.

```python
# Analisa a proporção de discordância total por fonte e idioma
disagree = test_df[test_df['diff']==2]
disagree.value_counts(subset=['font', 'lang'])/len(disagree)

# Analisa a proporção de discordância parcial por fonte e idioma
partial_disagree = test_df[test_df['diff']==1]
partial_disagree.value_counts(subset=['font', 'lang'])/len(partial_disagree)

# Analisa a proporção de concordância por fonte e idioma
agree = test_df[test_df['diff']==0]
agree.value_counts(subset=['font', 'lang'])/len(agree)
```

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
