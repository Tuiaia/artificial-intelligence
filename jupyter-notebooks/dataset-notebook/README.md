# Jupyter Notebooks - Datasets

## Sumário

- [Descrição](#descrição)
- [Infomoney](#infomoney)
- [Financial PhraseBank](#financial-phrasebank)
- [B3 - Bora Investir](#b3---bora-investir)
- [Google News](#google-news)
- [Construção do Dataset Final](#construção-do-dataset-final)

## Descrição

Este diretório contém os Jupyter Notebooks utilizados para manipulação, análise e pré-processamentos dos datasets que alimentam o treinamento, validação e teste do modelo de classificação. Cada fonte de dados possui tratamentos específicos registrados em seu respectivo Jupyter Notebook. Os passos para os processos realizados com cada conjunto estão detalhados abaixo e em código.

## Infomoney

InfoMoney é uma das principais fontes de notícias financeiras e de negócios no Brasil. O site fornece informações atualizadas sobre os mercados financeiros, economia, investimentos e outros tópicos relevantes. Para coletar e tratar os dados da InfoMoney, foram realizados os passos abaixo.

1. Configurar a busca no Twitter usando a biblioteca Twint para coletar tweets do perfil da InfoMoney.

    ```python
    def run_scrap(date=None):
        # Configuração do objeto twint
        config = twint.Config()
        # Coleta perfil completo
        config.Profile_full = True  
        # Habilita o armazenamento dos resultados em um DataFrame
        config.Pandas = True  
        # Define o usuário-alvo para a busca de tweets
        config.Username = "infomoney"  

        if date:
            # Define a data limite para coletar tweets, caso seja fornecida
            config.Until = date  
        
        # Executa a busca de tweets com a configuração definida
        twint.run.Search(config)

        # Retorna o DataFrame com os tweets coletados
        return twint.storage.panda.Tweets_df
    ```

2. Executar o processo de busca e coleta de tweets até obter um total de 20.000 tweets, tratando o DataFrame a cada iteração.

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

3. Tratamento da coluna de 'urls', remoção dos urls dos tweets e remoção de registros duplicados.

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

4. Coleta dos corpos das notícias através da url presente em cada tweet utilizando Requests e BeautifulSoup.

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

5. Coleta e armazenamento de todos os corpos de todos os urls presentes no DataFrame.

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

6. Formatação dos rótulos e remoção de valores nulos.

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

    # Salva o DataFrame em .csv
    infomoney_news_labelled.to_csv('datasets/infomoney/infomoney_news_labelled_cleaned.csv', index=False)
    ```

## Financial PhraseBank

O financial-phrase-bank-portuguese-translation é uma tradução para o português do FinancialPhraseBank, um dataset originalmente criado por Malo et al. em 2013, que tem como foco a análise de sentimentos no contexto financeiro. Ele contém mais de 5000 frases e sentenças extraídas de notícias financeiras em português, rotuladas de acordo com o sentimento subjacente (positivo, negativo ou neutro) e tem sido amplamente utilizado na pesquisa e desenvolvimento de modelos de aprendizado de máquina e PLN. Ambos foram utilizados para compor a base de dados.

Segue abaixo os procedimentos realizados com estas bases.

1. Download do dataset em português através do [Kaggle](https://www.kaggle.com/datasets/mateuspicanco/financial-phrase-bank-portuguese-translation)
2. Conversão dos rótulos para rótulos numéricos.

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

3. Remoção de registros duplicados:

    ```python
    # Remove as linhas duplicadas no DataFrame 'fpb_ptbr_df' com base na coluna 'text'
    fpb_ptbr_df = fpb_ptbr_df.drop_duplicates(subset='text')
    ```

4. Download do dataset em inglês através do [Research Gate](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10) (Malo, Pekka & Sinha, Ankur & Takala, Pyry & Korhonen, Pekka & Wallenius, Jyrki. (2013). FinancialPhraseBank-v1.0.)
5. Carregamento do dataset e divisão dos textos e rótulos.

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

6. Conversão de rótulos para rótulos numéricos.

    ```python
    # Define o dicionário 'label_dict' para mapear as strings das classes aos valores numéricos correspondentes
    label_encode = {
        'positive': 2,
        'neutral': 1,
        'negative': 0
    }

    # Aplica a função lambda para converter as strings das classes em valores numéricos usando o dicionário 'label_dict'
    fpb_eng_df['label'] = fpb_eng_df['label'].apply(lambda label: label_encode[label])
    ```

7. Remoção de registros duplicados.

    ```python
    fpb_eng_df.drop_duplicates(subset='text', ignore_index=True, inplace=True)
    ```

8. União e formatação das colunas e informações do dataset.

    ```python
    # Concatena os DataFrames 'fpb_ptbr_df' e 'fpb_eng_df' em um único DataFrame 'fpb_df'
    fpb_df = pd.concat([fpb_ptbr_df, fpb_eng_df], ignore_index=True)

    # Adiciona uma coluna 'font' ao DataFrame 'fpb_df' e atribui o valor 'financial-phrase-bank' a todas as linhas
    fpb_df['font'] = 'financial-phrase-bank'

    # Salva o DataFrame em .csv
    fpb_df.to_csv('datasets/financial-phrase-bank/financial-phrase-bank.csv', sep='|')
    ```

## B3 - Bora Investir

A plataforma Bora Investir da B3 é uma iniciativa voltada para a promoção da educação financeira e a difusão da cultura de investimentos no Brasil. Além de oferecer recursos como cursos, artigos, vídeos e ferramentas interativas, a plataforma também disponibiliza notícias relevantes do mercado financeiro aos seus usuários.

O procedimento de raspagem das notícias da plataforma Bora Investir da B3 estão presentes no repositório [backend](https://github.com/Tuiaia/backend) da aplicação. Segue abaixo as manipulações realizados com as notícias coletadas.

1. Carregamento do dataset após o scrap.

    ```python
    raw_b3_df = pd.read_csv('b3/bora-investir.csv', sep='|')
    ```

2. Formatação de texto, remoção de colunas não utilizadas e concatenação de título com corpo da notícia.

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

3. Conversão de rótulos e adição de informações a respeito do dataset.

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

    # Salva o DataFrame sem .csv
    b3_df.to_csv('b3/b3-labelled.csv', sep='|', index=False)
    ```

## Google News

Google News é um serviço de notícias agregado e personalizado, desenvolvido pela Google. Ele apresenta uma seleção contínua e atualizada de artigos de notícias de diversas fontes de informação, como jornais, revistas, sites de notícias e agências de notícias. Por esse motivo, o escolhemos como uma das fontes de coleta.

A coleta foi feita a partir do uso do Requests e BeautifulSoup, definindo uma rotina que seria executada periodicamente para a coleta das notícias em português e inglês. Segue abaixo os procedimentos realizados para coleta e tratamento das notícias do Google News.

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
    # Agenda o raspador para executar a cada 10 minutos para cada idioma
    schedule.every(10).minutes.do(get_news, pt_url, 'ptbr')
    schedule.every(10).minutes.do(get_news, eng_url, 'eng')
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

5. Formatação do dataset após a rotulação.

    ```python
    # Remove os registros com textos duplicados
    df.drop_duplicates(subset='text', inplace=True)
    # Remove registros com valores nulos
    df.dropna(inplace=True)

    # Codifica os rótulos para rótulos numéricos
    label_dict = {
        'positive': 2,
        'neutral': 1,
        'negative': 0
    }

    # Define o dicionário 'label_dict' para mapear as strings das classes aos valores numéricos correspondentes
    df['label'] = df['label'].apply(lambda label: label_dict[label.lower()])

    # Salva o DataFrame em .csv
    df.to_csv('google-news-labelled.csv', sep='|', index=False)
    ```

## Construção do dataset final

Após preparar cada conjunto anterior, esses foram unidos e devidamente tratados para desenvolver os conjuntos de treino, validação e teste que alimentam o modelo de classificação. Abaixo constam os passos realizados no Jupyter-Notebook `final-dataset-build.ipynb`

1. Carregamento, formatação inicial e concatenação dos datasets.

    ```python
    # Define os caminhos dos arquivos CSV que serão lidos
    infomoney_path = 'datasets/infomoney/infomoney_news_labelled_cleaned.csv'
    google_news_path = 'datasets/google-news/google-news-labelled.csv'
    financial_phrase_bank_path = 'datasets/financial-phrase-bank/financial-phrase-bank.csv'
    b3_path = 'datasets/b3/b3-labelled.csv'

    # Lê os arquivos CSV e os armazena em DataFrames do pandas
    infomoney_df = pd.read_csv(infomoney_path)
    google_news_df = pd.read_csv(google_news_path, sep='|').loc[:, ['text', 'label', 'lang', 'font']]
    fpb_df = pd.read_csv(financial_phrase_bank_path, sep='|', index_col=0)
    b3_df = pd.read_csv(b3_path, sep='|')

    # Concatena os DataFrames infomoney_df, google_news_df, fpb_df e b3_df em um único DataFrame
    df = pd.concat([infomoney_df, google_news_df, fpb_df, b3_df], axis=0)
    # Reinicia o índice do DataFrame concatenado, descartando o índice anterior
    df = df.reset_index(drop=True)
    ```

2. Remoção de stopwords e caracteres indesejados

    ```python
    # Aplica uma função lambda para remover caracteres indesejados das strings da coluna 'text'
    df['text'] = df['text'].apply(lambda text: text.replace('|', '').replace('\n', ''))

    # Define conjuntos de palavras irrelevantes (stopwords) em inglês e português
    stopwords_en = set(stopwords.words('english'))
    stopwords_pt = set(stopwords.words('portuguese'))

    # Função para remover stopwords do texto, baseado no idioma
    def remove_stopwords(text, lang):
        if lang == 'eng':
            stopwords_set = stopwords_en
        elif lang == 'ptbr':
            stopwords_set = stopwords_pt

        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stopwords_set]
        return ' '.join(filtered_words)

    # Função para remover stopwords do DataFrame
    def remove_stopwords_from_df(df):
        df['cleaned_text'] = df.apply(lambda row: remove_stopwords(row['text'], row['lang']), axis=1)
        return df

    # Aplica a função remove_stopwords_from_df ao DataFrame
    df = remove_stopwords_from_df(df)

    # Armazena o texto original na coluna 'raw_text' e o texto limpo na coluna 'text'
    df['raw_text'] = df['text']
    df['text'] = df['cleaned_text']
    # Remove a coluna 'cleaned_text' do DataFrame
    df.drop(columns=['cleaned_text'], inplace=True)
    ```

3. Divisão de treino, validação e teste

    ```python
    # Define a proporção de dados para treino, validação e teste
    train_temp_ratio = 0.7
    val_test_ratio = 0.5

    # Dividir o conjunto de dados em treino e teste, mantendo a proporção de classes e idiomas
    train_df, test_df = train_test_split(df, test_size=1-train_temp_ratio, stratify=df[['label', 'lang']], random_state=seed)
    # Dividir o conjunto de dados de teste em validação e teste, mantendo a proporção de classes e idiomas
    test_df, val_df = train_test_split(test_df, test_size=val_test_ratio, stratify=test_df[['label', 'lang']], random_state=seed)

    # Salvar os conjuntos de dados como .csv
    df.to_csv('datasets/df.csv', sep='|', index=False)
    train_df.to_csv('datasets/train_df.csv', sep='|', index=False)
    val_df.to_csv('datasets/val_df.csv', sep='|', index=False)
    test_df.to_csv('datasets/test_df.csv', sep='|', index=False)
    ```
