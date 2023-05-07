# Jupyter Notebooks

## Sumário

- [Descrição](#descrição)
- [Construção do modelo](#construção-do-modelo)
- [Avaliação do modelo](#avaliação-e-teste-do-modelo)
- [Extração de insights](#extração-de-insights)

## Descrição

Neste diretório, você encontrará os Jupyter Notebooks que abordam o desenvolvimento do modelo de classificação e a implementação de técnicas para extrair informações relevantes a partir dos resultados das classificações. Esses notebooks fornecem uma visão detalhada do processo de construção, avaliação e interpretação do modelo, auxiliando na compreensão das decisões tomadas e na identificação de áreas de melhoria ou oportunidades de otimização.

## Construção do modelo

O treinamento do modelo de classificação de sentimentos foi realizado utilizando o modelo BERT pré-treinado "bert-base-multilingual-cased" no Jupyter Notebook [model-build.ipynb](https://github.com/Tuiaia/artificial-intelligence/blob/docs-readme/jupyter-notebooks/model-build.ipynb). Os passos a seguir descrevem o processo de treinamento do modelo, incluindo técnicas, parâmetros e configurações utilizadas:

1. Instalar as dependências.

    ```python
    !pip install datasets transformers huggingface_hub wandb
    ```

2. Carregar os conjuntos de dados de treinamento, validação e teste a partir de arquivos CSV e converter para objetos Dataset.

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

3. Carregar o modelo pré-treinado e seu tokenizador correspondente utilizando a biblioteca Transformers.

    ```python
    # Carrega o modelo BERT pré-treinado para classificação de sequência e o tokenizador correspondente
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    ```

4. Criar um DataCollator com Padding para lidar com o preenchimento de sequências durante o treinamento.

    ```python
    # Cria um coletor de dados com preenchimento (padding) para lidar com sequências de comprimentos variáveis
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ```

5. Tokenizar os conjuntos de dados de treinamento, validação e teste.

    ```python
    # Tokeniza os conjuntos de dados de treinamento, validação e teste
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    ```

6. Configurar os argumentos de treinamento com os parâmetros e estratégias desejadas.

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

7. Criar um objeto Trainer para gerenciar o treinamento e a avaliação do modelo, fornecendo os componentes necessários, como o modelo, argumentos de treinamento, conjuntos de dados, DataCollator, tokenizador e métricas de avaliação.

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

8. Realizar o treinamento do modelo e, em seguida, salvar o modelo treinado.

    ```python
    # Treina o modelo e salva-o no diretório especificado
    trainer.train()
    trainer.save_model(f'./trainings/{repo_name}/')
    ```

9. Encerrar a execução do registro do experimento no Wandb.

    ```python
    # Encerra o registro do experimento no Weights & Biases (WandB)
    wandb.finish()
    ```

## Avaliação e teste do modelo

Nesta seção está descrito o processo de avaliação de performance do modelo treinado, incluindo as etapas e análises realizadas para medir o desempenho do modelo e entender como ele se comporta em diferentes fontes e idiomas. Esses procedimentos também estão contidos no Jupyter Notebook [model-build.ipynb](https://github.com/Tuiaia/artificial-intelligence/blob/docs-readme/jupyter-notebooks/model-build.ipynb).

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

## Extração de insights

A extração de insights é uma etapa fundamental para entender e interpretar as classificações realizadas pelo modelo. O código apresentado nesta seção utiliza várias funções auxiliares, para analisar a importância de cada palavra na classificação de um texto a partir nosso modelo. Abaixo estão os procedimentos para realizar a extração.

1. Instalar dependências.

    ```python
    !pip install transformers

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    ```

2. Carregar o modelo e o respectivo tokenizador.

    ```python
    # Carregar o tokenizer e o modelo BERT
    model = BertForSequenceClassification.from_pretrained('./trainings/bert-base-multilingual-cased-06/', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('./trainings/bert-base-multilingual-cased-06/')
    ```

3. Definir função de remoção de stopwords.

    ```python
    # Função para remover stopwords de um texto
    def remove_stopwords(text: str, language: str = "auto") -> str:
        """
        Remove stopwords de um texto em português ou inglês.

        Args:
            text (str): O texto para remover as stopwords.
            language (str, optional): O idioma das stopwords. Por padrão, é utilizado "auto" para detectar o idioma automaticamente. O texto é tratado com o idioma português caso o detector detecte uma linguagem diferente de português ou inglês.

        Returns:
            str: O texto sem stopwords.

        Exemplo de uso:
            >>> texto = "Este é um exemplo de texto em português que será processado para remoção de stopwords."
            >>> texto_sem_stopwords = remove_stopwords(texto, language="portuguese")
            >>> print(texto_sem_stopwords)
            "exemplo texto português processado remoção stopwords."
        """
        if language == 'auto':
            language = detect(text)
            if language == 'en':
                language = 'english'
            else:
                language = 'ptbr'

        words = nltk.word_tokenize(text)
        stopwords_list = set(stopwords.words(language))
        filtered_words = [word for word in words if word.lower() not in stopwords_list]
        return " ".join(filtered_words)
    ```

4. Definir função de mescla de subtokens.

    ```python
    def merge_subtokens(tokenizer, word_attributions: list[tuple]) -> list[tuple]:
        """
        Agrupa as pontuações de tokens divididos do BERT em um único token.
        
        Args:
            tokenizer: Tokenizer BERT para usar.
            word_attributions (List[tuple]): As atribuições de palavras do tokenizador.

        Returns:
            List[tuple]: Atribuições de palavras com tokens divididos mesclados.
        """
        merged_attributions = []
        merged_token = ''
        merged_value = 0.0

        for token, value in word_attributions:
            detokenized_token = tokenizer.convert_tokens_to_string([token]).strip()
            if detokenized_token:
                if detokenized_token in punctuation or detokenized_token in {'[CLS]', '[SEP]', '[UNK]'}:
                    continue
                if token.startswith('##'):
                    merged_token += token[2:]
                    merged_value += value
                else:
                    if merged_token:
                        merged_attributions.append((merged_token, merged_value))
                        merged_token = ''
                        merged_value = 0.0
                    merged_token = detokenized_token
                    merged_value = value
            else:
                merged_token += token.replace('##', '')
                merged_value += value

        if merged_token:
            merged_attributions.append((merged_token, merged_value))

        return merged_attributions
    ```

5. Definir função para formatar uma representação mais intuitiva.

    ```python
    def format_attributions(word_attributions: list[tuple[str, float]]) -> list[tuple[str, str]]:
        """
        Formata as atribuições de palavras para uma representação mais intuitiva.

        Args:
            word_attributions: Uma lista de tuplas contendo as atribuições de palavras. Cada tupla contém uma palavra/token
                e um valor float representando sua importância.

        Returns:
            Uma lista de tuplas contendo as palavras formatadas e suas atribuições em formato de porcentagem, ordenadas por
            importância.
        """
        # Obter o valor total das atribuições
        total = sum(abs(score) for _, score in word_attributions)

        # Formatar cada atribuição como uma tupla (token, porcentagem)
        formatted_attributions = []
        for token, score in word_attributions:
            # Calcular a porcentagem e arredondar para duas casas decimais
            percentage = round((abs(score) / total) * 100, 2)
            formatted_attributions.append((token, percentage))

        # Ordenar as atribuições por porcentagem descendente
        formatted_attributions = sorted(formatted_attributions, key=lambda x: x[1], reverse=True)

        return formatted_attributions
    ```

6. Initializar o interpretador

    ```python
    # Inicializar o SequenceClassificationExplainer
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    ```

7. Definir função para classificar o texto de entrada e extrair os insights.

    ```python
    # Função para classificar o texto e extrair insights
    def classify_text(input_text):
        # Remover stopwords do texto
        input_text = remove_stopwords(input_text)

        # Tokenizar o texto de entrada
        tokenized_input = tokenizer(input_text, truncation=True, padding=True, max_length=100, return_tensors='pt')
        truncated_input_text = tokenizer.decode(tokenized_input['input_ids'][0])

        # Obter as atribuições de palavras usando o SequenceClassificationExplainer
        word_attributions = cls_explainer(truncated_input_text)

        # Mesclar os subtokens e formatar as atribuições de palavras
        word_attributions = merge_subtokens(tokenizer, word_attributions)
        word_attributions = sorted(word_attributions, key=lambda x: (-x[1], x[0]))
        word_attributions = format_attributions(word_attributions)

        # Retornar os resultados
        return {
            'predicition_class_name': cls_explainer.predicted_class_name,
            'prediction_index': cls_explainer.predicted_class_index,
            'prediction_probatility': cls_explainer.pred_probs,
            'influential_words': word_attributions
        }
    ```

Dessa forma, o texto de entrada é processado, classificado e o `SequenceClassificationExplainer` é utilizado para obter as atribuições de palavras, que são os valores que indicam a importância de cada palavra na classificação.

As atribuições de palavras são então processadas e formatadas usando as funções auxiliares, resultando em uma lista de palavras e suas respectivas importâncias em formato de porcentagem. Essa lista permite identificar quais palavras tiveram maior influência na decisão do modelo, ajudando a compreender e avaliar a classificação gerada.

Ao analisar os insights extraídos, é possível ter uma maior compreensão do funcionamento interno do modelo, assim como identificar possíveis melhorias ou ajustes no processo de treinamento e pré-processamento. Essa análise contribui para a otimização do desempenho do modelo e aprimoramento das classificações realizadas.
