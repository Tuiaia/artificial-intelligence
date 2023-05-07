# Treinamentos

## Sumário

- [Descrição](#descrição)
- [Como acessar os modelos no Hugging Face](#como-acessar-os-modelos-no-hugging-face)
- [Execuções de treinamento](#execuções-de-treinamento)

## Descrição

Este diretório é utilizado para armazenar os artefatos associados ao treinamento dos modelos de classificação durante as execuções. No entanto, os modelos treinados ao longo do projeto não estão sob controle de versão no GitHub devido ao espaço de armazenamento que ocupam. As versões dos modelos treinados são armazenadas no projeto na plataforma Hugging Face.

## Como acessar os modelos no Hugging Face

Para acessar nossos modelos treinados para análise de sentimentos, siga as instruções abaixo:

1. Visite a página do nosso perfil no Hugging Face: [https://huggingface.co/Tuiaia](https://huggingface.co/Tuiaia)
2. Navegue até a seção de "Modelos" e clique no modelo desejado.
3. Siga as instruções abaixo para carregar o modelo no seu projeto.

    - Instale as bibliotecas necessárias.

    ```shell
    pip install transformers huggingface_hub
    ```

    - Importe as classes necessárias.

    ```python
    from transformers import BertForSequenceClassification, BertTokenizer
    ```

    - Carregue o modelo e o respectivo tokenizer.

    ```python
    model = BertForSequenceClassification.from_pretrained('Tuiaia/bert-base-multilingual-cased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('Tuiaia/bert-base-multilingual-cased')
    ```

    - Execute a predição com o modelo.

    ```python
    # Verifica se a GPU está disponível e atribui à variável 'device', caso contrário, usa a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move o modelo para o dispositivo escolhido (GPU ou CPU)
    model.to(device) # GPU ou CPU

    # Tokeniza o texto de entrada e cria tensores PyTorch, limitando a sequência a um comprimento máximo de 512
    input_tokens = tokenizer.encode_plus(input_text, return_tensors="pt", max_length=512)
    # Move os tokens de entrada para o dispositivo escolhido (GPU ou CPU)
    input_tokens.to(device) # GPU ou CPU

    # Desativa o cálculo do gradiente para melhorar o desempenho durante a inferência
    with torch.no_grad():
        # Passa os tokens de entrada pelo modelo
        output = model(**input_tokens)
    
    # Atribui os logits (saída não normalizada) do modelo à variável 'logits'
    logits = output.logits
    # Aplica a função softmax nos logits para obter probabilidades
    probabilities = F.softmax(logits, dim=-1)

    # Obtém o índice da classe predita com a maior probabilidade
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    ```

## Execuções de treinamento

Durante o desenvolvimento do projeto, várias execuções de treinamento com distintas configurações foram conduzidas para aperfeiçoar o modelo. Essas execuções estão registradas na [página do projeto no Weights & Biases](https://wandb.ai/tiagosanti/Tuiaia?workspace=user-tiagosanti), uma plataforma valiosa para monitorar, comparar e examinar experimentos em aprendizado de máquina.
