######################## Todos os Imports necessários ########################
import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
# Imports necessários para DistilBert NER
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForTokenClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import evaluate
# Imports necessários para a interface Gradio
import gradio as gr

# Definir dispositivo (CPU ou GPU, se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')

# Carregar dados
file_path = "base_info_produtos.csv"
df = pd.read_csv(file_path, sep='\t')

# Configurar pré-processamento de texto
stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()

def preprocess_text(text):
    """Preprocessa o texto removendo stopwords e aplicando stemming."""
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Concatenar colunas para enriquecer as informações
df.fillna('n/a', inplace=True)
df['concatenated'] = (df['nome'] + ' ' + df['tipo'] + ' ' + df['marca'] + ' ' + df['categoria'] + ' ' +
                      df['cor'] + ' ' + df['modelo'])

# Aplicar preprocessamento de texto
df['processed_text'] = df['concatenated'].apply(preprocess_text)

######################## TF-IDF ########################

# Verificar se os arquivos do modelo TF-IDF já existem
tfidf_dir = "tfidf_model"
vectorizer_path = os.path.join(tfidf_dir, "tfidf_vectorizer.pkl")
matrix_path = os.path.join(tfidf_dir, "tfidf_matrix.pkl")

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
with open(matrix_path, 'rb') as f:
    tfidf_matrix = pickle.load(f)
print("Modelo TF-IDF carregado com sucesso.")

def calculate_similarity(product1, product2):
    """Calcula a similaridade entre dois produtos."""
    product1_processed = preprocess_text(product1)
    product2_processed = preprocess_text(product2)
    product1_tfidf = vectorizer.transform([product1_processed])
    product2_tfidf = vectorizer.transform([product2_processed])
    similarity = cosine_similarity(product1_tfidf, product2_tfidf)
    return min(similarity[0][0], 1.0)

def search_products(query, top_n=5):
    """Realiza busca de produtos com base na similaridade TF-IDF."""
    query = preprocess_text(query)
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results['probabilidade'] = [calculate_similarity(query, results.iloc[i]['concatenated']) for i in range(len(results))]
    return results[['nome', 'tipo', 'marca', 'categoria', 'cor', 'modelo', 'probabilidade']]

def extract_info_from_title(title):
    """Extrai informações de um título usando TF-IDF."""
    processed_title = preprocess_text(title)
    query_tfidf = vectorizer.transform([processed_title])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_index = similarities.argsort()[::-1][0]
    return df.iloc[top_index][['tipo', 'marca', 'categoria', 'cor', 'modelo']]

######################## NER DISTILBERT ########################

model_path = "ner_model"
tokenizer = "ner_model"

from collections import defaultdict
from transformers import pipeline

def get_most_cited_label_for_strings(string, model_path, tokenizer, device):
    strings = string.split(" ")
    classifier = pipeline("ner", model=model_path, tokenizer=tokenizer, device=device)
    results = {}
    
    # Initialize a list to keep track of entities and their positions
    entities = []
    
    for idx, string in enumerate(strings):
        classifier_output = classifier(string)
        label_scores = defaultdict(float)
        
        # Aggregate scores for each label
        for item in classifier_output:
            entity = item['entity']
            score = item['score']
            label_scores[entity] += score
        
        # Find the label with the highest cumulative score
        most_cited_label = max(label_scores, key=label_scores.get)
        
        # Store the entity and its position
        entities.append((idx, most_cited_label))
    
    # Sort entities by their original position in the input string
    entities.sort(key=lambda x: x[0])
    
    # Build the results dictionary aligned with the original input
    for position, label in entities:
        results[strings[position]] = label
    
    return results

######################## GRADIO INTERFACE ########################

# Habilitar modo de debug com a variável de ambiente GRADIO_DEBUG=1
os.environ["GRADIO_DEBUG"] = "1"

def search_interface(query):
    results = search_products(query)
    return results

def ner_interface(input_text):
    ner_predictions = get_most_cited_label_for_strings(input_text, model_path, tokenizer, device)
    return ner_predictions

search_demo = gr.Interface(fn=search_interface, inputs="text", outputs="dataframe", title="Busca de produtos")
ner_demo = gr.Interface(fn=ner_interface, inputs="text", outputs="json", title="NER Extraction")

demo = gr.TabbedInterface([search_demo, ner_demo], ["Busca de produtos", "Extração de features NER"])
demo.launch()