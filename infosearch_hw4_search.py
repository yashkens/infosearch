import json
from tqdm import tqdm
import stanza
from nltk.corpus import stopwords
import nltk
import numpy as np
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import argparse


parser = argparse.ArgumentParser(description='Find most similar mail.ru answers to input using Bert of FastText')
parser.add_argument('--dir',
                    default='',
                    required=True,
                    help='path to data')
parser.add_argument('--vectorizer',
                    required=True,
                    help='type of vectorizer: bert or fasttext')
parser.add_argument('--top_n',
                    default=-1,
                    required=False,
                    type=int,
                    help='how many results to show')
args = parser.parse_args()


stanza.download('ru')
nlp = stanza.Pipeline('ru')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

gensim_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
bert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
bert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


# function to sort by value
def get_value(x):
    if x['author_rating']['value'] == '':
        return 0
    else:
        return int(x['author_rating']['value'])


# read file and get 50.000 answers with highest value
def collect_texts(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in list(f)[:60000]]
    corpus = []
    for question in data:
        if len(question['answers']) < 1:
            continue
        text = sorted(question['answers'],
                      key=lambda x: get_value(x), reverse=True)[0]['text']
        corpus.append(text)
    return corpus[:50000]


def preprocess_text(text):
    doc = nlp(text)
    words = list(doc.iter_words())
    document_lemmas = []
    for word in words:
        if word.upos == 'PUNCT' or word.lemma.lower() in stop_words:
            continue
        document_lemmas.append(word.lemma.lower())
    return document_lemmas


def preprocess_corpus(corpus):
    lemmatized_corpus = []
    for i, text in enumerate(tqdm(corpus)):
        lemmas = preprocess_text(text)
        if len(lemmas) == 0:
            lemmatized_corpus.append([''])
            continue
        lemmatized_corpus.append(lemmas)
    return lemmatized_corpus


def get_fasttext_vectors(lemmatized_corpus):
    matrix = []
    for text in lemmatized_corpus:
        vectors = [gensim_model.get_vector(word) for word in text]
        text_vector = np.mean(vectors, axis=0)
        matrix.append(text_vector)
    return np.array(matrix)


def get_bert_embeddings(corpus):
    batch_size = 100
    outputs = []
    for i in tqdm(range(0, len(corpus), batch_size)):
        texts = corpus[i:i + batch_size]
        encoded_input = bert_tokenizer(texts, padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = bert_model(**encoded_input)
        outputs.append(model_output)

    matrix = []
    for model_output in outputs:
        for text_emb in model_output[0]:
            matrix.append(text_emb[0].numpy())
    return np.array(matrix)


def find_most_similar(corpus_matrix, query_matrix, corpus, top_n=-1):
    similarities = cosine_similarity(corpus_matrix, query_matrix).ravel()
    sorted_inds = np.argsort(similarities)[::-1]
    nonzero_inds = sorted_inds[similarities[sorted_inds] > 0]
    best_matches = np.array(corpus)[nonzero_inds]
    if top_n == -1:
        top_n = len(best_matches)
    if len(best_matches) == 1:
        print('Подходящих ответов нет.')
    else:
        print('Выдача на запрос:')
        print('\n'.join(best_matches[:top_n]))
    return best_matches[:top_n]


def search(vectorizer, corpus, corpus_vec, query, top_n):
    similar = []
    if vectorizer == 'fasttext':
        lemmatized_query = preprocess_text(query)
        query_vec = get_fasttext_vectors([lemmatized_query])
        similar = find_most_similar(corpus_vec, query_vec, corpus, top_n)
    elif vectorizer == 'bert':
        query_vec = get_bert_embeddings([query])
        similar = find_most_similar(corpus_vec, query_vec, corpus, top_n)
    return similar


def run_search():
    filepath = args.dir
    vectorizer = args.vectorizer
    top_n = args.top_n
    corpus = collect_texts(filepath)
    corpus_vec = []
    if vectorizer == 'fasttext':
        lemmatized_corpus = preprocess_corpus(corpus)
        corpus_vec = get_fasttext_vectors(lemmatized_corpus)
    elif vectorizer == 'bert':
        corpus_vec = get_bert_embeddings(corpus)
    else:
        print('Пожалуйста, введите корректное название векторизатора (fasttext или bert).')
        exit()
    user_input = input('Enter your text. Type STOP if you want to stop searching.\n')
    while user_input != 'STOP':
        similar = search(vectorizer, corpus, corpus_vec, user_input, top_n)
        user_input = input('Enter your text:\n')


if __name__ == '__main__':
    run_search()
