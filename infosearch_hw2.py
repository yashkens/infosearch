import os
import argparse
from tqdm import tqdm
import stanza
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


parser = argparse.ArgumentParser(description='Find most similar series to input for 2nd homework')
parser.add_argument('--dir',
                    default='',
                    required=True,
                    help='path to data')
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
vectorizer = TfidfVectorizer()


# preprocess texts: lemmatize, lower case, delete punctuation and stop words
def preprocess(text, stop_words):
    doc = nlp(text)
    words = list(doc.iter_words())
    document_lemmas = []
    for word in words:
        if word.upos == 'PUNCT' or word.lemma.lower() in stop_words:
            continue
        document_lemmas.append(word.lemma.lower())
    return document_lemmas


# preprocess and index texts with TfIdfVectorizer
def index(episode_texts):
    episode_lemmas = []
    print('Preprocessing the corpus... (takes about 25 min)')
    for episode_text in tqdm(episode_texts):
        lemmas = preprocess(episode_text, stop_words)
        episode_lemmas.append(' '.join(lemmas))
    X = vectorizer.fit_transform(episode_lemmas)
    return X


# read all of the documents and save their names
def collect_texts_with_names(directory):
    episode_texts = []
    episode_names = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                text = f.read()
            episode_texts.append(text)
            episode_names.append(name)
    return episode_texts, episode_names


# count cosine similarity of input with texts in corpus
def count_similarity(user_input, corpus_vectors):
    input_text = preprocess(user_input, stop_words)
    user_input_vectors = vectorizer.transform([' '.join(input_text)])
    similarities = cosine_similarity(corpus_vectors, user_input_vectors )
    similarities = similarities.ravel()
    return similarities


# preprocess all texts in corpus
def prepare_corpus(directory):
    episode_texts, episode_names = collect_texts_with_names(directory)
    corpus_vectors = index(episode_texts)
    return episode_names, corpus_vectors


def find_most_similar(corpus_vectors, episode_names, user_input, top_n):
    similarities = count_similarity(user_input, corpus_vectors)
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_names = np.array(episode_names)[sorted_indices]
    if top_n == -1:
        top_n = len(sorted_names)
    print('These are {} most similar series to your input:'.format(top_n))
    print('\n'.join(sorted_names[:top_n]))
    return sorted_names[:top_n]


def run_search():
    directory = args.dir
    top_n = args.top_n
    episode_names, corpus_vectors = prepare_corpus(directory)
    user_input = input('Enter your text. Type STOP if you want to stop searching.\n')
    while user_input != 'STOP':
        similar = find_most_similar(corpus_vectors, episode_names, user_input, top_n)
        user_input = user_input = input('Enter your text:\n')


run_search()
