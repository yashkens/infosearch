import os
import argparse
from tqdm import tqdm
import stanza
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

parser = argparse.ArgumentParser(description='Finds some info about Friends series for the 1st homework')
parser.add_argument('--dir',
                    default='',
                    required=True,
                    help='path to data')
args = parser.parse_args()

stanza.download('ru')
nlp = stanza.Pipeline('ru')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
vectorizer = CountVectorizer(analyzer='word')


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


# preprocess and index texts with CountVectorizer
def index(episode_texts):
    episode_lemmas = []
    print('Preprocessing...')
    for episode_text in tqdm(episode_texts):
        lemmas = preprocess(episode_text, stop_words)
        episode_lemmas.append(' '.join(lemmas))
    X = vectorizer.fit_transform(episode_lemmas)
    return X, vectorizer.get_feature_names()


# read all of the documents
def collect_texts(directory):
    episode_texts = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='utf-8') as f:
                text = f.read()
            episode_texts.append(text)
    return episode_texts


# find out which character is the most frequently mentioned
def find_most_popular_friend(df):
    friends_dict = {"Моника": ["моника", "мон"], "Рэйчел": ["рэйчел", "рейч"], "Чендлер": ["чендлер", "чэндлер", "чен"],
                    "Фиби": ["фиби", "фибс"], "Росс": ["росс"], "Джоуи": ["джоуи", "джои", "джо"]}
    popular_friend_score = 0
    popular_friend = ''
    for friend in friends_dict:
        for name in friends_dict[friend]:
            if name not in df:
                friends_dict[friend].remove(name)
        instances = df.sum()[friends_dict[friend]]
        friend_score = sum(list(instances))
        if friend_score > popular_friend_score:
            popular_friend_score = friend_score
            popular_friend = friend
    return popular_friend, popular_friend_score


def find_series_info(directory):
    episode_texts = collect_texts(directory)
    X, features = index(episode_texts)
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    # Найдем наиболее частотное слово
    max_ind = np.argmax(matrix_freq)
    print('Most frequent word is "{}" with {} instances'.format(features[max_ind], matrix_freq[max_ind]))

    # Найдем наименее частотное слово
    min_ind = np.argmin(matrix_freq)
    print('Least frequent word is "{}" with {} instances'.format(features[min_ind], matrix_freq[min_ind]))

    # Найдем слова, которые есть во всех документах
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    words_everywhere = df.columns[(df > 0).all()]
    print('Lemmas that are present in all of the documents: {}'.format(', '.join(words_everywhere)))

    # Выясним, кто из персонажей чаще всего упоминается
    popular_friend, popular_friend_score = find_most_popular_friend(df)
    print('The most popular friend is {} with {} instances'.format(popular_friend, popular_friend_score))
    return features[max_ind], features[min_ind], ', '.join(words_everywhere), popular_friend


# directory = 'C:\\Users\\user\\friends-data\\'
directory = args.dir
most_freq_word, least_freq_word, words_everywhere, popular_friend = find_series_info(directory)
