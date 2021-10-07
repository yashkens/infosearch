import json
import argparse
from tqdm import tqdm
import stanza
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


parser = argparse.ArgumentParser(description='Find most similar mail.ru answers to input for 3rd homework')
parser.add_argument('--dir',
                    default='',
                    required=True,
                    help='path to data')
parser.add_argument('--top_n',
                    default=-1,
                    required=False,
                    type=int,
                    help='how many results to show')
parser.add_argument('--shorten',
                    default=True,
                    required=False,
                    type=bool,
                    help='cut the answers by 200 symbols')
args = parser.parse_args()

stanza.download('ru')
nlp = stanza.Pipeline('ru', processors='tokenize,pos,lemma', use_gpu=True)
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

vectorizer_tfidf = TfidfVectorizer()
vectorizer_tf = TfidfVectorizer(use_idf=False)
count_vectorizer = CountVectorizer()


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


# preprocess texts: lemmatize, lower case, delete punctuation and stop words
def preprocess(text):
    doc = nlp(text)
    words = list(doc.iter_words())
    document_lemmas = []
    for word in words:
        if word.upos == 'PUNCT' or word.lemma.lower() in stop_words:
            continue
        document_lemmas.append(word.lemma.lower())
    return ' '.join(document_lemmas)


def index_corpus(lemmatized_corpus, k=2.0, b=0.75):
    word_count = count_vectorizer.fit_transform(lemmatized_corpus).sum(axis=1)
    average_word_count = np.mean(word_count)
    vectorizer_tfidf.fit(lemmatized_corpus)
    tf_vectors = vectorizer_tf.fit_transform(lemmatized_corpus)
    upper_part = tf_vectors.multiply(vectorizer_tfidf.idf_) * (k + 1)
    upper_part = upper_part.tocsr()
    lower_part = tf_vectors + k * (1 - b + b * (word_count / average_word_count))
    upper_part[upper_part.nonzero()] = upper_part[upper_part.nonzero()] / lower_part[upper_part.nonzero()]
    return upper_part


def index_query(text):
    lemmatized_text = preprocess(text)
    query_vector = count_vectorizer.transform([lemmatized_text])
    return query_vector


def count_bm25_similarity(corpus_matrix, query):
    query_vector = index_query(query)
    bm = corpus_matrix.multiply(query_vector).toarray().sum(axis=1)
    return bm


def find_most_similar(corpus_matrix, corpus, query, top_n=-1, shorten=True):
    bm = count_bm25_similarity(corpus_matrix, query)
    sorted_inds = np.argsort(bm)[::-1]
    nonzero_inds = sorted_inds[bm[sorted_inds] != 0]
    best_matches = np.array(corpus)[nonzero_inds]
    best_matches_shortened = best_matches
    if shorten:
        best_matches_shortened = \
            [text[:200] + '[...]' if len(text) > 100 else text for text in best_matches]
    if top_n == -1:
        top_n = len(best_matches_shortened)
    if not best_matches_shortened:
        print('Подходящих ответов нет.')
    else:
        print('Выдача на запрос:')
        print('\n'.join(best_matches_shortened[:top_n]))
    return best_matches_shortened[:top_n]


def run_search():
    filepath = args.dir
    top_n = args.top_n
    shorten = args.shorten
    corpus = collect_texts(filepath)
    lemmatized_corpus = []
    print('Preprocessing... (takes about 30 min)')
    for text in tqdm(corpus):
        lemmas = preprocess(text)
        lemmatized_corpus.append(lemmas)
    corpus_matrix = index_corpus(lemmatized_corpus)
    user_input = input('Enter your text. Type STOP if you want to stop searching.\n')
    while user_input != 'STOP':
        similar = find_most_similar(corpus_matrix, corpus, user_input, top_n=top_n, shorten=shorten)
        user_input = input('Enter your text:\n')


if __name__ == '__main__':
    run_search()
