import json
from tqdm import tqdm
import stanza
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
import torch
import argparse


parser = argparse.ArgumentParser(description='Evaluate 5 different vectorization methods')
parser.add_argument('--dir',
                    default='',
                    required=True,
                    help='path to data')
args = parser.parse_args()


tfidf_vectorizer = TfidfVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False)
count_vectorizer = CountVectorizer()

stanza.download('ru')
nlp = stanza.Pipeline('ru', processors='tokenize,pos,lemma', use_gpu=True)
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

bert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
bert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
gensim_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')


def get_value(x):
    if x['author_rating']['value'] == '':
        return 0
    else:
        return int(x['author_rating']['value'])


# read file and get 50.000 answers with highest value
def collect_texts_and_questions(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in list(f)[:60000]]
    corpus = []
    questions = []
    for question in data:
        if len(question['answers']) < 1:
            continue
        text = sorted(question['answers'],
                      key=lambda x: get_value(x), reverse=True)[0]['text']
        questions.append(question['question'])
        corpus.append(text)
    return corpus[:50000], questions[:50000]


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


def get_count_vectors(lemmatized_corpus, lemmatized_query):
    lemmatized_corpus = [' '.join(lemmas) for lemmas in lemmatized_corpus]
    corpus_vec = count_vectorizer.fit_transform(lemmatized_corpus)
    lemmatized_query = [' '.join(lemmas) for lemmas in lemmatized_query]
    query_vec = count_vectorizer.transform(lemmatized_query)
    return corpus_vec, query_vec


def get_tfidf_vectors(lemmatized_corpus, lemmatized_query):
    lemmatized_corpus = [' '.join(lemmas) for lemmas in lemmatized_corpus]
    corpus_vec = tfidf_vectorizer.fit_transform(lemmatized_corpus)
    lemmatized_query = [' '.join(lemmas) for lemmas in lemmatized_query]
    query_vec = tfidf_vectorizer.transform(lemmatized_query)
    return corpus_vec, query_vec


# index answers corpus
def get_bm25_corpus(lemmatized_corpus, k=2.0, b=0.75):
    lemmatized_corpus = [' '.join(lemmas) for lemmas in lemmatized_corpus]
    word_count = count_vectorizer.fit_transform(lemmatized_corpus).sum(axis=1)
    average_word_count = np.mean(word_count)
    tfidf_vectorizer.fit(lemmatized_corpus)
    tf_vectors = tf_vectorizer.fit_transform(lemmatized_corpus)
    upper_part = tf_vectors.multiply(tfidf_vectorizer.idf_) * (k + 1)
    upper_part = upper_part.tocsr()
    lower_part = tf_vectors + k * (1 - b + b * (word_count / average_word_count))
    upper_part[upper_part.nonzero()] = upper_part[upper_part.nonzero()] / lower_part[upper_part.nonzero()]
    return upper_part


# get count vector of questions corpus
def get_bm25_query(lemmatized_queries):
    lemmatized_queries = [' '.join(lemmas) for lemmas in lemmatized_queries]
    query_vector = count_vectorizer.transform(lemmatized_queries)
    return query_vector


def get_fasttext_vectors(lemmatized_corpus):
    matrix = []
    for text in lemmatized_corpus:
        vectors = [gensim_model.get_vector(word) for word in text]
        text_vector = np.mean(vectors, axis=0)
        matrix.append(text_vector)
    return matrix


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


def evaluate(corpus_vectors_norm, query_vectors_norm):
    length = query_vectors_norm.shape[0]
    similarities = np.dot(query_vectors_norm, corpus_vectors_norm.T)
    sorted_indices = np.argsort(similarities, axis=1)
    places_of_indices = np.where(sorted_indices == np.arange(length)[:, None])[1]
    in_top_5 = np.sum(places_of_indices >= length  - 5)
    score = in_top_5/length
    return score


def get_5_scores():
    filepath = args.dir
    corpus, questions = collect_texts_and_questions(filepath)
    print('Preprocessing corpus...')
    lemmatized_corpus = preprocess_corpus(corpus)
    print('Preprocessing queries...')
    lemmatized_queries = preprocess_corpus(questions)
    score_dict = {}

    # CountVectorizer
    corpus_vectors, query_vectors = get_count_vectors(lemmatized_corpus, lemmatized_queries)
    corpus_vectors_norm = normalize(corpus_vectors).todense()
    query_vectors_norm = normalize(query_vectors).todense()
    score = evaluate(corpus_vectors_norm, query_vectors_norm)
    score_dict['CountVectorizer'] = score

    # TfIdfVectorizer
    corpus_vectors, query_vectors = get_tfidf_vectors(lemmatized_corpus, lemmatized_queries)
    corpus_vectors_norm = normalize(corpus_vectors).todense()
    query_vectors_norm = normalize(query_vectors).todense()
    score = evaluate(corpus_vectors_norm, query_vectors_norm)
    score_dict['TfIdfVectorizer'] = score

    # BM25
    corpus_vectors = get_bm25_corpus(lemmatized_corpus).todense()
    query_vectors = get_bm25_query(lemmatized_queries).todense()
    score = evaluate(corpus_vectors, query_vectors)
    score_dict['BM-25'] = score

    # FastText
    corpus_matrix = get_fasttext_vectors(lemmatized_corpus)
    query_matrix = get_fasttext_vectors(lemmatized_queries)
    corpus_vectors_norm = normalize(corpus_matrix)
    query_vectors_norm = normalize(query_matrix)
    score = evaluate(corpus_vectors_norm, query_vectors_norm)
    score_dict['FastText'] = score

    # Bert
    print('Extracting Bert embeddings...')
    corpus_vectors = get_bert_embeddings(corpus)
    query_vectors = get_bert_embeddings(questions)
    corpus_vectors_norm = normalize(corpus_vectors)
    query_vectors_norm = normalize(query_vectors)
    score = evaluate(corpus_vectors_norm, query_vectors_norm)
    score_dict['Bert'] = score

    final_output = ''
    format_str = '{} score: {}\n'
    for key in score_dict:
        final_output = final_output + format_str.format(key, str(score_dict[key]))
    print(final_output)
    return score_dict


if __name__ == '__main__':
    get_5_scores()
