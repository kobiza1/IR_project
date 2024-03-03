import re
import sys
import io
import os
import re
import gzip
import csv
import time
import json
import pickle
import numpy as np
import pandas as pd
import builtins
import math
from time import time
import itertools
from pathlib import Path
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.corpus import wordnet

nltk.download("wordnet")
from datetime import datetime
import operator
from itertools import islice, count, islice, count, groupby
from contextlib import closing
from io import StringIO
from pathlib import Path
from operator import itemgetter
from google.cloud import storage

from scipy.sparse import csr_matrix
from numpy import dot
from numpy.linalg import norm

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, OrderedDict, defaultdict

from inverted_index_gcp import InvertedIndex, MultiFileReader
from nltk.corpus import stopwords

TUPLE_SIZE = 6
english_stopwords = frozenset(stopwords.words("english"))
corpus_stopwords = [
    "category",
    "references",
    "also",
    "external",
    "links",
    "may",
    "first",
    "see",
    "history",
    "people",
    "one",
    "two",
    "part",
    "thumb",
    "including",
    "second",
    "following",
    "many",
    "however",
    "would",
    "became",
    "make",
    "accordingly",
    "hence",
    "namely",
    "therefore",
    "thus",
    "consequently",
    "meanwhile",
    "accordingly",
    "likewise",
    "similarly",
    "notwithstanding",
    "nonetheless",
    "despite",
    "whereas",
    "furthermore",
    "moreover",
    "nevertheless",
    "although",
    "notably",
    "notwithstanding",
    "nonetheless",
    "despite",
    "whereas",
    "furthermore",
    "moreover",
    "notably",
    "hence",
]
SIZE_OF_WIKI = 6348910
DOCUMENT_NORMALIZATION_SIZE = 20000

ALL_STOP_WORDS = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
BUCKET_NAME = "tfidf_bucket_318437159"
STEMMING_BODY_FOLDER = 'inverted_test_text_with_stemming'
STEMMING_TITLE_FOLDER = 'inverted_test_title_with_stemming'
BIGRAM_BODY_FOLDER = 'inverted_test_text_with_bigram'
BIGRAM_TITLE_FOLDER = 'inverted_test_title_with_bigram'
BIGRAM_BODY_MOST_COMMON_FOLDER = 'inverted_test_body_most_common_with_stemming'
BIGRAM_TITLE_MOST_COMMON_FOLDER = 'inverted_test_title_most_common_with_stemming'
INDEX_FOLDER = 'inverted_indices'


class search_engine:
    def __init__(self):
        self.inverted_indexes_dict = dict()
        self.term_idf_dict = dict()
        self.caching_docs = dict()
        self.pr = dict()
        self.averageDl = dict()
        self._load_indices()
        self.stemmer = PorterStemmer()

    def init_average_dl(self, index):
        total_sum = sum(index.dl.values())

        # Calculate the average
        return total_sum / len(index.dl)

    def load_index(self, bucket, path_to_folder, index_name):
        pickle_index = pickle.loads(bucket.get_blob(path_to_folder).download_as_string())

        self.inverted_indexes_dict[index_name] = pickle_index
        self.calculate_idf_from_index(pickle_index, index_name)
        self.averageDl[index_name] = self.init_average_dl(pickle_index)

    def calculate_idf_from_index(self, index, index_name):
        """
        calc index idf and keep it in term_idf_dic
        """
        try:
            cur_idf_dic = {}
            for term, freq in index.df:
                cur_idf_dic[term] = math.log10(SIZE_OF_WIKI / (freq + 1))

            self.term_idf_dict[index_name] = cur_idf_dic
        except Exception as e:
            print("error when calling calculate_index_idf")
            raise e

    def _load_indices(self):
        """
        load all indices from buckets and keep them in a dict objects.
        """
        try:
            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)

            self.load_index(bucket, f'{INDEX_FOLDER}/{STEMMING_BODY_FOLDER}.pkl', STEMMING_BODY_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_BODY_MOST_COMMON_FOLDER}.pkl',
                            BIGRAM_BODY_MOST_COMMON_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_BODY_MOST_COMMON_FOLDER}.pkl',
                            BIGRAM_BODY_MOST_COMMON_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{STEMMING_TITLE_FOLDER}.pkl', STEMMING_TITLE_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_TITLE_FOLDER}.pkl', BIGRAM_TITLE_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_TITLE_MOST_COMMON_FOLDER}.pkl',
                            BIGRAM_TITLE_MOST_COMMON_FOLDER)

            print("loading page views")
            # loading pageview
            self.pv = pickle.loads(bucket.get_blob("wid2pv.pkl").download_as_string())

            # page ranks to dict
            print("loading page rank")
            decompressed_file = gzip.decompress(bucket.get_blob("pagerank.csv.gz").download_as_string())
            csv_reader = csv.reader(io.StringIO(decompressed_file.decode("utf-8")))
            self.pr = {int(page_rank_tup[0]): float(page_rank_tup[1]) for page_rank_tup in csv_reader}

        except Exception as e:
            print("error when calling load_all_indices_from_bucket")
            raise e

    def calculate_bm25(self, idf, tf, dl, ave_dl, k=1.5, b=0.75):
        return idf * (tf * (k + 1) / (tf + k * (1 - b + b * (dl / ave_dl))))

    def calc_rank(self, tf, idf, tfidf_or_bm25, index_name, doc_id):

        if tfidf_or_bm25:
            rank = tf * idf
        else:
            ave_dl = self.averageDl[index_name]
            dl = self.get_doc_len(doc_id, index_name)
            rank = self.calculate_bm25(idf, tf, dl, ave_dl)
        return rank

    def get_doc_len(self, doc_id, index_name):
        index = self.inverted_indexes_dict[index_name]
        return index.dl[doc_id]

    def rank_docs_by_fast_cosin(self, rel_docs, index_name, tfidf_or_bm25):

        ranked_docs = Counter()
        idf_dic = self.term_idf_dict[index_name]
        for doc_id, tups in rel_docs:  # tups = [(term1, tf1), (term2, tf2)...]
            dl = self.get_doc_len(doc_id, index_name)
            for term, tf in tups:
                idf = idf_dic[term]
                rank = self.calc_rank(tf, idf, tfidf_or_bm25, index_name, doc_id)
                if doc_id in ranked_docs:
                    ranked_docs[doc_id] += rank / dl
                else:
                    ranked_docs[doc_id] = rank / dl

        return ranked_docs

    def sort_ranked_docs(self, ranked_docs, limit):
        sorted_ranked_docs = list(sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True))
        if limit > len(sorted_ranked_docs):
            return sorted_ranked_docs
        return sorted_ranked_docs[:limit]

    def search(self, query):
        query_words = self.fit_query(query, True)
        bigram_body_first_res = self.search_by_index(query_words, BIGRAM_BODY_FOLDER)
        bigram_title_first_res = self.search_by_index(query_words, BIGRAM_TITLE_FOLDER)
        # stem_title_first_res = self.search_by_index(query_words, STEMMING_BODY_FOLDER)
        # stem_title_first_res = self.search_by_index(query_words, STEMMING_TITLE_FOLDER)
        # bigram_mcw_title_first_res = self.search_by_index(query_words, BIGRAM_BODY_MOST_COMMON_FOLDER)
        # bigram_mcw_title_first_res = self.search_by_index(query_words, BIGRAM_TITLE_MOST_COMMON_FOLDER)
        ranked_docs_body = self.rank_docs_by_fast_cosin(bigram_body_first_res, BIGRAM_BODY_FOLDER, True)
        sorted_ranked_docs = self.sort_ranked_docs(ranked_docs_body, 100)
        ranked_docs_title = self.rank_docs_by_fast_cosin(bigram_title_first_res, BIGRAM_BODY_FOLDER, True)
        sorted_ranked_docs = self.sort_ranked_docs(ranked_docs_title, 100)

    def fit_query(self, query, bigram):
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        tokens = [tok for tok in tokens if tok not in ALL_STOP_WORDS]

        tokens = [self.stemmer.stem(token) for token in tokens]
        if bigram:
            tokens = list(ngrams(tokens, 2))
            tokens = [' '.join(bigram) for bigram in tokens]
        return tokens

    def search_by_index(self, query_words, index_name):
        # TODO cashing!!!!!
        index = self.inverted_indexes_dict[index_name]
        unique_words = np.unique(query_words)
        rel_docs = dict()
        for term in unique_words:
            posting_list = index.read_a_posting_list(index_name, term)
            for doc_id, tf in posting_list:
                if doc_id in rel_docs:
                    rel_docs[doc_id] += [(term, tf)]
                else:
                    rel_docs[doc_id] = [(term, tf)]
        return rel_docs
