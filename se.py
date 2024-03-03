import io
import gzip
import csv
import pickle
import numpy as np
import math
from nltk.stem.porter import *
from nltk.util import ngrams
from google.cloud import storage
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from google.oauth2 import service_account


TUPLE_SIZE = 6
english_stopwords = frozenset(stopwords.words("english"))
corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first", "see", "history", "people",
                    "one", "two", "part", "thumb", "including", "second", "following", "many", "however", "would",
                    "became", "make", "accordingly", "hence", "namely", "therefore", "thus", "consequently",
                    "meanwhile", "accordingly", "likewise", "similarly", "notwithstanding", "nonetheless", "despite",
                    "whereas", "furthermore", "moreover", "nevertheless", "although", "notably", "notwithstanding",
                    "nonetheless", "despite", "whereas", "furthermore", "moreover", "notably", "hence"]
SIZE_OF_WIKI = 6348910
DOCUMENT_NORMALIZATION_SIZE = 20000

WEIGHTS = {'pr': 0.4, 'tfidf-cosin': 0.3, 'bm25-cosin': 0.3}

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


def calculate_bm25(idf, tf, dl, ave_dl, k=1.5, b=0.75):
    return idf * (tf * (k + 1) / (tf + k * (1 - b + b * (dl / ave_dl))))


def calc_average_dl(index):
    print(index.dl)
    total_sum = sum(index.dl.values())
    # Calculate the average
    return total_sum / len(index.dl)


def merge_ranking(lst_of_ranked_docs, weights, limit=100):
    merged_ranking = defaultdict(float)

    # Merge rankings
    for ranked_docs, weight in zip(lst_of_ranked_docs, weights):
        for doc_id, score in ranked_docs.items():
            merged_ranking[doc_id] += weight * score

    # Sort the merged ranking by score
    sorted_merged_ranking = sorted(merged_ranking.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_merged_ranking) > limit:
        sorted_merged_ranking = sorted_merged_ranking[:limit]
    return sorted_merged_ranking


class search_engine:

    def __init__(self):
        self.inverted_indexes_dict = dict()
        self.term_idf_dict = dict()
        self.caching_docs = dict()  # (key) (index_name, term) : (val) posting_list
        self.pr = dict()
        self.averageDl = dict()
        self._load_indices()
        self.stemmer = PorterStemmer()

    def _load_indices(self):
        """
        load all indices from buckets and keep them in main memory.
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(
                "/Users/tomsheinman/Desktop/final-project-415618-52467f8aee42.json")
            client = storage.Client(credentials=credentials)
            bucket = client.get_bucket(BUCKET_NAME)
            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_BODY_MOST_COMMON_FOLDER}.pkl',
                            BIGRAM_BODY_MOST_COMMON_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{STEMMING_BODY_FOLDER}.pkl', STEMMING_BODY_FOLDER)

            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_BODY_MOST_COMMON_FOLDER}.pkl',
                            BIGRAM_BODY_MOST_COMMON_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{STEMMING_TITLE_FOLDER}.pkl', STEMMING_TITLE_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_TITLE_FOLDER}.pkl', BIGRAM_TITLE_FOLDER)
            self.load_index(bucket, f'{INDEX_FOLDER}/{BIGRAM_TITLE_MOST_COMMON_FOLDER}.pkl',
                            BIGRAM_TITLE_MOST_COMMON_FOLDER)

            # print("loading page views")
            # # loading page_view
            # self.pv = pickle.loads(bucket.get_blob("wid2pv.pkl").download_as_string())

            # page ranks to dict
            print("loading page rank")
            decompressed_file = gzip.decompress(bucket.get_blob("pagerank.csv.gz").download_as_string())
            csv_reader = csv.reader(io.StringIO(decompressed_file.decode("utf-8")))
            self.pr = {int(page_rank_tup[0]): float(page_rank_tup[1]) for page_rank_tup in csv_reader}

        except Exception as e:
            print("error when calling load_all_indices_from_bucket")
            raise e

    def load_index(self, bucket, path_to_folder, index_name):
        pickle_index = pickle.loads(bucket.get_blob(path_to_folder).download_as_string())
        print(pickle_index.dl)
        self.inverted_indexes_dict[index_name] = pickle_index
        self.calculate_idf_from_index(pickle_index, index_name)
        self.averageDl[index_name] = calc_average_dl(pickle_index)

    def search(self, query):

        query_words = self.fit_query(query, True)

        # # only stemming:
        # stem_body_first_res = self.search_by_index(query_words, STEMMING_BODY_FOLDER)
        # stem_title_first_res = self.search_by_index(query_words, STEMMING_TITLE_FOLDER)
        #
        # # tf - idf and cosin:
        # body_ranked_docs = self.rank_docs_by_fast_cosin(stem_body_first_res, STEMMING_BODY_FOLDER, True)
        # sorted_ranked_docs = self.sort_ranked_docs(body_ranked_docs)
        # title_ranked_docs = self.rank_docs_by_fast_cosin(stem_title_first_res, STEMMING_TITLE_FOLDER, True)
        # sorted_ranked_docs = self.sort_ranked_docs(title_ranked_docs)
        #
        # #bm-25 and cosin:
        # body_ranked_docs = self.rank_docs_by_fast_cosin(stem_body_first_res, STEMMING_BODY_FOLDER, False)
        # sorted_ranked_docs = self.sort_ranked_docs(body_ranked_docs)
        # title_ranked_docs = self.rank_docs_by_fast_cosin(stem_title_first_res, STEMMING_TITLE_FOLDER, False)
        # sorted_ranked_docs = self.sort_ranked_docs(title_ranked_docs)
        #
        # # page rank:
        # body_pr_docs = self.pr_docs_from_relevant_docs(stem_body_first_res)
        # title_pr_docs = self.pr_docs_from_relevant_docs(stem_title_first_res)

        # stem and bigram:
        bigram_body_first_res = self.search_by_index(query_words, BIGRAM_BODY_FOLDER)
        bigram_title_first_res = self.search_by_index(query_words, BIGRAM_TITLE_FOLDER)

        # tf - idf and cosin doc len normalization:
        body_ranked_docs = self.rank_docs_by_fast_cosin(bigram_body_first_res, BIGRAM_BODY_FOLDER, True)
        title_ranked_docs = self.rank_docs_by_fast_cosin(bigram_title_first_res, BIGRAM_TITLE_FOLDER, True)

        # bm-25 and cosin:
        body_ranked_docs = self.rank_docs_by_fast_cosin(bigram_body_first_res, BIGRAM_BODY_FOLDER, False)
        title_ranked_docs = self.rank_docs_by_fast_cosin(bigram_title_first_res, BIGRAM_TITLE_FOLDER, False)

        # page rank:
        body_pr_docs = self.pr_docs_from_relevant_docs(bigram_body_first_res)
        title_pr_docs = self.pr_docs_from_relevant_docs(bigram_title_first_res)

        # # stem and bigram most common word normalization: # TODO: KEEP UP
        # bigram_body_first_res = self.search_by_index(query_words, BIGRAM_BODY_MOST_COMMON_FOLDER)
        # bigram_title_first_res = self.search_by_index(query_words, BIGRAM_TITLE_MOST_COMMON_FOLDER)
        #
        # # tf - idf and cosin:
        # body_ranked_docs = self.rank_docs_by_fast_cosin(bigram_body_first_res, BIGRAM_BODY_MOST_COMMON_FOLDER, True)
        # sorted_ranked_docs = self.sort_ranked_docs(body_ranked_docs)
        # title_ranked_docs = self.rank_docs_by_fast_cosin(bigram_title_first_res,
        # BIGRAM_TITLE_MOST_COMMON_FOLDER, True)
        # sorted_ranked_docs = self.sort_ranked_docs(title_ranked_docs)
        #
        # # bm-25 and cosin:
        # body_ranked_docs = self.rank_docs_by_fast_cosin(bigram_body_first_res, BIGRAM_BODY_MOST_COMMON_FOLDER, False)
        # sorted_ranked_docs = self.sort_ranked_docs(body_ranked_docs)
        # title_ranked_docs = self.rank_docs_by_fast_cosin(bigram_title_first_res,
        # BIGRAM_TITLE_MOST_COMMON_FOLDER, False)
        # sorted_ranked_docs = self.sort_ranked_docs(title_ranked_docs)
        #
        # # page rank:
        # body_pr_docs = self.pr_docs_from_relevant_docs(bigram_body_first_res)
        # title_pr_docs = self.pr_docs_from_relevant_docs(bigram_title_first_res)

        # bigram_mcw_title_first_res = self.search_by_index(query_words, BIGRAM_BODY_MOST_COMMON_FOLDER)
        # bigram_mcw_title_first_res = self.search_by_index(query_words, BIGRAM_TITLE_MOST_COMMON_FOLDER)
        body_ranked_docs = self.rank_docs_by_fast_cosin(bigram_body_first_res, BIGRAM_BODY_FOLDER, True)
        title_ranked_docs = self.rank_docs_by_fast_cosin(bigram_title_first_res, BIGRAM_BODY_FOLDER, True)
        body_pr_docs = self.pr_docs_from_relevant_docs(bigram_body_first_res)
        title_pr_docs = self.pr_docs_from_relevant_docs(bigram_title_first_res)
        return merge_ranking([body_ranked_docs, title_ranked_docs], [0.3, 0.7])


    def calculate_idf_from_index(self, index, index_name):
        """
        calc index idf and keep it in term_idf_dic
        """
        try:
            cur_idf_dic = {}
            for term, freq in index.df.items():
                cur_idf_dic[term] = math.log10(SIZE_OF_WIKI / (freq + 1))

            self.term_idf_dict[index_name] = cur_idf_dic
        except Exception as e:
            print("error when calling calculate_index_idf")
            raise e


    def pr_docs_from_relevant_docs(self, rel_docs):
        pr_docs = Counter()
        for doc_id in rel_docs.keys():
            if doc_id in self.pr:
                page_rank = self.pr[doc_id]
                pr_docs[doc_id] = page_rank
            else:
                pr_docs[doc_id] = 0

        return pr_docs

    def calc_rank(self, tf, idf, tfidf_or_bm25, index_name, doc_id):

        if tfidf_or_bm25:
            rank = tf * idf
        else:
            ave_dl = self.averageDl[index_name]
            dl = self.get_doc_len(doc_id, index_name)
            rank = calculate_bm25(idf, tf, dl, ave_dl)
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

    def fit_query(self, query, bigram):
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        tokens = [tok for tok in tokens if tok not in ALL_STOP_WORDS]

        tokens = [self.stemmer.stem(token) for token in tokens]
        if bigram:
            tokens = list(ngrams(tokens, 2))
            tokens = [' '.join(bigram) for bigram in tokens]
        return tokens

    def search_by_index(self, query_words, index_name):
        index = self.inverted_indexes_dict[index_name]
        unique_words = np.unique(query_words)
        rel_docs = dict()
        for term in unique_words:
            if (index_name, term) in self.caching_docs and (
                    len(self.caching_docs) <= 2000):  # TODO: NEED LIMIT? IF YES HOW MUCH?
                posting_list = self.caching_docs[(index_name, term)]
            else:
                posting_list = index.read_a_posting_list(index_name, term)
                self.caching_docs[(index_name, term)] = posting_list
            for doc_id, tf in posting_list:
                if doc_id in rel_docs:
                    rel_docs[doc_id] += [(term, tf)]
                else:
                    rel_docs[doc_id] = [(term, tf)]
        return rel_docs

    def rank_docs_by_euclidean_dis(self, rel_docs, index_name, tfidf_or_bm25):

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
