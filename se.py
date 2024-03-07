import io
import gzip
import csv
import os
import pickle
import numpy as np
import math
from nltk.stem.porter import *
from nltk.util import ngrams
from google.cloud import storage
from collections import Counter
from nltk.corpus import stopwords
from google.oauth2 import service_account
from inverted_index_gcp import InvertedIndex


# Get the current working directory
current_directory = os.getcwd()
# Construct the full file path
json_cred_file_path = os.path.join(current_directory, 'final-project-415618-52467f8aee42.json')

TUPLE_SIZE = 6
english_stopwords = frozenset(stopwords.words("english"))
corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first", "see", "history", "people",
                    "one", "two", "part", "thumb", "including", "second", "following", "many", "however", "would",
                    "became", "make", "accordingly", "hence", "namely", "therefore", "thus", "consequently",
                    "meanwhile", "accordingly", "likewise", "similarly", "notwithstanding", "nonetheless", "despite",
                    "whereas", "furthermore", "moreover", "nevertheless", "although", "notably", "notwithstanding",
                    "nonetheless", "despite", "whereas", "furthermore", "moreover", "notably", "hence"]

SIZE_OF_WIKI = 6348910

params = {'max_docs_from_binary_title': 848, 'max_docs_from_binary_body': 859, 'bm25_body_weight': 8.798177098065569,
          'bm25_title_weight': 0.49852365380857405, 'bm25_body_bi_weight': 0.3017628167759724,
          'bm25_title_bi_weight': 7.194777117032163,
          'body_cosine_score': 3.3006609064263843, 'title_cosine_score': 4.252156540716102,
          'pr_weight': 2.0879280338073336, 'pv_weight': 5.483394450683551}

# WEIGHTS_PER_METHOD = {'pr': 0.4, 'tfidf-cosin': 0.3, 'bm25-cosin': 0.3}
# our weights = [body_bm25_bi, title_bm25_bi, body_bm25_stem, title_bm25_stem,
# title_binary_stem, body_bm25_no_stem, title_bm25_no_stem, title_binary_no_stem pr, pv]
our_weights = [0, 0, 0, 0, 0, 8, 3, 4, 3, 3]

ALL_STOP_WORDS = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
BUCKET_NAME = "tfidf_bucket_318437159"

STEMMING_BODY_FOLDER = 'inverted_text_with_stemming'
STEMMING_TITLE_FOLDER = 'inverted_title_with_stemming'

BIGRAM_BODY_FOLDER = 'inverted_text_with_bigram'
BIGRAM_TITLE_FOLDER = 'inverted_title_with_bigram'

NO_STEM_BODY_FOLDER = 'inverted_title_no_stem'
NO_STEM_TITLE_FOLDER = 'inverted_text_no_stem'

INDEX_FOLDER = 'inverted_indices'

DOC_ID_TO_TITLE_FILE = 'id2title.pkl'
PR_FILE = "pr_full_run/part-00000-d70a55fc-ebc4-4920-8b29-b57541c978c0-c000.csv.gz"
PV_FILE = "wid2pv.pkl"


class search_engine:

    def __init__(self):
        self.inverted_indexes_dict = dict()
        self.term_idf_dict = dict()
        self.pr = dict()
        self.averageDl = dict()
        self.stemmer = PorterStemmer()
        self.id_to_title = dict()
        self._load_indices()

    def _load_indices(self):
        """
        load all indices from buckets and keep them in main memory.
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(
                json_cred_file_path)
            client = storage.Client(credentials=credentials)
            bucket = client.get_bucket(BUCKET_NAME)

            print("loading indexes")
            self.load_index(INDEX_FOLDER, STEMMING_BODY_FOLDER)
            self.load_index(INDEX_FOLDER, STEMMING_TITLE_FOLDER)
            self.load_index(INDEX_FOLDER, BIGRAM_TITLE_FOLDER)
            self.load_index(INDEX_FOLDER, BIGRAM_BODY_FOLDER)
            self.load_index(INDEX_FOLDER, NO_STEM_BODY_FOLDER)
            self.load_index(INDEX_FOLDER, NO_STEM_TITLE_FOLDER)

            print("loading doc id 2 title index")
            self.id_to_title = pickle.loads(bucket.get_blob(DOC_ID_TO_TITLE_FILE).download_as_string())
            print("loading page views")
            # loading page_view
            self.pv = pickle.loads(bucket.get_blob(PV_FILE).download_as_string())

            # page ranks to dict
            print("loading page rank")
            decompressed_file = gzip.decompress(bucket.get_blob(PR_FILE).download_as_string())
            csv_reader = csv.reader(io.StringIO(decompressed_file.decode("utf-8")))
            self.pr = {int(page_rank_tup[0]): float(page_rank_tup[1]) for page_rank_tup in csv_reader}

        except Exception as e:
            print("error when calling load_all_indices_from_bucket")
            raise e

    def load_index(self, path_to_folder, index_name):

        pickle_index = InvertedIndex.read_index(path_to_folder, index_name, BUCKET_NAME)

        self.inverted_indexes_dict[index_name] = pickle_index
        self.calculate_idf_from_index(pickle_index, index_name)
        self.averageDl[index_name] = self.calc_average_dl(pickle_index)

    def find_candidates_by_index(self, query_words, tfidf_or_bm25, index_name_body, index_name_title):

        # stem and bigram with doc len normalization:
        bigram_body_first_res = self.search_by_index(query_words, index_name_body)
        bigram_title_first_res = self.search_by_index(query_words, index_name_title)

        body_ranked_docs, title_ranked_docs, title_binary_docs = self.rank_candidates_by_index(bigram_body_first_res,
                                                                                bigram_title_first_res, index_name_body,
                                                                                index_name_title, tfidf_or_bm25)



        # pr_body_tfidf_bigram, pr_title_tfidf_bigram = self.page_rank_title_and_body(body_ranked_docs, title_ranked_docs)

        return body_ranked_docs, title_ranked_docs, title_binary_docs

    # def page_rank_title_and_body(self, rel_docs_body, rel_docs_title):
    #
    #     pr_docs_body = self.pr_docs_from_relevant_docs(rel_docs_body)
    #     pr_docs_title = self.pr_docs_from_relevant_docs(rel_docs_title)
    #     return pr_docs_body, pr_docs_title

    def rank_candidates_by_index(self, rel_docs_body, rel_doc_title, index_name_body, index_name_title, tfidf_or_bm25):

        # bm-25 and cosin:
        body_ranked_docs = self.rank_docs_by_fast_cosin(rel_docs_body, index_name_body, tfidf_or_bm25)
        title_ranked_docs = self.rank_docs_by_fast_cosin(rel_doc_title, index_name_title, tfidf_or_bm25)
        title_binary_docs = self.rank_titles_by_binary(rel_doc_title)

        return body_ranked_docs, title_ranked_docs, title_binary_docs

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
        for doc_id in rel_docs:
            if doc_id in self.pr:
                page_rank = self.pr[doc_id]
                pr_docs[doc_id] = page_rank
            else:
                pr_docs[doc_id] = 0

        return pr_docs

    def pv_docs_from_relevant_docs(self, rel_docs):
        pv_docs = Counter()
        for doc_id in rel_docs:
            if doc_id in self.pv:
                page_rank = self.pv[doc_id]
                pv_docs[doc_id] = page_rank
            else:
                pv_docs[doc_id] = 0

        return pv_docs

    def calc_rank_doc_len_norm(self, tf, idf, tfidf_or_bm25, index_name, dl):
        if dl != 0:
            tf = tf / dl
        if tfidf_or_bm25:
            rank = tf * idf
        else:
            ave_dl = self.averageDl[index_name]
            rank = self.calculate_bm25(idf, tf, dl, ave_dl)
        return rank

    def get_doc_len(self, doc_id, index_name):
        index = self.inverted_indexes_dict[index_name]
        if doc_id not in index.dl:
            print(f"could not find {doc_id} length in {index_name}")
            return 1
        return index.dl[doc_id]

    def rank_docs_by_fast_cosin(self, rel_docs, index_name, tfidf_or_bm25):
        ranked_docs = Counter()
        idf_dic = self.term_idf_dict[index_name]
        for doc_id, tups in rel_docs.items():  # tups = [(term1, tf1), (term2, tf2)...]
            dl = self.get_doc_len(doc_id, index_name)
            # counter = counter + 1
            # if dl is None:
            #     counter_of_missed = counter_of_missed + 1
            for term, tf in tups:
                idf = idf_dic[term]
                rank = self.calc_rank_doc_len_norm(tf, idf, tfidf_or_bm25, index_name, dl)
                # if doc_id in ranked_docs:
                #     ranked_docs[doc_id] += rank #rank / dl
                # else:
                #      ranked_docs[doc_id] = rank # rank / dl
                ranked_docs.update({doc_id: rank})
        return ranked_docs

    def fit_query(self, query, bigram, stem):
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        tokens = [tok for tok in tokens if tok not in ALL_STOP_WORDS]
        if stem:
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
            posting_list = index.read_a_posting_list(term, BUCKET_NAME)
            for doc_id, tf in posting_list:
                rel_docs.setdefault(doc_id, []).append((term, tf))
        return rel_docs

    @staticmethod
    def calculate_bm25(idf, tf, dl, ave_dl, k=1.5, b=0.75):
        return idf * (tf * (k + 1) / (tf + k * (1 - b + b * (dl / ave_dl))))

    @staticmethod
    def calc_average_dl(index):
        total_sum = sum(index.dl.values())
        # Calculate the average
        return total_sum / len(index.dl)

    def merge_ranking(self, lst_of_ranked_docs, weights, limit=100):
        merged_ranking = Counter()
        # Merge rankings
        for ranked_docs, weight in zip(lst_of_ranked_docs, weights):
            for doc_id, score in ranked_docs.items():
                merged_ranking.update({doc_id: weight * score})

        # Sort the merged ranking by score
        return self.sort_ranked_docs(merged_ranking, limit)

    @staticmethod
    def sort_ranked_docs(ranked_docs, limit=2000):  # TODO: NEED LIMIT? IF YES HOW MUCH?
        sorted_ranked_docs = list(sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True))
        if limit > len(sorted_ranked_docs):
            return sorted_ranked_docs
        return sorted_ranked_docs[:limit]


    @staticmethod
    def rank_titles_by_binary(rel_docs):

        ranked_docs = Counter()
        for doc_id, tups in rel_docs.items():  # tups = [(term1, tf1), (term2, tf2)...]
            for _ in tups:
                ranked_docs.update({doc_id: 1})
        return ranked_docs

    @staticmethod
    def normalize_scores(scores):
        if len(scores) > 0:
            min_score = min(scores)
            max_score = max(scores)

            if max_score == 0:
                return [0 for _ in range(len(scores))]

            if min_score == max_score:
                normalized_scores = [score / max_score for score in scores]
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

            return normalized_scores

    def search(self, query):
        # body_rel_docs_tfidf_bigram, title_rel_docs_tfidf_bigram = \
        # (self.find_candidates_by_index(query_words, True, BIGRAM_BODY_FOLDER, BIGRAM_TITLE_FOLDER))
        # body_rel_docs_tfidf_stem, title_rel_docs_tfidf_bigram = \
        # (self.find_candidates_by_index(query_words, True, STEMMING_BODY_FOLDER, STEMMING_TITLE_FOLDER))

        # True = tfidf False = bm25
        query_words = self.fit_query(query, True, True)

        if len(query_words) <= 1:
            our_weights[1] = 9
            our_weights[0] = 6
            our_weights[2] = 2
        else:
            our_weights[1] = 1
            our_weights[0] = 7
            our_weights[2] = 3


        # body_rel_docs_bm25_bigram, title_rel_docs_bm25_bigram, title_binary_docs_bigram = (self.find_candidates_by_index
        #                                                          (query_words, False,
        #                                                           BIGRAM_BODY_FOLDER, BIGRAM_TITLE_FOLDER))
        #
        # query_words_no_bigram = self.fit_query(query, False, True)
        #
        # body_rel_docs_bm25_stem, title_rel_docs_bm25_stem, title_binary_docs_stem = (self.find_candidates_by_index
        #                                                      (query_words_no_bigram, False,
        #                                                       STEMMING_BODY_FOLDER, STEMMING_TITLE_FOLDER))

        query_words_no_stem = self.fit_query(query, False, False)

        body_rel_docs_bm25_no_stem, title_rel_docs_bm25_no_stem, title_binary_docs_stem_no_stem = (self.find_candidates_by_index
                                                                                     (query_words_no_stem, False,
                                                                                      NO_STEM_BODY_FOLDER,
                                                                                      NO_STEM_TITLE_FOLDER))
        weights = our_weights
        filtered_scores = []
        filtered_weights = []
        all_scores = [body_rel_docs_bm25_no_stem, title_rel_docs_bm25_no_stem, title_binary_docs_stem_no_stem]
        # all_scores = [body_rel_docs_bm25_bigram, title_rel_docs_bm25_bigram, body_rel_docs_bm25_stem,
        #               title_rel_docs_bm25_stem, title_binary_docs_stem, body_rel_docs_bm25_no_stem,
        #               title_rel_docs_bm25_no_stem, title_binary_docs_stem_no_stem]

        for scores_dict in all_scores:
            print(len(scores_dict))

        for index, scores_dict in enumerate(all_scores):
            if scores_dict and len(scores_dict) > 0:
                scores_values = list(scores_dict.values())
                # Normalize the values
                normalized_values = self.normalize_scores(scores_values)

                # Update the dictionary with the normalized values
                normalized_scores_dict = {key: value for key, value in zip(scores_dict.keys(), normalized_values)}

                filtered_scores.append(normalized_scores_dict)
                filtered_weights.append(weights[index])

        all_docs = Counter()
        for d in filtered_scores:
            for doc_id, score in d.items():
                all_docs.update({doc_id: score})

        sorted_docs_dict = self.sort_ranked_docs(all_docs, limit=100)

        pr_all_rel_docs = self.pr_docs_from_relevant_docs(sorted_docs_dict)
        pv_all_rel_docs = self.pv_docs_from_relevant_docs(sorted_docs_dict)
        pr_pv_list = [pr_all_rel_docs, pv_all_rel_docs]

        for index, scores_dict in enumerate(pr_pv_list):

            if scores_dict and len(scores_dict) > 0:
                scores_values = list(scores_dict.values())
                # Normalize the values
                normalized_values = self.normalize_scores(scores_values)

                # Update the dictionary with the normalized values
                normalized_scores_dict = {key: value for key, value in zip(scores_dict.keys(), normalized_values)}

                filtered_scores.append(normalized_scores_dict)
                filtered_weights.append(weights[index])

        rankings = self.merge_ranking(filtered_scores, filtered_weights)

        # add titles
        res = list(map(lambda x: (str(x[0]), self.id_to_title.get(x[0], 'Unknown')), rankings))
        return res
