import io
import gzip
import csv
import pickle
import threading
import concurrent.futures
import numpy as np
import math
from nltk.stem.porter import *
from nltk.util import ngrams
from google.cloud import storage
from collections import Counter
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex

PROJECT_ID = 'final-project-415618'

TUPLE_SIZE = 6
english_stopwords = frozenset(stopwords.words("english"))
corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first", "see",
                    "one", "two", "part", "thumb", "including", "second", "following", "many", "however", "would",
                    "became", "make", "accordingly", "hence", "namely", "therefore", "thus", "consequently",
                    "meanwhile", "accordingly", "likewise", "similarly", "notwithstanding", "nonetheless", "despite",
                    "whereas", "furthermore", "moreover", "nevertheless", "although", "notably", "notwithstanding",
                    "nonetheless", "despite", "whereas", "furthermore", "moreover", "notably", "hence", 'considered',
                    'consider']

SIZE_OF_WIKI = 6348910

# params = {'max_docs_from_binary_title': 848, 'max_docs_from_binary_body': 859, 'bm25_body_weight': 8.798177098065569,
#           'bm25_title_weight': 0.49852365380857405, 'bm25_body_bi_weight': 0.3017628167759724,
#           'bm25_title_bi_weight': 7.194777117032163,
#           'body_cosine_score': 3.3006609064263843, 'title_cosine_score': 4.252156540716102,
#           'pr_weight': 2.0879280338073336, 'pv_weight': 5.483394450683551}

ALL_STOP_WORDS = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
BUCKET_NAME = "tfidf_bucket_318437159"

STEMMING_BODY_FOLDER = 'inverted_text_with_stemming'
STEMMING_TITLE_FOLDER = 'inverted_title_with_stemming'

BIGRAM_BODY_FOLDER = 'inverted_text_with_bigram'
BIGRAM_TITLE_FOLDER = 'inverted_title_with_bigram'

NO_STEM_BODY_FOLDER = 'inverted_title_no_stem'
NO_STEM_TITLE_FOLDER = 'inverted_text_no_stem'

NO_STEM_ANCHOR_FOLDER = 'inverted_anchor_no_stem'
STEM_ANCHOR_FOLDER = 'inverted_anchor_stem'

INDEX_FOLDER = 'inverted_indices'

DOC_ID_TO_TITLE_FILE = 'id2title.pkl'
PR_FILE = "pr_full_run/part-00000-d70a55fc-ebc4-4920-8b29-b57541c978c0-c000.csv.gz"
PV_FILE = "wid2pv.pkl"

lock = threading.Lock()


# PV_MEAN = 674.0488855666746
# PV_STD = 55916.041828254696
# PR_MEAN = 0.9999999999999324
# PR_STD = 12.416830511713206


class search_engine:

    def __init__(self):

        self.inverted_indexes_lock = threading.Lock()
        self.inverted_indexes_dict = dict()  # dict of inverted indexes
        self.term_idf_dict = dict()  # tfidf dict
        self.term_idf_lock = threading.Lock()
        self.pr = dict()  # page rank dict
        self.averageDl_lock = threading.Lock()
        self.averageDl = dict()  # document len dict
        self.stemmer = PorterStemmer()
        self.id_to_title = dict()
        self._load_indices()

    def _load_indices(self):
        """
        load all indices from buckets and keep them in main memory.
        """
        try:
            client = storage.Client(project=PROJECT_ID)
            bucket = client.get_bucket(BUCKET_NAME)

            print("loading indexes")

            # self.load_index(INDEX_FOLDER, BIGRAM_TITLE_FOLDER)
            # self.load_index(INDEX_FOLDER, BIGRAM_BODY_FOLDER)
            # self.load_index(INDEX_FOLDER, NO_STEM_ANCHOR_FOLDER)
            # self.load_index(INDEX_FOLDER, NO_STEM_BODY_FOLDER)

            self.load_index(INDEX_FOLDER, STEMMING_BODY_FOLDER)
            self.load_index(INDEX_FOLDER, STEMMING_TITLE_FOLDER)
            self.load_index(INDEX_FOLDER, NO_STEM_TITLE_FOLDER)
            self.load_index(INDEX_FOLDER, STEM_ANCHOR_FOLDER)

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
        """
        Loads an inverted index from a specified folder and updates the search engine's internal state.

        Args:
        - path_to_folder (str): Path to the folder containing the index data.
        - index_name (str): Name of the index.
        """
        pickle_index = InvertedIndex.read_index(path_to_folder, index_name, BUCKET_NAME)

        self.inverted_indexes_dict[index_name] = pickle_index
        self.calculate_idf_from_index(pickle_index, index_name)
        self.averageDl[index_name] = self.calc_average_dl(pickle_index)

    def find_candidates_by_index(self, query_words, index_name, binary=False):
        """
        Finds candidates by index based on the provided query words and index names.

        Args:
        - query_words (list): List of query words.
        - index_name(str): Name of the index.
        - binary (bool): If we use binary ranking or fast cosin

        Returns:
        - ranked_res (dict): Dictionary containing ranked documents for the index.

        """
        # stem and bigram with doc len normalization:

        first_res = self.find_relevant_docs(query_words, index_name)
        ranked_res = self.rank_candidates_by_index(first_res, index_name)
        if binary:
            binary_ranked_res = self.rank_candidates_by_index(first_res, index_name, binary=binary)
            return ranked_res, binary_ranked_res

        return ranked_res

    def rank_candidates_by_index(self, rel_docs, index_name, binary=False):
        """
        Ranks candidates by index based on the provided relevant documents, index names, and scoring method.

        Args:
        - rel_docs_body (dict): Dictionary containing relevant documents for the body index.
        - rel_docs_title (dict): Dictionary containing relevant documents for the title index.
        - index_name_body (str): Name of the index for the body.
        - index_name_title (str): Name of the index for the title.
        - tfidf_or_bm25 (bool): Flag indicating whether to use TF-IDF or BM25 scoring.

        Returns:
        - body_ranked_docs (dict): Dictionary containing ranked documents for the body index.
        - title_ranked_docs (dict): Dictionary containing ranked documents for the title index.
        - title_binary_docs (dict): Dictionary containing ranked documents for the title index by binary.
        """

        if binary:
            binary_docs = self.rank_by_binary(rel_docs)
            return binary_docs

        ranked_docs = self.rank_docs_by_fast_cosin(rel_docs, index_name)
        return ranked_docs

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
        """
        Retrieves PageRank scores for relevant documents.

        Args:
        - rel_docs (list): List of relevant document IDs.

        Returns:
        - pr_docs (Counter): Counter object containing PageRank scores for relevant documents.

        """
        pr_docs = Counter()
        for doc_id in rel_docs:
            if doc_id in self.pr:
                page_rank = self.pr[doc_id]
                pr_docs[doc_id] = page_rank
            else:
                pr_docs[doc_id] = 0
        return pr_docs

    def pv_docs_from_relevant_docs(self, rel_docs):
        """
        Retrieves PageView scores for relevant documents.

        Args:
        - rel_docs (list): List of relevant document IDs.

        Returns:
        - pv_docs (Counter): Counter object containing PageView scores for relevant documents.

        """
        pv_docs = Counter()
        for doc_id in rel_docs:
            if doc_id in self.pv:
                page_view = self.pv[doc_id]
                pv_docs[doc_id] = page_view
            else:
                pv_docs[doc_id] = 0
        return pv_docs

    def calc_rank_doc_len_norm(self, tf, idf, index_name, dl):
        """
        Calculates the document ranking score with length normalization.

        Args:
        - tf (float): Term Frequency.
        - idf (float): Inverse Document Frequency.
        - index_name (str): Name of the index.
        - dl (int): Document Length.

        Returns:
        - rank (float): Document ranking score with length normalization.

        Notes:
        - If tfidf_or_bm25 is True, The ranking score is calculated using the tfidf.
        -
        """
        # if tfidf_or_bm25:
        #     if dl != 0:
        #         tf = tf / dl
        #     rank = tf * idf

        with self.averageDl_lock:
            ave_dl = self.averageDl[index_name]
        rank = self.calculate_bm25(idf, tf, dl, ave_dl)
        return rank

    def get_doc_len(self, doc_id, index_name):
        """
        Retrieves the length of a document from the specified index.

        Args:
        - doc_id (str): Document ID.
        - index_name (str): Name of the index.

        Returns:
        - doc_len (int): Length of the document.

        """
        with self.inverted_indexes_lock:
            index = self.inverted_indexes_dict[index_name]
        if doc_id not in index.dl:
            return 1
        return index.dl[doc_id]

    def rank_docs_by_fast_cosin(self, rel_docs, index_name):
        """
        Ranks documents by cosine similarity with TF-IDF or BM25 scoring.

        Args:
        - rel_docs (dict): Dictionary containing relevant documents and their term frequencies.
        - index_name (str): Name of the index.
        - tfidf_or_bm25 (bool): Flag indicating whether to use TF-IDF or BM25 scoring.

        Returns:
        - ranked_docs (Counter): Counter containing the ranked documents.
        """

        ranked_docs = Counter()
        with self.term_idf_lock:
            idf_dic = self.term_idf_dict[index_name]
        for doc_id, tups in rel_docs.items():
            dl = self.get_doc_len(doc_id, index_name)
            for term, tf in tups:
                with self.term_idf_lock:
                    idf = idf_dic[term]
                rank = self.calc_rank_doc_len_norm(tf, idf, index_name, dl)
                ranked_docs.update({doc_id: rank})
        return ranked_docs

    def fit_query(self, query, bigram, stem):
        """
        Prepares and processes a query based on specified options.

        Args:
        - query (str): Input query string.
        - bigram (bool): If True, generates bigrams from the query.
        - stem (bool): If True, applies stemming to the query.

        Returns:
        - tokens (list): List of processed query tokens.

        """
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        tokens = [tok for tok in tokens if tok not in ALL_STOP_WORDS]

        if stem:
            tokens = [self.stemmer.stem(token) for token in tokens]

        if bigram:
            tokens = list(ngrams(tokens, 2))
            tokens = [' '.join(bigram) for bigram in tokens]

        return tokens

    @staticmethod
    def read_posting_list(term, index, posting_list_lists):
        """
        Reads a posting list for a given term from the index and appends it to a list.

        Args:
        - term (str): Term for which the posting list is retrieved.
        - index (object): Index object with a method 'read_a_posting_list'.
        - posting_list_lists (list): List to which the posting list information is appended.

        """
        posting_list = index.read_a_posting_list(term, BUCKET_NAME)
        with lock:
            posting_list_lists.append((term, posting_list))

    def find_relevant_docs(self, query_words, index_name):
        """
        Searches for documents containing the query words within a specific index.
        Args:
        - query_words (list): List of query words.
        - index_name (str): Name of the index to search within.
        - stem (bool): If we use stem or no
        Returns:
        - rel_docs (dict): Dictionary containing relevant documents and their corresponding term frequencies.
        """
        with self.inverted_indexes_lock:
            index = self.inverted_indexes_dict[index_name]
        unique_words = np.unique(query_words)
        rel_docs = dict()
        threads = []
        posting_list_lists = []

        for term in unique_words:
            thread = threading.Thread(target=self.read_posting_list, args=(term, index, posting_list_lists))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        for term, posting_list in posting_list_lists:
            for doc_id, tf in posting_list:
                rel_docs.setdefault(doc_id, []).append((term, tf))
        return rel_docs

    @staticmethod
    def calculate_bm25(idf, tf, dl, avg_dl, k=1.5, b=0.75):
        """
        Calculates the BM25 ranking score for a term in a document.

        Args:
        - idf (float): Inverse Document Frequency.
        - tf (float): Term Frequency.
        - dl (int): Document Length.
        - avg_dl (float): Average Document Length.
        - k (float): BM25 parameter controlling saturation, default is 1.5.
        - b (float): BM25 parameter controlling length normalization, default is 0.75.

        Returns:
        - bm25_score (float): BM25 ranking score.

        """
        return idf * (tf * (k + 1) / (tf + k * (1 - b + b * (dl / avg_dl))))

    @staticmethod
    def calc_average_dl(index):
        """
        Calculates the average document length for an index.

        Args:
        - index (object): Index object with a 'dl' attribute containing document lengths.

        Returns:
        - avg_dl (float): Average document length.

        """
        total_sum = sum(index.dl.values())
        # Calculate the average
        return total_sum / len(index.dl)

    def merge_ranking(self, lst_of_ranked_docs, weights, limit=100):
        """
        Merges multiple rankings using weighted scores and returns a sorted merged ranking.

        Args:
        - lst_of_ranked_docs (list): List of dictionaries containing document IDs and scores.
        - weights (list): List of weights corresponding to each ranking.
        - limit (int): Maximum number of documents to include in the merged ranking, default is 100.

        Returns:
        - merged_ranking (Counter): Counter object containing merged and weighted document scores.

        """
        merged_ranking = Counter()

        # Merge rankings
        for ranked_docs, weight in zip(lst_of_ranked_docs, weights):
            for doc_id, score in ranked_docs.items():
                merged_ranking.update({doc_id: weight * score})

        # Sort the merged ranking by score
        return self.sort_ranked_docs(merged_ranking, limit)

    @staticmethod
    def sort_ranked_docs(ranked_docs, limit=2000):
        """
        Sorts a dictionary of ranked documents by score in descending order.

        Args:
        - ranked_docs (dict): Dictionary containing document IDs and scores.
        - limit (int): Maximum number of documents to include in the sorted list, default is 2000.

        Returns:
        - sorted_ranked_docs (list): List of tuples containing sorted document IDs and scores.

        """
        sorted_ranked_docs = list(sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True))
        if limit > len(sorted_ranked_docs):
            return sorted_ranked_docs
        return sorted_ranked_docs[:limit]

    @staticmethod
    def rank_by_binary(rel_docs):
        """
        Ranks documents based on binary relevance.

        Parameters:
        - rel_docs (dict): A dictionary containing document IDs as keys and lists of tuples
                           (term, tf) representing relevant terms and their frequencies in the document.

        Returns:
        - Counter: A Counter object containing document IDs as keys and binary relevance scores (1 for relevant,
                   0 for non-relevant) as values.
        """
        ranked_docs = Counter()
        for doc_id, tups in rel_docs.items():
            for _ in tups:
                ranked_docs.update({doc_id: 1})
        return ranked_docs

    @staticmethod
    def normalize_scores(scores):
        """
        Normalizes a list of scores to the range [0, 1].

        Args:
        - scores (list): List of numerical scores.

        Returns:
        - normalized_scores (list): List of normalized scores.

        """
        if len(scores) > 0:
            min_score = min(scores)
            max_score = max(scores)

            if max_score == 0:
                return [0 for _ in scores]

            if min_score == max_score:
                normalized_scores = [score / max_score for score in scores]
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

            return normalized_scores

    # @staticmethod
    # def normalize_scores(scores):
    #     """
    #     Normalize a list of scores using robust normalization (interquartile range).
    #
    #     Args:
    #         scores (list of float): List of scores to be normalized.
    #
    #     Returns:
    #         list of float: Normalized scores where each score has been robustly normalized.
    #     """
    #     if len(scores) > 0:
    #         scores_np = np.array(scores)
    #
    #         # Calculate first quartile (Q1) and third quartile (Q3)
    #         q1 = np.percentile(scores_np, 25)
    #         q3 = np.percentile(scores_np, 75)
    #
    #         # Calculate interquartile range (IQR)
    #         iqr = q3 - q1
    #
    #         # Check if interquartile range is zero to avoid division by zero
    #         if iqr == 0:
    #             return [0 for _ in range(len(scores))]
    #
    #         # Robust normalization
    #         normalized_scores = (scores_np - q1) / iqr
    #
    #         return normalized_scores.tolist()

    def get_non_empty_scores(self, all_scores, all_weights):
        """
        Filters out non-empty scores and their corresponding weights from the input dictionaries.

        Parameters:
        - all_scores (dict): A dictionary containing scores for each key.
        - all_weights (dict): A dictionary containing weights for each key.

        Returns:
        - tuple: A tuple containing two lists:
            - filtered_scores (list): A list of dictionaries containing normalized scores for non-empty keys.
            - filtered_weights (list): A list containing weights corresponding to the non-empty keys.
        """
        filtered_scores = []
        filtered_weights = []

        for key, scores_dict in all_scores.items():
            if scores_dict and len(scores_dict) > 0:
                scores_values = list(scores_dict.values())
                # Normalize the values
                normalized_values = self.normalize_scores(scores_values)
                # Update the dictionary with the normalized values
                normalized_scores_dict = {k: v for k, v in zip(scores_dict.keys(), normalized_values)}
                filtered_scores.append(normalized_scores_dict)
                filtered_weights.append(all_weights[key])

        return filtered_scores, filtered_weights

    def anchor_candidates(self, query_words, index_name, binary=True):
        """
        Retrieves anchor candidates based on the provided query words and index.

        Args:
        - query_words (list): List of query words.
        - index_name (str): Name of the index.
        - binary (bool): If True, generates binary ranking.

        Returns:
        - ranked_res (dict): Dictionary containing ranked anchor candidates.

        """
        first_res = self.find_relevant_docs(query_words, index_name)

        if binary:
            ranked_res = self.rank_candidates_by_index(first_res, index_name, binary=binary)
            return ranked_res
        else:
            ranked_res = self.rank_candidates_by_index(first_res, index_name)
            return ranked_res

    def find_candidates_parallel(self, query):
        """
        Finds candidates in parallel based on the provided query.

        Args:
        - query (str): Input query string.

        Returns:
        - body_rel_docs_bm25_stem (dict): Body relevant documents with BM25 stemming.
        - title_rel_docs_bm25_stem (dict): Title relevant documents with BM25 stemming.
        - anchor_binary_docs_stem (dict): Anchor binary documents with BM25 stemming.
        - title_rel_docs_bm25_no_stem (dict): Title relevant documents with BM25 and no stemming.

        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # first: True = bigram,  False = no bigram,  second: True = stem, false = no stem
            query_words = self.fit_query(query, False, True)
            body_future = executor.submit(self.find_candidates_by_index, query_words, STEMMING_BODY_FOLDER)
            title_future = executor.submit(self.anchor_candidates, query_words, STEMMING_TITLE_FOLDER, False)
            anchor_future = executor.submit(self.anchor_candidates, query_words, STEM_ANCHOR_FOLDER)

            query_words_no_stem = self.fit_query(query, False, False)
            title_future_no_stem = executor.submit(self.find_candidates_by_index, query_words_no_stem,
                                                   NO_STEM_TITLE_FOLDER)

            # Retrieve results when ready
            body_rel_docs_bm25_stem = body_future.result()
            title_rel_docs_bm25_stem = title_future.result()
            anchor_binary_docs_stem = anchor_future.result()
            title_rel_docs_bm25_no_stem = title_future_no_stem.result()

            return body_rel_docs_bm25_stem, title_rel_docs_bm25_stem, anchor_binary_docs_stem, title_rel_docs_bm25_no_stem

    def search(self, query, weights):
        """
        Executes a search query and returns ranked search results.

        Parameters:
        - query (str): The search query to be executed.

        Returns:
        - list: A list of tuples containing document IDs and their corresponding titles.
        """

        body_rel_docs_bm25_stem, title_rel_docs_bm25_stem, \
            anchor_binary_docs_stem, title_rel_docs_bm25_no_stem = self.find_candidates_parallel(query)

        all_scores = {
            'body_bm25_stem': body_rel_docs_bm25_stem, 'title_bm25_stem': title_rel_docs_bm25_stem,
             'anchor_stem': anchor_binary_docs_stem,
            'title_bm25_no_stem': title_rel_docs_bm25_no_stem}

        filtered_scores, filtered_weights = self.get_non_empty_scores(all_scores, weights)

        rankings = self.merge_ranking(filtered_scores, filtered_weights, 700)

        first_1000_docs = []
        first_1000_docs_dict = dict()

        for doc_id, score in rankings:
            first_1000_docs.append(doc_id)
            first_1000_docs_dict[doc_id] = score

        already_weighted = [1]

        pr_rel_docs = self.pr_docs_from_relevant_docs(first_1000_docs)
        pv_rel_docs = self.pv_docs_from_relevant_docs(first_1000_docs)

        pr_pv_map = {"pr": pr_rel_docs, "pv": pv_rel_docs}

        filtered_scores_pr_pv, filtered_weights_pr_pv = self.get_non_empty_scores(pr_pv_map, weights)

        if len(filtered_weights_pr_pv) == 2:
            for doc_id, score in filtered_scores_pr_pv[1].items():
                if doc_id in title_rel_docs_bm25_stem.keys():
                    filtered_scores_pr_pv[1][doc_id] = score * 2

        final_scores = [first_1000_docs_dict] + filtered_scores_pr_pv
        finals_weights = already_weighted + filtered_weights_pr_pv

        rankings = self.merge_ranking(final_scores, finals_weights)

        # add titles
        res = list(map(lambda x: (str(x[0]), self.id_to_title.get(x[0], 'Unknown')), rankings))
        return res
