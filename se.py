import io
import gzip
import csv
import pickle
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
                    "nonetheless", "despite", "whereas", "furthermore", "moreover", "notably", "hence"]

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

INDEX_FOLDER = 'inverted_indices'

DOC_ID_TO_TITLE_FILE = 'id2title.pkl'
PR_FILE = "pr_full_run/part-00000-d70a55fc-ebc4-4920-8b29-b57541c978c0-c000.csv.gz"
PV_FILE = "wid2pv.pkl"

PV_MEAN = 674.0488855666746
PV_STD = 55916.041828254696
PR_MEAN = 0.9999999999999324
PR_STD = 12.416830511713206


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
            client = storage.Client(project=PROJECT_ID)
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
        """
        Finds candidates by index based on the provided query words, index types, and index names.

        Args:
        - query_words (list): List of query words.
        - tfidf_or_bm25 (bool): Flag indicating whether to use TF-IDF or BM25 scoring.
        - index_name_body (str): Name of the index for the body.
        - index_name_title (str): Name of the index for the title.

        Returns:
        - body_ranked_docs (dict): Dictionary containing ranked documents for the body index.
        - title_ranked_docs (dict): Dictionary containing ranked documents for the title index.
        - title_binary_docs (dict): Dictionary containing binary documents for the title index.
        """
        # stem and bigram with doc len normalization:
        bigram_body_first_res = self.find_relevant_docs(query_words, index_name_body)
        bigram_title_first_res = self.find_relevant_docs(query_words, index_name_title)

        body_ranked_docs, title_ranked_docs, title_binary_docs = self.rank_candidates_by_index(bigram_body_first_res,
                                                                                               bigram_title_first_res,
                                                                                               index_name_body,
                                                                                               index_name_title,
                                                                                               tfidf_or_bm25)

        return body_ranked_docs, title_ranked_docs, title_binary_docs

    def rank_candidates_by_index(self, rel_docs_body, rel_doc_title, index_name_body, index_name_title, tfidf_or_bm25):
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
        if tfidf_or_bm25:
            if dl != 0:
                tf = tf / dl
            rank = tf * idf
        else:
            ave_dl = self.averageDl[index_name]
            rank = self.calculate_bm25(idf, tf, dl, ave_dl)
        return rank

    def get_doc_len(self, doc_id, index_name):
        index = self.inverted_indexes_dict[index_name]
        if doc_id not in index.dl:
            return 1
        return index.dl[doc_id]

    def rank_docs_by_fast_cosin(self, rel_docs, index_name, tfidf_or_bm25):
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
        idf_dic = self.term_idf_dict[index_name]
        for doc_id, tups in rel_docs.items():
            dl = self.get_doc_len(doc_id, index_name)
            for term, tf in tups:
                idf = idf_dic[term]
                rank = self.calc_rank_doc_len_norm(tf, idf, tfidf_or_bm25, index_name, dl)
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

    def find_relevant_docs(self, query_words, index_name):
        """
        Searches for documents containing the query words within a specific index.
        Args:
        - query_words (list): List of query words.
        - index_name (str): Name of the index to search within.
        Returns:
        - rel_docs (dict): Dictionary containing relevant documents and their corresponding term frequencies.
        """
        index = self.inverted_indexes_dict[index_name]
        unique_words = np.unique(query_words)
        rel_docs = dict()
        for term in unique_words:
            posting_list = index.read_a_posting_list(term, BUCKET_NAME)
            for doc_id, tf in posting_list:
                rel_docs.setdefault(doc_id, []).append((term, tf))
        return rel_docs

    @staticmethod
    def calculate_bm25(idf, tf, dl, avg_dl, k=1.5, b=0.75):
        return idf * (tf * (k + 1) / (tf + k * (1 - b + b * (dl / avg_dl))))

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
        if len(scores) > 0:
            min_score = min(scores)
            max_score = max(scores)
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


    def search(self, query, weights):
        """
        Executes a search query and returns ranked search results.

        Parameters:
        - query (str): The search query to be executed.

        Returns:
        - list: A list of tuples containing document IDs and their corresponding titles.
        """

        # body_rel_docs_tfidf_bigram, title_rel_docs_tfidf_bigram = \
        # (self.find_candidates_by_index(query_words, True, BIGRAM_BODY_FOLDER, BIGRAM_TITLE_FOLDER))
        # body_rel_docs_tfidf_stem, title_rel_docs_tfidf_bigram = \
        # (self.find_candidates_by_index(query_words, True, STEMMING_BODY_FOLDER, STEMMING_TITLE_FOLDER))

        # True = tfidf False = bm25

        query_words = self.fit_query(query, True, True)


        # body_rel_docs_bm25_bigram, title_rel_docs_bm25_bigram, title_binary_docs_bigram = (self.find_candidates_by_index
        #                                                                                    (query_words, False,
        #                                                                                     BIGRAM_BODY_FOLDER,
        #                                                                                     BIGRAM_TITLE_FOLDER))
        # print(f'number of docs bm25 bigram body: {len(body_rel_docs_bm25_bigram)}')
        # print(f'number of docs bm25 bigram title: {len(title_rel_docs_bm25_bigram)}')
        # print(f'number of docs binary bigram title: {len(title_binary_docs_bigram)}')

        query_words_no_bigram = self.fit_query(query, False, True)

        body_rel_docs_bm25_stem, title_rel_docs_bm25_stem, title_binary_docs_stem = (self.find_candidates_by_index
                                                                                     (query_words_no_bigram, False,
                                                                                      STEMMING_BODY_FOLDER,
                                                                                      STEMMING_TITLE_FOLDER))

        query_words_no_stem_no_bigram = self.fit_query(query, False, False)

        body_rel_docs_bm25_no_stem, title_rel_docs_bm25_no_stem, title_binary_docs_stem_no_stem = (
            self.find_candidates_by_index
            (query_words_no_stem_no_bigram, False,
             NO_STEM_BODY_FOLDER,
             NO_STEM_TITLE_FOLDER))


        all_scores = {
            # 'body_bm25_bi': body_rel_docs_bm25_bigram, 'title_bm25_bi': title_rel_docs_bm25_bigram,
                      'body_bm25_stem': body_rel_docs_bm25_stem, 'title_bm25_stem': title_rel_docs_bm25_stem,
                      'title_binary_stem': title_binary_docs_stem, 'body_bm25_no_stem': body_rel_docs_bm25_no_stem,
                      # 'title_bm25_no_stem': title_rel_docs_bm25_no_stem,
                      'title_binary_no_stem': title_binary_docs_stem_no_stem}


        filtered_scores, filtered_weights = self.get_non_empty_scores(all_scores, weights)

        all_docs = set()
        for d in filtered_scores:
            for doc_id, _ in d.items():
                all_docs.add(doc_id)

        all_docs_list = list(all_docs)

        pr_rel_docs = self.pr_docs_from_relevant_docs(all_docs_list)
        pv_rel_docs = self.pv_docs_from_relevant_docs(all_docs_list)

        pr_pv_map = {"pr": pr_rel_docs, "pv": pv_rel_docs}

        filtered_scores_pr_pv, filtered_weights_pr_pv = self.get_non_empty_scores(pr_pv_map, weights)

        pv_dict = {'pv': 0}
        if len(filtered_weights_pr_pv) == 2:
            for doc_id, score in filtered_scores_pr_pv[1].items():
                if doc_id in title_rel_docs_bm25_stem.keys() or doc_id in title_binary_docs_stem_no_stem.keys():
                    filtered_scores_pr_pv[1][doc_id] = score * 2

        final_scores = filtered_scores + filtered_scores_pr_pv
        finals_weights = filtered_weights + filtered_weights_pr_pv


        rankings = self.merge_ranking(final_scores, finals_weights)


        # add titles
        res = list(map(lambda x: (str(x[0]), self.id_to_title.get(x[0], 'Unknown')), rankings))
        return res
