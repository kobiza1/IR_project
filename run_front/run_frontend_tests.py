import json
import pickle

import requests
from time import time
import random
import pickle
from google.cloud import storage

from google.cloud import storage

with open('queries_train.json', 'rt') as f:
    queries = json.load(f)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for index, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (index + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)


def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)


def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p + 1.0 / r), 3)


def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 5)
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0 / p5 + 1.0 / f1_30), 3)


url = 'http://35.225.147.92:8080'

# good_very_good_mucho_good = ['bioinformatics', 'Who painted "Starry Night"?', 'artificial intelligence',
#                              'nanotechnology', 'neuroscience', 'snowboard']
DOC_ID_TO_TITLE_FILE = 'id2title.pkl'
PROJECT_ID = 'final-project-415618'
BUCKET_NAME = "tfidf_bucket_318437159"

client = storage.Client(project=PROJECT_ID)
bucket = client.get_bucket(BUCKET_NAME)
id_to_title = pickle.loads(bucket.get_blob(DOC_ID_TO_TITLE_FILE).download_as_string())

with open('results.txt', 'a') as f:
    for i in range(1000):

        all_weights_keys = [
            'body_bm25_stem', 'title_bm25_stem',
            'title_binary_stem', 'body_bm25_no_stem',
            'title_binary_no_stem',
            'anchor_stem',
             'pr', 'pv']

        # random_weights = [random.randint(1, 50) * 0.02 for _ in range(8)]
        r = random.randint(1, 50) * 0.02
        random_weights = [0.88, 1.0, 0.74, 0.7000000000000001, 0.1] + [1] + [0.38, 0.9400000000000001]

        weights_map = {key: value for key, value in zip(all_weights_keys, random_weights)}
        print(weights_map)
        jsoned_rw = json.dumps(weights_map)
        total_duration = 0
        total_result = 0
        total_q = 0
        total_pre = 0
        pre = 0
        qs_res = []
        for q, true_wids in queries.items():
            duration, rq = 0, 0
            t_start = time()
            try:
                res = requests.get(url + '/search', {'query': q, 'random_weights': jsoned_rw}, timeout=150)
                duration = time() - t_start
                if res.status_code == 200:
                    pred_wids, _ = zip(*res.json())
                    rq = results_quality(true_wids, pred_wids)
                    pre = precision_at_k(true_wids, pred_wids, 10)
                    # our_titles = list(map(lambda x: id_to_title[int(x)], pred_wids))
                    # right_titles = list(map(lambda x: id_to_title[int(x)], true_wids))
                    # print(our_titles)
                    # print(right_titles)
                    print(q)
                    print(rq)

            except Exception as e:
                # print(e)
                pass

            total_pre += pre
            total_q += 1
            total_duration += duration
            total_result += rq

            # qs_res.append((q, duration, rq))

        avg_duration = total_duration / total_q
        avg_result = total_result / total_q
        avg_pre = total_pre / total_q

        curr_to_write = (random_weights, avg_duration, avg_result, avg_pre)
        print(f'finished running loop {i}')
        f.write(str(curr_to_write) + '\n')
        f.flush()

# url = 'http://34.42.220.52:8080'
#
#
# all_weights_keys = [
#                     # 'body_bm25_bi', 'title_bm25_bi',
#                     'body_bm25_stem', 'title_bm25_stem',
#                     'title_binary_stem', 'body_bm25_no_stem',
#                     # 'title_bm25_no_stem',
#                     'title_binary_no_stem',
#                     'pr', 'pv']
# random_weights = [0.12, 0.48, 0.72, 0.2, 0.98, 0.88, 0.44]
# weights_map = {key: value for key, value in zip(all_weights_keys, random_weights)}
# jsoned_rw = json.dumps(weights_map)
# total_duration = 0
# total_result = 0
# total_q = 0
# total_pre = 0
# qs_res = []
#
# for q, true_wids in queries.items():
#     pre = 0
#     duration, rq = 0, 0
#     t_start = time()
#     try:
#         res = requests.get(url + '/search', {'query': q, 'random_weights': jsoned_rw}, timeout=50)
#         duration = time() - t_start
#         if res.status_code == 200:
#             pred_wids, _ = zip(*res.json())
#             rq = results_quality(true_wids, pred_wids)
#             not_found = [x for x in pred_wids if x not in true_wids]
#             pre = precision_at_k(true_wids, pred_wids, 10)
#             # our_titles = list(map(lambda x: id_to_title[int(x)], pred_wids))
#             # right_titles = list(map(lambda x: id_to_title[int(x)], true_wids))
#             # print(our_titles)
#             # print(right_titles)
#             print(q)
#             print(rq)
#             print(pre)
#
#
#     except Exception as e:
#         # print(e)
#         pass
#
#     total_pre += pre
#     total_q += 1
#     total_duration += duration
#     total_result += rq
#     # qs_res.append((q, duration, rq))
#
# avg_duration = total_duration / total_q
# avg_result = total_result / total_q
# avg_pre = total_pre / total_q
# print(avg_duration, avg_result, avg_pre)
