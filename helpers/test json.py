import json
import os
import pickle

from google.cloud import storage
from google.oauth2 import service_account

good_very_good_mucho_good = ['bioinformatics', 'Who painted "Starry Night"?', 'artificial intelligence', 'nanotechnology', 'neuroscience', 'snowboard']

DOC_ID_TO_TITLE_FILE = 'id2title.pkl'
# Get the current working directory
# Construct the full file path

PROJECT_ID = 'final-project-415618'
BUCKET_NAME = "tfidf_bucket_318437159"

with open('../run_front/queries_train.json', 'rt') as f:
    queries = json.load(f)

genetics = queries["genetics"]
print(genetics)

client = storage.Client(project=PROJECT_ID)
bucket = client.get_bucket(BUCKET_NAME)
id_to_title = pickle.loads(bucket.get_blob(DOC_ID_TO_TITLE_FILE).download_as_string())

for title_num in genetics:
    print(title_num, id_to_title[int(title_num)])
