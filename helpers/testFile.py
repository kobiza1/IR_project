import pickle
from collections import Counter

from google.cloud import storage

BUCKET_NAME = "tfidf_bucket_318437159"
INVERTED_INDEX_FOLDER = 'inverted_indices'
NO_STEM_BODY_PATH = f'{INVERTED_INDEX_FOLDER}/inverted_text_no_stem.pkl'
NO_STEM_TITLE_PATH = f'{INVERTED_INDEX_FOLDER}/inverted_title_no_stem.pkl'
PROJECT_ID = 'final-project-415618'
DOC_ID_TO_TITLE_FILE = 'id2title.pkl'

client = storage.Client(project=PROJECT_ID)
bucket = client.get_bucket(BUCKET_NAME)
no_stem_body = pickle.loads(bucket.get_blob(NO_STEM_BODY_PATH).download_as_string())
no_stem_title = pickle.loads(bucket.get_blob(NO_STEM_TITLE_PATH).download_as_string())
title_id_to_text = pickle.loads(bucket.get_blob(DOC_ID_TO_TITLE_FILE).download_as_string())

print(Counter(no_stem_body.dl).most_common(1))
print(Counter(no_stem_title.dl).most_common(1))
print(title_id_to_text[4214])
