import json
import os
import pickle

from google.cloud import storage
from google.oauth2 import service_account

DOC_ID_TO_TITLE_FILE = 'id2title.pkl'
# Get the current working directory
current_directory = os.getcwd()
# Construct the full file path
json_cred_file_path = os.path.join(current_directory, '../final-project-415618-52467f8aee42.json')
BUCKET_NAME = "tfidf_bucket_318437159"

with open('queries_train.json', 'rt') as f:
    queries = json.load(f)

genetics = queries["genetics"]
print(genetics)

credentials = service_account.Credentials.from_service_account_file(
    json_cred_file_path)
client = storage.Client(credentials=credentials)
bucket = client.get_bucket(BUCKET_NAME)
id_to_title = pickle.loads(bucket.get_blob(DOC_ID_TO_TITLE_FILE).download_as_string())

for title_num in genetics:
    print(title_num, id_to_title[int(title_num)])
