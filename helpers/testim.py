import io
import gzip
import csv
import pickle
import numpy as np
from google.cloud import storage


PR_FILE = "pr_full_run/part-00000-d70a55fc-ebc4-4920-8b29-b57541c978c0-c000.csv.gz"
PV_FILE = "wid2pv.pkl"
PROJECT_ID = 'final-project-415618'
BUCKET_NAME = "tfidf_bucket_318437159"

client = storage.Client(project=PROJECT_ID)
bucket = client.get_bucket(BUCKET_NAME)
pv = pickle.loads(bucket.get_blob(PV_FILE).download_as_string())

decompressed_file = gzip.decompress(bucket.get_blob(PR_FILE).download_as_string())
csv_reader = csv.reader(io.StringIO(decompressed_file.decode("utf-8")))
pr = {int(page_rank_tup[0]): float(page_rank_tup[1]) for page_rank_tup in csv_reader}

# Convert dictionary values to NumPy arrays
pv_values_np = np.array(list(pv.values()))
pr_values_np = np.array(list(pr.values()))

# Calculate mean and standard deviation for pv
pv_mean_np = np.mean(pv_values_np)
pv_std_deviation_np = np.std(pv_values_np)

print("PV Mean (NumPy):", pv_mean_np)
print("PV Standard Deviation (NumPy):", pv_std_deviation_np)

# Calculate mean and standard deviation for pr
pr_mean_np = np.mean(pr_values_np)
pr_std_deviation_np = np.std(pr_values_np)

print("PR Mean (NumPy):", pr_mean_np)
print("PR Standard Deviation (NumPy):", pr_std_deviation_np)