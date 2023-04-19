from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv('.env')


chkpt_path = os.environ["CHKPT_PATH"]

storage_client = storage.Client()
bucket = storage_client.bucket('merlotreserve')
blob = bucket.blob(f'ckpts/base')
print(f"DOWNLOADING! gs://merlotreserve/ckpts/base", flush=True)
blob.download_to_filename(chkpt_path)