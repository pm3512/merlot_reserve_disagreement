# runs make_mp3.py, make_frames.py, make_json.py to preprocess data for SiQ 1.0
# fill in correct paths in .env

import subprocess
import os
import argparse
import make_mp3
import make_frames
import make_json
from dotenv import load_dotenv

load_dotenv('/home/aobolens/merlot_reserve/.env')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mp3',
    help='convert wav to mp3',
    action='store_true'
)
parser.add_argument(
    '--frames',
    help='convert (mp4) videos to frames',
    action='store_true'
)
parser.add_argument(
    '--json',
    help='convert qa files to train/val json files',
    action='store_true'
)
parser.add_argument('--binary', 
    help='for making json only; if the task is binary q&a',
    action='store_true'
)

args = parser.parse_args()

if args.mp3:
    make_mp3.make_mp3_from(os.environ["WAV_PATH"], os.path.join(os.environ["DATA_DIR"], "acoustic_mp3"))

if args.frames:
    make_frames.video_to_frames(os.environ["VIDEO_PATH"], os.environ["DATA_DIR"])

if args.json:
    # make json for train and val vids
    make_json.make_json_for(os.environ["VIDEO_PATH"], 
                os.environ["DATA_DIR"],
                os.environ["QA_PATH"], args.binary)
