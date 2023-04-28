import tensorflow as tf 
raw_dataset = tf.data.TFRecordDataset("/home/aobolens/urfunny/finetune/train000of128.tfrecord")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
'''
from socialiq_std_folds import *

import pandas as pd

folds = standard_valid_fold

df = pd.DataFrame(folds, columns=['video_id'])

df.to_csv('folds_val.csv', index=False)
'''