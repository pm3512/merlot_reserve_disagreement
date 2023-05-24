"""
This is the training script for the first 10 epochs of MERLOT Reserve (at low resolution)
"""

import sys

sys.path.append('../')
import os
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder
from pretrain.pretrain_model import *
from flax import jax_utils
from pretrain.optimization import construct_train_state
from mreserve.checkpoint import save_checkpoint, load_checkpoint, bf16_to_f32
#from mreserve.lowercase_encoder import get_encoder
import argparse
import numpy as np
import functools
import time
from pretrain.pretrain_model import exclusions_list

# DEBUG
#jax.config.update('jax_disable_jit', True)


#encoder = get_encoder()
jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
if not is_on_gpu:
    assert any([x.platform == 'tpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')
parser.add_argument(
    'config_file',
    help='Where the config.yaml is located',
    type=str,
)
parser.add_argument(
    '-output_dir',
    help='Override output directory (otherwise we do whats in the config file and add timestamp).',
    dest='output_dir',
    default='/home/aobolens/ckpt',
    type=str,
)

parser.add_argument(
    '--dataset',
    type=str,
)

parser.add_argument(
    '-wandb_name',
    help='wandb_name',
    type=str,
    default='agreement-reweight',
)

parser.add_argument(
    '--text_threshold',
    help='Threshold for text loss',
    type=float,
    default=0,
)

parser.add_argument(
    '--video_threshold',
    help='Threshold for video loss',
    type=float,
    default=0,
)

parser.add_argument(
    '--sim_threshold_text',
    help='Threshold for similarity loss',
    type=float,
    default=0,
)
parser.add_argument(
    '--sim_threshold_video',
    help='Threshold for similarity loss',
    type=float,
    default=0,
)
parser.add_argument(
    '--sim_threshold_audio',
    help='Threshold for similarity loss',
    type=float,
    default=0,
)

parser.add_argument(
    '--audio_threshold',
    help='Threshold for audio loss',
    type=float,
    default=0,
)
parser.add_argument(
    '--run_name',
    type=str
)

parser.add_argument(
    '--reweight',
    help='do reweighting based on agreement',
    action='store_true'
)
args = parser.parse_args()

print(f"Loading from {args.config_file}", flush=True)
with open(args.config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

    if is_on_gpu:
        config['data']['num_train_files'] = 1
        config['device']['output_dir'] = 'temp'
        config['model']['use_bfloat16'] = False
        config['device']['batch_size'] = 6

        config['optimizer']['num_train_steps_override'] = 1000
    elif args.output_dir == '':
        config['device']['output_dir'] = os.path.join(config['device']['output_dir'], args.run_name)
    else:
        config['device']['output_subdir'] = os.path.join(config['device']['output_dir'], args.run_name)
        config['device']['output_dir'] = args.output_dir
    config['model']['reweight'] = args.reweight
    config['model']['sim_threshold_text'] = args.sim_threshold_text
    config['model']['sim_threshold_video'] = args.sim_threshold_video
    config['model']['sim_threshold_audio'] = args.sim_threshold_audio
    config['model']['audio_threshold'] = args.audio_threshold
    config['model']['video_threshold'] = args.video_threshold
    config['model']['text_threshold'] = args.text_threshold
    if args.dataset == 'urfunny':
        config['data']['num_train_files'] = 128
        config['data']['train_fns'] = '/home/aobolens/urfunny/tfrecords/train{:03d}of128.tfrecord'
    elif args.dataset == 'social_iq':
        config['data']['num_train_files'] = 128
        config['data']['train_fns'] = '/home/aobolens/social_iq/tfrecords/train{:03d}of128.tfrecord'
    else:
        assert args.dataset == 'mustard'
        config['data']['num_train_files'] = 8
        config['data']['train_fns'] = '/home/aobolens/mustard/tfrecords/train{:03d}of008.tfrecord'


config['_path'] = args.config_file
'''
if (jax.process_index() == 0) and (not is_on_gpu) and (not args.disable_wandb):
    #import wandb
    #wandb.init( add your info here )
else:
    wandb = None
'''

ds_train_iter = input_fn_builder(config)
dummy_batch = next(ds_train_iter)

for k, v in dummy_batch.items():
    print("{}: {} {}".format(k, v.shape, v.dtype), flush=True)

ablation_type = config['model'].get('ablation','')
if ablation_type:
    print(f"Using {ablation_type}")
    model = getattr(sys.modules[__name__], ablation_type).from_config(config)
else:
    model = MerlotReservePretrainer.from_config(config)

if is_on_gpu:
    print("DEBUG GPU BATCH!", flush=True)
    model.init(jax.random.PRNGKey(0), {k: jnp.asarray(v[0]) for k, v in dummy_batch.items()})

params = model.init_from_dummy_batch(dummy_batch)
state = construct_train_state(opt_config=config['optimizer'], model=model, params=params)

# load if we can
state = load_checkpoint(state=state, path=os.path.join(config['device']['output_dir'], 'social_iq_t1'), step=None,
                        use_bfloat16_weights=config['optimizer'].get('use_bfloat16_weights', False))
start_step = int(state.step)
state = jax_utils.replicate(state)

thresholds = ['sim_threshold_text','sim_threshold_video', 'sim_threshold_audio','audio_threshold', 'video_threshold', 'text_threshold']
p_train_step = jax.pmap(functools.partial(train_step, use_bfloat16_grads=config['model']['use_bfloat16'],
                                        reweight=config['model']['reweight'], thresholds={k: config['model'][k] for k in thresholds}),
                        axis_name='batch', donate_argnums=(0, 1,))

similarities = []
for batch in ds_train_iter:
    already_seen = False
    _, loss_info = p_train_step(state, batch)
    loss_info, _, _ = loss_info
    for i in range(config['device']['batch_size']):
        for j in range(16):
            similarity = 0
            for k in loss_info:
                if k.startswith('detailed_similarities_vision_text'):
                    print(k, loss_info[k])
                    print(loss_info[k][i])
                    print(loss_info[k][i][j])
                    similarity += loss_info[k][i][j]
            print('spans', batch['text_spans'][i][0])
            spans = jnp.concatenate([batch['text_spans'][i][0][3 * j + k] for k in range(3)])
            #decoded =  encoder.decode(spans['text_spans'][i][0])
            print('spans', spans)
            print('decoded', spans)
            similarities.append((similarities, spans))