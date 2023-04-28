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
import argparse
import numpy as np
import functools
import time
import wandb

# DEBUG
#jax.config.update('jax_disable_jit', True)



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
    '-wandb_name',
    help='wandb_name',
    type=str,
    default='agreement-reweight',
)

parser.add_argument(
    '--video_threshold',
    help='Threshold for video loss',
    type=float,
    default=0,
)

parser.add_argument(
    '--sim_threshold',
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
    config['model']['sim_threshold'] = args.sim_threshold
    config['model']['audio_threshold'] = args.audio_threshold
    config['model']['video_threshold'] = args.video_threshold

config['_path'] = args.config_file
wandb.init(config=config, project=args.wandb_name, tags=['pretrain'], name=args.run_name)
wandb.run.log_code('.')
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
state = load_checkpoint(state=state, path=os.path.join(config['device']['output_dir'], 'ckpt_zzzzzzzzzz'), step=None,
                        use_bfloat16_weights=config['optimizer'].get('use_bfloat16_weights', False))
start_step = int(state.step)
state = jax_utils.replicate(state)

p_train_step = jax.pmap(functools.partial(train_step, use_bfloat16_grads=config['model']['use_bfloat16'],
                                        reweight=config['model']['reweight'], thresholds={k: config['model'][k] for k in ['sim_threshold', 'audio_threshold', 'video_threshold']}),
                        axis_name='batch', donate_argnums=(0, 1,))

'''
for batch in ds_train_iter:
    pass
'''
train_metrics = []
time_elapsed = []
num_train_steps = config['optimizer'].get('num_train_steps_override', config['optimizer']['num_train_steps'])
log_every = config['device'].get('commit_every_nsteps', 50)
wandb_logs = {
    'loss_vision': [],
    'loss_no_vision': [],
    'delta_loss': []
}
if config['model']['reweight']:
    wandb_logs['reweighted'] = []
for n in range(num_train_steps):
    st = time.time()
    batch = next(ds_train_iter)
    state, loss_info = p_train_step(state, batch)
    if config['model']['reweight']:
        (loss_info, loss, loss_nv, loss_reweighted) = loss_info
    else:
        (loss_info, loss, loss_nv) = loss_info
    for i in range(config['device']['batch_size']):

        wandb_logs['loss_vision'].append([loss[i].item()])
        wandb_logs['loss_no_vision'].append([loss_nv[i].item()])
        wandb_logs['delta_loss'].append([loss_nv[i].item() - loss[i].item()])
        if config['model']['reweight']:
            wandb_logs['reweighted'].append([loss_reweighted[i].item()])

        log_data = {
            'loss_vision': loss[i],
            'loss_no_vision': loss_nv[i],
            'delta_loss': loss_nv[i] - loss[i],
        }
        if config['model']['reweight']:
            log_data['reweighted'] = loss_reweighted[i]
        wandb.log(log_data)

    # Async transfer. Basically we queue the last thing, then log the thing from `log_every` iterations ago
    if jax.process_index() == 0:
        train_metrics.append(jax.tree_map(lambda x: x[0], loss_info))
        jax.tree_map(lambda x: x.copy_to_host_async(), train_metrics[-1])

        step_for_logging = n - log_every
        if step_for_logging >= 0:
            train_metrics[step_for_logging] = {k: float(v) for k, v in train_metrics[step_for_logging].items()}

            wandb.log(train_metrics[step_for_logging], commit=(n + 1) % log_every == 0)

    if (n + 1) % config['device']['iterations_per_loop'] == 0:
        save_checkpoint(state, path=config['device']['output_subdir'])
        print(f"Saving @iter {n:03d}.", flush=True)
        temps = {}
        for i, k in enumerate(['imgs_to_audio', 'text_to_audio', 'stuff_to_span']):
            temps[k] = state.params._dict['contrastive_scales'][0, i].astype(jnp.float32)
        temps = jax.device_get(temps)
        for k, v in temps.items():
            print("{} temperature: log={:.3f}  exp={:.3f}".format(k, v, np.exp(v)), flush=True)

    time_elapsed.append(time.time() - st)
    if len(time_elapsed) >= 100:
        tsum = sum(time_elapsed)
        print("Completed 100 batches in {:.3f}sec, avg {:.3f} it/sec".format(tsum, 100.0/tsum), flush=True)
        time_elapsed = []

for k, v in wandb_logs.items():
    data = v
    table = wandb.Table(data=data, columns=[k])
    wandb.log({k: wandb.plot.histogram(table, k, title=None)})
