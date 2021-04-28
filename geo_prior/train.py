# Copyright 2021 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Tool to train Geo Prior Model.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import os
import random

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

from models import FCNet
import dataloader
import losses

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'train_data_json', default=None,
    help=('Path to json file containing the training data json'))

flags.DEFINE_string(
    'train_location_info_json', default=None,
    help=('Path to json file containing the location info for training data'))

flags.DEFINE_string(
    'val_data_json', default=None,
    help=('Path to json file containing the validation data json'))

flags.DEFINE_string(
    'val_location_info_json', default=None,
    help=('Path to json file containing the location info for validation data'))

flags.DEFINE_integer(
    'max_instances_per_class', default=100,
    help=('Max number of instances per class sampled at each epoch.'))

flags.DEFINE_string(
    'loc_encode', default='encode_cos_sin',
    help=('Encoding type for location coordinates'))

flags.DEFINE_string(
    'date_encode', default='encode_cos_sin',
    help=('Encoding type for date'))

flags.DEFINE_bool(
    'use_date_feats', default=True,
    help=('Include date features to the inputs'))

flags.DEFINE_bool(
    'use_photographers', default=False,
    help=('Include photographers classifier to the model'))

flags.DEFINE_bool(
    'use_batch_normalization', default=False,
    help=('Include Batch Normalization to the model'))

flags.DEFINE_bool(
    'use_data_augmentation', default=False,
    help=('Use data agumentation on coordinates and date during training'))

flags.DEFINE_integer(
    'batch_size', default=1024,
    help=('Batch size used during training.'))

flags.DEFINE_integer(
    'embed_dim', default=256,
    help=('Embedding dimension for geo prior model'))

flags.DEFINE_float(
    'lr', default=0.0005,
    help=('Initial learning rate'))

flags.DEFINE_float(
    'lr_decay', default=0.98,
    help=('Learning rate decay at each epoch'))

flags.DEFINE_integer(
    'epochs', default=30,
    help=('Number of epochs to training for'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Location of the model checkpoint files'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('train_data_json')
flags.mark_flag_as_required('model_dir')

def build_input_data(data_json,
                     location_info_json,
                     is_training,
                     num_classes=None):
  input_data = dataloader.JsonInatInputProcessor(
      data_json,
      location_info_json=location_info_json,
      batch_size=FLAGS.batch_size,
      is_training=is_training,
      max_instances_per_class=(FLAGS.max_instances_per_class if is_training \
                                                             else -1),
      loc_encode=FLAGS.loc_encode,
      date_encode=FLAGS.date_encode,
      use_date_feats=FLAGS.use_date_feats,
      num_classes=num_classes,
      use_photographers=(FLAGS.use_photographers if is_training else False),
      use_data_augmentation=FLAGS.use_data_augmentation)

  return input_data.make_source_dataset()

def lr_scheduler(epoch, lr):
  if epoch < 1:
      return lr
  else:
      return lr * FLAGS.lr_decay

def train_model(model, dataset, val_dataset, loss_fn):
  summary_dir = os.path.join(FLAGS.model_dir, "summaries")
  summary_callback = tf.keras.callbacks.TensorBoard(summary_dir,
                                                    profile_batch=0)

  checkpoint_filepath = os.path.join(FLAGS.model_dir, "ckp")
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_freq='epoch')

  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
  lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

  callbacks = [summary_callback, checkpoint_callback, lr_callback]

  model.compile(optimizer=optimizer, loss=loss_fn)

  return model.fit(dataset,
                   epochs=FLAGS.epochs,
                   callbacks=callbacks,
                   validation_data=val_dataset)

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  dataset, _, num_classes, num_users, num_feats = build_input_data(
    FLAGS.train_data_json, FLAGS.train_location_info_json, is_training=True)
  randgen = dataloader.RandSpatioTemporalGenerator(
      loc_encode=FLAGS.loc_encode,
      date_encode=FLAGS.date_encode,
      use_date_feats=FLAGS.use_date_feats)

  if FLAGS.val_data_json is not None:
    val_dataset, _, _, _, _ = build_input_data(FLAGS.val_data_json,
      FLAGS.val_location_info_json, is_training=False, num_classes=num_classes)
  else:
    val_dataset = None

  model = FCNet(num_inputs=num_feats,
                embed_dim=FLAGS.embed_dim,
                num_classes=num_classes,
                rand_sample_generator=randgen,
                num_users=(num_users if FLAGS.use_photographers else 0),
                use_bn=FLAGS.use_batch_normalization)

  loss_o_loc = losses.weighted_binary_cross_entropy(pos_weight=num_classes)

  model.build((None, num_feats))
  model.summary()

  train_model(model, dataset, val_dataset, loss_o_loc)

if __name__ == '__main__':
  app.run(main)
