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

from data_utils import RandSpatioTemporalGenerator
from model_builder import FCNet
import dataloader

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'train_data_json', default=None,
    help=('Path to json file containing the training data json'))

flags.DEFINE_string(
    'train_location_info_json', default=None,
    help=('Path to json file containing the location info for training data'))

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

flags.DEFINE_integer(
    'batch_size', default=1024,
    help=('Batch size used during training.'))

flags.DEFINE_integer(
    'embed_dim', default=256,
    help=('Embedding dimension for geo prior model'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('train_data_json')
flags.mark_flag_as_required('train_location_info_json')

def build_input_data():
  input_data = dataloader.JsonInatInputProcessor(
      FLAGS.train_data_json,
      FLAGS.train_location_info_json,
      batch_size=FLAGS.batch_size,
      is_training=True,
      max_instances_per_class=FLAGS.max_instances_per_class,
      loc_encode=FLAGS.loc_encode,
      date_encode=FLAGS.date_encode,
      use_date_feats=FLAGS.use_date_feats,
      use_photographers=FLAGS.use_photographers)

  return input_data.make_source_dataset()

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  dataset, _, num_classes, num_users, num_feats = build_input_data()
  randgen = RandSpatioTemporalGenerator(
      loc_encode=FLAGS.loc_encode,
      date_encode=FLAGS.date_encode,
      use_date_feats=FLAGS.use_date_feats)
  
  model = FCNet(num_inputs=num_feats,
                embed_dim=FLAGS.embed_dim,
                num_classes=num_classes,
                rand_sample_generator=randgen,
                num_users=(num_users if FLAGS.use_photographers else 0))
  model.build((None, num_feats))
  model.summary()

if __name__ == '__main__':
  app.run(main)
