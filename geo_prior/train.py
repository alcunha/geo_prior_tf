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

import dataloader
import model_builder

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
      is_training=True,
      max_instances_per_class=FLAGS.max_instances_per_class,
      loc_encode=FLAGS.loc_encode,
      date_encode=FLAGS.date_encode,
      use_date_feats=FLAGS.use_date_feats)

  return input_data.make_source_dataset()

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  dataset, _, num_classes, _ = build_input_data()

  model = model_builder.create_FCNET(6, num_classes, 256)
  model.summary()

if __name__ == '__main__':
  app.run(main)