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

r"""Tool to evaluate models.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import json
import os
import random

from absl import app
from absl import flags
from scipy import sparse
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

import dataloader

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'test_data_json', default=None,
    help=('Path to json file containing the test data json'))

flags.DEFINE_string(
    'test_location_info_json', default=None,
    help=('Path to json file containing the location info for test data'))

flags.DEFINE_string(
    'loc_encode', default='encode_cos_sin',
    help=('Encoding type for location coordinates'))

flags.DEFINE_string(
    'date_encode', default='encode_cos_sin',
    help=('Encoding type for date'))

flags.DEFINE_bool(
    'use_date_feats', default=True,
    help=('Include date features to the inputs'))

flags.DEFINE_integer(
    'batch_size', default=1024,
    help=('Batch size used during prediction.'))

flags.DEFINE_integer(
    'num_classes', default=8142,
    help=('Number of classes of the model.'))

flags.DEFINE_string(
    'prior_type', default='geo_prior',
    help=('Type of prior to be used for prediction'))

flags.DEFINE_string(
    'cnn_predictions_file', default=None,
    help=('File .npz containing class predictions for images on test dataset'))

flags.DEFINE_integer(
    'log_frequence', default=10,
    help=('Log prediction every n steps'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('test_data_json')
flags.mark_flag_as_required('test_location_info_json')
flags.mark_flag_as_required('cnn_predictions_file')

class CNNPredictor:
  def __init__(self, cnn_predictions_npz, data_json):
    with open(data_json) as json_file:
      json_data = json.load(json_file)
    samples = json_data['images']

    preds = sparse.load_npz(cnn_predictions_npz)
    preds = np.array(preds.todense(), dtype=np.float32)

    preds_dict = {sample['id']: pred for sample, pred in zip(samples, preds)}
    self.preds_dict = preds_dict
  
  def get_predictions(self, batch_ids):
    ids_list = list(batch_ids.numpy())
    preds = [self.preds_dict[instance_id] for instance_id in ids_list]
    return tf.convert_to_tensor(preds)


def build_input_data():
  input_data = dataloader.JsonInatInputProcessor(
    FLAGS.test_data_json,
    FLAGS.test_location_info_json,
    batch_size=FLAGS.batch_size,
    loc_encode=FLAGS.loc_encode,
    date_encode=FLAGS.date_encode,
    use_date_feats=FLAGS.use_date_feats,
    use_photographers=False,
    is_training=False,
    remove_invalid=False,
    provide_validity_info_output=True,
    num_classes=FLAGS.num_classes,
    provide_instance_id=True,
    batch_drop_remainder=False)

  dataset, _, _, _, num_feats = input_data.make_source_dataset()

  return dataset, num_feats

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor, axis=1).numpy()

def eval_model(cnn_model, dataset):
  labels = []
  predictions = []
  count = 0

  for batch, metadata in dataset:
    label, valid, ids = metadata
    preds = cnn_model.get_predictions(ids)

    labels += list(_decode_one_hot(label))
    predictions += list(_decode_one_hot(preds))

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1
  
  return accuracy_score(labels, predictions)

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  dataset, num_feats = build_input_data()
  cnn_model = CNNPredictor(FLAGS.cnn_predictions_file, FLAGS.test_data_json)

  acc = eval_model(cnn_model, dataset)

  print("Accuracy: %.2f" % (acc*100))

if __name__ == '__main__':
  app.run(main)
