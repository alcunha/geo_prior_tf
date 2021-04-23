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

import json
import math

import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class JsonInatInputProcessor:
  def __init__(self,
              dataset_json,
              location_info_json,
              loc_encode='encode_cos_sin',
              date_encode='encode_cos_sin',
              use_date_feats=True,
              is_training=False,
              remove_invalid=True,
              max_instances_per_class=-1,
              default_empty_label=0):
    self.dataset_json = dataset_json
    self.location_info_json = location_info_json
    self.loc_encode = loc_encode
    self.date_encode = date_encode
    self.use_date_feats = use_date_feats
    self.is_training = is_training
    self.default_empty_label = default_empty_label
    self.remove_invalid = remove_invalid
    self.max_instances_per_class = max_instances_per_class
    self.num_instances = 0
    self.num_classes = 0
    self.num_users = 1

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])
    if 'annotations' in json_data.keys():
      annotations = pd.DataFrame(json_data['annotations'])
      images = pd.merge(images,
                        annotations[["image_id", "category_id"]],
                        how='left',
                        left_on='id',
                        right_on='image_id')
    else:
      images['category_id'] = self.default_empty_label
    
    num_classes = len(json_data['categories'])

    with tf.io.gfile.GFile(self.location_info_json, 'r') as json_file:
      json_data = json.load(json_file)
    location_info = pd.DataFrame(json_data)
    images = pd.merge(images,
                      location_info,
                      how='left',
                      on='id')

    return images, num_classes

  def _get_balanced_dataset(self, metadata):
    num_instances = 0
    datasets = []
    for category in list(metadata.category_id.unique()):
      cat_metadata = metadata[metadata.category_id == category]
      num_instances_cat = min(len(cat_metadata), self.max_instances_per_class)
      num_instances += num_instances_cat

      cat_ds = tf.data.Dataset.from_tensor_slices((
        cat_metadata.id,
        cat_metadata.lat,
        cat_metadata.lon,
        cat_metadata.date_c,
        cat_metadata.user_id,
        cat_metadata.category_id))
      cat_ds = cat_ds.shuffle(len(cat_metadata)).take(num_instances_cat)
      datasets.append(cat_ds)
    
    self.num_instances = num_instances

    return tf.data.experimental.sample_from_datasets(datasets)

  def make_source_dataset(self):
    metadata, num_classes = self._load_metadata()
    self.num_classes = num_classes

    if self.remove_invalid:
      metadata = metadata[metadata.valid].copy()
    self.num_instances = len(metadata)
    self.num_users = len(metadata.user_id.unique())

    if self.max_instances_per_class == -1:
      dataset = tf.data.Dataset.from_tensor_slices((
        metadata.id,
        metadata.lat,
        metadata.lon,
        metadata.date_c,
        metadata.user_id,
        metadata.category_id))
    else:
      dataset = self._get_balanced_dataset(metadata)
    
    if self.is_training:
      dataset.shuffle(self.num_instances)
    
    def _encode_feat(feat, encode):
      if encode == 'encode_cos_sin':
        return tf.cos(math.pi*feat), tf.sin(math.pi*feat)
      else:
        raise RuntimeError('%s not implemented' % encode)

      return feat 

    def _preprocess_inputs(id, lat, lon, date_c, user_id, category_id):
      lat = lat/90.0
      lon = lon/180.0
      date_c = date_c*2.0 - 1.0

      lat = _encode_feat(lat, self.loc_encode)
      lon = _encode_feat(lon, self.loc_encode)
      date_c = _encode_feat(date_c, self.date_encode)

      if self.use_date_feats:
        inputs = tf.concat([lat, lon, date_c], axis=0)
      else:
        inputs = tf.concat([lat, lon], axis=0)

      return id, inputs, user_id, category_id
    dataset = dataset.map(_preprocess_inputs, num_parallel_calls=AUTOTUNE)

    return dataset, self.num_instances, self.num_classes, self.num_users
