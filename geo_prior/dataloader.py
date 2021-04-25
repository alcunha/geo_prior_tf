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
              batch_size,
              loc_encode='encode_cos_sin',
              date_encode='encode_cos_sin',
              use_date_feats=True,
              use_photographers=False,
              is_training=False,
              remove_invalid=True,
              max_instances_per_class=-1,
              default_empty_label=0,
              num_classes=None,
              provide_instance_id=False,
              batch_drop_remainder=True):
    self.dataset_json = dataset_json
    self.location_info_json = location_info_json
    self.batch_size = batch_size
    self.loc_encode = loc_encode
    self.date_encode = date_encode
    self.use_date_feats = use_date_feats
    self.use_photographers = use_photographers
    self.is_training = is_training
    self.default_empty_label = default_empty_label
    self.remove_invalid = remove_invalid
    self.max_instances_per_class = max_instances_per_class
    self.provide_instance_id = provide_instance_id
    self.batch_drop_remainder = batch_drop_remainder
    self.num_instances = 0
    self.num_classes = num_classes
    self.num_users = 1
    self.num_feats = 0

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

  def _calculate_num_features(self):
    num_feats = 0

    if self.loc_encode == 'encode_cos_sin':
      num_feats += 4
    
    if self.use_date_feats:
      if self.date_encode == 'encode_cos_sin':
        num_feats += 2
    
    self.num_feats = num_feats

  def make_source_dataset(self):
    metadata, num_classes = self._load_metadata()
    self._calculate_num_features()
    if self.num_classes is None:
      self.num_classes = num_classes

    if self.remove_invalid:
      metadata = metadata[metadata.valid].copy()
    self.num_instances = len(metadata)
    self.num_users = len(metadata.user_id.unique())
    if self.use_photographers and self.num_users < 2:
      raise RuntimeError('To add photographers branch to the model, data must'
                         ' have more than one photographer')

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
        return tf.sin(math.pi*feat), tf.cos(math.pi*feat)
      else:
        raise RuntimeError('%s not implemented' % encode)

      return feat 

    def _preprocess_data(id, lat, lon, date_c, user_id, category_id):
      lat = lat/90.0
      lon = lon/180.0
      date_c = date_c*2.0 - 1.0

      lat = _encode_feat(lat, self.loc_encode)
      lon = _encode_feat(lon, self.loc_encode)
      date_c = _encode_feat(date_c, self.date_encode)

      if self.use_date_feats:
        inputs = tf.concat([lon, lat, date_c], axis=0)
      else:
        inputs = tf.concat([lon, lat], axis=0)
      inputs = tf.cast(inputs, tf.float32)

      category_id = tf.one_hot(category_id, self.num_classes)
      if self.num_users > 1:
        user_id = tf.one_hot(user_id, self.num_users)

      return id, inputs, user_id, category_id
    dataset = dataset.map(_preprocess_data, num_parallel_calls=AUTOTUNE)

    def _select_inputs_outputs(id, inputs, user_id, category_id):
      if self.use_photographers:
        outputs = (category_id, user_id, id) if self.provide_instance_id \
                                             else (category_id, user_id)
      else:
        outputs = (category_id, id) if self.provide_instance_id else category_id
      
      return inputs, outputs
    dataset = dataset.map(_select_inputs_outputs, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.batch_drop_remainder)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return (dataset, self.num_instances, self.num_classes, self.num_users, \
            self.num_feats)

class RandSpatioTemporalGenerator:
  def __init__(self,
               rand_type='spherical',
               loc_encode='encode_cos_sin',
               date_encode='encode_cos_sin',
               use_date_feats=True):
    self.rand_type = rand_type
    self.loc_encode = loc_encode
    self.date_encode = date_encode
    self.use_date_feats = use_date_feats

  def _encode_feat(self, feat, encode):
    if encode == 'encode_cos_sin':
      feats = tf.concat([
        tf.sin(math.pi*feat),
        tf.cos(math.pi*feat)], axis=1)
    else:
      raise RuntimeError('%s not implemented' % encode)

    return feats

  def get_rand_samples(self, batch_size):
    if self.rand_type == 'spherical':
      rand_feats = tf.random.uniform(shape=(batch_size, 3),
                                    dtype=tf.float32)
      theta1 = 2.0*math.pi*rand_feats[:,0]
      theta2 = tf.acos(2.0*rand_feats[:,1] - 1.0)
      lat = 1.0 - 2.0*theta2/math.pi
      lon = (theta1/math.pi) - 1.0
      time = rand_feats[:,2]*2.0 - 1.0

      lon = tf.expand_dims(lon, axis=-1)
      lat = tf.expand_dims(lat, axis=-1)
      time = tf.expand_dims(time, axis=-1)
    else:
      raise RuntimeError('%s rand type not implemented' % self.rand_type)

    lon = self._encode_feat(lon, self.loc_encode)
    lat = self._encode_feat(lat, self.loc_encode)
    time = self._encode_feat(time, self.date_encode)

    if self.use_date_feats:
      return tf.concat([lon, lat, time], axis=1)
    else:
      return tf.concat([lon, lat], axis=1)
