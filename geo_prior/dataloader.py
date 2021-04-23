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

import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class JsonInatInputProcessor:
  def __init__(self,
              dataset_json,
              location_info_json,
              remove_invalid=True,
              default_empty_label=0):
    self.dataset_json = dataset_json
    self.location_info_json = location_info_json
    self.default_empty_label = default_empty_label
    self.remove_invalid = remove_invalid

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

    with tf.io.gfile.GFile(self.location_info_json, 'r') as json_file:
      json_data = json.load(json_file)
    location_info = pd.DataFrame(json_data)
    images = pd.merge(images,
                      location_info,
                      how='left',
                      on='id')

    return images

  def make_source_dataset(self):
    metadata = self._load_metadata()

    if self.remove_invalid:
      metadata = metadata[metadata.valid].copy()
    
    return metadata
