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

import math

import tensorflow as tf

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
