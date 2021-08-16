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

import tensorflow as tf

def weighted_binary_cross_entropy(pos_weight = 1, epsilon=1e-5):
  def _log(value):
    return (-1)*(tf.math.log(value + epsilon))

  def _call(y_true, y_pred):
    log_loss = pos_weight * y_true * _log(y_pred) \
               + (1 - y_true) * _log(1 - y_pred)

    return tf.reduce_mean(log_loss, axis=-1)

  return _call

def log_loss(epsilon=1e-5):
  def _log(value):
    return (-1)*(tf.math.log(value + epsilon))

  def _call(y_true, y_pred):
    _log_loss = y_true * _log(y_pred)

    return tf.reduce_mean(_log_loss, axis=-1)

  return _call
