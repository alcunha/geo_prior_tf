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

class WeightedBinaryCrossEntropy:
  def __init__(self, pos_weight = 1, epsilon=1e-5):
    self.pos_weight = pos_weight
    self.epsilon = epsilon

  def _log(self, value):
    return (-1)*(tf.math.log(value + self.epsilon))

  def __call__(self, y_true, y_pred):
    log_loss = self.pos_weight * y_true * self._log(y_pred) \
               + (1 - y_true) * self._log(1 - y_pred)

    return tf.reduce_mean(log_loss, axis=-1)
