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

def _createResLayer(inputs, embed_dim):
  x = tf.keras.layers.Dense(embed_dim)(inputs)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dropout(rate=0.5)(x)
  x = tf.keras.layers.Dense(embed_dim)(x)
  x = tf.keras.layers.Activation('relu')(x)
  outputs = tf.keras.layers.add([inputs, x])

  return outputs

def create_FCNET(num_inputs, num_classes, embed_dim, num_res_blocks=4):
  inputs = tf.keras.Input(shape=(num_inputs,))
  x = tf.keras.layers.Dense(embed_dim)(inputs)
  x = tf.keras.layers.Activation('relu')(x)
  for _ in range(num_res_blocks):
    x = _createResLayer(x, embed_dim)

  class_embed = tf.keras.layers.Dense(num_classes, use_bias=False)(x)

  model = tf.keras.models.Model(inputs=inputs, outputs=[class_embed])

  return model
