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

import calendar
import datetime
import math

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'datetime_format', default='%Y-%m-%d %H:%M:%S+00:00',
    help=('Datetime format used to convert to days to float'))

flags.DEFINE_integer(
    'rand_date_magnitute', default=10,
    help=('Number of days for date data augmentation'))

flags.DEFINE_integer(
    'rand_location_magnitute', default=5000,
    help=('Magnitude in meters for location data augmentation'))

def date2float(date):
    dt = datetime.datetime.strptime(date, FLAGS.datetime_format).timetuple()
    year_days = 366 if calendar.isleap(dt.tm_year) else 365
    
    return dt.tm_yday/year_days

def random_date(date):
    date = tf.cast(date, tf.float32)
    rand = tf.random.uniform(shape=[], minval=-1, maxval=1, dtype=tf.float32)
    rand = (rand*FLAGS.rand_date_magnitute)/365.0
    date = tf.math.floormod(tf.abs(date + rand), 1)

    return tf.cast(date, tf.float64)

def random_loc(lon, lat):
    lon = tf.cast(lon, tf.float32)
    lat = tf.cast(lat, tf.float32)
    radiusdegree = FLAGS.rand_location_magnitute/111320.0

    rand_u = tf.random.uniform(shape=[], dtype=tf.float32)
    rand_v = tf.random.uniform(shape=[], dtype=tf.float32)
    w = radiusdegree*tf.sqrt(rand_u)
    t = 2*math.pi*rand_v
    dx = w*tf.cos(t)
    dy = w*tf.sin(t)
    
    dx = dx/tf.cos((lat*math.pi)/180.0)
    
    lon = lon + dy
    lat = lat + dx

    return tf.cast(lon, tf.float64), tf.cast(lat, tf.float64)
