import cPickle
import itertools
import random

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack
import scipy

from data_access import DataAccess
import heading
from model_def import Model_LR, Model_RFC, Model_SVC
import settings
import util

def get_data_basic_big(model_id, driver_id, repeat, test=False, version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed))

  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  build_features = util.build_features_big if version == 1 else util.build_features_big_v2
  set1 = [build_features(ride) for ride in set1]
  set2 = [build_features(ride) for ride in set2]
  return np.array(set1), np.array(set2)

def get_data_basic_big_v2(model_id, driver_id, repeat, test=False):
  return get_data_basic_big(model_id, driver_id, repeat, test=test, version=2)

def get_data_basic(model_id, driver_id, repeat, test=False, normalized=False, version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed))

  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  set1 = [util.build_features(ride, normalized=normalized, version=version) for ride in set1]
  set2 = [util.build_features(ride, normalized=normalized, version=version) for ride in set2]
  return np.array(set1), np.array(set2)

def get_data_basic_accel(model_id, driver_id, repeat, test=False, version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed))

  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  set1 = [util.build_features_acc(ride, version=version) for ride in set1]
  set2 = [util.build_features_acc(ride, version=version) for ride in set2]
  return np.array(set1), np.array(set2)

def get_data_basic_accel_v2(model_id, driver_id, repeat, test=False):
  return get_data_basic_accel(model_id, driver_id, repeat, test=test, version=2)

def get_data_basic_v2(model_id, driver_id, repeat, test=False):
  return get_data_basic(model_id, driver_id, repeat, test=test, normalized=True, version=1)

def get_data_basic_v3(model_id, driver_id, repeat, test=False):
  return get_data_basic(model_id, driver_id, repeat, test=test, version=2)

def get_data_basic_v4(model_id, driver_id, repeat, test=False):
  return get_data_basic(model_id, driver_id, repeat, test=test, normalized=True, version=2)

def get_data_basic_v5(model_id, driver_id, repeat, test=False):
  return get_data_basic(model_id, driver_id, repeat, test=test, normalized=True, version=3)

def get_data_segment_lengths(model_id, driver_id, repeat, test=False, segment_version=1, extra=((1,8),1)):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  ngram_range, min_df = extra

  if test:
    set1 = list(da.get_rides_segments(driver_id, version=segment_version))
    set2 = list(da.get_random_rides(
        settings.BIG_CHUNK_TEST * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
  else:
    driver_train, driver_test = da.get_rides_split(
        driver_id,
        settings.BIG_CHUNK,
        segments=True,
        version=segment_version
    )
    other_train = list(da.get_random_rides(
        settings.BIG_CHUNK * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
    other_test = list(da.get_random_rides(
        settings.SMALL_CHUNK,
        driver_id,
        segments=True,
        version=segment_version
    ))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  # keep only lengths
  set1 = [d[0] for d in set1]
  set2 = [d[0] for d in set2]

  # convert to text
  set1 = [util.get_list_string(d) for d in set1]
  set2 = [util.get_list_string(d) for d in set2]

  vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)
  return set1, set2

def get_data_segment_times(model_id, driver_id, repeat, test=False, segment_version=1, extra=((1,8),1)):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  ngram_range, min_df = extra

  if test:
    set1 = list(da.get_rides_segments(driver_id, version=segment_version))
    set2 = list(da.get_random_rides(
        settings.BIG_CHUNK_TEST * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
  else:
    driver_train, driver_test = da.get_rides_split(
        driver_id,
        settings.BIG_CHUNK,
        segments=True,
        version=segment_version
    )
    other_train = list(da.get_random_rides(
        settings.BIG_CHUNK * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
    other_test = list(da.get_random_rides(
        settings.SMALL_CHUNK,
        driver_id,
        segments=True,
        version=segment_version
    ))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  set1 = [util.get_list_string(d[2]) for d in set1]
  set2 = [util.get_list_string(d[2]) for d in set2]

  vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)
  return set1, set2

def get_data_segment_angles(model_id, driver_id, repeat, test=False, segment_version=1, extra=((1,1),2)):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  ngram_range, min_df = extra

  if test:
    set1 = list(da.get_rides_segments(driver_id, version=segment_version))
    set2 = list(da.get_random_rides(
        settings.BIG_CHUNK_TEST * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
  else:
    driver_train, driver_test = da.get_rides_split(
        driver_id,
        settings.BIG_CHUNK,
        segments=True,
        version=segment_version
    )
    other_train = list(da.get_random_rides(
        settings.BIG_CHUNK * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
    other_test = list(da.get_random_rides(
        settings.SMALL_CHUNK,
        driver_id,
        segments=True,
        version=segment_version
    ))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  # create features for each (segment, angle, segment) tuple
  set1 = [['%s_%s_%s' % (d[0][i-1], d[1][i-1], d[0][i]) for i in xrange(1, len(d[0]))] for d in set1]
  set2 = [['%s_%s_%s' % (d[0][i-1], d[1][i-1], d[0][i]) for i in xrange(1, len(d[0]))] for d in set2]

  set1 = [util.get_list_string(d) for d in set1]
  set2 = [util.get_list_string(d) for d in set2]

  vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)
  return set1, set2

def get_data_segment_angles_v1(model_id, driver_id, repeat, test=False):
  return get_data_segment_lengths(model_id, driver_id, repeat, test=test, segment_version=2, extra=((1,8),1))

def get_data_segment_angles_v2(model_id, driver_id, repeat, test=False, segment_version=1, extra=((1,3),1)):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  ngram_range, min_df = extra

  if test:
    set1 = list(da.get_rides_segments(driver_id, version=segment_version))
    set2 = list(da.get_random_rides(
        settings.BIG_CHUNK_TEST * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
  else:
    driver_train, driver_test = da.get_rides_split(
        driver_id,
        settings.BIG_CHUNK,
        segments=True,
        version=segment_version
    )
    other_train = list(da.get_random_rides(
        settings.BIG_CHUNK * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
    other_test = list(da.get_random_rides(
        settings.SMALL_CHUNK,
        driver_id,
        segments=True,
        version=segment_version
    ))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  # create features for each (segment, angle, segment) tuple
  set1 = [['%s_%s' % (d[0][i-1], d[1][i-1]) for i in xrange(1, len(d[0]))] for d in set1]
  set2 = [['%s_%s' % (d[0][i-1], d[1][i-1]) for i in xrange(1, len(d[0]))] for d in set2]

  set1 = [util.get_list_string(d) for d in set1]
  set2 = [util.get_list_string(d) for d in set2]

  vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)
  return set1, set2

def get_data_segment_v2(model_id, driver_id, repeat, test=False, segment_version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()

  if test:
    set1 = list(da.get_rides_segments(driver_id, version=segment_version))
    set2 = list(da.get_random_rides(
        settings.BIG_CHUNK_TEST * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
  else:
    driver_train, driver_test = da.get_rides_split(
        driver_id,
        settings.BIG_CHUNK,
        segments=True,
        version=segment_version
    )
    other_train = list(da.get_random_rides(
        settings.BIG_CHUNK * repeat,
        driver_id,
        segments=True,
        version=segment_version,
        seed=seed
    ))
    other_test = list(da.get_random_rides(
        settings.SMALL_CHUNK,
        driver_id,
        segments=True,
        version=segment_version
    ))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  set1 = [['%s_%s_%s_%s' % (d[0][i-1], d[2][i-1], d[1][i-1], d[0][i]) for i in xrange(1, len(d[0]))] for d in set1]
  set2 = [['%s_%s_%s_%s' % (d[0][i-1], d[2][i-1], d[1][i-1], d[0][i]) for i in xrange(1, len(d[0]))] for d in set2]

  set1 = [util.get_list_string(d) for d in set1]
  set2 = [util.get_list_string(d) for d in set2]

  vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)
  return set1, set2

def get_data_movements_v1(model_id, driver_id, repeat, test=False, step=5, tf=False, version=1, extra=((1,5),2)):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  ngram_range, min_df = extra

  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, segments=False, seed=seed))
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK, segments=False)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, segments=False, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id, segments=False))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  # keep only lengths and convert to text
  set1 = [util.build_features3(r, step=step, version=version) for r in set1]
  set2 = [util.build_features3(r, step=step, version=version) for r in set2]

  if tf:
    vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
  else:
    vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)

  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_movements_v2(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=5, extra=((1,15),1))

def get_data_movements_v4(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=5, version=2, extra=((1,15),1))

def get_data_movements_v5(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=3, version=2, extra=((1,15),1))

def get_data_movements_v6(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=5, version=2)

def get_data_movements_v7(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=3, version=2)

def get_data_movements_v8(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=3, version=3)

def get_data_movements_v2_svd(model_id, driver_id, repeat, test=False):
  set1, set2 = get_data_movements_v2(model_id, driver_id, repeat, test=test)

  svd = TruncatedSVD(n_components=20, random_state=driver_id+model_id)
  set1 = svd.fit_transform(set1)
  set2 = svd.transform(set2)

  return set1, set2

def get_data_movements_v3(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=3, extra=((1,15),1))

def get_data_movements_v1_tf(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, tf=True)

def get_data_movements_v2_tf(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=5, tf=True, extra=((1,15),1))

def get_data_movements_v3_tf(model_id, driver_id, repeat, test=False):
  return get_data_movements_v1(model_id, driver_id, repeat, test=test, step=3, tf=True, extra=((1,15),1))

def get_data_movements_accel(model_id, driver_id, repeat, test=False, step=3, tf=False, extra=((1,15),2), version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  ngram_range, min_df = extra

  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, segments=False, seed=seed))
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK, segments=False)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, segments=False, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id, segments=False))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  set1 = [util.build_features4(r, step=step, version=version) for r in set1]
  set2 = [util.build_features4(r, step=step, version=version) for r in set2]

  if tf:
    vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
  else:
    vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)

  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_movements_accel_v2(model_id, driver_id, repeat, test=False):
  return get_data_movements_accel(model_id, driver_id, repeat, test=test, step=3, tf=False, extra=((1,15),1), version=2)

def get_data_movements_accel_svd(model_id, driver_id, repeat, test=False):
  set1, set2 = get_data_movements_accel(model_id, driver_id, repeat, test=test)

  svd = TruncatedSVD(n_components=20, random_state=driver_id+model_id)
  set1 = svd.fit_transform(set1)
  set2 = svd.transform(set2)

  return set1, set2

def get_data_accel(model_id, driver_id, repeat, test=False):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id)) # first half of the train set
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed)) # second half of the train set
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train # used for training
    set2 = driver_test + other_test # used for testing

  set1 = [util.get_list_string(util.get_accelerations(ride)) for ride in set1]
  set2 = [util.get_list_string(util.get_accelerations(ride)) for ride in set2]

  vectorizer = CountVectorizer(min_df=1)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_accel_v2(model_id, driver_id, repeat, test=False):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id)) # first half of the train set
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed)) # second half of the train set
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train # used for training
    set2 = driver_test + other_test # used for testing

  set1 = [util.get_list_string(util.get_accelerations_v2(ride)) for ride in set1]
  set2 = [util.get_list_string(util.get_accelerations_v2(ride)) for ride in set2]

  vectorizer = CountVectorizer(min_df=1)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_accel_v2_svd(model_id, driver_id, repeat, test=False):
  set1, set2 = get_data_accel_v2(model_id, driver_id, repeat, test=test)

  svd = TruncatedSVD(n_components=20, random_state=driver_id+model_id)
  set1 = svd.fit_transform(set1)
  set2 = svd.transform(set2)

  return set1, set2

def get_data_g_forces_v1(model_id, driver_id, repeat, test=False, min_df=1, ngram_range=(1,10), digitize=0):
  def process(ride, digitize):
    g_forces = util.get_g_forces(ride)
    if digitize:
      g_forces = np.digitize(g_forces, range(0, 800, digitize))
    return util.get_list_string(g_forces)

  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id)) # first half of the train set
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed)) # second half of the train set
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train # used for training
    set2 = driver_test + other_test # used for testing

  set1 = [process(ride, digitize) for ride in set1]
  set2 = [process(ride, digitize) for ride in set2]

  vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_g_forces_v2(model_id, driver_id, repeat, test=False):
  return get_data_g_forces_v1(model_id, driver_id, repeat, test=test, min_df=1, ngram_range=(1,20), digitize=2)

def get_data_g_forces_v3(model_id, driver_id, repeat, test=False):
  return get_data_g_forces_v1(model_id, driver_id, repeat, test=test, min_df=2, ngram_range=(1,5), digitize=1)

def get_data_g_forces_v4(model_id, driver_id, repeat, test=False):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id)) # first half of the train set
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed)) # second half of the train set
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train # used for training
    set2 = driver_test + other_test # used for testing

  set1 = [util.get_g_forces_v2(ride) for ride in set1]
  set2 = [util.get_g_forces_v2(ride) for ride in set2]

  vectorizer = CountVectorizer(min_df=1, ngram_range=(1,10))
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_g_forces_v5(model_id, driver_id, repeat, test=False):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id)) # first half of the train set
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed)) # second half of the train set
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train # used for training
    set2 = driver_test + other_test # used for testing

  set1 = [util.get_g_forces_v3(ride) for ride in set1]
  set2 = [util.get_g_forces_v3(ride) for ride in set2]

  vectorizer = CountVectorizer(min_df=1, ngram_range=(1,10))
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_g_forces_v6(model_id, driver_id, repeat, test=False, version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id)) # first half of the train set
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed)) # second half of the train set
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train # used for training
    set2 = driver_test + other_test # used for testing

  set1 = [util.get_g_forces_v4(ride, version=version) for ride in set1]
  set2 = [util.get_g_forces_v4(ride, version=version) for ride in set2]

  vectorizer = CountVectorizer(min_df=1, ngram_range=(1,20))
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_g_forces_v7(model_id, driver_id, repeat, test=False):
  return get_data_g_forces_v6(model_id, driver_id, repeat, test=test, version=2)

def get_data_dist_acc(model_id, driver_id, repeat, test=False):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed))
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  set1 = [util.get_distance_acc_words(ride, step=3) for ride in set1]
  set2 = [util.get_distance_acc_words(ride, step=3) for ride in set2]

  vectorizer = CountVectorizer(min_df=1, ngram_range=(1,15))
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_acc4acc(model_id, driver_id, repeat, test=False, version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed))
  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  set1 = [util.get_acc4acc_words(ride, step=3, version=version) for ride in set1]
  set2 = [util.get_acc4acc_words(ride, step=3, version=version) for ride in set2]

  max_ngram = 15 if version == 1 else 20
  vectorizer = CountVectorizer(min_df=1, ngram_range=(1,max_ngram))
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_acc4acc_v2(model_id, driver_id, repeat, test=False):
  return get_data_acc4acc(model_id, driver_id, repeat, test=test, version=2)

def get_data_fft(model_id, driver_id, repeat, test=False, version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    set1 = list(da.get_rides(driver_id))
    set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed))

  else:
    driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
    other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
    other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

    set1 = driver_train + other_train
    set2 = driver_test + other_test

  if version == 1:
    set1 = [util.fft(ride) for ride in set1]
    set2 = [util.fft(ride) for ride in set2]
  else:
    set1 = [util.fft_strip(ride) for ride in set1]
    set2 = [util.fft_strip(ride) for ride in set2]

  return np.array(set1), np.array(set2)

def get_data_fft_v2(model_id, driver_id, repeat, test=False):
  return get_data_fft(model_id, driver_id, repeat, test=test, version=2)

def get_data_heading(model_id, driver_id, repeat, test=False, moving_average_window=3, stops=False, version=1):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()
  if test:
    raise Exception

  driver_train, driver_test = da.get_rides_split(driver_id, settings.BIG_CHUNK)
  other_train = list(da.get_random_rides(settings.BIG_CHUNK * repeat, driver_id, seed=seed))
  other_test = list(da.get_random_rides(settings.SMALL_CHUNK, driver_id))

  set1 = driver_train + other_train # used for training
  set2 = driver_test + other_test # used for testing

  set1 = [heading.get_ride_heading(ride, variations=True, \
      moving_average_window=moving_average_window, stops=stops, version=version) for ride in set1]
  set2 = [util.get_list_string(heading.get_ride_heading(ride, \
      moving_average_window=moving_average_window, stops=stops, version=version)) for ride in set2]

  set1 = list(itertools.chain(*set1))

  set1 = [util.get_list_string(r) for r in set1]

  vectorizer = CountVectorizer(min_df=2, ngram_range=(1,15), max_df=1000000)
  set1 = vectorizer.fit_transform(set1)
  set2 = vectorizer.transform(set2)

  return set1, set2

def get_data_heading_svd(model_id, driver_id, repeat, test=False):
  set1, set2 = get_data_heading(model_id, driver_id, repeat, test=test)

  svd = TruncatedSVD(n_components=20, random_state=driver_id+model_id)
  set1 = svd.fit_transform(set1)
  set2 = svd.transform(set2)

  return set1, set2

def get_data_heading_v2(model_id, driver_id, repeat, test=False):
  return get_data_heading(model_id, driver_id, repeat, test=test, moving_average_window=6)

def get_data_heading_stops(model_id, driver_id, repeat, test=False):
  return get_data_heading(model_id, driver_id, repeat, test=test, stops=True)

def get_data_heading_v3(model_id, driver_id, repeat, test=False):
  return get_data_heading(model_id, driver_id, repeat, test=test, moving_average_window=0, version=2)

def get_data_heading_stops_v2(model_id, driver_id, repeat, test=False):
  return get_data_heading(model_id, driver_id, repeat, test=test, stops=True, moving_average_window=0, version=2)

HEADING_DATA_FUNCTIONS = (get_data_heading, get_data_heading_v2, get_data_heading_svd, \
    get_data_heading_stops, get_data_heading_v3, get_data_heading_stops_v2)

def run_model((model_id, driver_id, Model, get_data, repeat)):
  testY = [1] * settings.SMALL_CHUNK + [0] * settings.SMALL_CHUNK

  if settings.ENABLE_CACHE:
    predictions = util.get_results(Model, get_data, driver_id, False, repeat)
    if predictions is not False:
      return predictions, testY

  multiplier = 4 if get_data in HEADING_DATA_FUNCTIONS else 1

  trainY = [1] * settings.BIG_CHUNK * multiplier * repeat + \
      [0] * settings.BIG_CHUNK * multiplier * repeat
  trainX, testX = get_data(model_id, driver_id, repeat)

  if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
    trainX = scipy.sparse.vstack(
        [trainX[:settings.BIG_CHUNK * multiplier]] * repeat +
        [trainX[settings.BIG_CHUNK * multiplier:]]
    )
  else:
    trainX = np.vstack((
        np.tile(np.array(trainX[:settings.BIG_CHUNK * multiplier]).T, repeat).T,
        trainX[settings.BIG_CHUNK * multiplier:]
    ))

  assert(trainX.shape[0] == len(trainY))
  assert(testX.shape[0] == len(testY))

  model = Model(trainX, trainY, driver_id)
  predictions = model.predict(testX)

  if settings.ENABLE_CACHE:
    util.cache_results(Model, get_data, driver_id, False, predictions, repeat)

  return predictions, testY

def test_model_heading(model_id, driver_id, Model, get_data, repeat):
  seed = random.Random(x=driver_id+model_id)
  da = DataAccess()

  set1 = list(da.get_rides(driver_id)) # first half of the train set
  set2 = list(da.get_random_rides(settings.BIG_CHUNK_TEST * repeat, driver_id, seed=seed)) # second half of the train set

  moving_average_window = 6 if get_data == get_data_heading_v2 else 3
  set1 = [heading.get_ride_heading(ride, variations=True, \
      moving_average_window=moving_average_window) for ride in set1]
  set2 = [heading.get_ride_heading(ride, variations=True, \
      moving_average_window=moving_average_window) for ride in set2]

  set1 = [[util.get_list_string(r) for r in four_pack] for four_pack in set1]
  set2 = [[util.get_list_string(r) for r in four_pack] for four_pack in set2]

  vectorizer = CountVectorizer(min_df=2, ngram_range=(1,15), max_df=1000000)
  vectorizer.fit([r[0] for r in set1])
  rides = [[vectorizer.transform([r])[0] for r in four_pack] for four_pack in set1]
  other_rides = [[vectorizer.transform([r])[0] for r in four_pack] for four_pack in set2]
  other_rides = list(itertools.chain(*other_rides))

  rides = np.array(rides)

  trainY = [1] * settings.BIG_CHUNK_TEST * 4 * repeat + [0] * settings.BIG_CHUNK_TEST * 4 * repeat
  kf = KFold(200, n_folds=settings.FOLDS, shuffle=True, random_state=driver_id)
  predictions = ['bug'] * 200
  for train_fold, test_fold in kf:
    trainX = rides[train_fold]
    trainX = scipy.sparse.vstack(
        list(itertools.chain(*trainX)) * repeat + \
        other_rides
    )
    testX = scipy.sparse.vstack([r[0] for r in rides[test_fold]])

    assert(trainX.shape[0] == len(trainY))
    assert(testX.shape[0] == settings.SMALL_CHUNK_TEST)

    model = Model(trainX, trainY, driver_id)
    fold_predictions = model.predict(testX)
    for i, v in enumerate(test_fold):
      predictions[v] = fold_predictions[i]

  predictions = np.array(predictions)
  if settings.ENABLE_CACHE:
    util.cache_results(Model, get_data, driver_id, True, predictions, repeat)
  return driver_id, predictions

def test_model((model_id, driver_id, Model, get_data, repeat)):
  if settings.ENABLE_CACHE:
    predictions = util.get_results(Model, get_data, driver_id, True, repeat)
    if predictions is not False:
      return driver_id, predictions

  if get_data in HEADING_DATA_FUNCTIONS:
    return test_model_heading(model_id, driver_id, Model, get_data, repeat)

  rides, other_rides = get_data(model_id, driver_id, repeat, test=True)
  trainY = [1] * settings.BIG_CHUNK_TEST * repeat + [0] * settings.BIG_CHUNK_TEST * repeat
  kf = KFold(200, n_folds=settings.FOLDS, shuffle=True, random_state=driver_id)
  predictions = ['bug'] * 200
  for train_fold, test_fold in kf:
    trainX = rides[train_fold]
    testX = rides[test_fold]

    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = scipy.sparse.vstack([trainX] * repeat + [other_rides])
    else:
      trainX = np.vstack((
          np.tile(np.array(trainX).T, repeat).T,
          other_rides
      ))

    assert(trainX.shape[0] == len(trainY))
    assert(testX.shape[0] == settings.SMALL_CHUNK_TEST)

    model = Model(trainX, trainY, driver_id)
    fold_predictions = model.predict(testX)
    for i, v in enumerate(test_fold):
      predictions[v] = fold_predictions[i]

  predictions = np.array(predictions)
  if settings.ENABLE_CACHE:
    util.cache_results(Model, get_data, driver_id, True, predictions, repeat)
  return driver_id, predictions
