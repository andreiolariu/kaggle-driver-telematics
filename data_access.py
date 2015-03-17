import os
import os.path
import random

import util
import settings

class DataAccess:

  def get_drivers(self):
    return settings.DRIVER_IDS

  def get_ride(self, driver_id, ride_id):
    filename = '%s/%s/%s.csv' % (settings.DATA_FOLDER, driver_id, ride_id)
    data = open(filename, 'r').read()
    data = [[float(x) for x in row.split(',')] for row in data.split('\n')[1:-1]]
    return data

  def get_rides(self, driver_id):
    for ride_id in range(1, 201):
      yield self.get_ride(driver_id, ride_id)

  def get_ride_segments(self, driver_id, ride_id, version=1):
    filename = '%s/%s_%s.csv' % (settings.SEGMENTS_FOLDER[version], driver_id, ride_id)
    data = open(filename, 'r').read()
    data = [[int(x) for x in row.split(',')] if row else [] for row in data.split('\n')]
    if data == [[]]:
      print driver_id
      print ride_id
    return data

  def get_rides_segments(self, driver_id, version=1):
    for ride_id in range(1, 201):
      yield self.get_ride_segments(driver_id, ride_id, version=version)

  def write_ride_segments(self, driver_id, ride_id, lengths, times, angles, version=1):
    filename = '%s/%s_%s.csv' % (settings.SEGMENTS_FOLDER[version], driver_id, ride_id)
    f = open(filename, 'w')
    f.write('%s\n' % util.get_list_string(lengths))
    f.write('%s\n' % util.get_list_string(times))
    f.write('%s' % util.get_list_string(angles))
    f.close()

  def skip_segment(self, driver_id, ride_id, version=1):
    filename = '%s/%s_%s.csv' % (settings.SEGMENTS_FOLDER[version], driver_id, ride_id)
    return os.path.isfile(filename)

  def get_random_drivers(self, size, seed, exception):
    # old version for small samples (without replacement)
    if size <= 2000:
      sample = [settings.DRIVER_IDS[i] for i in seed.sample(xrange(len(settings.DRIVER_IDS)), size+1)]
      if exception in sample:
        sample.remove(exception)
      else:
        sample = sample[:-1]
      return sample

    # new version - large numbers, with replacement
    sample = []
    while len(sample) < size:
      new = settings.DRIVER_IDS[seed.randint(0, len(settings.DRIVER_IDS) - 1)]
      if new != exception:
        sample.append(new)
    return sample


  def get_rides_split(self, driver_id, size_train, segments=False, version=1):
    seed = random.Random(x=driver_id)
    if not segments:
      rides = list(self.get_rides(driver_id))
    else:
      rides = list(self.get_rides_segments(driver_id, version=version))

    split_train = set([i for i in seed.sample(xrange(200), size_train)])
    rides_train = [rides[i] for i in split_train]
    rides_test = [rides[i] for i in xrange(200) if i not in split_train]
    return rides_train, rides_test

  def get_random_rides(self, size, driver_id, seed=None, segments=False, version=1):
    if not seed:
      seed = random.Random(x=driver_id)
    drivers = self.get_random_drivers(size, seed, driver_id)
    for driver_id in drivers:
      ride_id = seed.randint(1, 200)
      if not segments:
        yield self.get_ride(driver_id, ride_id)
      else:
        yield self.get_ride_segments(driver_id, ride_id, version=version)


