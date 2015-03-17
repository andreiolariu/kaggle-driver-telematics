import logging
import multiprocessing

from rdp_alg import rdp # Ramer-Douglas-Peucker Algorithm # sudo pip install rdp
import numpy as np

from data_access import DataAccess
import settings
import util

logging.root.setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

def smoothen(ride): # TODO: also try without smoothing
  for i in range(2, len(ride)):
    if util.euclidian_distance(ride[i-2], ride[i]) < max( \
        util.euclidian_distance(ride[i-2], ride[i-1]),
        util.euclidian_distance(ride[i-1], ride[i])):
      ride[i-1] = [(ride[i-2][0] + ride[i][0]) / 2, (ride[i-2][1] + ride[i][1]) / 2]
  return ride

def segment_driver(driver_id):
  ''' this generated the segments in settings.SEGMENTS_FOLDER[1] '''
  da = DataAccess()
  for ride_id_minus_1, ride in enumerate(da.get_rides(driver_id)):
    ride_id = ride_id_minus_1 + 1
    if da.skip_segment(driver_id, ride_id):
      continue

    # apply the Ramer-Douglas-Peucker algorithm
    ride = [p + [i]  for i, p in enumerate(smoothen(ride))] # enrich with timestamp
    ride = rdp(ride, epsilon=10)

    lengths = [util.euclidian_distance(ride[i-1], ride[i]) for i in xrange(1, len(ride))]
    times = [ride[i][2] - ride[i-1][2] for i in xrange(1, len(ride))]
    angles = [util.get_angle(ride[i-2], ride[i-1], ride[i]) for i in xrange(2, len(ride))]

    # bucket the values
    lengths = util.bucket(np.log(lengths), 25, [2.2,8]) # [int(l) for l in lengths]
    times = util.bucket(np.log(times), 20, [1,5.5]) # [int(t) for t in times]
    angles = util.bucket(angles, 30, [0,180]) # [int(a) for a in angles]

    # write results
    da.write_ride_segments(driver_id, ride_id, lengths, times, angles)

  logging.info('finished segmenting driver %s' % driver_id)

def segment_driver_v2(driver_id):
  ''' this generated the segments in settings.SEGMENTS_FOLDER[2] '''
  da = DataAccess()
  for ride_id_minus_1, ride in enumerate(da.get_rides(driver_id)):
    ride_id = ride_id_minus_1 + 1
    if da.skip_segment(driver_id, ride_id, version=2):
      continue

    # apply the Ramer-Douglas-Peucker algorithm
    ride = [p + [i]  for i, p in enumerate(ride)] # enrich with timestamp
    ride = rdp(ride, epsilon=4)

    lengths = [util.euclidian_distance(ride[i-1], ride[i]) for i in xrange(1, len(ride))]
    times = [ride[i][2] - ride[i-1][2] for i in xrange(1, len(ride))]
    angles = [util.get_angle(ride[i-2], ride[i-1], ride[i]) for i in xrange(2, len(ride))]

    lengths = np.histogram(lengths, bins=range(0, 700, 20) + [1000000000])[0]
    times = np.histogram(times, bins=range(0, 60, 4) + [1000000000])[0]
    angles = np.histogram(angles, bins=range(0, 181, 20))[0]

    # write results
    da.write_ride_segments(driver_id, ride_id, lengths, times, angles, version=2)

  logging.info('finished segmenting driver %s' % driver_id)

if __name__ == '__main__':
  logging.info('starting segmentation')
  pool = multiprocessing.Pool(processes=4)
  pool.map(
      segment_driver,
      settings.DRIVER_IDS
  )

