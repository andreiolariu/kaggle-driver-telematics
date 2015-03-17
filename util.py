import math
import os.path
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter
from sklearn.metrics import roc_curve, auc

import settings

OUTLIER_LIMIT = 60
FLOAT_ERROR = 0.000001

def movingaverage(interval, window_size):
  window = np.ones(int(window_size))/float(window_size)
  return np.vstack((
      np.convolve(interval[:,0], window, 'same'),
      np.convolve(interval[:,1], window, 'same'),
  )).T

def get_list_string(l):
  return ','.join([str(e) for e in l])

def compute_auc(y1, y2):
  fpr, tpr, thresholds = roc_curve(y1, y2)
  roc_auc = auc(fpr, tpr)
  return roc_auc

def view_rides(*rides):
  colors = ['b', 'r', 'g', 'm', 'y', 'c', 'k']
  for i, ride in enumerate(rides):
    plt.plot([p[0] for p in ride], [p[1] for p in ride], '%s-' % colors[i % len(colors)])
  plt.show()

def euclidian_distance(p1, p2):
  return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def euclidian_distances(ride):
  return [euclidian_distance(ride[i], ride[i+1]) for i in xrange(len(ride) - 1)]

def view_ride_speed(ride):
  sm_ride = savgol_filter(np.array(ride).T, 7, 2).T
  distances = euclidian_distances(ride)
  #smoothed = [np.mean(distances[max(0, i-1):min(i+2, len(distances))]) for i in range(len(distances))]
  #smoothed = np.array(smoothed)
  smoothed = euclidian_distances(sm_ride)
  acc = np.hstack((smoothed, [0])) - np.hstack(([0], smoothed))
  acc = acc[1:-1]
  plt.plot(range(len(distances)), distances, 'b-')
  plt.plot(range(len(smoothed)), smoothed, 'r-')
  plt.plot(range(len(acc)), acc, 'g-')
  plt.plot(range(len(distances)), [0] * len(distances), 'm-')
  plt.show()

def get_ride_histograms(distances, normalized=False, version=1):
  numbers1 = np.array(distances)
  numbers2 = (np.hstack((numbers1, [0])) - np.hstack(([0], numbers1)))[1:-1]

  if version == 1:
    hists = [
        np.histogram(numbers1, bins=range(0, 50, 4))[0],
        np.histogram(numbers1[numbers1 < 20], bins=range(0, 20, 2))[0],
        np.histogram(numbers2[-(numbers2>-4) -(numbers2<3)], bins=[-4 + i * 0.7 for i in range(10)])[0],
    ]
  else:
    hists = [
        np.histogram(numbers1, bins=range(0, 40, 4))[0],
        np.histogram(numbers1, bins=range(0, 20, 1))[0],
        np.histogram(numbers2, bins=[-100] + [-4 + i * 0.6 for i in range(14)] + [100])[0],
    ]

  if normalized:
    hists = [
        hists[0] / (len(numbers1) + 1.0),
        hists[1] / (len(numbers1) + 1.0),
        hists[2] / (len(numbers2) + 1.0),
    ]

  return list(itertools.chain(*hists))

def get_g_forces(ride, distances=None):
  if distances is None:
    distances = np.array(euclidian_distances(ride))
  angles = [get_angle(ride[i-2], ride[i-1], ride[i]) for i in range(2, len(ride))]
  g_forces = [(180-angles[i-1]) * (distances[i-1] + distances[i]) for i in range(1, len(distances))]
  return np.array(g_forces)

def get_g_forces_v2(ride):
  distances = np.array(euclidian_distances(ride))
  lateral_g_forces = get_g_forces(ride, distances=distances)
  acc = np.hstack((distances, [0])) - np.hstack(([0], distances))
  acc = acc[1:-1]
  distances = distances[1:]
  forward_g_forces = distances * acc

  LAT_TH = [1, 5, 10, 30, 70, 110, 150]
  FW_TH = [-30, -15, -7, -3, -1, 1, 3, 7, 15, 30]
  DIST_TH = [1, 3, 8, 13, 20, 35]

  # print np.percentile(forward_g_forces, [1, 5, 25, 75, 95, 99])
  # print ''

  lateral_g_forces = np.digitize(lateral_g_forces, LAT_TH)
  forward_g_forces = np.digitize(forward_g_forces, FW_TH)
  distances = np.digitize(distances, DIST_TH)

  g_forces = np.vstack((distances, lateral_g_forces, forward_g_forces)).transpose()
  g_force_string = ' '.join(['%s_%s_%s' % (m[0], m[1], m[2]) for m in g_forces])
  return g_force_string

def get_g_forces_v3(ride, step=5):
  ride2 = np.array(ride)
  ride1 = np.roll(ride2, step, axis=0)
  ride0 = np.roll(ride1, step, axis=0)

  ride0 = ride0[step*2:]
  ride1 = ride1[step*2:]
  ride2 = ride2[step*2:]

  a1 = ride1 - ride0
  a2 = ride2 - ride1

  distances1 = np.linalg.norm(a1, axis=1)
  distances2 = np.linalg.norm(a2, axis=1)
  distances = distances1 + distances2

  np.seterr(all='ignore')
  angles = np.arccos((a1 * a2).sum(1) / (distances1 * distances2))
  np.seterr(all='print')
  angles[distances1 < 0.5] = 0
  angles[distances2 < 0.5] = 0
  angles = angles * 180 / math.pi

  lateral_g_forces = angles * distances

  acc = distances2 - distances1
  forward_g_forces = acc * distances

  LAT_TH = [2, 33, 88, 164, 524, 1275, 1693, 2615, 3996]
  FW_TH = [-3952, -1963, -1081, -576, 0, 652, 1034, 1718, 3279]
  DIST_TH = [1, 47, 108, 146, 200, 250]

  lateral_g_forces = np.digitize(lateral_g_forces, LAT_TH)
  forward_g_forces = np.digitize(forward_g_forces, FW_TH)
  distances = np.digitize(distances, DIST_TH)

  g_forces = np.vstack((distances, lateral_g_forces, forward_g_forces)).transpose()
  g_force_string = ' '.join(['%s_%s' % (m[0], m[1]) for m in g_forces])
  return g_force_string

def get_g_forces_v4(ride, version=1):
  ride = np.array(ride)
  ride = savgol_filter(ride.T, 7, 3).T

  # http://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
  dx_dt = np.gradient(ride[:, 0])
  dy_dt = np.gradient(ride[:, 1])
  velocity = np.vstack((dx_dt, dy_dt)).T
  ds_dt = np.linalg.norm(velocity, axis=1)
  np.seterr(all='ignore')
  tangent = np.array([1/ds_dt] * 2).T
  np.seterr(all='print')
  tangent = np.nan_to_num(tangent)
  tangent = tangent * velocity
  tangent_x = tangent[:, 0]
  tangent_y = tangent[:, 1]

  deriv_tangent_x = np.gradient(tangent_x)
  deriv_tangent_y = np.gradient(tangent_y)
  dT_dt = np.vstack((deriv_tangent_x, deriv_tangent_y)).T
  length_dT_dt = np.linalg.norm(dT_dt, axis=1)

  np.seterr(all='ignore')
  normal = np.array([1/length_dT_dt] * 2).T
  np.seterr(all='print')
  normal = np.nan_to_num(normal)
  normal = normal * dT_dt
  d2s_dt2 = np.gradient(ds_dt)
  d2x_dt2 = np.gradient(dx_dt)
  d2y_dt2 = np.gradient(dy_dt)

  np.seterr(all='ignore')
  curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
  np.seterr(all='print')
  curvature = np.nan_to_num(curvature)

  t_comp = d2s_dt2
  n_comp = curvature * ds_dt * ds_dt
  # t_component = np.array([t_comp] * 2).T
  # n_component = np.array([n_comp] * 2).T

  # acceleration = t_component * tangent + n_component * normal

  N_TH = [0.001, 0.01, 0.1, 0.5, 1]
  T_TH = [-1.5, -1, -0.5, -0.1, 0.1, 0.5, 1]
  D_TH = [1, 3, 8, 15, 30]
  C_TH = [0.001, 0.1, 0.8]

  if version == 1:
    n_comp = np.digitize(n_comp, N_TH)
    t_comp = np.digitize(t_comp, T_TH)

    acc_vectors = np.vstack((n_comp, t_comp)).transpose()
  else:
    d_comp = np.digitize(ds_dt, D_TH)
    c_comp = np.digitize(curvature, C_TH)

    acc_vectors = np.vstack((d_comp, c_comp)).transpose()

  acc_string = ' '.join(['%s_%s' % (m[0], m[1]) for m in acc_vectors])
  return acc_string

def get_distance_acc_words(ride, step=5):
  ride = np.array(ride)
  ride1 = savgol_filter(ride.T, 7, 2).T
  ride0 = np.roll(ride1, step, axis=0)[step:]
  ride1 = ride1[step:]

  distance_vectors = ride1 - ride0
  acc_vectors = np.vstack((distance_vectors, [0,0])) - \
      np.vstack(([0,0], distance_vectors))
  acc_vectors = acc_vectors[1:-1]
  distance_vectors = distance_vectors[:-1]

  distances = np.linalg.norm(distance_vectors, axis=1)
  acc_projection = (distance_vectors[:,0] * acc_vectors[:,0] + \
      distance_vectors[:,1] * acc_vectors[:,1]) / np.maximum(distances, 0.01)
  acc = np.linalg.norm(acc_vectors, axis=1)
  acc_rejection = np.sqrt(np.maximum(acc**2 - acc_projection**2,0))

  DIST_TH = np.array([0.5, 3, 8, 12, 22, 30]) * step
  PROJ_TH = [-8, -4, -1, -0.1, 0.1, 1, 3, 5]
  REJ_TH = [0.1, 0.8, 3, 6, 10]

  features = np.vstack((
      np.digitize(distances, DIST_TH),
      np.digitize(acc_projection, PROJ_TH),
      np.digitize(acc_rejection, REJ_TH)
  )).T
  features = ' '.join(['%s_%s_%s' % (f[0], f[1], f[2]) for f in features])
  return features

def get_acc4acc_words(ride, step=5, version=1):
  ride = np.array(ride)
  ride1 = savgol_filter(ride.T, 7, 2).T
  ride0 = np.roll(ride1, step, axis=0)[step:]
  ride1 = ride1[step:]

  distance_vectors = ride1 - ride0
  acc_vectors = distance_vectors[1:] - distance_vectors[:-1]
  acc4acc_vectors = acc_vectors[1:] - acc_vectors[:-1]
  acc_vectors = acc_vectors[:-1]

  acc = np.linalg.norm(acc_vectors, axis=1)
  acc4acc = np.linalg.norm(acc4acc_vectors, axis=1)
  ACC_TH = [0.1, 0.3, 0.7, 1.1, 1.6, 2.3, 3.5, 5, 6.5, 9]
  ACC4ACC_TH = [0.1, 0.3, 0.7, 1.2, 2, 2.8]

  if version == 1:
    features = np.vstack((
        np.digitize(acc, ACC_TH),
        np.digitize(acc4acc, ACC4ACC_TH),
    )).T
    features = ' '.join(['%s_%s' % (f[0], f[1]) for f in features])

  else:
    features = ' '.join(['a%s' % f for f in np.digitize(acc, ACC_TH)])

  return features

def build_features_acc(ride, version=1):
  IS_MOVING_TH = 0.7 if version == 1 else 0.3
  distances = euclidian_distances(ride)
  if version == 1:
    smoothed = [np.mean(distances[max(0, i-1):min(i+2, len(distances))] or [0]) for i in range(len(distances))]
    smoothed = np.array(smoothed)
  else:
    smoothed = np.array(distances)

  acc = np.hstack((smoothed, [0])) - np.hstack(([0], smoothed))
  acc = acc[1:-1]
  windows = []
  current_window = []
  current_window_type = 0
  for i in range(len(acc)):
    current_window.append(acc[i])
    current_window = current_window[-3:]
    t = np.mean(current_window)
    if current_window_type == 0:
      if np.abs(t) > IS_MOVING_TH:
        current_window_type = np.sign(t)
    else:
      if np.sign(current_window[-1]) != current_window_type:
        current_window_type = 0

    windows.append(current_window_type)

  windows[0] = windows[1]
  for i in range(1, len(windows) - 1):
    if windows[i] != windows[i-1] and windows[i] != windows[i+1]:
      windows[i] = windows[i+1]

  features = []
  # features to compute:
  # - percent accelerating, contant, decelerating
  # features.extend(np.histogram(windows, [-1, 0, 1, 2])[0] / (1.0 * len(windows))) # eventual normalizat
  # - average acceleration, deceleration
  mean_acc = np.mean([acc[i] for i in range(len(acc)) if windows[i] == 1] or [0])
  mean_dec = np.mean([acc[i] for i in range(len(acc)) if windows[i] == -1] or [0])
  features.extend([mean_acc, mean_dec])
  # - average acceleration, deceleration relative to speed
  SPEED_TH = list(range(0, 50, 3)) + [10000]
  for sp in range(len(SPEED_TH)-1):
    mean_acc = np.mean([acc[i] for i in range(len(acc)) if windows[i] == 1 and SPEED_TH[sp] <= smoothed[i] < SPEED_TH[sp+1]] or [0])
    mean_dec = np.mean([acc[i] for i in range(len(acc)) if windows[i] == -1 and SPEED_TH[sp] <= smoothed[i] < SPEED_TH[sp+1]] or [0])
    features.extend([mean_acc, mean_dec])
  # - average number of acc/dec changes in a trip
  changes = 0
  current_type = 1
  for w in windows:
    if w == -current_type:
      changes += 1
      current_type = w
  features.append(changes) # eventual normalizat
  features.append(1.0 * changes / len(windows))
  # - the maximum, minimum, and average values of speed multiplied by acceleration
  # - their standard deviations
  speed_times_acc = np.hstack((acc, [0])) * smoothed
  if version == 1:
    sta_hist = np.histogram(speed_times_acc, bins=range(-400, 400, 40))[0]
  else:
    sta_hist = np.histogram(speed_times_acc, bins=range(-500, 500, 20))[0]
  if version == 1:
    features.extend(sta_hist * 1.0 / len(speed_times_acc))
  else:
    features.extend(sta_hist)
  if version != 1:
    features.extend(np.percentile(speed_times_acc, [1, 3, 5, 7, 25, 50, 75, 93, 95, 97, 99]))
  features.append(np.std(speed_times_acc))

  # max acceleration per window
  max_windows = []
  current_max = 0
  is_accelerating = 0
  for i in range(len(acc)):
    if windows[i] == 1:
      is_accelerating = 1
      current_max = max(current_max, acc[i])
    else:
      if current_max:
        max_windows.append(current_max)
        current_max = 0
      is_accelerating = 0
  features.append(np.mean(max_windows or [0]))

  acc_for_acc = (np.hstack((acc, [0])) - np.hstack(([0], acc)))[1:-1]
  acc_for_acc_hist = np.histogram(acc_for_acc, bins=[-3 + i * 0.3 for i in range(21)])[0]
  if version == 1:
    features.extend(acc_for_acc_hist * 1.0 / len(acc_for_acc))
  else:
    features.extend(acc_for_acc_hist)

  # #standing start
  # standing_starts = []
  # for i in range(1, len(windows) - 4):
  #   if not (windows[i] == 1 and windows[i-1] == 0):
  #     continue
  #   if distances[i-1] > 1.5:
  #     continue
  #   d = sum(distances[i:i+5])
  #   standing_starts.append(d)
  # features.append(np.max(standing_starts or [0]))

  csw_lengths = []
  current_window_lenght = 0
  tbs_lengths = []
  current_stop_length = 0
  for i in range(1, len(windows)):
    # time at constant speed
    if windows[i] == 0 and smoothed[i] > 4:
      current_window_lenght += 1
    else:
      if current_window_lenght:
        csw_lengths.append(current_window_lenght)
        current_window_lenght = 0

    # time between stops
    if windows[i] == 0 and smoothed[i] < 3:
      current_stop_length += 1
    else:
      if current_stop_length:
        tbs_lengths.append(current_stop_length)
        current_stop_length = 0
  if version == 1:
    features.append(np.mean(csw_lengths or [0]))
  features.append(np.std(csw_lengths or [0]))
  features.append(np.mean(tbs_lengths or [0]))

  if version == 1:
    csw_length_hist = np.histogram(csw_lengths, bins=[0, 5, 15, 35, 70, 200, 10000])[0]
    features.extend(csw_length_hist * 1.0 / (len(csw_lengths) + 1))

  return features

def build_features(ride, normalized=False, version=1):
  if version == 3:
    ride = savgol_filter(np.array(ride).T, 7, 2).T

  distances = np.array(euclidian_distances(ride))
  #ride_length = distances.sum()
  #ride_speed = ride_length / len(ride)
  distances_no_stops = distances[distances > 1.5]
  #stops_ratio = len(distances[distances < 1.5]) / (len(distances) + 1.0)
  ride_length_no_stops = distances_no_stops.sum()
  ride_speed_no_stops = ride_length_no_stops / (len(distances_no_stops) + 1)
  features = [
      #ride_length,
      #ride_speed,
      ride_length_no_stops,
      #stops_ratio,
      euclidian_distance(ride[0], ride[-1]),
  ]
  if version == 1:
    features.append(ride_speed_no_stops)

  features.extend(get_ride_histograms(distances, normalized=normalized, version=version))

  g_forces = get_g_forces(ride, distances=distances)
  if version == 1:
    h_g_forces = np.histogram(g_forces, bins=range(0, 600, 50))[0]
  else:
    h_g_forces = np.histogram(g_forces, bins=range(0, 600, 10))[0]
  features.extend(h_g_forces)

  return np.array(features)

def build_features_big(ride_orig):
  ride = savgol_filter(np.array(ride_orig).T, 7, 2).T

  distances = np.linalg.norm(
      (np.vstack((ride, [0,0])) - np.vstack(([0,0], ride)))[1:-1],
      axis=1
  )
  acc = (np.hstack((distances, [0])) - np.hstack(([0], distances)))[1:-1]

  ride_length = distances.sum()
  ride_speed = ride_length / len(ride)
  distances_no_stops = distances[distances > 1.5]
  stops_ratio = len(distances[distances < 1.5]) / (len(distances) + 1.0)
  ride_length_no_stops = distances_no_stops.sum()
  ride_speed_no_stops = ride_length_no_stops / (len(distances_no_stops) + 1)
  features = [
      ride_length,
      ride_speed,
      ride_length_no_stops,
      stops_ratio,
      euclidian_distance(ride[0], ride[-1]),
  ]

  move_vectors = (np.vstack((ride, [0,0])) - np.vstack(([0,0], ride)))[1:-1]
  m1 = move_vectors[1:]
  m2 = move_vectors[:-1]

  distances1 = np.linalg.norm(m1, axis=1)
  distances2 = np.linalg.norm(m2, axis=1)

  dot_product = (m1 * m2).sum(1)
  denominator = np.maximum(distances1 * distances2, 0.01)
  angles = np.arccos(np.maximum(np.minimum(dot_product / denominator, 1.0), -1.0))
  angles = angles * 180 / math.pi

  g_forces = angles * (distances1 + distances2)
  features.extend(np.percentile(angles, [25, 50, 75, 90, 95, 99]))

  acc_for_acc = (np.hstack((acc, [0])) - np.hstack(([0], acc)))[1:-1]

  hists = [
      np.histogram(distances, bins=range(0, 50, 4))[0] / (len(distances) + 1.0),
      np.histogram(distances[distances < 20], bins=range(0, 20, 2))[0],
      np.histogram(acc, bins=[-4 + i * 0.7 for i in range(10)])[0] / (len(acc) + 1.0),
      np.histogram(g_forces, bins=range(0, 600, 10))[0],
      np.histogram(acc * distances2, bins=range(-500, 500, 20))[0],
      np.histogram(acc_for_acc, bins=[-2.1 + i * 0.3 for i in range(15)])[0] / (len(acc_for_acc) + 1.0),
  ]
  features.extend(list(itertools.chain(*hists)))

  return np.array(features)

def build_features_big_v2(ride_orig):
  ride_orig = np.array(ride_orig)
  ride = savgol_filter(ride_orig.T, 11, 2).T

  distances_orig = np.linalg.norm(
      (np.vstack((ride_orig, [0,0])) - np.vstack(([0,0], ride_orig)))[1:-1],
      axis=1
  )
  acc_orig = (np.hstack((distances_orig, [0])) - np.hstack(([0], distances_orig)))[1:-1]

  distances = np.linalg.norm(
      (np.vstack((ride, [0,0])) - np.vstack(([0,0], ride)))[1:-1],
      axis=1
  )
  acc = (np.hstack((distances, [0])) - np.hstack(([0], distances)))[1:-1]

  ride_length = distances.sum()
  ride_speed = ride_length / len(ride)
  distances_no_stops = distances[distances > 1.5]
  stops_ratio = len(distances[distances < 1.5]) / (len(distances) + 1.0)
  ride_length_no_stops = distances_no_stops.sum()
  ride_speed_no_stops = ride_length_no_stops / (len(distances_no_stops) + 1)
  features = [
      ride_length,
      ride_speed,
      ride_length_no_stops,
      stops_ratio,
      euclidian_distance(ride[0], ride[-1]),
  ]

  move_vectors = (np.vstack((ride, [0,0])) - np.vstack(([0,0], ride)))[1:-1]
  m1 = move_vectors[1:]
  m2 = move_vectors[:-1]

  distances1 = np.linalg.norm(m1, axis=1)
  distances2 = np.linalg.norm(m2, axis=1)

  dot_product = (m1 * m2).sum(1)
  denominator = np.maximum(distances1 * distances2, 0.01)
  angles = np.arccos(np.maximum(np.minimum(dot_product / denominator, 1.0), -1.0))
  angles = angles * 180 / math.pi

  g_forces = angles * (distances1 + distances2)
  features.extend(np.percentile(angles, [1, 5, 25, 50, 75, 90, 95, 99]))

  acc_for_acc = (np.hstack((acc, [0])) - np.hstack(([0], acc)))[1:-1]
  acc_for_acc_orig = (np.hstack((acc_orig, [0])) - np.hstack(([0], acc_orig)))[1:-1]

  acc5 = np.pad(acc, (0,4), 'constant') + \
      np.pad(acc, (1,3), 'constant') + \
      np.pad(acc, (2,2), 'constant') + \
      np.pad(acc, (3,1), 'constant') + \
      np.pad(acc, (4,0), 'constant')
  acc5 = acc5[5:-5]

  hists = [
      np.histogram(distances, bins=range(0, 50, 3))[0] / (len(distances) + 1.0),
      np.histogram(distances[distances < 20], bins=range(0, 20, 1))[0],
      np.histogram(acc_orig, bins=[-4 + i * 0.7 for i in range(10)])[0],
      np.histogram(acc, bins=[-4 + i * 0.7 for i in range(10)])[0],
      np.percentile(acc_orig, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
      np.percentile(acc, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
      np.histogram(g_forces, bins=range(0, 600, 10))[0],
      np.histogram(acc * distances2, bins=range(-500, 500, 20))[0],
      np.histogram(acc_orig * distances2, bins=range(-500, 500, 20))[0],
      np.histogram(acc_for_acc, bins=[-2.1 + i * 0.3 for i in range(15)])[0],
      np.percentile(acc_for_acc, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
      np.percentile(acc_for_acc_orig, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
      np.percentile(acc5, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
      np.histogram(acc_for_acc, bins=[-1.2 + i * 0.2 for i in range(12)])[0],
      np.histogram(acc_for_acc_orig, bins=[-1.2 + i * 0.2 for i in range(12)])[0],
  ]
  features.extend(list(itertools.chain(*hists)))

  for step in [10, 30, 50]:
    distances = np.linalg.norm((np.roll(ride, step) - ride)[step:], axis=1)
    features.extend(np.percentile(distances, [1, 5, 20, 50, 80, 95, 99]))

    dist_slice = distances[distances < 10 * step]
    if not len(dist_slice):
      dist_slice = [0]
    features.extend(np.percentile(dist_slice, [1, 5, 20, 50, 80, 95, 99]))

  return np.array(features)

def get_angle(p1, p2, p3):
  dot_product = (p1[0] - p2[0]) * (p3[0] - p2[0]) + (p1[1] - p2[1]) * (p3[1] - p2[1])
  denominator = max(euclidian_distance(p1, p2) * euclidian_distance(p2, p3), 0.1)

  # just in case dot_product is infinitesimaly larger than denominator
  ratio = dot_product / denominator
  if ratio > 1:
    ratio = 1
  if ratio < -1:
    ratio = -1
  angle = math.acos(ratio)

  return angle * 180 / math.pi

def bucket(values, bins, cutoff):
  bucketed = []
  diff = cutoff[1] - cutoff[0]
  for value in values:
    if value < cutoff[0]:
      bucketed.append(0)
      continue
    if value >= cutoff[1]:
      bucketed.append(bins - 1)
      continue

    ratio = (value - cutoff[0]) / diff
    bin = int(ratio * bins)
    bucketed.append(bin)
  return bucketed

def get_accelerations(ride):
  distances = euclidian_distances(ride)
  accelerations = [distances[i] - distances[i-1] for i in xrange(1, len(distances))]
  bucketed = bucket(accelerations, 10, [-2,2])
  words = ['a%s_%s' % (bucketed[i-1], bucketed[i]) for i in xrange(1, len(bucketed))]
  return words

def get_accelerations_v2(ride):
  distances = euclidian_distances(ride)
  accelerations = [distances[i] - distances[i-1] for i in xrange(1, len(distances))]
  bucketed = np.digitize(accelerations, np.array(range(-30, 30, 3)) / 10.0)
  words = ['a%s_%s' % (bucketed[i-1], bucketed[i]) for i in xrange(1, len(bucketed))]
  return words

def _get_cache_file(model, get_data, driver_id, test, repeat):
  cache_folder = settings.CACHE[repeat]
  filename = '%s/%s_%s_%s/%s.npy' % (
      cache_folder,
      'TEST' if test else 'TRAIN',
      get_data.func_name,
      model.__name__,
      driver_id
  )
  d = os.path.dirname(filename)
  if not os.path.exists(d):
    os.makedirs(d)
  return filename

def get_results(model, get_data, driver_id, test, repeat):
  filename = _get_cache_file(model, get_data, driver_id, test, repeat)
  if not os.path.isfile(filename):
    return False
  return np.load(filename)

def cache_results(model, get_data, driver_id, test, data, repeat):
  filename = _get_cache_file(model, get_data, driver_id, test, repeat)
  np.save(filename, data)

def build_features3(ride, step=5, version=1):
  if version == 3:
    ride = savgol_filter(np.array(ride).T, 7, 3).T

  ride2 = np.array(ride)
  ride1 = np.roll(ride2, step, axis=0)
  ride0 = np.roll(ride1, step, axis=0)

  ride0 = ride0[step*2:]
  ride1 = ride1[step*2:]
  ride2 = ride2[step*2:]

  a1 = ride1 - ride0
  a2 = ride2 - ride1

  distances1 = np.linalg.norm(a1, axis=1)
  distances2 = np.linalg.norm(a2, axis=1)
  distances = distances1 + distances2

  np.seterr(all='ignore')
  angles = np.arccos((a1 * a2).sum(1) / (distances1 * distances2))
  np.seterr(all='print')
  if version == 1:
    angles[distances1 < 7] = 0
    angles[distances2 < 7] = 0
  else:
    angles[distances1 < 0.5] = 0
    angles[distances2 < 0.5] = 0
  angles = angles * 180 / math.pi

  if version == 1:
    DIST_THR = np.array([1, 11, 16, 26, 36, 56, 80]) * step
    ANGL_THR = np.array([10, 30, 60, 100])
  else:
    DIST_THR = np.array([1, 3, 7, 15, 24, 35, 56, 80]) * step
    ANGL_THR = np.array([5, 15, 35, 55, 75, 95, 110])
  distances = np.digitize(distances, DIST_THR)
  angles = np.digitize(angles, ANGL_THR)

  movements = np.vstack((distances, angles)).transpose()
  movement_string = ' '.join(['%s_%s' % (m[0], m[1]) for m in movements])

  return movement_string

def build_features4(ride, step=3, version=1):
  MIN_DIST_TH = 7 if version == 1 else 0.2
  ride2 = np.array(ride)
  ride1 = np.roll(ride2, step, axis=0)
  ride0 = np.roll(ride1, step, axis=0)

  ride0 = ride0[step*2:]
  ride1 = ride1[step*2:]
  ride2 = ride2[step*2:]

  a1 = ride1 - ride0
  a2 = ride2 - ride1

  distances1 = np.linalg.norm(a1, axis=1)
  distances2 = np.linalg.norm(a2, axis=1)
  distances = distances1 + distances2
  accel = distances2 - distances1

  np.seterr(all='ignore')
  angles = np.arccos((a1 * a2).sum(1) / (distances1 * distances2))
  np.seterr(all='print')
  angles[distances1 < MIN_DIST_TH] = 0
  angles[distances2 < MIN_DIST_TH] = 0
  angles = angles * 180 / math.pi

  if version == 1:
    DIST_THR = np.array([1, 11, 16, 26, 36, 56, 80]) * step
  else:
    DIST_THR = np.array([1, 11, 25, 45, 70]) * step
  distances = np.digitize(distances, DIST_THR)
  ANGL_THR = np.array([10, 30, 60, 100])
  angles = np.digitize(angles, ANGL_THR)
  ACCEL_THR = np.array([-3, -1.5, -0.3, 0.3, 1.5, 3]) * step
  accel = np.digitize(accel, ACCEL_THR)

  movements = np.vstack((distances, angles, accel)).transpose()
  movement_string = ' '.join(['%s_%s_%s' % (m[0], m[1], m[2]) for m in movements])

  return movement_string

def get_similarities(rides):

  def get_words(ride):
    GAP = 15
    ride = [p + [i]  for i, p in enumerate(ride)] # enrich with timestamp
    #ride = rdp(ride, epsilon=20)
    lengths = [euclidian_distance(ride[i-GAP], ride[i]) for i in xrange(GAP, len(ride))]
    #times = [ride[i][2] - ride[i-1][2] for i in xrange(1, len(ride))]
    angles = [get_angle(ride[i-2 * GAP], ride[i- GAP], ride[i]) for i in xrange(2 * GAP, len(ride))]

    # # average window
    # lengths2 = []
    # for i in range(len(lengths)):
    #   start = max(i - 2, 0)
    #   end = min(i + 3, len(lengths))
    #   lengths2.append(sum(lengths[start:end]) / (end - start))
    # lengths = lengths2
    lengths = np.array(lengths)

    # bucket the values
    lengths = np.digitize(lengths * 0.3, range(0, 200 * GAP, 1 * GAP))
    #times = [int(t) for t in times]
    angles = np.digitize(angles, range(0, 180, 10))

    words = np.array([(lengths[i-GAP], angles[i-GAP], lengths[i]) for i in range(GAP, len(lengths))])

    # filter small values
    #print len(words)
    words = words[words[:,0] > 1]
    words = words[words[:,2] > 1]
    #print len(words)
    # print ''

    # ngrams
    ngrams = 3
    words = ['/'.join([str(e) for e in words[i-ngrams:i]]) for i in range(ngrams, len(words))]
    return words

  def get_dict(words):
    d = {}
    for i, w in enumerate(words):
      if w not in d:
        d[w] = []
      d[w].append(i)
    return d

  def get_sim(dict1, dict2, words1, words2):
    true = 0
    total = len(words1) * len(words2)
    for w in dict1:
      if w not in dict2:
        continue
      for p1 in dict1[w]:
        if p1 == 0:
          continue
        for p2 in dict2[w]:
          if p2 == 0:
            continue
          if words1[p1-1] == words2[p2-1]:
            true += 1
    return true * 100.0 / max(total, 1)

  processed_rides = []
  for r in rides:
    words = get_words(r)
    d = get_dict(words)
    processed_rides.append((d, words))

  results = {}
  for i1, p1 in enumerate(processed_rides):
    if i1 not in results:
      results[i1] = {}
    for i2, p2 in enumerate(processed_rides):
      if i1 >= i2:
        continue
      if i2 not in results:
        results[i2] = {}
      score = get_sim(p1[0], p2[0], p1[1], p2[1])
      results[i1][i2] = score
      results[i2][i1] = score

  # avg_results = {}
  # for k, rez in results.iteritems():
  #   avg_results[k] = sum(rez.values()) / len(rez.values())

  top_results = ['bug'] * len(results)
  for k, rez in results.iteritems():
    top_results[k] = sum(sorted(rez.values())[-10:]) / 10

  return top_results

def fft(ride):
  distances = np.array(euclidian_distances(ride))
  #acc = np.hstack((distances, [0])) - np.hstack(([0], distances))
  #acc = acc[1:-1]

  window_size = 15
  overlap = 12
  no_windows = (len(distances) - overlap) / (window_size - overlap)

  windows = np.lib.stride_tricks.as_strided(
      distances,
      shape=(no_windows, window_size),
      strides=(8 * (window_size - overlap), 8)
  )

  fourier = scipy.fft(windows) / window_size
  fourier = np.abs(fourier)

  percentiles = np.percentile(fourier, [1, 3, 5, 10, 25, 75, 90, 95, 97, 99], axis=0)
  return np.array(percentiles).flatten()

def fft_strip(ride):
  distances = np.array(euclidian_distances(ride))

  window_size = 5
  overlap = 4
  no_windows = (len(distances) - overlap) / (window_size - overlap)

  windows = np.lib.stride_tricks.as_strided(
      distances,
      shape=(no_windows, window_size),
      strides=(8 * (window_size - overlap), 8)
  )

  fourier = scipy.fft(windows) / window_size
  fourier = np.abs(fourier)
  fourier = fourier[:,1:] # remove the component with frequency 0

  percentiles = np.percentile(fourier, [1, 3, 5, 10, 25, 75, 90, 95, 97, 99], axis=0)
  return np.array(percentiles).flatten()