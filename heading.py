import numpy as np
import matplotlib.pyplot as plt

from data_access import DataAccess
import util

def get_angle(angle_window):
  angle_window = np.array(angle_window)
  normalized = angle_window.T / np.maximum(np.linalg.norm(angle_window, axis=1), 0.1)
  average = np.mean(normalized, axis=1)
  angle = util.get_angle([1,0], [0,0], average)
  if average[1] < 0:
    angle = 360 - angle
  return angle

def get_ride_heading(ride, variations=False, moving_average_window=3, stops=False, version=1):
  '''
  I don't know exactly what this does. I was drunk when I wrote it.
  '''
  ride = np.array(ride)
  if moving_average_window:
    ride = util.movingaverage(ride, moving_average_window)

  ROLL_STEP = 3
  ride2 = np.array(ride)
  ride1 = np.roll(ride2, ROLL_STEP, axis=0)
  l = len(ride)
  ride0 = np.hstack((np.ones((l,1)), np.zeros((l,1))))

  ride0 = ride0[ROLL_STEP:]
  ride1 = ride1[ROLL_STEP:]
  ride2 = ride2[ROLL_STEP:]

  a1 = ride0
  a2 = ride2 - ride1

  distances1 = np.linalg.norm(a1, axis=1)
  distances = np.linalg.norm(a2, axis=1)

  x = (a1 * a2).sum(1) / np.maximum(distances1 * distances, 0.1)
  y = np.sign(a2[:,1])
  np.seterr(all='ignore')
  angles = np.arccos(x) * 180 / np.pi
  np.seterr(all='print')
  angles[y<0] = 360 - angles[y<0]
  angles[distances < 2] = np.nan

  is_stopped = []
  angle = []
  angle_window = []
  WINDOW_SIZE = 2
  MIN_SPEED = 2
  direction = []
  for i, dist in enumerate(distances):
    if dist > MIN_SPEED:
      angle_window.append(a2[i])
      angle_window = angle_window[-WINDOW_SIZE:]
      if len(angle_window) < WINDOW_SIZE:
        direction.append(np.nan)
      else:
        d = get_angle(angle_window[-WINDOW_SIZE/2:]) - get_angle(angle_window[:WINDOW_SIZE/2])
        if d > 180:
          d -= 360
        if d < -180:
          d += 360
        direction.append(d)
    else:
      direction.append(np.nan)

    if dist < MIN_SPEED or len(angle_window) < WINDOW_SIZE:
      is_stopped.append(True)
    else:
      is_stopped.append(False)
      angle.append(get_angle(angle_window))

  windows = []
  current_window = []
  current_window_type = 0
  for i in range(len(direction)):
    if np.isnan(direction[i]):
      current_window = []
      windows.append(0)
      continue

    current_window.append(direction[i])
    current_window = current_window[-4:]
    t = np.mean(current_window)
    if current_window_type == 0:
      if np.abs(t) > 3:
        current_window_type = np.sign(t)
    else:
      if np.sign(current_window[-1]) != current_window_type:
        current_window_type = 0

    windows.append(current_window_type)

  windows = windows[2:] + [0, 0]
  sw = True
  while sw:
    sw = False
    for i in range(1, len(windows) - 1):
      if windows[i] != windows[i-1] and windows[i] != windows[i+1]:
        windows[i] = windows[i+1]
        sw = True
    for i in range(3, len(windows)):
      if windows[i-3] != windows[i-2] and \
          windows[i-2] == windows[i-1] and \
          windows[i-1] != windows[i]:
        windows[i-2] = windows[i-3]
        windows[i-1] = windows[i]
        sw = True

  description = []
  current_window_type = 'stop' if stops else 'straight'
  new_type = current_window_type
  current_window_length = 0
  for i in range(1, len(windows)):
    if stops:
      if is_stopped[i]:
        if current_window_type != 'stop':
          new_type = 'stop'

      if windows[i] == 0 and not is_stopped[i]:
        if current_window_type != 'straight':
          new_type = 'straight'
    else:
      if windows[i] == 0 or is_stopped[i]:
        if current_window_type != 'straight':
          new_type = 'straight'

    if windows[i] == 1:
      if current_window_type != 'left':
        new_type = 'left'

    if windows[i] == -1:
      if current_window_type != 'right':
        new_type = 'right'

    if new_type == current_window_type:
      current_window_length += 1
    else:
      if current_window_length:
        description.append([
            current_window_type,
            current_window_length,
            util.euclidian_distance(ride2[i-current_window_length], ride2[i]),
            -np.sum(direction[i-current_window_length : i-1])
        ])
      current_window_type = new_type
      current_window_length = 0

  i = 0
  while i < len(description) - 1:
    if description[i][0] in ['right', 'left'] and np.abs(description[i][3]) < 15:
      description[i+1][2] += description[i][2]
      description.pop(i)
    else:
      i += 1

  i = 0
  while i < len(description) - 1:
    if description[i][0] == description[i+1][0]:
      description[i+1][1] += description[i][1]
      description[i+1][2] += description[i][2]
      description[i+1][3] += description[i][3]
      description.pop(i)
    else:
      i += 1

  # convert to words
  DIST_TH = [0, 10, 50, 100, 250, 500, 1500]
  if version == 1:
    TIME_TH = [0, 2, 8, 50]
    ANGLE_TH = [0, 35, 70, 110, 145]
  else:
    TIME_TH = [0, 8, 50]
    ANGLE_TH = [0, 10, 30, 55, 80, 110, 145]
  mirrored = {'straight': 'straight', 'left': 'right', 'right': 'left', 'stop': 'stop'}
  words_original = []
  words_mirror = []
  for row in description:
    if row[0] == 'stop':
      v = np.digitize([row[1]], TIME_TH)[0]
    elif row[0] == 'straight':
      v = np.digitize([row[2]], DIST_TH)[0]
    else:
      v = np.digitize([np.abs(row[3])], ANGLE_TH)[0]

    word = '%s_%s' % (row[0], v)
    words_original.append(word)

    word = '%s_%s' % (mirrored[row[0]], v)
    words_mirror.append(word)

  if variations:
    words_inverted = list(reversed(words_mirror))
    words_mirror_inverted = list(reversed(words_original))
    return [words_original, words_mirror, words_inverted, words_mirror_inverted]

  return words_original

