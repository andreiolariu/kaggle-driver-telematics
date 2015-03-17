import itertools
import logging
import multiprocessing
import sys
import time

import numpy as np
from sklearn.linear_model import Lasso

import model_def
from model_run import run_model, test_model
import model_run
import settings
import util
from weights import WEIGHTS, MODELS, STACK

logging.root.setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

# --------------------------------------------------
# locally test ensemble with the precomputed weights

def run_ensemble(params):
  driver_id, verbose = params
  results = []
  for i, get_data, model, repeat in STACK:
    results.append((
        run_model((i, driver_id, model, get_data, repeat)),
        WEIGHTS[i]
    ))

  predictions = np.array([r[0][0] * r[1] for r in results]).sum(0)
  testY = results[0][0][1]

  if verbose >= 1:
    logging.info('finished driver %s' % driver_id)

  return predictions, testY

def train(verbose=0):
  logging.info('running ensemble')
  pool = multiprocessing.Pool(processes=4)
  drivers = settings.DRIVER_IDS #[1000:]
  results = pool.map(
      run_ensemble,
      map(lambda x: (x, verbose), drivers)
  )

  predictions = np.array(list(itertools.chain(*[r[0] for r in results])))
  testY = list(itertools.chain(*[r[1] for r in results]))

  S = sum([s[-1] for s in STACK])
  logging.info(util.compute_auc(testY, predictions))

# ----------------------------
# compute weights for ensemble

def compute_weights(params):
  driver_id, verbose, stack_option = params

  stack = STACK if stack_option == 's' else MODELS

  predictions = {}
  for i, get_data, model, repeat in stack:
    start_time = time.time()
    predictions[i], testY = run_model((i, driver_id, model, get_data, repeat))

    if verbose == 2:
      logging.info('%s: %.2f' % (i, time.time() - start_time))

  if verbose >= 1:
    logging.info('finished driver %s' % driver_id)

  return driver_id, predictions, testY

def weight_analysis(verbose=0, stack_option='s'):
  logging.info('starting ensemble weight analysis')

  stack = STACK if stack_option == 's' else MODELS

  pool = multiprocessing.Pool(processes=4)
  drivers = settings.DRIVER_IDS#[:1000]
  CUTOFF = -1
  results = pool.map(
      compute_weights,
      map(lambda x: (x, verbose, stack_option), drivers)
  )

  predictions = {}
  for i, get_data, model, _ in stack:
    predictions[i] = np.array(list(itertools.chain(*[r[1][i] for r in results])))
  testY = list(itertools.chain(*[r[2] for r in results]))

  model_names = [
      ('%s.%s.%s' % (get_data.func_name, model.__name__, i), i)
      for i, get_data, model, repeat in stack
  ]
  model_names.sort(key=lambda x: x[0])
  keys = [x[1] for x in model_names]
  model_names = [x[0] for x in model_names]

  lasso = Lasso(alpha=0.0, positive=True)
  trainX = []
  for row_id in xrange(len(testY)):
    train_row = [predictions[i][row_id] for i in keys]
    trainX.append(train_row)

  a, b = trainX[:CUTOFF], trainX[CUTOFF:]
  c, d = testY[:CUTOFF], testY[CUTOFF:]
  lasso.fit(a, c)
  pred = lasso.predict(b)
  pred_train = lasso.predict(a)
  #logging.info('auc: %s' % util.compute_auc(d, pred))

  logging.info('coefficients:')
  weights = {}
  for i, name in enumerate(model_names):
    logging.info('%s: %.3f' % (model_names[i], lasso.coef_[i]))
    weights[keys[i]] = lasso.coef_[i]

  logging.info('individual scores:')
  for i, key in enumerate(keys):
    logging.info('%s: %.3f' % (
        model_names[i],
        util.compute_auc(testY, predictions[key])
    ))

  logging.info('weights dictionary: %s' % weights)

  # and again in the end, so you don't have to scroll
  logging.info('------------')
  #logging.info('auc: %s' % util.compute_auc(d, pred))
  logging.info('auc train: %s' % util.compute_auc(c, pred_train))


# -----------------------------------------------
# prepare submission with the precomputed weights

def submit_ensemble(params):
  driver_id, verbose = params
  results = []
  for i, get_data, model, repeat in STACK:
    start_time = time.time()
    model_results = test_model((i, driver_id, model, get_data, repeat))
    results.append((model_results, WEIGHTS[i]))

    # check against nans in results
    if np.isnan(np.sum(model_results[1])):
      logging.info('nan found in results for driver %s, model %s' % (driver_id, i))

    if verbose == 2:
      logging.info('%s: %.2f' % (i, time.time() - start_time))

  predictions = np.array([r[0][1] * r[1] for r in results]).sum(0)

  if verbose >= 1:
    logging.info('finished driver %s' % driver_id)

  return driver_id, predictions

def submit(verbose=0):
  logging.info('submission ensemble')
  drivers = settings.DRIVER_IDS

  pool = multiprocessing.Pool(processes=4)
  results = pool.map(
      submit_ensemble,
      map(lambda x: (x, verbose), drivers)
  )
  results = {d: r for d, r in results}

  # write to file
  f = open('submission.csv', 'w')
  f.write('driver_trip,prob\n')
  for driver_id in drivers:
    driver_results = results[driver_id]
    for ride_id, ride_prob in enumerate(driver_results):
      f.writelines('%s_%s,%s\n' % (driver_id, ride_id + 1, ride_prob))
  f.close()

if __name__ == '__main__':
  option = sys.argv[1]

  # 0 for very little output
  # 1 for per driver output
  # 2 for per driver+model output
  verbose = 0
  if len(sys.argv) > 2:
    verbose = int(sys.argv[2])

  if option == 'train':
    train(verbose=verbose)

  elif option == 'submit':
    submit(verbose=verbose)

  elif option == 'weights':
    if len(sys.argv) < 3:
      raise Exception('stack not mentioned')
    stack_option = sys.argv[3]
    if stack_option not in ['s', 'e']: # submit or extended
      raise Exception('illegal stack option')

    weight_analysis(verbose=verbose, stack_option=stack_option)

  else:
    logging.info('invalid option')

