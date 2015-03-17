import logging
import multiprocessing
import itertools

import numpy as np

from model_run import run_model, test_model
import model_run
import model_def
import settings
import util

logging.root.setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

if __name__ == '__main__':
  logging.info('starting main.py')

  #run_model((100, 203, model_def.Model_GBC, model_run.get_data_accel_v2_svd, 1)); raise Exception

  pool = multiprocessing.Pool(processes=1)
  results = pool.map(
      run_model,
      map(lambda x: (100, x, model_def.Model_LR2, model_run.get_data_movements_accel, 1), settings.DRIVER_IDS[:10])
  )
  predictions = np.array(list(itertools.chain(*[r[0] for r in results])))
  testY = list(itertools.chain(*[r[-1] for r in results]))
  logging.info(util.compute_auc(testY, predictions))
