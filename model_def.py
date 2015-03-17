from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.special import expit
import scipy

class Model_RFC:

  def __init__(self, trainX, trainY, seed):
    self.model = RandomForestClassifier(
        n_estimators=500,
        random_state=seed
    )
    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = trainX.toarray()
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    if type(testX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      testX = testX.toarray()
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_ABC1:

  def __init__(self, trainX, trainY, seed):
    self.model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=0.1,
        random_state=seed
    )
    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = trainX.toarray()
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    if type(testX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      testX = testX.toarray()
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_ABC2:

  def __init__(self, trainX, trainY, seed):
    self.model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
        learning_rate=0.03,
        random_state=seed
    )
    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = trainX.toarray()
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    if type(testX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      testX = testX.toarray()
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_ETC:

  def __init__(self, trainX, trainY, seed):
    self.model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=seed
    )
    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = trainX.toarray()
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    if type(testX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      testX = testX.toarray()
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_GBC:

  def __init__(self, trainX, trainY, seed):
    self.model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=seed
    )
    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = trainX.toarray()
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    if type(testX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      testX = testX.toarray()
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_GBC2:

  def __init__(self, trainX, trainY, seed):
    self.model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=5,
        subsample=0.5,
        max_features='log2',
        random_state=seed
    )
    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = trainX.toarray()
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    if type(testX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      testX = testX.toarray()
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_GBC3:

  def __init__(self, trainX, trainY, seed):
    self.model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=2,
        subsample=0.5,
        max_features='log2',
        random_state=seed
    )
    if type(trainX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      trainX = trainX.toarray()
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    if type(testX) in [scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix]:
      testX = testX.toarray()
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_KNN:

  def __init__(self, trainX, trainY, seed):
    self.model = KNeighborsClassifier(n_neighbors=7)
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_LR:

  def __init__(self, trainX, trainY, seed):
    self.model = LogisticRegression(
        C=10,
        random_state=seed
    )
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_LR2:

  def __init__(self, trainX, trainY, seed):
    self.model = LogisticRegression(
        C=0.001,
        random_state=seed
    )

    self.model.fit(trainX, trainY)

  def predict(self, testX):
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_LR3:

  def __init__(self, trainX, trainY, seed):
    self.model = LogisticRegression(
        penalty='l1',
        C=10,
        random_state=seed
    )
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_LR4:

  def __init__(self, trainX, trainY, seed):
    self.model = LogisticRegression(
        C=0.0001,
        random_state=seed
    )

    self.model.fit(trainX, trainY)

  def predict(self, testX):
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions

class Model_SVC:

  def __init__(self, trainX, trainY, seed):
    self.model = LinearSVC(
        loss='l2',
        C=0.1,
        fit_intercept=True,
        intercept_scaling=1,
        random_state=seed
    )
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    predictions = self.model.decision_function(testX)
    predictions[predictions > 100] = 100 # large number bug in expit
    return expit(predictions)

class Model_SVC2:

  def __init__(self, trainX, trainY, seed):
    self.model = SVC(
        kernel='rbf',
        C=100,
        random_state=seed,
        probability=True
    )
    self.model.fit(trainX, trainY)

  def predict(self, testX):
    predictions = self.model.predict_proba(testX)[:,1]
    return predictions
