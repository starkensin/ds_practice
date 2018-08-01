# Random Forest
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(21)

# 유방암 데이터 로드
dataset = datasets.load_breast_cancer()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.25, random_state=21)

# (1) Random Forest 모델을 만들고 데이터에 fit 하세요.
model_randomForest = RandomForestClassifier()
model_randomForest.fit(X_train,y_train)
print(model_randomForest)

# (2) Logistic Regression 모델을 만들고 데이터에 fit 하세요.
model_logisticReg = LogisticRegression()
model_logisticReg.fit(X_train,y_train)
print(model_logisticReg)

# (3) Gaussian Naive Bayes 모델을 만들고 데이터에 fit 하세요.
model_gaussianNB = GaussianNB()
model_gaussianNB.fit(X_train,y_train)
print(model_gaussianNB)

# (4) Fitting 된 모델을 이용해 test 데이터의 label을 predict 하세요.
expected = y_test
pred_randomForest = model_randomForest.predict(X_test)
pred_logisticReg = model_logisticReg.predict(X_test)
pred_gaussianNB = model_gaussianNB.predict(X_test)

# Prediction 확인
print(metrics.classification_report(expected, pred_randomForest))
print(metrics.classification_report(expected, pred_logisticReg))
print(metrics.classification_report(expected, pred_gaussianNB))

############################

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

np.random.seed(21)

# 유방암 데이터 로드
dataset = datasets.load_breast_cancer()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.25, random_state=21)

# (1) GradientBoostingClassifier 모델을 만들고 데이터에 fit 하세요. 
model_boosting = GradientBoostingClassifier().fit(X_train,y_train)
print(model_boosting)

# (2) XGBClassifier 모델을 만들고 데이터에 fit 하세요. 
model_xgb = XGBClassifier().fit(X_train,y_train)
print(model_xgb)

# (3) Fitting 된 모델을 이용해 test 데이터의 label을 predict 하세요.
expected = y_test
pred_boosting = model_boosting.predict(X_test)
pred_xgb = model_xgb.predict(X_test)

# Prediction 확인
print(metrics.classification_report(expected, pred_boosting))
print(metrics.classification_report(expected, pred_xgb))
