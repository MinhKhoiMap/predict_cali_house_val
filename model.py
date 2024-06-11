from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import sys
import json
import ast

models = {
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(algorithm='brute', weights='distance'),
    "RandomForestRegressor": RandomForestRegressor(max_features='sqrt', min_samples_leaf=2),
}

class Stacking():
  def __init__(self, baseModels=models, meta_model=LinearRegression(), dataset=pd.notnull):
    self.models = baseModels
    self.meta_model = meta_model
    self.ds = dataset
    self.X = self.ds.drop(columns="median_house_value", axis = 1)
    self.y = self.ds['median_house_value']
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 42)

  # Cross validation để chia bộ dữ liệu thành các phần train và test nhỏ hơn
  def split(self):
    partXs = []
    partYs = []
    sampled_set = []
    partX_size = len(self.X_train) // 10 if len(self.X_train) & 0 else len(self.X_train) // 10 + 1
    partY_size = len(self.y_train) // 10 if len(self.y_train) & 0 else len(self.y_train) // 10 + 1

    for i in range(10):
      start = i * partX_size
      end = (i + 1) * partX_size
      partXs.append(self.X_train[start:end])
      partYs.append(self.y_train[start:end])

    for i in range(10):
      trainingX_set = partXs[0:i] + partXs[i+1:]
      trainingY_set = partYs[0:i] + partYs[i+1:]
      testX_set = partXs[i]

      sampled_set.append({"trainingX_set": pd.concat(trainingX_set, ignore_index=True), "testX_set": testX_set, "trainingY_set": pd.concat(trainingY_set, ignore_index=True)})
    return sampled_set

  # Thực hiện stacking
  def fit(self):
    sampled_set = self.split()

    for name, model in self.models.items():
      new_train_set = []
      new_test_set = []
      for s in sampled_set:
        X_train = s['trainingX_set']
        X_test = s['testX_set']
        y_train = s['trainingY_set']

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        new_train_set.append(y_pred)

      new_test_set.append(pd.DataFrame(model.predict(self.X_test)))
      X_meta = np.concatenate(new_train_set).tolist()
      # Add new columns to X_train (stack y_pred of model as a feature)
      self.X_train[name] = X_meta

    self.meta_model.fit(self.X_train, self.y_train)

  def predict(self, X_new):
    X_new_meta = X_new.copy()
    for name, model in self.models.items():
      X_new_meta[name] = model.predict(X_new)

    return self.meta_model.predict(X_new_meta)


# df = pd.read_csv("./file3.csv")

# stacking = Stacking(dataset=df)
# stacking.fit()

# ref_cols = list(df.columns)
# target = 'median_house_value'

# joblib.dump(value=[stacking, ref_cols, target], filename="./model.pkl")

input = pd.read_json(sys.argv[1], orient='index').T
stacking, ref_cols, target = joblib.load("./model.pkl")

prediction = stacking.predict(input).tolist()

print(json.dumps(prediction))


sys.stdout.flush()