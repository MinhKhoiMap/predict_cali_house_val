import sys
import json
import ast
import pandas as pd
import joblib

ref_cols, target = joblib.load("./frame.pkl")

print(json.dumps(ref_cols))

sys.stdout.flush()