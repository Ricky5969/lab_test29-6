import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

Path("data").mkdir(exist_ok=True)

expdata = load_breast_cancer(as_frame=True)
df = expdata.frame

df.to_csv("data/breast-cancer.csv", index=False)

print("Dataset saved to data/breast-cancer.csv")
