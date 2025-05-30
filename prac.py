import pandas as pd

df = pd.read_csv("static/data.csv",header=0)
print(df.columns)