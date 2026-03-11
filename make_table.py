import pandas as pd

df = pd.read_csv("results.csv")

print("\n=== Final Round Performance ===\n")
print(df.tail(1))

print("\n=== Full FL Progress Table ===\n")
print(df)
