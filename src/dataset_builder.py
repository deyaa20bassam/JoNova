import pandas as pd

df = pd.read_csv("landmarks.csv")

print("Shape:", df.shape)

print("\nFirst rows:")
print(df.head())
