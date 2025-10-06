import pandas as pd
import numpy as np

print("median initial")
print(pd.__version__)

df = pd.read_csv('car_fuel_efficiency.csv')

print("unique fuel types")
print(df.fuel_type.unique())

print("Asia origin max fuel efficiency")
filtered_df_by_origin = df[df['origin'] == 'Asia']
print(filtered_df_by_origin.fuel_efficiency_mpg.max())

print("count of columns with missing values")
print(len([col for col in df.columns if df[col].isnull().any()]))

print("most frequent horsepower value")
mostFrequent = df.horsepower.value_counts().idxmax()
print(mostFrequent)

print("median initial")
print(df.horsepower.median())

hp = df.horsepower
#print(hp)
hp.fillna(mostFrequent, inplace=True)
#print(hp)
print("median after filling missing values")
print(hp.median())


filtered_df_by_origin_columns = filtered_df_by_origin[['vehicle_weight', 'model_year']].head(7)

print(filtered_df_by_origin_columns)

X = filtered_df_by_origin_columns.to_numpy()
print(X)

XT = X.T

XTX = XT @ X

print(XTX)

XTX_inv = np.linalg.inv(XTX)

print(XTX_inv)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ XT @ y

print(w.sum())