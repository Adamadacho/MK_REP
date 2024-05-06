import pandas as pd
import numpy as np

data = pd.read_csv('war6.csv', sep=';')

print(data.dtypes)

X = data[['x1', 'x2']].astype(float) 
y = data['y'].str.replace(',', '.').astype(float) 

X = np.column_stack([np.ones(len(X)), X])

X_pinv = np.linalg.pinv(X)

coefficients = X_pinv.dot(y)

print("Współczynniki modelu regresji to:", coefficients)
