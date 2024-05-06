import pandas as pd
import numpy as np

def convert_to_float(value):
    return float(value.replace(',', '.'))

try:
    data = pd.read_csv('6.csv', sep=',', dtype=str, header=None)
    print("Dane wczytane pomyślnie.")
except Exception as e:
    print("Nie udało się wczytać danych:", e)
    exit()

if data.shape[1] == 1:
    data = data[0].str.split(',', expand=True)

data = data.applymap(convert_to_float)

data = data.T

print("Dane po transpozycji:", data.head())
print("Rozmiar danych po transpozycji:", data.shape)

X = data.values
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Macierz kowariancji:", cov_matrix)
print("Wartości własne:", eigenvalues)
print("Wektory własne:", eigenvectors)
