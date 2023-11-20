import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms


data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head(5))

print(data.shape)
print(data.describe())

# En este caso, supongamos que consideramos valores atípicos aquellos por encima de 2 desviaciones estándar de la media
mean_price = data['price'].mean()
std_price = data['price'].std()
data = data[(data['price'] <= mean_price + 2 * std_price) & (data['price'] >= mean_price - 2 * std_price)]

# Lidiar con datos faltantes
# Puedes eliminar filas con datos faltantes o imputar valores, dependiendo de tu estrategia
data = data.dropna()  # Elimina filas con datos faltantes

print(data.shape)


# Seleccionar las variables independientes relevantes
# Supongamos que seleccionamos 'horsepower', 'curbweight', 'enginesize' como variables independientes
X = data[['horsepower', 'curbweight', 'enginesize']]

# Codificar variables categóricas
# Supongamos que queremos codificar la variable 'fueltype' usando Label Encoding
label_encoder = LabelEncoder()
data['fueltype'] = label_encoder.fit_transform(data['fueltype'])

# Normalizar variables numéricas
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Ahora, X contiene las variables independientes seleccionadas y normalizadas, y df contiene el DataFrame limpio y procesado.

from sklearn.linear_model import LinearRegression
# Carga los datos desde un DataFrame (asegúrate de que ya hayas realizado la preparación de datos)
# Supongamos que ya tienes las variables independientes en X y la variable de destino (precio) en y
# X y y deben ser matrices NumPy
X = np.array(X)
y = np.array(data['price'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal múltiple
model = LinearRegression()

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

pickle.dump(model, open("model.pkl", "wb"))



# Evaluación del modelo

feature_names = ['Horsepower', 'Curbweight', 'Enginesize']  # Reemplaza con tus nombres de variables

from sklearn.metrics import mean_squared_error, r2_score

# Calcula el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calcula el coeficiente de determinación (R^2)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R^2): {r2:.2f}")

# Interpreta los coeficientes del modelo
coeficientes = model.coef_
intercepto = model.intercept_

print("Coeficientes del modelo:")
for i, coef in enumerate(coeficientes):
    print(f"{feature_names[i]}: {coef:.2f}")

print(f"Intercepto: {intercepto:.2f}")