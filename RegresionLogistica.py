import pandas as pd
from sklearn.linear_model import LogisticRegression

# Crear dataset
data = {
    "edad":[25,30,35,40,45,50,23,28],
    "ingreso":[2000,2500,3000,3500,4000,4500,1800,2200],
    "compra":[0,0,1,1,1,1,0,0]
}

df = pd.DataFrame(data)

# Variables
X = df[["edad","ingreso"]]
y = df["compra"]

# Modelo
modelo = LogisticRegression()

# Entrenamiento
modelo.fit(X,y)

# Predicción
nuevo_cliente = pd.DataFrame({
    "edad":[38],
    "ingreso":[2800]
})
# Predicción
prediccion  = modelo.predict(nuevo_cliente)

# Probabilidad
probabilidad = modelo.predict_proba(nuevo_cliente)

print("Clase predicha:", prediccion[0])
print("Probabilidad de NO comprar:", probabilidad[0][0])
print("Probabilidad de COMPRAR:", probabilidad[0][1])

print("¿Crédito aprobado?", prediccion [0])