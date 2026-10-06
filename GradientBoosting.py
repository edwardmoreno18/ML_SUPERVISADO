import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# 1. Crear dataset
data = {
    "edad": [25, 30, 35, 40, 45, 50, 23, 28],
    "ingreso": [2000, 2500, 3000, 3500, 4000, 4500, 1800, 2200],
    "credito": [0, 0, 1, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

# 2. Variables
X = df[["edad", "ingreso"]]
y = df["credito"]

# 3. Modelo
modelo = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 4. Entrenamiento
modelo.fit(X, y)

# 5. Predicción para nuevo cliente
nuevo_cliente = pd.DataFrame({
    "edad": [18],
    "ingreso": [2800]
})

prediccion = modelo.predict(nuevo_cliente)
probabilidad = modelo.predict_proba(nuevo_cliente)

# 6. Resultados
print("Clase predicha:", prediccion[0])
print("Probabilidad de NO crédito:", probabilidad[0][0])
print("Probabilidad de crédito:", probabilidad[0][1])

print("\n¿Crédito aprobado?")
if prediccion[0] == 1:
    print("Aprobado")
else:
    print("Rechazado")