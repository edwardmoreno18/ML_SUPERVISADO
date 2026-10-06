from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Crear dataset
data = {
    "edad":[25,30,35,40,45,50,23,28],
    "ingreso":[2000,2500,3000,3500,4000,4500,1800,2200],
    "compra":[0,0,1,1,1,1,0,0]
}

df = pd.DataFrame(data)

# 2. Definir variables
X = df[["edad","ingreso"]]
y = df["compra"]

# 3. Crear modelo
modelo = RandomForestClassifier(n_estimators=10, random_state=42)

# 4. Entrenar modelo
modelo.fit(X,y)

# 5. Nuevo dato
nuevo = pd.DataFrame({"edad":[38],"ingreso":[2800]})

# 6. Predicción
prediccion = modelo.predict(nuevo)
print("Predicción:")
if prediccion[0] == 1:
    print("Resultado: Compra")
else:
    print("Resultado: No compra")