import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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

# Estandarización
scaler = StandardScaler()

X_escalado = scaler.fit_transform(X)

# 3. Crear modelo
modelo = KNeighborsClassifier(n_neighbors=3)

# 4. Entrenar modelo
modelo.fit(X_escalado,y)

# 5. Nuevo dato
nuevo = pd.DataFrame({"edad":[38],"ingreso":[2800]})

nuevo_escalado = scaler.transform(nuevo)

# 6. Predicción
print("Predicción:", modelo.predict(nuevo_escalado))

