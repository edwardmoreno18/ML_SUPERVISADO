# Importar librerías
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import matplotlib.pyplot as plt

# 1. Crear el dataset
data = {
    "edad": [25, 30, 35, 40, 45, 50, 23, 28],
    "ingreso": [2000, 2500, 3000, 3500, 4000, 4500, 1800, 2200],
    "compra": [0, 0, 1, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

# Mostrar los datos
print("Dataset:")
print(df)

# 2. Definir variables independientes y dependiente
X = df[["edad", "ingreso"]]
y = df["compra"]

# 3. Crear el modelo de árbol de decisión
modelo = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# 4. Entrenar el modelo
modelo.fit(X, y)

# 5. Realizar una predicción

# --------------------------------------------------
# 2. Definir variables independientes y dependiente
# --------------------------------------------------

X = df[["edad", "ingreso"]]
y = df["compra"]


# --------------------------------------------------
# 3. Crear el modelo de árbol de decisión
# --------------------------------------------------

modelo = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


# --------------------------------------------------
# 4. Entrenar el modelo
# --------------------------------------------------

modelo.fit(X, y)


# --------------------------------------------------
# 5. Realizar una predicción
# --------------------------------------------------

nuevo = pd.DataFrame({
    "edad": [38],
    "ingreso": [2800]
})

prediccion = modelo.predict(nuevo)

print("\nPredicción:")
print(prediccion)

if prediccion[0] == 1:
    print("Resultado: Compra")
else:
    print("Resultado: No compra")

# 6. Visualizar el árbol de decisión
fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
plot_tree(
    modelo,
    feature_names=["Edad", "Ingreso"],
    class_names=["No compra", "Compra"],
    filled=True,
    rounded=True,
    # Ocultar información que no es necesaria
    impurity=False,
    # Mostrar proporción de muestras
    proportion=True,
    # Número de decimales
    precision=1,
    # Tamaño de letra
    fontsize=11,
    ax=ax
)

# Título
ax.set_title(
    "Árbol de decisión: Predicción de compra",
    fontsize=16,
    pad=20
)

# Ajustar automáticamente los elementos
plt.tight_layout()

# Mostrar gráfico
plt.show()
