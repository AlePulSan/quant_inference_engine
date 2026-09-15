import pandas as pd
import numpy as np
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# 1. Carga de datos
DATA_PATH = "data/datos_mercado.csv"
MODEL_PATH = "models/quant_model.onnx"

print("Cargando datos crudos...")
df = pd.read_csv(DATA_PATH)

if len(df) < 1000:
    print("Error: Necesitas al menos 100 transacciones. Deja el WebSocket corriendo más tiempo.")
    exit()

# 2. Feature Engineering (Creación de variables predictivas)
print(f"Procesando {len(df)} transacciones...")
# Ordenar por tiempo por si acaso
df = df.sort_values('timestamp')

# Calcular retornos y medias móviles rápidas
df['returns'] = df['price'].pct_change()
df['ma_5'] = df['price'].rolling(window=5).mean()
df['ma_15'] = df['price'].rolling(window=15).mean()

# Variable Objetivo (Target): 1 si el precio sube en el siguiente tick, 0 si baja o se mantiene
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)

# Limpieza de nulos por los cálculos
df = df.dropna()

# Seleccionar features para el modelo
features = ['price', 'size', 'returns', 'ma_5', 'ma_15']
X = df[features].values.astype(np.float32) # ONNX prefiere float32
y = df['target'].values

# 3. Entrenamiento del Modelo (XGBoost)
print("Entrenando clasificador XGBoost...")
model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X, y)

accuracy = model.score(X, y)
print(f"Precisión del modelo en entrenamiento: {accuracy:.2f}")

# 4. Exportación a ONNX
print("Convirtiendo modelo a formato ONNX...")
# Definimos el tensor de entrada (5 variables flotantes)
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

with open(MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"¡Modelo exportado con éxito a {MODEL_PATH}!")