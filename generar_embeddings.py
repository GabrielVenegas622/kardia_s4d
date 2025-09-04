import torch
import numpy as np
from tqdm import tqdm
# Asegúrate de importar tu S4Model y MyDataset desde tus otros archivos
from Train import S4Model, MyDataset 

# --- CONFIGURACIÓN ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = './s4_results/S4D/model.pt' # Ruta a tu modelo S4D-ECG entrenado
OUTPUT_FILE = './s4_results/embeddings/ecg_embeddings_testset.pt' # Archivo de salida

# --- CARGAR MODELO S4D-ECG ---
print("Cargando modelo S4D-ECG...")
# Recrea la arquitectura del modelo (asegúrate de que los parámetros coincidan con tu entrenamiento)
model = S4Model(d_input=12, d_output=8, d_model=128, n_layers=4)
# Carga los pesos entrenados
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()
print("Modelo cargado.")

# --- CARGAR DATOS ---
# Carga los datos que quieres explicar (p. ej., el conjunto de prueba)
# (Este código es de tu script Train.py)
import h5py
import pandas as pd
with h5py.File('x.hdf5', 'r') as f:
    X = f['tracings'][:]
y = pd.read_csv('y.csv').values
testset = MyDataset(X[-500:], y[-500:])
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
print("Datos de prueba cargados.")

# --- GENERAR Y GUARDAR EMBEDDINGS ---
all_embeddings = []
all_original_signals = []
all_labels = []

print("Generando embeddings...")
with torch.no_grad():
    for inputs, targets in tqdm(testloader):
        inputs = inputs.to(DEVICE)
        
        # Obtén la predicción y los embeddings de tu modelo
        print(model(inputs))
        prediction, embedding_secuencia = model(inputs)
        
        # Guarda los resultados (moviéndolos a la CPU para almacenarlos)
        all_embeddings.append(embedding_secuencia.cpu())
        all_original_signals.append(inputs.cpu())
        all_labels.append(targets.cpu())

# Concatena los resultados de todos los batches
all_embeddings = torch.cat(all_embeddings, dim=0)
all_original_signals = torch.cat(all_original_signals, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print(f"Embeddings generados. Forma: {all_embeddings.shape}")

# Guarda todo en un único archivo
torch.save({
    'embeddings': all_embeddings,
    'original_signals': all_original_signals,
    'labels': all_labels
}, OUTPUT_FILE)

print(f"¡Éxito! Embeddings guardados en '{OUTPUT_FILE}'")
