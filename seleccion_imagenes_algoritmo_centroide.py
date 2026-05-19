# -*- coding: utf-8 -*-
"""
SELECCIÓN DE IMÁGENES GENERADAS - Versión LOCAL
Para ejecutar en Windows 

Flujo:
1. Lee imágenes reales de total_images/all_unified_images
2. Extrae características con ResNet50
3. Calcula centroides por clase
4. Procesa TODOS los GAN-outputs
5. Selecciona las TOP-N por distancia al centroide
6. Genera CSVs y guarda en salida-CNN/imagenes-seleccionadas-N/
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print(" Librerías importadas correctamente\n")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN - AJUSTA SOLO ESTOS PATHS
# ═════════════════════════════════════════════════════════════════════════════

TFG_ROOT = r'C:\Users\lucia\Desktop\TFG-DRIVE\TFG'

# Rutas base - CORREGIDAS PARA TU PC
REAL_IMAGES_FOLDER = os.path.join(TFG_ROOT, 'total_images', 'all_unified_images')
GENERATED_IMAGES_FOLDER = os.path.join(TFG_ROOT, 'todas_las_generadas')
# CSV corregido (añadido para el Paso 2)
CSV_PATH = os.path.join(GENERATED_IMAGES_FOLDER, 'metadata_todas_las_generadas.csv')
OUTPUT_ROOT_FOLDER = os.path.join(TFG_ROOT, 'salida-CNN')

# Device (GPU si está disponible)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}\n")

# Parámetros
IMAGE_SIZE = 224
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH = 256, 256, 3

# Epochs a procesar (detecta automáticamente)
EPOCHS_TO_PROCESS = [300, 310, 335, 385, 400]

# Números de imágenes a seleccionar para cada CNN
N_IMAGES_TO_SELECT_VALUES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

# Proporción train/valid en las imágenes sintéticas
TRAIN_RATIO = 0.70

print('═' * 80)
print('CONFIGURACIÓN')
print('═' * 80)
print(f'Imágenes reales: {REAL_IMAGES_FOLDER}')
print(f'Imágenes generadas: {GENERATED_IMAGES_FOLDER}')
print(f'Salida: {OUTPUT_ROOT_FOLDER}')
print(f'Epochs a procesar: {EPOCHS_TO_PROCESS}')
print(f'CNN a generar: CNN 2-11 (con {N_IMAGES_TO_SELECT_VALUES} imágenes)')
print('=' * 80 + '\n')

# ═════════════════════════════════════════════════════════════════════════════
# PASO 1: Extractor de características
# ═════════════════════════════════════════════════════════════════════════════

class DeepFeatureExtractor:
    """Extrae características con ResNet50"""
    
    def __init__(self, device='cpu'):
        print("Cargando ResNet50 preentrenado...")
        self.device = device
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        self.transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✓ ResNet50 cargado\n")
    
    def extract_features(self, image_path):
        """Extrae vector de características (2048-dim)"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
            return features.squeeze().cpu().numpy()
        except Exception as e:
            return None

extractor = DeepFeatureExtractor(device=DEVICE)

# ═════════════════════════════════════════════════════════════════════════════
# PASO 2: Cargar imágenes reales y calcular centroides
# ═════════════════════════════════════════════════════════════════════════════

print('=' * 80)
print('PASO 1: CARGAR IMÁGENES REALES Y CALCULAR CENTROIDES')
print('=' * 80 + '\n')

def load_real_images_and_compute_centroids():
    """Carga imágenes reales y calcula centroide para cada clase"""
    
    centroids = {}
    features_all = {}
    features_stats = {}
    
    for label, label_name in [(0, 'invasivo'), (1, 'in_situ')]:
        print(f"Procesando imágenes reales ({label_name})...")
        
        # BUSQUEDA FLEXIBLE DE CARPETA (Corregido)
        target_dir = None
        if os.path.exists(REAL_IMAGES_FOLDER):
            for d in os.listdir(REAL_IMAGES_FOLDER):
                if label_name.lower() in d.lower():
                    target_dir = os.path.join(REAL_IMAGES_FOLDER, d)
                    break
        
        if not target_dir:
            print(f" No se encontró carpeta para {label_name} en {REAL_IMAGES_FOLDER}")
            continue
        
        # Buscar directamente archivos en la carpeta encontrada
        image_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"  Encontradas {len(image_files)} imágenes en {os.path.basename(target_dir)}")
        
        # Extraer características
        features_list = []
        
        for img_path in tqdm(image_files[:100], desc=f"Extrayendo features ({label_name})"):
            features = extractor.extract_features(img_path)
            if features is not None:
                features_list.append(features)
        
        if not features_list:
            print(f" Error: No se extrajeron características para {label_name}")
            continue
        
        features_array = np.array(features_list)
        
        # Normalizar
        mean = features_array.mean(axis=0)
        std = features_array.std(axis=0)
        features_normalized = (features_array - mean) / (std + 1e-8)
        
        # Centroide
        centroid = features_normalized.mean(axis=0)
        
        centroids[label_name] = centroid
        features_all[label_name] = {
            'array': features_array,
            'normalized': features_normalized,
            'mean': mean,
            'std': std
        }
        
        distances_to_centroid = euclidean_distances(
            features_normalized,
            centroid.reshape(1, -1)
        ).flatten()
        
        features_stats[label_name] = {
            'n_images': len(features_list),
            'distances_mean': distances_to_centroid.mean(),
            'distances_std': distances_to_centroid.std()
        }
        
        print(f"✓ {label_name.upper()}")
        print(f"  Imágenes procesadas: {len(features_list)}")
        print(f"  Distancia promedio al centroide: {distances_to_centroid.mean():.4f}\n")
    
    return centroids, features_all, features_stats

centroids, features_real, stats_real = load_real_images_and_compute_centroids()

# ═════════════════════════════════════════════════════════════════════════════
# PASO 3: Cargar y procesar imágenes generadas (VÍA CSV CORREGIDA)
# ═════════════════════════════════════════════════════════════════════════════

print('=' * 80)
print('PASO 2: PROCESAR IMÁGENES GENERADAS (VÍA CSV)')
print('=' * 80 + '\n')

all_results = []

if os.path.exists(CSV_PATH):
    # Lectura corregida con sep=',' para evitar el KeyError 'label'
    df_meta = pd.read_csv(CSV_PATH, sep=',')
    
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Calculando distancias"):
        class_name = 'invasivo' if 'invasivo' in str(row['class']).lower() else 'in_situ'
        
        if class_name not in features_real:
            continue
            
        # Ruta corregida según tu carpeta: todas_las_generadas / clase / filename
        img_path = os.path.join(GENERATED_IMAGES_FOLDER, str(row['class']), str(row['filename']))
        
        features = extractor.extract_features(img_path)
        if features is None:
            continue
            
        # Normalizar con estadísticas de reales
        real_mean = features_real[class_name]['mean']
        real_std = features_real[class_name]['std']
        features_normalized = (features - real_mean) / (real_std + 1e-8)
        
        # Distancia al centroide
        centroid = centroids[class_name]
        distance = euclidean_distances(
            features_normalized.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0, 0]
        
        all_results.append({
            'image_path': img_path,
            'filename': row['filename'],
            'epoch': row['epoch'],
            'class': class_name,
            'label': row['label'],
            'distance': distance
        })
else:
    print(f" No se encontró el CSV en {CSV_PATH}")

print(f"\n Total imágenes procesadas: {len(all_results)}\n")

# ═════════════════════════════════════════════════════════════════════════════
# PASO 4: Crear DataFrame y ordenar por distancia
# ═════════════════════════════════════════════════════════════════════════════

print('=' * 80)
print('PASO 3: ORDENAR POR DISTANCIA')
print('=' * 80 + '\n')

df_all = pd.DataFrame(all_results)

if len(df_all) == 0:
    print(" ERROR: No se procesaron imágenes. Revisa las rutas.")
    exit(1)

print(f"Total imágenes cargadas: {len(df_all)}")
print(f"Distribución por clase:")
print(f"  invasivo: {(df_all['class']=='invasivo').sum()}")
print(f"  in_situ:  {(df_all['class']=='in_situ').sum()}")

# Ordenar por distancia
df_sorted = df_all.sort_values('distance').reset_index(drop=True)

print(f"\n DataFrame ordenado por distancia")
print(f"Distancia promedio: {df_sorted['distance'].mean():.4f}\n")

# ═════════════════════════════════════════════════════════════════════════════
# PASO 5: Seleccionar TOP-N e ir generando datasets
# ═════════════════════════════════════════════════════════════════════════════

print('=' * 80)
print('PASO 4: GENERAR DATASETS PARA CADA N')
print('=' * 80 + '\n')

for n_gen in N_IMAGES_TO_SELECT_VALUES:
    cnn_id = N_IMAGES_TO_SELECT_VALUES.index(n_gen) + 2
    
    print(f"\n{'─' * 80}")
    print(f"CNN {cnn_id} — Seleccionando {n_gen} imágenes")
    print(f"{'─' * 80}")
    
    # Seleccionar TOP-N
    df_selected = df_sorted.head(n_gen).copy()
    
    # Crear carpetas
    output_folder = os.path.join(OUTPUT_ROOT_FOLDER, f'imagenes-seleccionadas-{n_gen}')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'invasivo'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'in_situ'), exist_ok=True)
    
    # Copiar imágenes
    for class_name in ['invasivo', 'in_situ']:
        class_data = df_selected[df_selected['class'] == class_name]
        output_class_dir = os.path.join(output_folder, class_name)
        
        for _, row in tqdm(class_data.iterrows(), total=len(class_data), desc=f"  Copiando {class_name}"):
            src = row['image_path']
            dst = os.path.join(output_class_dir, row['filename'])
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                pass
    
    # Generar CSV
    df_csv = df_selected[['filename', 'class', 'label', 'epoch', 'distance']].copy()
    csv_path = os.path.join(output_folder, 'selected_images_metadata.csv')
    df_csv.to_csv(csv_path, index=False)
    
    print(f"✓ Guardado en: {output_folder}")

print(f"\n{'=' * 80}")
print(" TODOS LOS DATASETS GENERADOS CORRECTAMENTE")
print(f"{'=' * 80}\n")

print("RESUMEN FINALIZADO")