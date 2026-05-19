# -*- coding: utf-8 -*-
"""
# TFG - Generación de Imágenes de Melanoma con GAN Condicional

## Arquitectura: BigGAN + StyleGAN2 (Conditional)

**Clases objetivo:**
- Clase 0 → Melanoma **Invasivo**
- Clase 1 → Melanoma **In Situ**

**Datasets:**
1. ISIC 2024 (Kaggle) - ya en Colab
2. Argenciano - Google Drive
3. Polesie - Google Drive
4. Virgen del Rocío - Google Drive


"""

# CELDA 3A - Configuración Global
import os
import argparse

# argparse use parameter for learning rate, batch size, number of epochs, etc.
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate_g', type=float, default=1e-4)
parser.add_argument('--learning_rate_d', type=float, default=4e-4)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--output_dir', type=str, default='/home/jpdominguez/projects/TFGLuciaVela/src/output')
args = parser.parse_args()


BASE_DIR = '/home/jpdominguez/projects/TFGLuciaVela/data'
# Dataset ISIC 2024 (ya en Colab, no en Drive)
ISIC_DIR = f'{BASE_DIR}/ISIC'
ISIC_CSV= f'{ISIC_DIR}/metadata.csv'


# Dataset Argenciano (en Drive)
ARGENCIANO_DIR = f'{BASE_DIR}/Argenciano/Argenciano'
ARGENCIANO_CSV   = f'{ARGENCIANO_DIR}/images_paths_whole.csv'

# Dataset Polesie (en Drive)
POLESIE_DIR      = f'{BASE_DIR}/Polesie/Polesie'
POLESIE_CSV      = f'{POLESIE_DIR}/images_paths_whole.csv'

# Dataset Virgen del Rocío (en Drive)
ROCIO_DIR        = f'{BASE_DIR}/Virgen_del_Rocio/melanomas virgen del rocio'
ROCIO_CSV        = f'{ROCIO_DIR}/Total de imagenes con Breslow-VR/images_paths_whole.csv'
# Subcarpetas del Rocío (dentro de Total de imagenes con Breslow-VR):
ROCIO_SUBFOLDERS = ['Tis', 'T1 menor igual 1 mm', 'T2 mayor 1 mm menor igual 2 mm',
                    'T3 mayor 2 mm menor igual 4 mm', 'T4 mayor 4 mm']

# Directorio de salida (para guardar imágenes generadas y checkpoints) CAMBIAR
OUTPUT_DIR       = args.output_dir
CHECKPOINT_DIR   = f'{OUTPUT_DIR}/checkpoints'
GENERATED_DIR    = f'{OUTPUT_DIR}/generated_images'
SPLIT_DATA_DIR   = f'{OUTPUT_DIR}/split_data'  # Para guardar el split train/test


# HIPERPARÁMETROS GAN

IMAGE_SIZE    = 256    # Tamaño de imagen (256x256).
BATCH_SIZE    = 64    # Reducir a 32-64 si hay problemas de memoria GPU
LATENT_DIM    = 128    # Dimensión del vector latente z
NUM_CLASSES   = 2      # Invasivo (0) e In Situ (1)
LEARNING_RATE_G = args.learning_rate_g  # TTUR: generador aprende más lento
LEARNING_RATE_D = args.learning_rate_d  # TTUR: discriminador aprende más rápido
BETA1, BETA2  = 0.0, 0.999
N_EPOCHS      = args.num_epochs    #
SAVE_EVERY    = 5     # Guardar checkpoint cada N epochs
SAMPLE_EVERY  = 5      # Generar muestras cada N epochs
N_CRITIC      = 3      # Pasos del discriminador por cada paso del generador
GP_LAMBDA     = 10     # Peso del gradient penalty (WGAN-GP)

# SPLIT PARA CNN FUTURA
#Se reserva el 10% de cada clase para el test de la CNN. La GAN entrena con el 90% restante

TEST_HOLD_OUT = 0.10   # 10% bloqueado
RANDOM_SEED   = 42

# Crear directorios de salida
for d in [OUTPUT_DIR, CHECKPOINT_DIR, GENERATED_DIR, SPLIT_DATA_DIR,
           f'{GENERATED_DIR}/class_0_invasivo',
           f'{GENERATED_DIR}/class_1_in_situ']:
    os.makedirs(d, exist_ok=True)

print(' Configuración cargada correctamente')
print(f'   IMAGE_SIZE:  {IMAGE_SIZE}x{IMAGE_SIZE}')
print(f'   BATCH_SIZE:  {BATCH_SIZE}')
print(f'   N_EPOCHS:    {N_EPOCHS}')
print(f'   NUM_CLASSES: {NUM_CLASSES} (0=Invasivo, 1=In Situ)')

# CELDA 3B
print(f'ISIC_DIR:       {ISIC_DIR}')
print(f'ARGENCIANO_DIR: {ARGENCIANO_DIR}')
print(f'POLESIE_DIR:    {POLESIE_DIR}')
print(f'ROCIO_DIR:      {ROCIO_DIR}')
print(f'OUTPUT_DIR:     {OUTPUT_DIR}')
print(f'CHECKPOINT_DIR: {CHECKPOINT_DIR}')
print(f'GENERATED_DIR:  {GENERATED_DIR}')
print(f'SPLIT_DATA_DIR: {SPLIT_DATA_DIR}')

"""## **Imports globales**"""

# CELDA 4 - Imports

import os, re, math, random, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.utils as vutils
from sklearn.model_selection import train_test_split

# Reproducibilidad
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Configurar dispositivo (GPU si está disponible, sino CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {DEVICE}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

"""## **Carga del Dataset ISIC**

"""

# CELDA 5 - Carga ISIC con CSV de metadatos
# Melanoma Invasive = 0, Melanoma in situ = 1

def load_isic_from_csv(csv_path, images_dir):
    records = []
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    label_map = {
        'Melanoma Invasive': (0, 'invasivo'),
        'Melanoma in situ':  (1, 'in_situ')
    }

    # Cargar CSV
    df_meta = pd.read_csv(csv_path)

    # Filtrar solo las filas que nos interesan
    df_meta = df_meta[df_meta['diagnosis_3'].isin(label_map.keys())]

    if df_meta.empty:
        print(' No se encontraron casos de melanoma invasivo o in situ en el CSV.')
        return pd.DataFrame(columns=['image_path', 'label', 'label_name', 'source'])

    for _, row in df_meta.iterrows():
        isic_id = row['isic_id']
        diagnosis = row['diagnosis_3']
        label, label_name = label_map[diagnosis]

        # Buscar la imagen en el directorio
        image_path = None
        for ext in exts:
            candidate = os.path.join(images_dir, isic_id + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f' No se encontró imagen para {isic_id}')
            continue

        records.append({
            'image_path': image_path,
            'label':      label,
            'label_name': label_name,
            'source':     'ISIC'
        })

    df = pd.DataFrame(records)
    print(f' ISIC | invasivo: {(df.label==0).sum()} | in_situ: {(df.label==1).sum()}')
    return df

# Uso:
# csv_path    = ruta a tu archivo CSV de metadatos
# images_dir  = carpeta donde tienes las imágenes descargadas

df_isic = load_isic_from_csv(ISIC_CSV, ISIC_DIR)
df_isic.head(3)

"""## **Carga Dataset Argenciano**"""

# CELDA 6- Carga Dataset Argenciano

def load_argenciano(base_dir, csv_path):
    if not os.path.exists(csv_path):
        print(f' CSV no encontrado: {csv_path}')
        return pd.DataFrame(columns=['image_path', 'label', 'label_name', 'source'])

    df_csv = pd.read_csv(csv_path, header=0)
    print(f'   Columnas CSV Argenciano: {list(df_csv.columns)}')

    path_col  = 'IMAGE_PATH'
    label_col = 'IN_SITU'  # 0=invasivo, 1=in situ

    records = []
    for _, row in df_csv.iterrows():
        # Extraer solo el nombre del archivo (ignorar ruta de Windows)
        fname = os.path.basename(str(row[path_col]).replace('\\', '/'))
        full_path = os.path.join(base_dir, fname)

        label = int(row[label_col])
        label_name = 'invasivo' if label == 0 else 'in_situ'

        records.append({
            'image_path': full_path,
            'label':      label,
            'label_name': label_name,
            'source':     'Argenciano'
        })

    df = pd.DataFrame(records)
    df['exists'] = df['image_path'].apply(os.path.exists)
    missing = (~df['exists']).sum()
    if missing > 0:
        print(f' {missing} imágenes no encontradas en disco.')
    df = df[df['exists']].drop(columns=['exists'])

    print(f' Argenciano | invasivo: {(df.label==0).sum()} | in_situ: {(df.label==1).sum()}')
    return df

df_argenciano = load_argenciano(ARGENCIANO_DIR, ARGENCIANO_CSV)
df_argenciano.head(3)

"""## **Carga Dataset Polesie**"""

# CELDA 7 - Carga Dataset Polesie

def load_polesie(base_dir, csv_path):
    if not os.path.exists(csv_path):
        print(f' CSV no encontrado: {csv_path}')
        return pd.DataFrame(columns=['image_path', 'label', 'label_name', 'source'])

    df_csv = pd.read_csv(csv_path, header=0)
    print(f'   Columnas CSV Polesie: {list(df_csv.columns)}')

    records = []
    for _, row in df_csv.iterrows():
        fname = os.path.basename(str(row['IMAGE_PATH']).replace('\\', '/'))
        full_path = os.path.join(base_dir, fname)
        label = int(row['IN_SITU'])
        label_name = 'invasivo' if label == 0 else 'in_situ'
        records.append({
            'image_path': full_path,
            'label':      label,
            'label_name': label_name,
            'source':     'Polesie'
        })

    df = pd.DataFrame(records)
    df['exists'] = df['image_path'].apply(os.path.exists)
    missing = (~df['exists']).sum()
    if missing > 0:
        print(f'{missing} imágenes no encontradas en disco.')
    df = df[df['exists']].drop(columns=['exists'])

    print(f'Polesie | invasivo: {(df.label==0).sum()} | in_situ: {(df.label==1).sum()}')
    return df

df_polesie = load_polesie(POLESIE_DIR, POLESIE_CSV)
df_polesie.head(3)

"""## **Carga Dataset Virgen del Rocío**

**Mapeo de etiquetas:**
- Breslow = `'Tis'` → **In Situ** (clase 1)
- Breslow = número > 0 → **Invasivo** (clase 0)
"""

# CELDA 8- Carga Dataset Virgen del Rocío


def load_virgen_del_rocio(base_dir, csv_path):
    if not os.path.exists(csv_path):
        print(f' CSV no encontrado: {csv_path}')
        return pd.DataFrame(columns=['image_path', 'label', 'label_name', 'source'])

    df_csv = pd.read_csv(csv_path, header=0)
    print(f'   Columnas: {list(df_csv.columns)}')

    # Construir índice de todas las imágenes disponibles: nombre -> ruta completa
    print('   Indexando imágenes en disco (puede tardar unos segundos)...')
    img_index = {}
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                img_index[f] = os.path.join(root, f)
    print(f'   {len(img_index)} imágenes indexadas en disco')

    def breslow_to_label(val):
        val_str = str(val).strip().lower().replace(',', '.')
        if val_str == 'tis':
            return 1, 'in_situ'
        try:
            float(val_str)
            return 0, 'invasivo'
        except ValueError:
            return None, None

    records = []
    skipped = 0
    not_found = 0
    for _, row in df_csv.iterrows():
        label, label_name = breslow_to_label(row['BRESLOW'])
        if label is None:
            skipped += 1
            continue

        fname = os.path.basename(str(row['IMAGE_PATH']).replace('\\', '/'))

        if fname in img_index:
            records.append({
                'image_path': img_index[fname],
                'label':      label,
                'label_name': label_name,
                'source':     'VirgenDelRocio'
            })
        else:
            not_found += 1

    if not_found > 0:
        print(f' {not_found} imágenes del CSV no encontradas en disco')
    if skipped > 0:
        print(f' {skipped} filas omitidas por Breslow no parseable')

    df = pd.DataFrame(records)
    print(f'Virgen Rocío | invasivo: {(df.label==0).sum()} | in_situ: {(df.label==1).sum()}')
    return df

df_rocio = load_virgen_del_rocio(ROCIO_DIR, ROCIO_CSV)
df_rocio.head(3)

"""##**Unificación de todos los datasets**



"""

# CELDA 9 - Unificación

dfs_to_merge = []
for name, df in [('ISIC', df_isic),
                  ('Argenciano', df_argenciano),
                  ('Polesie', df_polesie),
                  ('VirgenDelRocio', df_rocio)]:
    if df is not None and len(df) > 0:
        # Asegurarse de que tienen las columnas mínimas
        cols_needed = ['image_path', 'label', 'label_name', 'source']
        df_clean = df[cols_needed].copy()
        dfs_to_merge.append(df_clean)
        print(f'  {name:20s}: {len(df_clean):5d} imágenes')

df_all = pd.concat(dfs_to_merge, ignore_index=True)

# Verificar que los archivos existen (doble check)
df_all['exists'] = df_all['image_path'].apply(os.path.exists)
n_missing = (~df_all['exists']).sum()
if n_missing > 0:
    print(f'\n  {n_missing} rutas no válidas eliminadas del dataset unificado')
df_all = df_all[df_all['exists']].drop(columns=['exists']).reset_index(drop=True)

print(f'\n✅ Dataset unificado total: {len(df_all)} imágenes')
print(f'   Invasivo  (0): {(df_all.label==0).sum()}')
print(f'   In Situ   (1): {(df_all.label==1).sum()}')
df_all.head()

"""## **Análisis Exploratorio de Datos (EDA)**

El EDA es una fase previa obligatoria en cualquier pipeline de aprendizaje automático. Su objetivo es caracterizar la distribución empírica del conjunto de datos antes de cualquier modelado.
En este caso, se analizan la distribución de clases (balance entre invasivo e in situ), la procedencia de las muestras (heterogeneidad entre datasets).

"""

# CELDA 10 - Análisis de las imágenes

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Análisis Exploratorio del Dataset Unificado', fontsize=14, fontweight='bold')

# 1. Distribución de clases
class_counts = df_all['label_name'].value_counts()
colors = ['#e74c3c', '#3498db']
axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
axes[0].set_title('Distribución de Clases')
axes[0].set_xlabel('Clase')
axes[0].set_ylabel('Número de imágenes')
for i, (k, v) in enumerate(class_counts.items()):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# 2. Distribución por fuente
source_counts = df_all['source'].value_counts()
axes[1].pie(source_counts.values, labels=source_counts.index,
            autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Set3.colors[:len(source_counts)])
axes[1].set_title('Distribución por Dataset')

# 3. Clases por dataset (stacked bar)
pivot = df_all.groupby(['source', 'label_name']).size().unstack(fill_value=0)
pivot.plot(kind='bar', ax=axes[2], color=colors, edgecolor='black')
axes[2].set_title('Distribución de Clases por Dataset')
axes[2].set_xlabel('Dataset')
axes[2].set_ylabel('Número de imágenes')
axes[2].tick_params(axis='x', rotation=30)
axes[2].legend(title='Clase')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_distribucion.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n Resumen estadístico:')
print(df_all.groupby(['source', 'label_name']).size().reset_index(name='count').to_string())

# CELDA 10b - Visualizar muestra de imágenes por clase

def show_samples(df, label, label_name, n=8):
    subset = df[df['label'] == label].sample(min(n, len(df[df['label']==label])),
                                              random_state=RANDOM_SEED)
    fig, axes = plt.subplots(1, len(subset), figsize=(2.5*len(subset), 3))
    if len(subset) == 1: axes = [axes]
    for ax, (_, row) in zip(axes, subset.iterrows()):
        try:
            img = Image.open(row['image_path']).convert('RGB')
            ax.imshow(img)
            ax.set_title(row['source'], fontsize=8)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, 'Error', ha='center')
            ax.axis('off')
    plt.suptitle(f'Muestra: {label_name.upper()} (clase {label})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/muestra_{label_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

show_samples(df_all, 0, 'invasivo')
show_samples(df_all, 1, 'in_situ')

"""## **Split Train / Test (holdout para CNN futura)**

Se reserva el **10% de cada clase** como test set estratificado y bloqueado.
Este conjunto no se utiliza durante el entrenamiento de la GAN.

El objetivo es disponer de un conjunto completamente independiente para evaluar en el futuro la CNN, evitando **data leakage** y permitiendo medir la capacidad real de generalización del modelo.

El riesgo está en que si la GAN entrenara con todo el dataset, podría generar imágenes sintéticas que "memoricen" o repliquen sutilmente rasgos del test set.
Es por ello, que al evaluar la CNN con datos que ni la GAN ni la propia CNN conocen, la métrica de precisión será un reflejo fiel de la generalización real.
"""

# CELDA 11 - Train/Test Split estratificado


df_train, df_test = train_test_split(
    df_all,
    test_size=TEST_HOLD_OUT,
    stratify=df_all['label'],  # Estratificado por clase
    random_state=RANDOM_SEED
)

df_train = df_train.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

# Guardar los splits en Drive (para reproducibilidad)
df_train.to_csv(f'{SPLIT_DATA_DIR}/train_split.csv', index=False)
df_test.to_csv(f'{SPLIT_DATA_DIR}/test_split_BLOQUEADO_CNN.csv', index=False)

print(' Split realizado y guardado en Drive')
print(f'   TRAIN (para GAN):  {len(df_train)} imágenes')
print(f'     Invasivo:  {(df_train.label==0).sum()}')
print(f'     In Situ:   {(df_train.label==1).sum()}')
print(f'   TEST (bloqueado):  {len(df_test)} imágenes')
print(f'     Invasivo:  {(df_test.label==0).sum()}')
print(f'     In Situ:   {(df_test.label==1).sum()}')

# Visualizar desequilibrio
ratio = (df_train.label==0).sum() / max((df_train.label==1).sum(), 1)
print(f'\n  Ratio invasivo/in_situ en train: {ratio:.2f}x')
if ratio > 3:
    print('     Dataset desbalanceado. Se usará WeightedRandomSampler para el DataLoader.')

"""## **Dataset y DataLoader con augmentación**
La augmentación de datos es una técnica de regularización implícita que incrementa artificialmente la variabilidad del conjunto de entrenamiento mediante transformaciones que preservan la etiqueta.
Las transformaciones aplicadas (flips, rotaciones, jitter de color) son estándar en imágenes dermoscópicas y están respaldadas por la literatura.
El **WeightedRandomSampler** implementa un muestreo no uniforme con reemplazamiento que compensa el desbalance de clases a nivel de batch, equivalente a asignar un peso inversamente proporcional a la frecuencia de clase.
El parámetro *drop_last=True* es necesario por la Batch Normalization de la GAN es inestable con batches de tamaño 1 al final de una época.

---

Pytorch necesita que le des los datos en ""lotes" (batches) durante el entrenamiento, no todos a la vez porque no caben en la memoria de la GPU.
En esta celda:
1. Se aplica augmentation para que la red vea variaciones artificales de cada imagen y aprenda mejor.
2. Se usa un **WeigthtedRandomSampler** que compensa el desbalance entre clases asegurando que cada batch tenga aproximadamente el mismo número de imágenes invasivo e in situ;  
3. Organiza todo en un DataLoader que va alimentando a la red durante el entrenamiento.
"""


# ANÁLISIS DE COLORES REALES (para regularización del generador)
# ═══════════════════════════════════════════════════════════
 
print('\n Analizando distribución de colores del dataset real...')
 
# Tomar muestra de imágenes reales (no hace falta todas para rapidez)
n_samples = min(200, len(df_train))
sample_paths = df_train['image_path'].sample(n_samples, random_state=RANDOM_SEED).tolist()
 
r_values, g_values, b_values = [], [], []
errors_count = 0
 
from tqdm import tqdm
for img_path in tqdm(sample_paths, desc='Analizando colores'):
    try:
        img = Image.open(img_path).convert('RGB').resize((128, 128))
        arr = np.array(img) / 255.0
 
        r_values.append(arr[:, :, 0].mean())
        g_values.append(arr[:, :, 1].mean())
        b_values.append(arr[:, :, 2].mean())
    except Exception as e:
        errors_count += 1

if errors_count > 0:
    print(f'  {errors_count} imágenes fallaron al analizar colores')
 
# Calcular rangos de color válidos (media ± 2 desviaciones estándar)
r_mean, r_std = np.mean(r_values), np.std(r_values)
g_mean, g_std = np.mean(g_values), np.std(g_values)
b_mean, b_std = np.mean(b_values), np.std(b_values)
 
print(f'\n Distribución de colores reales:')
print(f'   Rojo:  {r_mean:.3f} ± {r_std:.3f}  → rango [{r_mean-2*r_std:.3f}, {r_mean+2*r_std:.3f}]')
print(f'   Verde: {g_mean:.3f} ± {g_std:.3f}  → rango [{g_mean-2*g_std:.3f}, {g_mean+2*g_std:.3f}]')
print(f'   Azul:  {b_mean:.3f} ± {b_std:.3f}  → rango [{b_mean-2*b_std:.3f}, {b_mean+2*b_std:.3f}]')
 
# Guardar estadísticas para usar durante entrenamiento
COLOR_STATS = {
    'r_mean': r_mean,
    'r_std': r_std,
    'g_mean': g_mean,
    'g_std': g_std,
    'b_mean': b_mean,
    'b_std': b_std
}
 
print('\n✅ Estadísticas de color calculadas y guardadas en COLOR_STATS')
#


# CELDA 12 - Dataset PyTorch con augmentación

class MelanomaDataset(Dataset):
    """
    Dataset PyTorch para imágenes de melanoma.
    Devuelve (imagen_tensor, label) donde label es 0 (invasivo) o 1 (in situ).
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except Exception:
            # Si la imagen está corrupta, devolver imagen negra
            img = Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row['label'], dtype=torch.long)
        return img, label


# Transformaciones para entrenamiento (con augmentación)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 30, IMAGE_SIZE + 30)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
])

# Dataset
train_dataset = MelanomaDataset(df_train, transform=train_transform)

# WeightedRandomSampler para balancear clases en cada batch
class_counts_train = df_train['label'].value_counts().sort_index().values
class_weights      = 1.0 / class_counts_train
sample_weights     = class_weights[df_train['label'].values]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    drop_last=True  # Importante para BatchNorm en la GAN
)

print(f' DataLoader creado')
print(f'   Imágenes en train:   {len(train_dataset)}')
print(f'   Batches por época:   {len(train_loader)}')
print(f'   Batch size:          {BATCH_SIZE}')
print(f'   Resolución:          {IMAGE_SIZE}x{IMAGE_SIZE}')

# Verificar un batch
sample_batch, sample_labels = next(iter(train_loader))
print(f'\n   Shape de un batch:   {sample_batch.shape}')
print(f'   Labels en el batch:  {sample_labels.tolist()}')

"""## **Arquitectura BigGAN + StyleGAN2**

**Concepto de la arquitectura híbrida:**
- **BigGAN**: Condicionalidad por clase (class embedding), Self-Attention, ortogonal regularization
- **StyleGAN2 ideas adoptadas**: Modulated convolutions (mapping network → style → AdaIN), skip connections en el generador, path length regularization (simplificado)

Esta combinación permite generar imágenes **condicionadas a la clase** (invasivo/in situ) con alta calidad fotorrealista.

---
La arquitectura implementa una **Generative Adversarial Network condicional (cGAN)** que combina dos familias de modelos estado del arte.

Del **BigGAN** (Brock et al., 2019) se adoptan 3 contribuciones: el *class embedding* que concatena la representación de clase al vector latente, el *Projection Discriminator* que condiciona al discriminador mediante producto escalar con el embedding de clase, y la *Self-Attention* que modela dependencias espaciales de largo alcance superando la limitación receptiva de las convoluciones locales.

Del **StyleGAN2** (Karras et al., 2020) se adopta el *Mapping Network*, una red fully-connected de 8 capas que transforma el espacio intermedio W de menos entrelazamiento (*disentanglement*), y la modulación de capas mediante *Conditional Batch Normalization*, donde los parámetros afines γ y β se predicen dinámicamente desde el vector de estilo w.
Las *skip connections* residuales facilitan el flujo de gradiente durante la retropropagación.

La *Spectral Normalization* en el discriminador controla la constante de Lipschitz de cada capa, estabilizando el entrenamiento.

"""

# CELDA 13 - Bloques base de la arquitectura

# MÓDULOS COMPARTIDOS

class SpectralNorm(nn.Module):
    """Wrapper para aplicar spectral normalization."""
    def __init__(self, module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)
    def forward(self, x):
        return self.module(x)


class SelfAttention(nn.Module):
    """Self-Attention de BigGAN para capturar dependencias globales."""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key   = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class ConditionalBatchNorm(nn.Module):
    """Conditional BatchNorm: afín params condicionados al class embedding (BigGAN)."""
    def __init__(self, num_features, num_classes, style_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        # gamma y beta son predichos desde el style vector (BigGAN + StyleGAN2 idea)
        self.style_gamma = nn.Linear(style_dim, num_features)
        self.style_beta  = nn.Linear(style_dim, num_features)
        # Inicialización: empezar con transformación identidad
        nn.init.ones_(self.style_gamma.weight)
        nn.init.zeros_(self.style_beta.weight)

    def forward(self, x, style):
        out = self.bn(x)
        gamma = self.style_gamma(style).unsqueeze(2).unsqueeze(3)
        beta  = self.style_beta(style).unsqueeze(2).unsqueeze(3)
        return gamma * out + beta


print(' Módulos base definidos (SpectralNorm, SelfAttention, ConditionalBatchNorm)')

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, num_classes, style_dim=512, n_layers=8):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        layers = [nn.Linear(latent_dim * 2, style_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2)]
        self.mapping = nn.Sequential(*layers)

    def forward(self, z, labels):
        c = self.class_embed(labels)
        zc = torch.cat([z, c], dim=1)
        return self.mapping(zc)


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.cbn1  = ConditionalBatchNorm(in_ch, None, style_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cbn2  = ConditionalBatchNorm(out_ch, None, style_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, style):
        # Skip antes del upsample
        skip = self.skip(x)
        if self.upsample:
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            x    = F.interpolate(x,    scale_factor=2, mode='bilinear', align_corners=False)
        h = F.relu(self.cbn1(x, style))
        h = self.conv1(h)
        h = F.relu(self.cbn2(h, style))
        h = self.conv2(h)
        return h + skip


class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=2, style_dim=512, image_size=256):
        super().__init__()
        self.style_dim  = style_dim
        self.init_size  = 4
        self.mapping    = MappingNetwork(latent_dim, num_classes, style_dim)
        self.input_proj = nn.Linear(latent_dim, 512 * 4 * 4)

        self.blocks = nn.ModuleList([
            GenBlock(512, 512, style_dim),  # 4  -> 8
            GenBlock(512, 256, style_dim),  # 8  -> 16
            GenBlock(256, 256, style_dim),  # 16 -> 32
            GenBlock(256, 128, style_dim),  # 32 -> 64
            GenBlock(128, 64,  style_dim),  # 64 -> 128
            GenBlock(64,  32,  style_dim),  # 128-> 256
        ])

        self.attn   = SelfAttention(128)  # En resolución 64x64 (bloque índice 3)

        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        style = self.mapping(z, labels)
        x = self.input_proj(z).view(z.size(0), 512, 4, 4)

        for i, block in enumerate(self.blocks):
            x = block(x, style)
            if i == 3:  # Self-attention tras el bloque 32->64 (resolución 64x64)
                x = self.attn(x)

        return self.to_rgb(x)


# Test
G = Generator(LATENT_DIM, NUM_CLASSES, style_dim=512, image_size=IMAGE_SIZE).to(DEVICE)
z_test = torch.randn(2, LATENT_DIM).to(DEVICE)
l_test = torch.tensor([0, 1]).to(DEVICE)
out_test = G(z_test, l_test)
print(f' Generador OK | Output shape: {out_test.shape}')
n_params_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
print(f'   Parámetros del Generador: {n_params_G/1e6:.1f}M')

# CELDA 13c - Discriminador (BigGAN + Projection Conditioning)

class DiscBlock(nn.Module):
    """Bloque del Discriminador con Spectral Norm y downsample."""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip  = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.downsample = downsample

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(self.skip(x), 2)
        else:
            x = self.skip(x)
        return h + x


class Discriminator(nn.Module):
    """
    Discriminador BigGAN con Projection Conditioning.
    Entrada: imagen [B, 3, H, W] + labels → Salida: score real/falso
    """
    def __init__(self, num_classes=2, image_size=256):
        super().__init__()

        self.blocks = nn.Sequential(
            DiscBlock(3,   32,  downsample=True),   # 256->128
            DiscBlock(32,  64,  downsample=True),   # 128->64
            DiscBlock(64,  128, downsample=True),   # 64->32
            DiscBlock(128, 256, downsample=True),   # 32->16
            DiscBlock(256, 512, downsample=True),   # 16->8
            DiscBlock(512, 512, downsample=True),   # 8->4
        )

        # Self-Attention tras el bloque 2 (resolución 32x32, 128 canales)
        self.attn = SelfAttention(128)

        # Projection conditioning (BigGAN)
        self.class_embed = nn.Embedding(num_classes, 512)
        self.linear = nn.utils.spectral_norm(nn.Linear(512, 1))

    def forward(self, x, labels):
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 2:  # Attention después del bloque 2 (32x32 spatial)
                h = self.attn(h)

        # Global sum pooling
        h = h.sum(dim=[2, 3])  # [B, 512]

        # Projection conditioning: dot product con class embedding
        c = self.class_embed(labels)  # [B, 512]
        proj = (h * c).sum(dim=1, keepdim=True)  # [B, 1]

        out = self.linear(h) + proj
        return out.squeeze(1)


# Test del discriminador
D = Discriminator(NUM_CLASSES, IMAGE_SIZE).to(DEVICE)
d_out = D(out_test, l_test)
print(f' Discriminador OK | Output shape: {d_out.shape}')
n_params_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
print(f'   Parámetros del Discriminador: {n_params_D/1e6:.1f}M')



"""## **Loss y Optimizadores (WGAN-GP)**

Cuando la red genera una imagen, necesita saber si lo ha hecho bien o mal. La función de pérdida es simplemente esa medida de error.
Cuanto más alto es el loss, es que el error es mayor.

No usamos la GAN clásica, que es la original de 2014, porque tiene un problema grave llamado **mode collapse**: el Generador aprende que si siempre genera el mismo tipo de imagen, el Discriminador se confunde, y deja de intentar mejorar.

Además, la GAN clásica, cuando el Discriminador es mucho mejor que el Generador, el error que llega al Generador se vuelve tan pequeño que deja de aprender. Es el problema del **gradiente que desaparece**.

Es por ello, que introducimos WGAN-GP.
La WGAN cambia la forma de medir el error. En lugar de preguntar "¿es real o falso?", que es una respuesta binaria, lo que hace es preguntar "¿cuánto de real parece?".
Esto hace que siempre haya información útil para que el Generador aprenda, aunque el Discriminador sea mucho mejor.

El **Gradient Penalty** es una restricción matemática extra que se añade al Discriminador para que no se vuelva demasiado extremo en sus puntuaciones. Sin esta restricción, el Discriminador podría dar puntuaciones de millones o de menos millones, lo cual desestabiliza todo el entrenamiento.

El **TTUR**, es simplemente hacer que el Discriminador aprenda más rápido que el Generador (4 veces en este caso).
Necesitamos que el discriminador sea lo suficientemente bueno para evaluar correctamente al generador.

Los **optimizadores**, es "quien le dice a la red cuánto y en qué dirección tiene que cambiar sus pesos".
Usamos el optimizador **Adam**, que es el más común en deep learning porque adapta automáticamente el tamaño de los pasos de aprendizaje para cada peso de la red de forma individual.  

------------------------
La función de pérdida implementada es la Wasserstein GAN con Gradient Penalty (Gulrajani et al., 2017). La métrica de Wasserstein, también llamada distancia Earth Mover, mide el coste mínimo de transformar una distribución en otra. A diferencia de la divergencia Jensen-Shannon de la GAN original, proporciona gradientes informativos incluso cuando las distribuciones real y generada tienen soporte disjunto, lo que elimina el problema del gradiente que desaparece y reduce el mode collapse.
La función de pérdida del discriminador es:
L_D = E[D(x̃)] − E[D(x)] + λ · E[(‖∇D(x̂)‖₂ − 1)²]
El primer término penaliza puntuaciones altas en imágenes falsas. El segundo premia puntuaciones altas en imágenes reales. El tercer término es el Gradient Penalty, que obliga al discriminador a ser una función 1-Lipschitz, condición necesaria para que la distancia de Wasserstein sea matemáticamente válida. λ=10 es el peso que controla la importancia relativa de esta penalización.
La función de pérdida del generador es simplemente:
L_G = −E[D(G(z, c))]
Es decir, el Generador intenta maximizar la puntuación que le da el Discriminador a sus imágenes falsas.
La optimización sigue el Two Time-Scale Update Rule (TTUR) (Heusel et al., 2017), con LR_D = 4×10⁻⁴ y LR_G = 1×10⁻⁴. Bajo condiciones teóricas este esquema garantiza convergencia a un equilibrio de Nash local. El parámetro N_CRITIC=3 asegura que el discriminador permanece próximo a su óptimo en cada actualización del generador, lo cual es requisito para que el gradiente que recibe el generador sea una estimación válida de la distancia de Wasserstein.
"""

# CELDA 14 - Loss WGAN-GP + Optimizadores

def gradient_penalty(D, real_imgs, fake_imgs, labels, device):
    """Gradient Penalty de WGAN-GP para estabilizar el entrenamiento."""
    B = real_imgs.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

    d_interp = D(interpolated, labels)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True
    )[0]

    grad_norm = grad.view(B, -1).norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


# Optimizadores con TTUR (Two Time-Scale Update Rule)
opt_G = optim.Adam(G.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))

# Learning rate schedulers (reducir LR a la mitad del entrenamiento)
# sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=N_EPOCHS, eta_min=1e-6)
# Reducir LR solo después de 60% del entrenamiento
milestone = int(N_EPOCHS * 0.6)
sched_G = optim.lr_scheduler.MultiStepLR(opt_G, milestones=[milestone], gamma=0.5)
sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=N_EPOCHS, eta_min=1e-6)

print(' Optimizadores y schedulers configurados')
print(f'   LR Generador:     {LEARNING_RATE_G}')
print(f'   LR Discriminador: {LEARNING_RATE_D}')
print(f'   Loss:             WGAN-GP (lambda={GP_LAMBDA})')
print(f'   N_CRITIC:         {N_CRITIC} (pasos D por cada paso G)')

# ═══════════════════════════════════════════════════════════
# FUNCIÓN DE REGULARIZACIÓN DE COLOR
# ═══════════════════════════════════════════════════════════

def color_realism_loss(fake_imgs, color_stats, weight=2.0):
    """
    Penaliza imágenes generadas que tengan colores fuera del rango
    de melanomas reales.

    Args:
        fake_imgs: Tensor [B, 3, H, W] en rango [-1, 1]
        color_stats: Dict con medias y stds de colores reales
        weight: Peso de la penalización

    Returns:
        loss: Scalar tensor
    """
    # Convertir de [-1, 1] a [0, 1]
    fake_imgs_01 = (fake_imgs * 0.5 + 0.5).clamp(0, 1)

    # Calcular la media de cada canal para cada imagen
    r_mean_fake = fake_imgs_01[:, 0, :, :].mean(dim=[1, 2])
    g_mean_fake = fake_imgs_01[:, 1, :, :].mean(dim=[1, 2])
    b_mean_fake = fake_imgs_01[:, 2, :, :].mean(dim=[1, 2])

    # Rangos de color válidos (media ± 2 desviaciones estándar)
    r_min, r_max = color_stats['r_mean'] - 2 * color_stats['r_std'], color_stats['r_mean'] + 2 * color_stats['r_std']
    g_min, g_max = color_stats['g_mean'] - 2 * color_stats['g_std'], color_stats['g_mean'] + 2 * color_stats['g_std']
    b_min, b_max = color_stats['b_mean'] - 2 * color_stats['b_std'], color_stats['b_mean'] + 2 * color_stats['b_std']

    # Penalización por colores fuera de rango (relu para penalizar solo si está fuera)
    penalty_r = torch.relu(r_mean_fake - r_max) + torch.relu(r_min - r_mean_fake)
    penalty_g = torch.relu(g_mean_fake - g_max) + torch.relu(g_min - g_mean_fake)
    penalty_b = torch.relu(b_mean_fake - b_max) + torch.relu(b_min - b_mean_fake)

    color_loss = (penalty_r + penalty_g + penalty_b).mean() * weight
    return color_loss

"""## **Bucle de Entrenamiento**

El entrenamiento se organiza en **épocas**, donde cada época significa que la red ha procesado todas las imágenes disponibles una vez.

En cada iteración del bucle se ejecutan dos fases diferenciadas.
Primero se entrenan los parámetros del **discriminador** durante 3 pasos consecutivos (N_CRITIC=3), exponiéndole a imágenes reales del dataset e imágenes sintéticas producidas por el generador.
Este desequilibrio de 3 a 1 es deliberado: necesitamos que el discriminador sea siempre un evaluador suficientemente competente antes de que el generador reciba su señal de aprendizaje. Si ambos se actualizaran a la misma velocidad, *el generador aprendería a engañar a un juez poco fiable*, lo cual no produce mejoras reales.


Después se actualiza el **generador** durante 1 solo paso, ajustando sus parámetros para que las imágenes que produce reciban puntuaciones más altas del discriminador.



**Los checkpoints** son guardados automáticos del estado completo de ambas redes cada 25 épocas, incluyendo también el estado interno de los optimizadores. Esto no solo protege ante posibles interrupciones del servidor, sino que permite retomar el entrenamiento de forma exactamente reproducible, como si nunca se hubiera interrumpido.

**Las imágenes de muestra** se generan cada 10 épocas usando siempre los mismos vectores de ruido fijos. Usar siempre el mismo ruido de entrada es clave: así las diferencias visuales entre la muestra de la época 10 y la de la época 200 se deben únicamente al aprendizaje del generador, y no a que simplemente le tocó un ruido de entrada más favorable. Es el equivalente a fotografiar siempre al mismo paciente para ver la evolución de un tratamiento, en lugar de fotografiar pacientes distintos cada vez

"""


# CELDA 15 - Entrenamiento

# Vectores fijos para monitorear progreso visual
N_FIXED = 8
fixed_z = torch.randn(N_FIXED * NUM_CLASSES, LATENT_DIM, device=DEVICE)
fixed_labels = torch.tensor(
    [c for c in range(NUM_CLASSES) for _ in range(N_FIXED)],
    device=DEVICE
)

history = {'d_loss': [], 'g_loss': [], 'g_color_loss': [], 'epoch': []}

# Cargar checkpoint si existe
start_epoch = 0
latest_ckpt = f'{CHECKPOINT_DIR}/latest_checkpoint.pt'
if os.path.exists(latest_ckpt):
    print('Checkpoint encontrado, reanudando entrenamiento...')
    ckpt = torch.load(latest_ckpt, map_location=DEVICE)
    G.load_state_dict(ckpt['G'])
    D.load_state_dict(ckpt['D'])
    opt_G.load_state_dict(ckpt['opt_G'])
    opt_D.load_state_dict(ckpt['opt_D'])
    start_epoch = ckpt['epoch'] + 1
    history = ckpt.get('history', history)
    print(f'   Reanudando desde época {start_epoch}')

print(f' Iniciando entrenamiento: épocas {start_epoch} → {N_EPOCHS}')
print(f'   Guardado de checkpoints cada {SAVE_EVERY} épocas')
print(f'   Generación de muestras cada {SAMPLE_EVERY} épocas')
print(f'   N_CRITIC = {N_CRITIC}')
print(f'   Regularización de color: ACTIVADA')
print('─' * 60)

for epoch in range(start_epoch, N_EPOCHS):
    G.train()
    D.train()
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_g_color_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f'Época {epoch+1}/{N_EPOCHS}', leave=False)

    for real_imgs, real_labels in pbar:
        real_imgs = real_imgs.to(DEVICE)
        real_labels = real_labels.to(DEVICE)
        B = real_imgs.size(0)

        # ═══════════════════════════════════════════════════════
        # PASO 1: Entrenar Discriminador (N_CRITIC veces)
        # ═══════════════════════════════════════════════════════
        for critic_step in range(N_CRITIC):
            opt_D.zero_grad()

            # Imágenes reales
            d_real = D(real_imgs, real_labels)

            # Imágenes falsas (NUEVO Z cada iteración)
            # IMPORTANTE: Usar las mismas labels que las reales para el GP
            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_labels = real_labels  # Mismo label para consistencia en GP
            fake_imgs = G(z, fake_labels).detach()
            d_fake = D(fake_imgs, fake_labels)

            # WGAN loss + Gradient Penalty
            gp = gradient_penalty(D, real_imgs, fake_imgs, real_labels, DEVICE)
            # No label smoothing en WGAN-GP (ya es estable por naturaleza)
            d_loss = d_fake.mean() - d_real.mean() + GP_LAMBDA * gp

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_D.step()

            # Solo guardamos el último d_loss para el promedio
            if critic_step == N_CRITIC - 1:
                epoch_d_loss += d_loss.item()

        # ═══════════════════════════════════════════════════════
        # PASO 2: Entrenar Generador CON REGULARIZACIÓN DE COLOR
        # ═══════════════════════════════════════════════════════
        opt_G.zero_grad()

        z = torch.randn(B, LATENT_DIM, device=DEVICE)
        # Usar labels aleatorias para que el generador aprenda ambas clases
        fake_labels_g = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)
        fake_imgs_g = G(z, fake_labels_g)
        d_fake_g = D(fake_imgs_g, fake_labels_g)

        # Loss adversarial (WGAN)
        g_loss_adversarial = -d_fake_g.mean()

        # Loss de color realista (decaimiento más suave)
        color_weight = max(2.0 * (1 - epoch/(2*N_EPOCHS)), 0.5)
        g_loss_color = color_realism_loss(fake_imgs_g, COLOR_STATS, weight=color_weight)

        # Loss total del generador
        g_loss = g_loss_adversarial + g_loss_color

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        opt_G.step()

        epoch_g_loss += g_loss.item()
        epoch_g_color_loss += g_loss_color.item()
        n_batches += 1

        pbar.set_postfix({
            'D': f'{d_loss.item():.3f}',
            'G': f'{g_loss.item():.3f}',
            'Color': f'{g_loss_color.item():.2f}'
        })

    sched_G.step()
    sched_D.step()

    avg_d = epoch_d_loss / n_batches
    avg_g = epoch_g_loss / n_batches
    avg_g_color = epoch_g_color_loss / n_batches

    history['epoch'].append(epoch + 1)
    history['d_loss'].append(avg_d)
    history['g_loss'].append(avg_g)
    history['g_color_loss'].append(avg_g_color)

    print(f'Época [{epoch+1:3d}/{N_EPOCHS}] | D_loss: {avg_d:7.4f} | G_loss: {avg_g:7.4f} | Color_loss: {avg_g_color:6.3f}')

    # Guardar imágenes de muestra
    if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == 0:
        G.eval()
        with torch.no_grad():
            samples = G(fixed_z, fixed_labels)
        # Desnormalizar de [-1,1] a [0,1]
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        grid = vutils.make_grid(samples, nrow=N_FIXED, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(figsize=(18, 5))
        ax.imshow(grid_np)
        ax.set_title(f'Época {epoch+1} | Fila 1: Invasivo (0) | Fila 2: In Situ (1)')
        ax.axis('off')
        plt.savefig(f'{OUTPUT_DIR}/samples_epoch_{epoch+1:04d}.png', dpi=100, bbox_inches='tight')
        plt.close()

    # Guardar checkpoint
    if (epoch + 1) % SAVE_EVERY == 0:
        ckpt_data = {
            'epoch': epoch,
            'G': G.state_dict(),
            'D': D.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'history': history
        }
        torch.save(ckpt_data, latest_ckpt)
        torch.save(ckpt_data, f'{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1:04d}.pt')
        print(f'    ✅ Checkpoint guardado: época {epoch+1}')

print('\n✅ Entrenamiento completado!')


"""## **Curvas de Pérdida**


Las curvas de pérdida son la principal herramienta de diagnóstico del entrenamiento de una GAN. A diferencia de los modelos supervisados clásicos, donde una pérdida decreciente siempre indica mejora, en las GANs la interpretación es más compleja porque las dos redes compiten entre sí.

**¿Qué debería verse en un entrenamiento saludable con WGAN-GP?**
La pérdida del discriminador debería estabilizarse en valores cercanos a cero o ligeramente negativos. Esto indica que el discriminador ve las imágenes reales y falsas como igualmente convincentes, que es exactamente el equilibrio que buscamos: el generador ha aprendido a engañarle bien.
La pérdida del generador debería ir haciéndose progresivamente más negativa a lo largo del entrenamiento, indicando que sus imágenes reciben puntuaciones cada vez más altas por parte del discriminador.

**¿Qué indicaría un problema?**
Si la pérdida del discriminador se va muy a negativo y no remonta, significa que el discriminador siempre gana y el generador no está aprendiendo nada útil. Si ambas pérdidas oscilan de forma caótica sin estabilizarse, indica inestabilidad numérica. Si la pérdida del generador colapsa de repente a un valor fijo y no cambia más, es un síntoma de mode collapse, es decir, que el generador ha aprendido a producir siempre la misma imagen.


"""

# CELDA 16 - Visualización de curvas de entrenamiento

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(history['epoch'], history['d_loss'], color='#e74c3c', label='Discriminador')
ax1.plot(history['epoch'], history['g_loss'], color='#3498db', label='Generador')
ax1.set_title('Pérdidas WGAN-GP')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(alpha=0.3)

# Suavizado (media móvil)
window = max(1, len(history['epoch']) // 20)
d_smooth = pd.Series(history['d_loss']).rolling(window, min_periods=1).mean()
g_smooth = pd.Series(history['g_loss']).rolling(window, min_periods=1).mean()
ax2.plot(history['epoch'], d_smooth, color='#e74c3c', label='D (suavizado)')
ax2.plot(history['epoch'], g_smooth, color='#3498db', label='G (suavizado)')
ax2.set_title('Pérdidas suavizadas (media móvil)')
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

"""##**Generación de Imágenes Etiquetadas**

Una vez la GAN haya aprendido, esta celda la usa para fabricar imágenes nuevas. Le dices cuántas quieres de cada clase y el Generador las produce.
Cada imagen se guarda en una carpeta según su clase (invasivo o in situ) y con un nombre de archivo que indica también la clase, así nunca se pierde la etiqueta.
Al final se genera también un CSV con la lista de todas las imágenes generadas y su clase correspondiente, para uqe la CNN futura pueda usarlas fácilmente.
"""

# CELDA 17 - Generación y guardado de imágenes etiquetadas

N_GENERATE_PER_CLASS = 2000  # Imágenes a generar por clase

G.eval()

class_dirs = {
    0: f'{GENERATED_DIR}/class_0_invasivo',
    1: f'{GENERATED_DIR}/class_1_in_situ'
}
class_names = {0: 'invasivo', 1: 'in_situ'}

# Metadata para guardar (útil para la CNN futura)
generated_records = []

with torch.no_grad():
    for class_idx in range(NUM_CLASSES):
        print(f'Generando clase {class_idx} ({class_names[class_idx]})...')
        n_generated = 0

        pbar = tqdm(total=N_GENERATE_PER_CLASS,
                    desc=f'  Clase {class_names[class_idx]}')

        while n_generated < N_GENERATE_PER_CLASS:
            batch_size = min(BATCH_SIZE, N_GENERATE_PER_CLASS - n_generated)
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=DEVICE)

            fake_imgs = G(z, labels)
            fake_imgs = (fake_imgs * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

            for i in range(batch_size):
                img_tensor = fake_imgs[i].cpu()
                img_pil = transforms.ToPILImage()(img_tensor)

                filename = f'gen_{class_names[class_idx]}_{n_generated:05d}.png'
                save_path = os.path.join(class_dirs[class_idx], filename)
                img_pil.save(save_path)

                generated_records.append({
                    'image_path': save_path,
                    'label': class_idx,
                    'label_name': class_names[class_idx],
                    'source': 'GAN_generated'
                })
                n_generated += 1
                pbar.update(1)

        pbar.close()
        print(f'   {n_generated} imágenes generadas para {class_names[class_idx]}')

# Guardar metadata CSV de imágenes generadas
df_generated = pd.DataFrame(generated_records)
csv_path = f'{GENERATED_DIR}/generated_metadata.csv'
df_generated.to_csv(csv_path, index=False)

print(f'\n Generación completada!')
print(f'   Total imágenes generadas: {len(df_generated)}')
print(f'   Guardadas en:             {GENERATED_DIR}')
print(f'   Metadata CSV:             {csv_path}')

"""##**Visualización final de imágenes generadas**

Esta celda genera una imagen comparativa con na fila de imágenes reales y debajo una fila de imágenes generadas por la GAN, para cada clase.
"""

# CELDA 18 - Comparación Real vs Generado

def show_real_vs_generated(df_real, df_gen, class_idx, class_name, n=6):
    fig, axes = plt.subplots(2, n, figsize=(2.5*n, 6))
    fig.suptitle(f'Real vs Generado | Clase: {class_name}', fontsize=13, fontweight='bold')

    # Fila 1: Reales
    sample_real = df_real[df_real['label'] == class_idx].sample(n, random_state=RANDOM_SEED)
    for ax, (_, row) in zip(axes[0], sample_real.iterrows()):
        try:
            img = Image.open(row['image_path']).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'Error', ha='center')
        ax.set_title('Real', fontsize=8, color='green')
        ax.axis('off')

    # Fila 2: Generadas
    sample_gen = df_gen[df_gen['label'] == class_idx].sample(n, random_state=RANDOM_SEED)
    for ax, (_, row) in zip(axes[1], sample_gen.iterrows()):
        try:
            img = Image.open(row['image_path']).convert('RGB')
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'Error', ha='center')
        ax.set_title('Generada', fontsize=8, color='red')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/comparison_{class_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

show_real_vs_generated(df_train, df_generated, 0, 'invasivo')
show_real_vs_generated(df_train, df_generated, 1, 'in_situ')

"""##**Resumen final**"""

# CELDA 19 - Resumen final

print('=' * 60)
print('  RESUMEN DEL EXPERIMENTO GAN')
print('=' * 60)
print(f'\n DATASETS USADOS:')
print(df_all.groupby(['source', 'label_name']).size().to_string())
print(f'\n   Total imágenes reales:     {len(df_all)}')
print(f'   Usadas para entrenar GAN:  {len(df_train)}')
print(f'   Reservadas para CNN (test):{len(df_test)}')

print(f'\n ARQUITECTURA:')
print(f'   Generador:      BigGAN + StyleGAN2 | {n_params_G/1e6:.1f}M params')
print(f'   Discriminador:  BigGAN Projection  | {n_params_D/1e6:.1f}M params')
print(f'   Loss:           WGAN-GP (λ={GP_LAMBDA})')
print(f'   Resolución:     {IMAGE_SIZE}x{IMAGE_SIZE}')

print(f'\n  IMÁGENES GENERADAS:')
if len(df_generated) > 0:
    print(df_generated.groupby('label_name').size().to_string())
    print(f'   Metadata:  {GENERATED_DIR}/generated_metadata.csv')

print(f'\n ARCHIVOS GUARDADOS:')
print(f'   {OUTPUT_DIR}/')
print(f'   ├── checkpoints/          # Modelos guardados')
print(f'   ├── generated_images/     # Imágenes generadas (etiquetadas por carpeta)')
print(f'   ├── split_data/           # CSVs de train/test splits')
print(f'   ├── eda_distribucion.png  # Gráficas EDA')
print(f'   ├── training_curves.png   # Curvas de pérdida')
print(f'   └── samples_epoch_*.png   # Muestras por época')

