# Documentación Técnica - TFG - Generación de Imágenes de Melanoma Invasivo vs In Situ con GAN y Clasificación con CNN

## Índice
1. [Arquitectura GAN](#arquitectura-gan)
2. [Arquitectura CNN](#arquitectura-cnn)
3. [Pipeline de Selección de Imágenes](#pipeline-de-selección)
4. [Estrategia de Datos](#estrategia-de-datos)
5. [Métricas y Evaluación](#métricas)

---

## Arquitectura GAN

### Descripción General

La **Generador es una combinación de BigGAN + StyleGAN2** optimizado para:
- Generar imágenes 256×256 de alta resolución
- Aprender características específicas de cada clase (invasivo vs in situ)
- Estabilidad en el entrenamiento mediante WGAN-GP

### Generador (G)

```
Input: z ~ N(0, 1) [latent vector, 128-dim]
       c [class label: 0=invasivo, 1=in situ]

Arquitectura:
  1. Embedding de clase → 128-dim
  2. Concatenar z + class embedding
  3. Dense: 256×256 [4096 dim] → reshape a 4×4×256
  
  4. Bloques de upsampling (4×4 → 256×256):
     - ReLU + Conv3×3 + BatchNorm
     - Nearest Neighbor Upsample (2×)
     - Repetir 6 veces
  
  5. Normalización final: Tanh → [-1, 1]

Total parámetros: ~52M
```

**Hiperparámetros GAN:**
- Learning rate G: 1e-4 (TTUR - Two Time-scale Update Rule)
- Learning rate D: 4e-4 (discriminador aprende más rápido)
- Optimizer: Adam(β₁=0.0, β₂=0.999)
- Batch size: 64
- Latent dimension: 128
- Image size: 256×256

### Discriminador (D)

```
Input: imagen real/falsa [256×256×3]
       c [class label]

Arquitectura:
  1. Conv 3×3 [256→128 channels] + ReLU
  2. Bloques de downsampling (256×256 → 4×4):
     - Conv 3×3 + ReLU + Spectral Norm
     - MaxPool 2×2
     - Repetir 6 veces
  
  3. Global Average Pooling
  4. Dense: 128 → 1 (prediction score)
  
  5. Class conditioning: Proyección del embedding de clase
     Score final = base_score + class_projection

Total parámetros: ~32M
```

**Loss Function: WGAN-GP (Wasserstein GAN - Gradient Penalty)**

```
L_D = -E[D(real)] + E[D(fake)] + λ * GP

donde:
- E[D(real)]: Score del discriminador en imágenes reales
- E[D(fake)]: Score del discriminador en imágenes falsas
- GP (Gradient Penalty): Penaliza gradientes muy grandes
  GP = E[(∇ₓ D(x̂) - 1)²]  donde x̂ = tReal + (1-t)tFake
- λ = 10 (peso del gradient penalty)

L_G = -E[D(fake)]

Objetivo:
- Discriminador: minimizar L_D
- Generador: maximizar L_G
```

**N_CRITIC = 3:** El discriminador realiza 3 pasos por cada paso del generador.

---

## Arquitectura CNN

### VGG16 Preentrenado + Transfer Learning

```
Input: imagen [256×256×3]
       (normalizado con ImageNet stats)

Arquitectura base (VGG16):
  ├── Bloques de convolución (13 capas)
  │   └── Extraen features progresivamente
  │
  ├── Clasificador original:
  │   ├── Dense 4096 → ReLU → Dropout 0.5
  │   ├── Dense 4096 → ReLU → Dropout 0.5
  │   └── Dense 1000 (ImageNet)
  │
  └── Eliminar clasificador

Capas añadidas (transfer learning):
  1. GlobalAveragePooling2D() [reduce 7×7×512 → 512]
  2. Dense 512 → ReLU
  3. Dropout 0.25
  4. Dense 2 → Softmax [invasivo, in_situ]

Entrenamiento:
  ├── Freeze: todas las capas convolucionales (VGG16 preentrenado)
  ├── Trainable: solo las 3 capas finales añadidas
  └── Razón: aprovechar features preaprendidas de ImageNet

Total parámetros:
  - VGG16 preentrenado: 134M (congelados)
  - Nuevas capas: ~262K (entrenables)
  - Total trainable: 262K
```

**Hiperparámetros CNN:**

```
Learning rate:        1e-3 (Adam)
Loss:                 Binary Crossentropy (2 clases)
Batch size:           32
Epochs:               500 (con early stopping)
Early stopping:       patience=20 (monitorear val_loss)
Data augmentation:    Sí (flag --augment)
  ├── RandomFlip (horizontal y vertical)
  ├── RandomRotation (20%)
  ├── RandomZoom (10%)
  └── RandomContrast (10%)

Callbacks:
  ├── ModelCheckpoint: guardar mejor modelo (val_loss mín)
  └── EarlyStopping: detener si val_loss no mejora en 20 épocas
```

---

## Pipeline de Selección de Imágenes

### Algoritmo de Centroide

**Objetivo:** Seleccionar las imágenes sintéticas más cercanas al espacio de características real.

```
PASO 1: EXTRAER CARACTERÍSTICAS DE IMÁGENES REALES
  ├── Cargar ResNet50 preentrenado (ImageNet)
  ├── Extraer features antes de clasificador: 2048-dim
  ├── Normalizar: (x - mean) / std
  ├── Calcular centroide por clase:
  │   centroid_class = mean(features_normalized)
  └── Guardar: mean, std (para normalizar sintéticas igual)

PASO 2: PROCESAR IMÁGENES GENERADAS
  ├── Para cada imagen generada:
  │   ├── Extraer features con ResNet50 (mismo modelo)
  │   ├── Normalizar con estadísticas de reales
  │   ├── Calcular distancia euclidiana al centroide:
  │   │   distance = ||features_normalized - centroid||₂
  │   └── Guardar (imagen_path, distancia, clase)

PASO 3: ORDENAR Y SELECCIONAR
  ├── Ordenar todas las imágenes por distancia (ascendente)
  ├── TOP-N = primeras N imágenes (más cercanas al centroide)
  ├── Generar múltiples datasets:
  │   CNN-1: 200 imágenes sintéticas
  │   CNN-2: 400 imágenes sintéticas
  │   ...
  │   CNN-11: 2000 imágenes sintéticas
  │
  ├── Para cada CNN-i:
  │   ├── Crear carpetas: class_0_invasivo/, class_1_in_situ/
  │   ├── Copiar imágenes seleccionadas
  │   └── Generar CSV metadata

PASO 4: ENTRENAR CNNs CON DIFERENTES TAMAÑOS
  └── Evaluar impacto del aumento sintético progresivo
```

**Distancia Euclidiana:**

```
features_real ∈ ℝ²⁰⁴⁸
centroid_class ∈ ℝ²⁰⁴⁸

distance = √( Σᵢ₌₁²⁰⁴⁸ (fᵢ - centroidᵢ)² )
```

**Normalización:**

```
Para mantener distribuciones consistentes:

features_normalized = (features - mean_real) / (std_real + ε)

donde ε = 1e-8 (evitar división por cero)

Se usa la misma mean y std de imágenes reales para ambas
(imágenes sintéticas y reales), garantizando consistencia.
```

---

## Estrategia de Datos

### División de Datos

```
Dataset Original: 3,945 imágenes
├── 20% (789) → Reservado para test final CNN
│   └── NO usado en entrenamiento GAN
│
└── 80% (3,156) → Entrenamiento GAN
    ├── Distribución:
    │   ├── Invasivo: 1,755 (55.6%)
    │   └── In Situ: 1,401 (44.4%)
    │
    └── Resultado GAN:
        ├── ~2,000 imágenes sintéticas por clase
        ├── Total: ~4,000 imágenes generadas
        │
        └── Para CNN: DATASETS AUMENTADOS
            ├── Dataset 1: 200 sintéticas + reales
            ├── Dataset 2: 400 sintéticas + reales
            ├── ...
            └── Dataset 11: 2,000 sintéticas + reales

CNN: División train/val/test
├── 70% Training   → ajustar pesos
├── 15% Validation → monitorear val_loss (early stopping)
└── 15% Test       → evaluar métricas finales
```

### Balanceo de Clases

**Problema:** Desbalance de clases (~1.25:1 invasivo/in_situ)

**Solución:**

1. **Weighted Loss en CNN:**
   ```
   class_weight = {
       0: n_total / (2 * n_invasivo),    # invasivo menos pesado
       1: n_total / (2 * n_in_situ)      # in_situ más pesado
   }
   ```
   
2. **Data Augmentation:**
   - RandomFlip, RandomRotation, RandomZoom, RandomContrast
   - Aumenta variabilidad en conjunto de entrenamiento

3. **Aumento Sintético (GAN):**
   - Genera ~2,000 imágenes por clase
   - Balanceo perfecto: 50% invasivo, 50% in_situ
   - Inyecta al conjunto de entrenamiento

---

## Métricas y Evaluación

### Métricas de Clasificación

```
TP  = True Positives  (predicción correcta: positivo)
TN  = True Negatives  (predicción correcta: negativo)
FP  = False Positives (error: predicción pos, real neg)
FN  = False Negatives (error: predicción neg, real pos)

Accuracy = (TP + TN) / (TP + TN + FP + FN)
           → Proporción general correcta

Precision = TP / (TP + FP)
            → De las predicciones positivas, ¿cuántas correctas?
            → Crítico si falso positivo es costoso

Recall = TP / (TP + FN)
         → De los positivos reales, ¿cuántos detectados?
         → Crítico si falso negativo es costoso

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
           → Media armónica, balance entre Precision y Recall

AUC-ROC = Área bajo la curva Receiver Operating Characteristic
          → Robustez a cambios de threshold
          → Rango [0, 1]: 1 = perfecto, 0.5 = aleatorio
```

### Matriz de Confusión

```
                Predicción
            Invasivo  In Situ
Invasivo      TP        FN
Real
In Situ       FP        TN

Interpretación:
  - Diagonal principal: predicciones correctas
  - Off-diagonal: errores por clase
```

### Curva ROC

```
- Eje X: False Positive Rate (1 - Specificity)
- Eje Y: True Positive Rate (Sensitivity)

El clasificador "ideal" tiene TPR ≈ 1, FPR ≈ 0
(esquina superior izquierda).

AUC = área bajo la curva
  - AUC = 1.0 → clasificador perfecto
  - AUC = 0.5 → clasificador aleatorio
```

---

## Referencias Internas

- `src/generator/entrenamiento_gan_melanoma.py` — Implementación GAN completa
- `src/classifier/cnn.py` — Entrenamiento CNN con VGG16
- `src/preprocessing/seleccion_imagenes_algoritmo_centroide.py` — Selección por centroide
- Outputs: gráficas de entrenamiento, matrices confusión, ROC curves

---

**Documento generado:** Mayo 2026  
