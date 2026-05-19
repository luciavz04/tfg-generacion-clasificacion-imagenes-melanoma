# Generación y Clasificación de Imágenes de Melanoma Mediante IA

**Trabajo Fin de Grado (TFG)** | Ingeniería de la Salud - mención en Ingeniería Biomédica
Autora: Lucía Vela  
Universidad: Universidad de Sevilla  
Año: 2026

---

## Resumen del Proyecto

Este trabajo implementa un pipeline completo basado en **inteligencia artificial generativa** para abordar el problema del desbalance de clases en la clasificación de imágenes dermoscópicas de melanoma.

### Módulo Principal: **Síntesis Generativa (GAN)**
- **Arquitectura:** GAN Condicional (BigGAN + StyleGAN2)
- **Loss:** WGAN-GP para estabilidad
- **Objetivo:** Generar imágenes sintéticas de melanoma invasivo e in situ de alta calidad

### Módulo Secundario: **Clasificación (CNN)**
- **Arquitectura:** VGG16 preentrenado
- **Objetivo:** Evaluar si el aumento sintético mejora la clasificación

---

## Estructura de Datos

### Datasets Utilizados (4 multi-institucionales)
1. **ISIC 2024**
2. **Hospital Universitario Virgen del Rocío** 
3. **Dataset Polesie**
4. **Dataset Argenciano y Kawahara**

### Distribución de Clases
| Clase | Cantidad | Porcentaje |
|-------|----------|-----------|
| Invasivo | 2,194 | 55.6% |
| In Situ | 1,751 | 44.4% |
| **Total** | **3,945** | **100%** |

### Estrategia de Split
```
Datos originales (3,945)
    ├── 20% Reservado para CNN (789 imágenes)
    └── 80% Para entrenar GAN (3,156 imágenes)
        └── CNN: 70% train, 15% val, 15% test
```

---

## Estructura del Repositorio

```
TFG-Melanoma-IA/
│
├── README.md                             
├── requirements.txt                       
├── .gitignore                            
│
├── src/
│   ├── generator/
│   │   ├── entrenamiento_gan_melanoma.py # Script principal GAN
│   │   ├── arquitectura_gan.py            # Definición de redes
│   │   └── utils_gan.py                   # Funciones auxiliares
│   │
│   ├── classifier/
│   │   ├── cnn.py                         # Script entrenamiento CNN
│   │   ├── arquitectura_cnn.py            # VGG16 adaptado
│   │   └── utils_classifier.py            # Funciones auxiliares
│   │
│   └── preprocessing/
│       ├── seleccion_imagenes_algoritmo_centroide.py
│       └── data_loader.py                 # Carga de datos
│
├── outputs/
│   ├── generated_images/                 
│   │   ├── class_0_invasivo/
│   │   └── class_1_in_situ/
│   │
│   ├── cnn_results/                       
│   │   ├── loss.png
│   │   ├── accuracy.png
│   │   ├── confusion_matrix.png
│   │   └── model.keras
│   │
│   └── figures/                          
│       ├── training_curves.png
│       ├── eda_distribucion.png
│       └── real_vs_generated.png
│
├── data/
│   └── README.md                          
│
└── docs/
    ├── metodologia.md                  
    └── resultados.md                      
```

---

## Instalación y Uso Rápido

### 1. Clonar el Repositorio
### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Entrenar el Generador GAN
```bash
python src/generator/entrenamiento_gan_melanoma.py \
  --learning_rate_g 1e-4 \
  --learning_rate_d 4e-4 \
  --num_epochs 500 \
  --output_dir ./outputs
```

**Parámetros principales:**
- `--learning_rate_g`: Learning rate del generador (default: 1e-4)
- `--learning_rate_d`: Learning rate del discriminador (default: 4e-4)
- `--num_epochs`: Número de épocas (default: 500)
- `--batch_size`: Tamaño de batch (default: 64)
- `--output_dir`: Directorio de salida

### 4. Seleccionar Imágenes Sintéticas por Calidad
```bash
python src/preprocessing/seleccion_imagenes_algoritmo_centroide.py
```

Este script:
- Extrae características con ResNet50
- Calcula centroides por clase
- Ordena imágenes generadas por similitud al espacio de características real
- Genera datasets de N imágenes seleccionadas

### 5. Entrenar Clasificador CNN
```bash
python src/classifier/cnn.py \
  --epochs 500 \
  --batch-size 32 \
  --augment
```

**Parámetros:**
- `--epochs`: Número de épocas (default: 500)
- `--batch-size`: Tamaño de batch (default: 32)
- `--augment`: Activar data augmentation (flag)


---

## Requisitos de Sistema

- **Python:** 3.9+
- **GPU:** CUDA-compatible (NVIDIA recomendado)
- **Memoria RAM:** 16 GB+
- **Espacio disco:** ~50 GB (para datasets + modelos + outputs)

### Dependencias Principales
```
torch >= 2.0.0
tensorflow >= 2.12.0
torchvision >= 0.15.0
scikit-learn >= 1.3.0
pandas >= 1.5.0
matplotlib >= 3.7.0
PIL >= 9.0.0
tqdm >= 4.65.0
```

---

## Figuras y Gráficas Incluidas

### Análisis Exploratorio (EDA)
- Distribución de clases por dataset
- Estadísticas de tamaño de imagen
- Análisis de desbalance

### Entrenamiento GAN
- Curvas de pérdida (discriminador vs generador)
- Muestras generadas por época
- Comparación real vs sintético

### Evaluación CNN
- Matriz de confusión
- Curvas ROC
- Gráficas de precisión/recall
- Ejemplos de predicciones correctas/incorrectas

---

## 🎓 Citación

Si utilizas este código o resultados en tu investigación, cita:

```bibtex
@thesis{vela2026melanoma,
  author = {Vela Zambrano, Lucía},
  title = {Generación de Imágenes de Melanoma Invasivo vs In Situ con GAN y Clasificación con CNN},
  school = {Universidad de Sevilla},
  year = {2026}
}
```

---

## Licencia

Este proyecto está bajo licencia **MIT**. Consulta `LICENSE` para detalles.

---

## 📧 Contacto

- **Autora:** Lucía Vela
- **Email:** luciavela04@gmail.com
- **GitHub:** [@tu-usuario](https://github.com/luciavz04)

---

## Limitaciones y Trabajos Futuros

### Limitaciones Actuales
- Imágenes sintéticas mejor para clase mayoritaria (invasivo)
- Tiempo de entrenamiento GAN elevado (48h GPU)
- Dependencia de arquitectura VGG16 (más antiguo)

### Trabajos Futuros
- Explorar Vision Transformers para clasificación
- Difusión en lugar de GAN (mejor calidad)
- Fine-tuning de parámetros GAN por clase
- Validación clínica con dermatólogos

---

**Última actualización:** Mayo 2026
