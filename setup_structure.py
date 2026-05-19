#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inicializar estructura de carpetas del repositorio TFG
Ejecutar una sola vez al clonar el repositorio
"""

import os
from pathlib import Path

# Estructura de directorios recomendada
STRUCTURE = {
    'src': {
        'generator': {
            'files': [
                'entrenamiento_gan_melanoma.py',
                'arquitectura_gan.py',
                'utils_gan.py',
                '__init__.py'
            ]
        },
        'classifier': {
            'files': [
                'cnn.py',
                'arquitectura_cnn.py',
                'utils_classifier.py',
                '__init__.py'
            ]
        },
        'preprocessing': {
            'files': [
                'seleccion_imagenes_algoritmo_centroide.py',
                'data_loader.py',
                'normalizacion.py',
                '__init__.py'
            ]
        }
    },
    'data': {
        'info': 'Datasets (no incluidos en repositorio, ver README.md)'
    },
    'outputs': {
        'generated_images': {
            'class_0_invasivo': {},
            'class_1_in_situ': {}
        },
        'cnn_results': {
            'info': 'Métricas y gráficas de CNN'
        },
        'figures': {
            'info': 'Gráficas finales para memoria TFG'
        }
    },
    'docs': {
        'files': [
            'metodologia.md',
            'resultados.md'
        ]
    },
    'notebooks': {
        'info': 'Jupyter notebooks para análisis exploratorio (opcional)'
    },
    'tests': {
        'files': [
            'test_gan.py',
            'test_cnn.py',
            '__init__.py'
        ]
    }
}

def create_structure(base_path='.', structure=STRUCTURE, level=0):
    """Crea recursivamente la estructura de directorios"""
    
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        # Crear directorio
        os.makedirs(path, exist_ok=True)
        indent = "  " * level
        print(f"{indent}📁 {name}/")
        
        if isinstance(content, dict):
            # Procesar archivos si existen
            if 'files' in content:
                for file in content['files']:
                    file_path = os.path.join(path, file)
                    # Solo crear si no existe
                    if not os.path.exists(file_path):
                        with open(file_path, 'w') as f:
                            if file.endswith('.py'):
                                f.write(f"# {file}\n# Autogenerado\n")
                            elif file.endswith('.md'):
                                f.write(f"# {file}\n\nTODO\n")
                    print(f"{indent}  └─ {file}")
            
            # Procesar subdirectorios
            if 'info' in content:
                print(f"{indent}  ℹ️  {content['info']}")
            
            # Recursividad
            subdirs = {k: v for k, v in content.items() 
                      if k not in ['files', 'info']}
            if subdirs:
                create_structure(path, subdirs, level + 1)
        else:
            print(f"{indent}  ℹ️  {content}")

def create_gitkeep(base_path='.', structure=STRUCTURE):
    """Crea archivos .gitkeep en directorios que deben versionarse pero están vacíos"""
    
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            # Crear .gitkeep si no hay archivos
            has_files = 'files' in content
            
            if not has_files:
                gitkeep_path = os.path.join(path, '.gitkeep')
                if not os.path.exists(gitkeep_path):
                    open(gitkeep_path, 'a').close()
            
            # Recursividad
            subdirs = {k: v for k, v in content.items() 
                      if k not in ['files', 'info']}
            if subdirs:
                create_gitkeep(path, subdirs)

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════╗
║       Inicializando estructura de directorios TFG              ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Crear estructura
    print("\n📂 Creando directorios:\n")
    create_structure()
    
    # Crear .gitkeep para directorios vacíos
    print("\n✓ Creando .gitkeep en directorios vacíos...\n")
    create_gitkeep()
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                    ¡ESTRUCTURA LISTA!                          ║
╚════════════════════════════════════════════════════════════════╝

Próximos pasos:

1️⃣  Copiar tus archivos .py en las carpetas correspondientes:
    • entrenamiento_gan_melanoma.py → src/generator/
    • cnn.py → src/classifier/
    • seleccion_imagenes_algoritmo_centroide.py → src/preprocessing/

2️⃣  Instalar dependencias:
    pip install -r requirements.txt

3️⃣  Preparar los datos (ver data/README.md)

4️⃣  Ejecutar pipeline:
    python reproduce.py --all

Para más información, consulta README.md
    """)
