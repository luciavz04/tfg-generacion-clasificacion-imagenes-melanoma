#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUÍA RÁPIDA DE REPRODUCCIÓN - TFG Melanoma IA
==============================================

Este script es una guía paso a paso para reproducir el experimento completo.
Incluye comentarios detallados sobre cómo ejecutar cada fase.

FASE 1: Generador GAN (Síntesis de imágenes)
FASE 2: Selector de imágenes por calidad (Centroide)
FASE 3: Clasificador CNN (Evaluación)

Autor: Lucía Vela
Fecha: 2024
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════╗
║          TFG - GENERACIÓN Y CLASIFICACIÓN DE MELANOMA         ║
║                  GUÍA DE REPRODUCCIÓN                         ║
╚════════════════════════════════════════════════════════════════╝
""")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ═════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# Crear directorios si no existen
for d in [SRC_DIR, DATA_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

print(f"\n📁 Directorio raíz: {PROJECT_ROOT}")
print(f"📁 Código fuente: {SRC_DIR}")
print(f"📁 Datos: {DATA_DIR}")
print(f"📁 Outputs: {OUTPUTS_DIR}\n")

# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═════════════════════════════════════════════════════════════════════════════

def print_section(title):
    """Imprime un título de sección formateado"""
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}\n")

def check_dependencies():
    """Verifica que están instaladas las dependencias principales"""
    print_section("VERIFICANDO DEPENDENCIAS")
    
    required = ['torch', 'tensorflow', 'sklearn', 'pandas', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - FALTA INSTALAR")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Faltan dependencias: {', '.join(missing)}")
        print(f"    Instálalas con: pip install -r requirements.txt")
        return False
    
    print(f"\n✅ Todas las dependencias están instaladas\n")
    return True

def run_phase_1():
    """FASE 1: Entrenar GAN"""
    print_section("FASE 1: ENTRENAR GENERADOR GAN")
    
    print("Este paso genera imágenes sintéticas de melanoma.")
    print("⏱️  Tiempo estimado: 24-48 horas en GPU")
    print("💾 Espacio requerido: ~30 GB\n")
    
    script = SRC_DIR / 'generator' / 'entrenamiento_gan_melanoma.py'
    
    if not script.exists():
        print(f"❌ Error: No se encontró {script}")
        return False
    
    # Parámetros por defecto (modificables)
    cmd = [
        'python', str(script),
        '--learning_rate_g', '1e-4',
        '--learning_rate_d', '4e-4',
        '--num_epochs', '500',
        '--output_dir', str(OUTPUTS_DIR)
    ]
    
    print(f"Ejecutando: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n✅ FASE 1 completada: imágenes generadas")
            return True
        else:
            print("\n❌ Error en FASE 1")
            return False
    except Exception as e:
        print(f"❌ Error al ejecutar FASE 1: {e}")
        return False

def run_phase_2():
    """FASE 2: Seleccionar imágenes por calidad"""
    print_section("FASE 2: SELECCIONAR IMÁGENES SINTÉTICAS POR CALIDAD")
    
    print("Ordena imágenes generadas por similitud al espacio de características real.")
    print("✅ Tiempo estimado: 1-2 horas")
    print("⚠️  Requiere: imágenes generadas en outputs/generated_images/\n")
    
    script = SRC_DIR / 'preprocessing' / 'seleccion_imagenes_algoritmo_centroide.py'
    
    if not script.exists():
        print(f"❌ Error: No se encontró {script}")
        return False
    
    # Este script no usa argparse, pero puedes modificar paths internos
    cmd = ['python', str(script)]
    
    print(f"Ejecutando: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n✅ FASE 2 completada: imágenes seleccionadas")
            return True
        else:
            print("\n❌ Error en FASE 2")
            return False
    except Exception as e:
        print(f"❌ Error al ejecutar FASE 2: {e}")
        return False

def run_phase_3():
    """FASE 3: Entrenar clasificador CNN"""
    print_section("FASE 3: ENTRENAR CLASIFICADOR CNN")
    
    print("Entrena VGG16 para clasificar melanoma invasivo vs in situ.")
    print("✅ Tiempo estimado: 2-4 horas")
    print("⚠️  Requiere: datos preprocessados en outputs/\n")
    
    script = SRC_DIR / 'classifier' / 'cnn.py'
    
    if not script.exists():
        print(f"❌ Error: No se encontró {script}")
        return False
    
    cmd = [
        'python', str(script),
        '--epochs', '500',
        '--batch-size', '32',
        '--augment'
    ]
    
    print(f"Ejecutando: {' '.join(cmd)}\n")
    print("Parámetros:")
    print("  • Epochs: 500")
    print("  • Batch size: 32")
    print("  • Data augmentation: Sí\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n✅ FASE 3 completada: CNN entrenada")
            return True
        else:
            print("\n❌ Error en FASE 3")
            return False
    except Exception as e:
        print(f"❌ Error al ejecutar FASE 3: {e}")
        return False

def run_quick_demo():
    """Ejecuta versión demo rápida (sin entrenamiento completo)"""
    print_section("MODO DEMO (Prueba rápida sin entrenamiento completo)")
    
    print("Este modo:")
    print("  • Entrena GAN con 10 épocas (verificar código)")
    print("  • Entrena CNN con 5 épocas (verificar código)")
    print("  • Genera gráficas de ejemplo")
    print("  ⏱️  Tiempo: ~30 minutos\n")
    
    # Para implementar demo, necesitarías scripts separados con parámetros reducidos
    # De momento, mostramos el flujo
    
    print("⚠️  Para activar modo demo, modifica los scripts con:")
    print("    N_EPOCHS = 10  # En lugar de 500")
    print("    BATCH_SIZE = 16")
    print("    N_GENERATE_PER_CLASS = 100  # En lugar de 2000\n")

# ═════════════════════════════════════════════════════════════════════════════
# MENÚ PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Guía de reproducción del TFG - Melanoma IA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS DE USO:

  # Ejecutar todas las fases en orden
  python reproduce.py --all

  # Ejecutar solo la FASE 1 (GAN)
  python reproduce.py --phase 1

  # Ejecutar FASE 2 y 3
  python reproduce.py --phase 2 3

  # Modo demo (rápido, para verificar setup)
  python reproduce.py --demo
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Ejecutar todas las fases (1→2→3)')
    parser.add_argument('--phase', type=int, nargs='+', choices=[1, 2, 3],
                       help='Ejecutar fases específicas (ej: --phase 1 3)')
    parser.add_argument('--demo', action='store_true',
                       help='Modo demo (prueba rápida)')
    parser.add_argument('--check', action='store_true',
                       help='Solo verificar dependencias')
    
    args = parser.parse_args()
    
    # Verificar dependencias siempre
    if not check_dependencies():
        print("\n❌ Por favor instala las dependencias primero:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Solo verificar
    if args.check:
        print("✅ Verificación completada")
        sys.exit(0)
    
    # Demo
    if args.demo:
        run_quick_demo()
        sys.exit(0)
    
    # Sin argumentos: mostrar help
    if not (args.all or args.phase):
        parser.print_help()
        sys.exit(0)
    
    phases_to_run = []
    
    if args.all:
        phases_to_run = [1, 2, 3]
    elif args.phase:
        phases_to_run = sorted(args.phase)
    
    # Ejecutar fases
    print(f"\n✅ Fases a ejecutar: {phases_to_run}\n")
    
    results = {}
    
    if 1 in phases_to_run:
        results[1] = run_phase_1()
        if not results[1]:
            print("⚠️  FASE 1 falló. Continuando...")
    
    if 2 in phases_to_run and (1 not in phases_to_run or results.get(1, True)):
        results[2] = run_phase_2()
        if not results[2]:
            print("⚠️  FASE 2 falló. Continuando...")
    
    if 3 in phases_to_run and (2 not in phases_to_run or results.get(2, True)):
        results[3] = run_phase_3()
    
    # Resumen final
    print_section("RESUMEN DE EJECUCIÓN")
    
    for phase, success in sorted(results.items()):
        status = "✅ Completada" if success else "❌ Falló"
        print(f"  FASE {phase}: {status}")
    
    if all(results.values()):
        print("\n🎉 ¡TODAS LAS FASES COMPLETADAS CORRECTAMENTE!")
        print("\n📊 Revisa los outputs en:")
        print(f"   {OUTPUTS_DIR}")
    else:
        print("\n⚠️  Algunas fases tuvieron errores. Revisa los logs arriba.")
    
    print("\n" + "═" * 70)

if __name__ == '__main__':
    main()
