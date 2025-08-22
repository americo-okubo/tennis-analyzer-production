#!/usr/bin/env python3
"""
Debug script para verificar se a detecção do ângulo da raquete está funcionando
"""

import sys
import os
from improved_biomech_classifier_2d import ImprovedBiomechClassifier2D

def debug_racket_angle_detection():
    """Debug da detecção do ângulo da raquete"""
    
    print("=" * 70)
    print("DEBUG: DETECÇÃO DO ÂNGULO DA RAQUETE")
    print("=" * 70)
    
    classifier = ImprovedBiomechClassifier2D()
    
    # Mostrar thresholds relevantes
    print(f"\n[CONFIG] Thresholds de ângulo da raquete:")
    print(f"  - BD mínimo: {classifier.biomech_thresholds['bd_finish_angle_min']}°")
    print(f"  - BD ideal: {classifier.biomech_thresholds['bd_finish_angle_ideal']}°")
    print(f"  - BP máximo: {classifier.biomech_thresholds['bp_finish_angle_max']}°")
    print(f"  - Confiança mínima: {classifier.biomech_thresholds['racket_angle_confidence_min']}")
    
    # Mostrar ordem dos discriminadores
    print(f"\n[INFO] Ordem dos discriminadores:")
    print(f"  1. BACKHAND DRIVE (prioridade por ângulo upward)")
    print(f"  2. BACKHAND PUSH (apenas se não upward)")
    print(f"  3. FOREHAND PUSH anômalo")
    print(f"  4. FOREHAND PUSH normal")
    
    print(f"\n[NEXT] Para debug completo, analise um vídeo:")
    print(f"  python improved_biomech_classifier_2d.py videos/seu_video.mp4")
    print(f"\n[LOOK] Procure por estas mensagens de debug:")
    print(f"  - [RACKET] Ângulo final médio: X.X°, Tipo: upward_finish/horizontal_finish")
    print(f"  - [OK] Ângulo típico de BD: X.X° (upward), bonus: +0.XX")
    print(f"  - [WARNING] Ângulo inconsistente com BD: X.X° (horizontal)")
    print(f"  - [OK] DISCRIMINADOR BD: ...")
    
    return True

if __name__ == "__main__":
    debug_racket_angle_detection()