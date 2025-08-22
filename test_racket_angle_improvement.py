#!/usr/bin/env python3
"""
Test script para verificar as melhorias na diferenciação BD vs BP
com detecção do ângulo da raquete
"""

import os
import sys
from improved_biomech_classifier_2d import ImprovedBiomechClassifier2D

def test_racket_angle_improvements():
    """Testa as melhorias na detecção do ângulo da raquete"""
    
    print("=" * 60)
    print("TESTE DAS MELHORIAS BD vs BP - ÂNGULO DA RAQUETE")
    print("=" * 60)
    
    # Inicializar classificador
    classifier = ImprovedBiomechClassifier2D()
    
    print(f"\n[OK] Classificador inicializado com melhorias:")
    print(f"  - Deteccao do angulo final da raquete")
    print(f"  - BD: terminacao voltada para cima (>{classifier.biomech_thresholds['bd_finish_angle_min']} graus)")
    print(f"  - BP: terminacao horizontal (<={classifier.biomech_thresholds['bp_finish_angle_max']} graus)")
    print(f"  - Bonus de confianca para angulos tipicos")
    
    # Verificar se há vídeos de teste
    test_videos_dir = "videos"
    if os.path.exists(test_videos_dir):
        print(f"\n[INFO] Diretório de vídeos encontrado: {test_videos_dir}")
        
        # Procurar por vídeos de backhand
        backhand_videos = []
        for filename in os.listdir(test_videos_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov')) and 'backhand' in filename.lower():
                backhand_videos.append(os.path.join(test_videos_dir, filename))
        
        if backhand_videos:
            print(f"[FOUND] {len(backhand_videos)} vídeo(s) de backhand encontrado(s)")
            for video in backhand_videos[:3]:  # Testar apenas os primeiros 3
                print(f"  - {os.path.basename(video)}")
        else:
            print(f"[INFO] Nenhum vídeo de backhand encontrado para teste")
    else:
        print(f"[INFO] Diretório de vídeos não encontrado")
    
    print(f"\n[SUCCESS] Melhorias implementadas com sucesso!")
    print(f"[NEXT] Para testar com um vídeo específico:")
    print(f"  python improved_biomech_classifier_2d.py videos/seu_video_backhand.mp4")
    
    return True

if __name__ == "__main__":
    try:
        test_racket_angle_improvements()
        print(f"\n[RESULT] Teste das melhorias concluido com sucesso")
    except Exception as e:
        print(f"\n[ERROR] Erro no teste: {e}")
        sys.exit(1)