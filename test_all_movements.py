#!/usr/bin/env python3
"""
Teste completo das implementações biomecânicas em todos os tipos de movimento
Testa 2 vídeos de cada tipo: FD, FP, BD, BP
"""

import sys
import os
from improved_biomech_classifier_2d import ImprovedBiomechClassifier2D

def test_video(video_path, expected_type):
    """Testa um vídeo específico"""
    print(f"\n{'='*70}")
    print(f"TESTANDO: {os.path.basename(video_path)}")
    print(f"ESPERADO: {expected_type}")
    print(f"{'='*70}")
    
    try:
        classifier = ImprovedBiomechClassifier2D()
        result = classifier.process_video(video_path)
        
        if result:
            detected_type = result.movement_type.value
            confidence = result.confidence
            logic = result.decision_logic
            
            # Verificar se está correto
            is_correct = detected_type == expected_type
            status = "CORRETO" if is_correct else "INCORRETO"
            
            print(f"\nRESULTADO: {detected_type}")
            print(f"CONFIANÇA: {confidence:.1%}")
            print(f"LÓGICA: {logic}")
            print(f"STATUS: {status}")
            
            return is_correct, detected_type, confidence
        else:
            print("ERRO: Nao foi possivel analisar o video")
            return False, "error", 0.0
            
    except Exception as e:
        print(f"ERRO: {e}")
        return False, "error", 0.0

def main():
    print("TESTE COMPLETO DAS IMPLEMENTACOES BIOMECANICAS")
    print("Testando 2 videos de cada tipo de movimento")
    
    # Definir vídeos de teste - 1 de cada tipo primeiro
    test_videos = [
        # Forehand Drive
        ("videos/Calderano_FD_D_D.mp4", "forehand_drive"),
        
        # Forehand Push  
        ("videos/Maharu_FP_D_E.mp4", "forehand_push"),
        
        # Backhand Drive
        ("videos/Fan_Zhendong_BD_D_D.mp4", "backhand_drive"),
        
        # Backhand Push
        ("videos/Japones_BP_D_D.mp4", "backhand_push"),
    ]
    
    results = []
    correct_count = 0
    total_count = len(test_videos)
    
    # Testar cada vídeo
    for video_path, expected_type in test_videos:
        if os.path.exists(video_path):
            is_correct, detected_type, confidence = test_video(video_path, expected_type)
            results.append({
                'video': os.path.basename(video_path),
                'expected': expected_type,
                'detected': detected_type,
                'confidence': confidence,
                'correct': is_correct
            })
            if is_correct:
                correct_count += 1
        else:
            print(f"VIDEO NAO ENCONTRADO: {video_path}")
    
    # Resumo final
    print(f"\n{'='*70}")
    print("RESUMO DOS RESULTADOS")
    print(f"{'='*70}")
    
    # Agrupar por tipo de movimento
    movement_types = {
        'forehand_drive': [],
        'forehand_push': [], 
        'backhand_drive': [],
        'backhand_push': []
    }
    
    for result in results:
        movement_types[result['expected']].append(result)
    
    # Exibir resultados por tipo
    for movement_type, tests in movement_types.items():
        if tests:
            print(f"\n{movement_type.upper()}:")
            correct_in_type = 0
            for test in tests:
                status = "[OK]" if test['correct'] else "[ERRO]"
                print(f"   {status} {test['video']}: {test['detected']} ({test['confidence']:.1%})")
                if test['correct']:
                    correct_in_type += 1
            print(f"   Precisao: {correct_in_type}/{len(tests)} ({correct_in_type/len(tests):.1%})")
    
    # Estatísticas gerais
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nRESULTADO GERAL:")
    print(f"   Acertos: {correct_count}/{total_count}")
    print(f"   Precisao: {accuracy:.1%}")
    
    if accuracy >= 0.75:
        print("IMPLEMENTACAO FUNCIONANDO BEM!")
    elif accuracy >= 0.50:
        print("IMPLEMENTACAO PRECISA DE AJUSTES")
    else:
        print("IMPLEMENTACAO PRECISA DE CORRECOES SIGNIFICATIVAS")

if __name__ == "__main__":
    main()