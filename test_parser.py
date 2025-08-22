#!/usr/bin/env python3
"""
Test parser for classifier output
"""

import re

# Exemplo do output real do classificador (do que vimos anteriormente)
sample_output = """
[TARGET] IMPROVED BIOMECH CLASSIFIER INICIALIZADO:
  [OK] Sistema hierárquico de confiança implementado
  [CONFIG] Threshold biomecânico: 0.50 (mais rigoroso)
  [TARGET] BD threshold: 100 (mais seletivo)
  [NEW] FP anômalo: detecção específica implementada
  [RESULT] Meta: 95%+ acurácia (resolver Maharu_FP + PingSkills_FP)

[TARGET] === IMPROVED BIOMECH CLASSIFIER ===
[FILE] Analisando: b178e4be38506d2136a8f0979d25e6e3.mp4

[PROCESS] Executando Fase 1...
[OK] Fase 1: voltado_para_esquerda, mao_direita, esquerda

[PROCESS] Executando Fase 2...

[RESULT] === RESULTADO HIERÁRQUICO MELHORADO ===
[TENNIS] Movimento: forehand_drive
[STATS] Confiança: 84.6% (medium)
[TARGET] Zona: complex
[BUILD] Nível hierárquico: fallback
[CONFIG] Regra aplicada: amplitude_based
[BIOMECH] Biomecânica usada: NÃO
[LOGIC] Lógica: Forehand Drive: zona complexa, Y=0.236 + V normal

[RESULT] === RESULTADO HIERÁRQUICO MELHORADO ===
[TENNIS] Tipo de movimento: forehand_drive
[STATS] Confiança: 84.6% (medium)
[BUILD] Nível hierárquico: fallback
[HAND] Mão ativa: direita
[TARGET] Zona: complex
[CONFIG] Regra aplicada: amplitude_based
[BIOMECH] Biomecânica: NÃO
[TIME] Padrão temporal: quick_push
[STATS] Assinatura movimento: 0.605
"""

def test_parser(output):
    """Test the parsing patterns"""
    print("=== TESTING PARSER ===")
    
    detected_info = {
        'movement_type': 'unknown',
        'dominant_hand': 'unknown',
        'orientation': 'unknown', 
        'camera_perspective': 'unknown'
    }

    # 1. MOVIMENTO (buscar por "[TENNIS] Tipo de movimento: forehand_drive")
    movement_patterns = [
        r'\[TENNIS\] Tipo de movimento:\s*([a-zA-Z_]+)',
        r'\[TENNIS\] Movimento:\s*([a-zA-Z_]+)',
        r'TENNIS Tipo de movimento:\s*([a-zA-Z_]+)',
        r'TENNIS Movimento:\s*([a-zA-Z_]+)',
        r'Movimento:\s*([a-zA-Z_]+)'
    ]
    
    for i, pattern in enumerate(movement_patterns):
        movement_match = re.search(pattern, output, re.IGNORECASE)
        print(f"   [DEBUG] Testando padrão {i+1}: {pattern}")
        if movement_match:
            movement_value = movement_match.group(1)
            print(f"   [MATCH] Encontrado: '{movement_value}' com padrão {i+1}")
            # Validar se é um movimento válido (não número ou string vazia)
            if movement_value and not movement_value.isdigit() and movement_value != '0':
                detected_info['movement_type'] = movement_value
                print(f"   [OK] Movimento detectado: {detected_info['movement_type']}")
                break
            else:
                print(f"   [WARNING] Movimento inválido detectado: {movement_value}, continuando busca...")
        else:
            print(f"   [NO_MATCH] Padrão {i+1} não encontrou nada")

    # 2. MAO DOMINANTE (buscar por "mao_direita" ou "mao_esquerda")
    if 'mao_direita' in output:
        detected_info['dominant_hand'] = 'D'
        print(f"   [OK] Mao dominante detectada: Direita (D)")
    elif 'mao_esquerda' in output:
        detected_info['dominant_hand'] = 'E'
        print(f"   [OK] Mao dominante detectada: Esquerda (E)")

    # 3. ORIENTACAO (buscar por "voltado_para_direita" ou "voltado_para_esquerda")
    if 'voltado_para_direita' in output:
        detected_info['orientation'] = 'direita'
        print(f"   [OK] Orientacao detectada: voltado para direita")
    elif 'voltado_para_esquerda' in output:
        detected_info['orientation'] = 'esquerda'
        print(f"   [OK] Orientacao detectada: voltado para esquerda")

    # 4. PERSPECTIVA DA CAMERA (buscar por "Perspectiva final: direita/esquerda")
    perspective_patterns = [
        r'\[TENNIS\] Perspectiva final:\s*(\w+)',
        r'Perspectiva final:\s*(\w+)',
        r'TENNIS Perspectiva final:\s*(\w+)'
    ]
    
    for pattern in perspective_patterns:
        perspective_match = re.search(pattern, output, re.IGNORECASE)
        if perspective_match:
            perspective = perspective_match.group(1).lower()
            if perspective == 'direita':
                detected_info['camera_perspective'] = 'D'
            elif perspective == 'esquerda':
                detected_info['camera_perspective'] = 'E'
            print(f"   [OK] Perspectiva detectada: {perspective} ({detected_info['camera_perspective']})")
            break

    print(f"\n=== RESULTADO FINAL ===")
    print(f"Info COMPLETA extraida: {detected_info}")
    return detected_info

if __name__ == "__main__":
    result = test_parser(sample_output)
    print(f"\nMovimento detectado: {result['movement_type']}")
    print(f"Esperado: forehand_drive")
    print(f"Funcionou: {'SIM' if result['movement_type'] == 'forehand_drive' else 'NÃO'}")