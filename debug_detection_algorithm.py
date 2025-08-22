"""
Debug do algoritmo de detecção de movimento
Investigar por que está detectando FP ao invés de FD
"""

def debug_detection_algorithm():
    """Examina detalhadamente o algoritmo de detecção"""
    
    print("🔍 INVESTIGANDO ALGORITMO DE DETECÇÃO...")
    
    # Testar detecção com debug detalhado
    test_code = '''
from tennis_comparison_backend import TennisComparisonEngine
import cv2
import numpy as np

def debug_single_frame_analysis():
    """Debug detalhado da análise de um frame"""
    
    analyzer = TennisComparisonEngine()
    
    # Verificar se método existe
    if hasattr(analyzer, '_analyze_single_frame_movement'):
        print("✅ Método _analyze_single_frame_movement encontrado")
    else:
        print("❌ Método _analyze_single_frame_movement NÃO encontrado")
        return
    
    # Verificar se método de detecção existe
    if hasattr(analyzer, 'analyze_movement_from_content'):
        print("✅ Método analyze_movement_from_content encontrado")
    else:
        print("❌ Método analyze_movement_from_content NÃO encontrado")
        return
    
    print("🔍 Executando análise com debug...")
    
    # Executar análise
    result = analyzer.analyze_movement_from_content('teste_neutro.mp4')
    
    print("📊 RESULTADO COMPLETO:")
    print(f"   Movimento: {result.get('movement', 'N/A')}")
    print(f"   Confiança: {result.get('confidence', 'N/A')}")
    print(f"   Método: {result.get('method', 'N/A')}")
    
    if 'stats' in result:
        stats = result['stats']
        print("📈 ESTATÍSTICAS:")
        print(f"   Total frames: {stats.get('total_frames', 'N/A')}")
        print(f"   Votos: {stats.get('votes', 'N/A')}")
        print(f"   Consistência: {stats.get('consistency', 'N/A')}")
    
    return result

# Executar debug
print("=== DEBUG DA DETECÇÃO ===")
result = debug_single_frame_analysis()

if result and result.get('movement') == 'FP':
    print("\\n🚨 CONFIRMADO: Sistema detecta FP incorretamente!")
    print("🔍 Vamos investigar a lógica de classificação...")
    
    # Verificar lógica de classificação Drive vs Push
    print("\\n=== INVESTIGANDO LÓGICA DRIVE vs PUSH ===")
    print("📋 CRITÉRIOS ESPERADOS:")
    print("   DRIVE: maior amplitude + pulso mais alto")
    print("   PUSH: movimento compacto + pulso mais baixo")
    print("\\n🔧 POSSÍVEIS PROBLEMAS:")
    print("   1. Thresholds incorretos para amplitude")
    print("   2. Lógica invertida de altura do pulso")
    print("   3. Análise de frames específicos incorreta")
    print("   4. Fallback para Push como padrão")

print("\\n" + "="*50)
'''
    
    # Salvar script de debug
    with open('debug_detection.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("✅ Script de debug criado: debug_detection.py")
    print("\n🔍 EXECUTE PARA INVESTIGAR:")
    print("python debug_detection.py")
    
    # Criar script de correção baseado na investigação
    correction_code = '''
"""
Correção do algoritmo de detecção Drive vs Push
Baseado na investigação dos thresholds incorretos
"""

def fix_detection_algorithm():
    """Corrige os thresholds de detecção Drive vs Push"""
    
    import re
    from datetime import datetime
    
    print("🔧 CORRIGINDO ALGORITMO DE DETECÇÃO...")
    
    # Backup
    backup_filename = f"tennis_comparison_backend_detection_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    with open('tennis_comparison_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📁 Backup criado: {backup_filename}")
    
    # Procurar pela lógica de classificação Drive vs Push
    # Padrão típico seria algo como:
    # if horizontal_amplitude > X and wrist_height_relative > Y:
    #     movement = 'FD'  # Drive
    # else:
    #     movement = 'FP'  # Push
    
    # Corrigir thresholds para melhor detecção de Drive
    fixes = [
        # Reduzir threshold de amplitude para Drive (era muito alto)
        (r'horizontal_amplitude > 0\\.3', 'horizontal_amplitude > 0.2'),
        
        # Reduzir threshold de altura para Drive (era muito alto)
        (r'wrist_height_relative > 0\\.1', 'wrist_height_relative > 0.05'),
        
        # Para Backhand Drive, ajustar também
        (r'horizontal_amplitude > 0\\.25 and wrist_height_relative > 0\\.05', 
         'horizontal_amplitude > 0.15 and wrist_height_relative > 0.02'),
        
        # Adicionar debug na classificação
        (r'# CLASSIFICAÇÃO', 
         '''# CLASSIFICAÇÃO - com debug detalhado
        print(f"🔍 DEBUG CLASSIFICAÇÃO:")
        print(f"   wrist_cross_body: {wrist_cross_body}")
        print(f"   horizontal_amplitude: {horizontal_amplitude:.3f}")
        print(f"   wrist_height_relative: {wrist_height_relative:.3f}")'''),
    ]
    
    changes_made = 0
    for pattern, replacement in fixes:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes_made += 1
            print(f"✅ Ajuste aplicado: {pattern}")
    
    if changes_made == 0:
        print("⚠️ Nenhuma alteração automática aplicada")
        print("📋 POSSÍVEIS PROBLEMAS:")
        print("1. Lógica de classificação diferente do esperado")
        print("2. Thresholds em local diferente")
        print("3. Algoritmo usa método completamente diferente")
        print("\\n🔍 MANUAL: Abra tennis_comparison_backend.py e procure por:")
        print("- _analyze_single_frame_movement")
        print("- horizontal_amplitude")
        print("- wrist_height_relative")
        print("- CLASSIFICAÇÃO")
    else:
        # Salvar arquivo corrigido
        with open('tennis_comparison_backend.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {changes_made} correções aplicadas!")
        print(f"💾 Backup disponível: {backup_filename}")
        
        print("\\n🧪 TESTE NOVAMENTE:")
        print("python -c \"")
        print("from tennis_comparison_backend import TennisComparisonEngine")
        print("analyzer = TennisComparisonEngine()")
        print("result = analyzer.analyze_movement_from_content('teste_neutro.mp4')")
        print("print('Novo resultado:', result['movement'])")
        print("\"")

if __name__ == "__main__":
    fix_detection_algorithm()
'''
    
    with open('fix_detection_algorithm.py', 'w', encoding='utf-8') as f:
        f.write(correction_code)
    
    print("✅ Script de correção criado: fix_detection_algorithm.py")
    
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Execute: python debug_detection.py")
    print("2. Analise os resultados")
    print("3. Execute: python fix_detection_algorithm.py")
    print("4. Teste novamente a detecção")
    
    print("\n🔍 O QUE VAMOS INVESTIGAR:")
    print("- Thresholds de amplitude muito altos")
    print("- Critérios de altura do pulso incorretos")
    print("- Lógica invertida Drive vs Push")
    print("- Fallbacks que forçam Push como padrão")

if __name__ == "__main__":
    debug_detection_algorithm()
