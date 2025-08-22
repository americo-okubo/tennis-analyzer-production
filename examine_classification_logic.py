"""
Examinar e corrigir a lógica de classificação Drive vs Push
O algoritmo está classificando Drive como Push incorretamente
"""

def examine_and_fix_classification():
    """Examina a lógica atual e corrige os critérios"""
    
    print("🔍 EXAMINANDO LÓGICA DE CLASSIFICAÇÃO DRIVE vs PUSH...")
    
    # Mostrar onde está a lógica atual
    examination_code = '''
# Vamos examinar o arquivo tennis_comparison_backend.py
import re

def show_current_classification_logic():
    """Mostra a lógica atual de classificação"""
    
    try:
        with open('tennis_comparison_backend.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Procurar pelo método _analyze_single_frame_movement
        method_match = re.search(
            r'def _analyze_single_frame_movement\(self, landmarks\):(.*?)(?=def|\Z)', 
            content, 
            re.DOTALL
        )
        
        if method_match:
            method_content = method_match.group(1)
            print("📋 MÉTODO _analyze_single_frame_movement ENCONTRADO:")
            print("="*60)
            
            # Procurar pela seção de classificação
            classification_match = re.search(
                r'# CLASSIFICAÇÃO(.*?)return {', 
                method_content, 
                re.DOTALL
            )
            
            if classification_match:
                classification_logic = classification_match.group(1)
                print("🎯 LÓGICA DE CLASSIFICAÇÃO ATUAL:")
                print(classification_logic)
                
                # Analisar problemas
                print("\\n🔍 ANÁLISE DOS PROBLEMAS:")
                
                if 'horizontal_amplitude > 0.3' in classification_logic:
                    print("❌ PROBLEMA 1: Threshold muito alto (0.3) para Drive")
                    print("   → Movimentos Drive ficam abaixo e viram Push")
                
                if 'wrist_height_relative > 0.1' in classification_logic:
                    print("❌ PROBLEMA 2: Altura muito alta (0.1) para Drive")
                    print("   → Pulsos que não atingem altura viram Push")
                
                if 'horizontal_amplitude > 0.25 and wrist_height_relative > 0.05' in classification_logic:
                    print("❌ PROBLEMA 3: Critérios Backhand também muito altos")
                
                return True
            else:
                print("❌ Seção CLASSIFICAÇÃO não encontrada")
                return False
        else:
            print("❌ Método _analyze_single_frame_movement não encontrado")
            return False
            
    except FileNotFoundError:
        print("❌ Arquivo tennis_comparison_backend.py não encontrado")
        return False

def fix_classification_thresholds():
    """Corrige os thresholds de classificação"""
    
    import re
    from datetime import datetime
    
    print("\\n🔧 CORRIGINDO THRESHOLDS DE CLASSIFICAÇÃO...")
    
    # Backup
    backup_filename = f"tennis_comparison_backend_threshold_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    try:
        with open('tennis_comparison_backend.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(backup_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📁 Backup criado: {backup_filename}")
        
        # Correções específicas
        fixes_applied = []
        
        # 1. Reduzir threshold horizontal para Forehand Drive
        if re.search(r'horizontal_amplitude > 0\\.3 and wrist_height_relative > 0\\.1', content):
            content = re.sub(
                r'horizontal_amplitude > 0\\.3 and wrist_height_relative > 0\\.1',
                'horizontal_amplitude > 0.15 and wrist_height_relative > 0.03',
                content
            )
            fixes_applied.append("Forehand Drive thresholds: 0.3→0.15, 0.1→0.03")
        
        # 2. Reduzir threshold para Backhand Drive
        if re.search(r'horizontal_amplitude > 0\\.25 and wrist_height_relative > 0\\.05', content):
            content = re.sub(
                r'horizontal_amplitude > 0\\.25 and wrist_height_relative > 0\\.05',
                'horizontal_amplitude > 0.12 and wrist_height_relative > 0.02',
                content
            )
            fixes_applied.append("Backhand Drive thresholds: 0.25→0.12, 0.05→0.02")
        
        # 3. Adicionar debug detalhado
        if '# CLASSIFICAÇÃO' in content:
            debug_code = '''# CLASSIFICAÇÃO - COM DEBUG DETALHADO
        print(f"🔍 FRAME DEBUG:")
        print(f"   wrist_cross_body: {wrist_cross_body}")
        print(f"   horizontal_amplitude: {horizontal_amplitude:.4f}")
        print(f"   wrist_height_relative: {wrist_height_relative:.4f}")
        print(f"   shoulder_center_x: {shoulder_center_x:.4f}")
        print(f"   dominant_wrist_x: {dominant_wrist.x:.4f}")'''
            
            content = re.sub(r'# CLASSIFICAÇÃO', debug_code, content)
            fixes_applied.append("Debug detalhado adicionado")
        
        # 4. Ajustar lógica para favorecer Drive sobre Push
        if 'confidence = 0.7' in content and 'FP' in content:
            # Push deveria ter confiança menor
            content = re.sub(r'confidence = 0\\.7', 'confidence = 0.6', content)
            fixes_applied.append("Confiança Push reduzida: 0.7→0.6")
        
        if fixes_applied:
            # Salvar arquivo corrigido
            with open('tennis_comparison_backend.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ CORREÇÕES APLICADAS:")
            for fix in fixes_applied:
                print(f"   • {fix}")
            
            print(f"\\n💾 Backup disponível: {backup_filename}")
            
            return True
        else:
            print("⚠️ Nenhuma correção aplicada - padrões não encontrados")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_corrected_detection():
    """Testa a detecção após correção"""
    
    print("\\n🧪 TESTANDO DETECÇÃO CORRIGIDA...")
    
    test_code = '''
from tennis_comparison_backend import TennisComparisonEngine

print("=== TESTE PÓS-CORREÇÃO ===")
analyzer = TennisComparisonEngine()
result = analyzer.analyze_movement_from_content('teste_neutro.mp4')

print(f"Movimento detectado: {result['movement']}")
print(f"Confiança: {result['confidence']}")

if result['movement'] == 'FD':
    print("✅ SUCESSO: Agora detecta FD (Forehand Drive) corretamente!")
elif result['movement'] == 'FP':
    print("❌ AINDA INCORRETO: Ainda detecta FP (Push)")
    print("🔧 NECESSÁRIO: Ajustes adicionais nos thresholds")
else:
    print(f"⚠️ INESPERADO: Detectou {result['movement']}")
'''
    
    with open('test_corrected_detection.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("📄 Script de teste criado: test_corrected_detection.py")

# Executar examinação e correção
if __name__ == "__main__":
    if show_current_classification_logic():
        if fix_classification_thresholds():
            test_corrected_detection()
            
            print("\\n🎯 PRÓXIMOS PASSOS:")
            print("1. Execute: python test_corrected_detection.py")
            print("2. Verifique se agora detecta FD")
            print("3. Se ainda detectar FP, ajustaremos mais")
        else:
            print("\\n⚠️ Correção automática falhou")
            print("🔧 AÇÃO MANUAL NECESSÁRIA:")
            print("1. Abra tennis_comparison_backend.py")
            print("2. Procure por _analyze_single_frame_movement")
            print("3. Reduza os thresholds manualmente:")
            print("   horizontal_amplitude: 0.3 → 0.15")
            print("   wrist_height_relative: 0.1 → 0.03")
    else:
        print("\\n❌ Não foi possível encontrar a lógica de classificação")
        print("🔍 Verifique se o arquivo tennis_comparison_backend.py existe")
'''
    
    with open('examine_classification.py', 'w', encoding='utf-8') as f:
        f.write(examination_code)
    
    print("✅ Script de examinação criado!")
    print("\n🎯 EXECUTE PARA CORRIGIR:")
    print("python examine_classification.py")
    
    print("\n🔍 O QUE VAMOS CORRIGIR:")
    print("1. Thresholds muito altos fazem Drive virar Push")
    print("2. Critérios de amplitude horizontal muito restritivos")
    print("3. Altura do pulso com limites muito altos")
    print("4. Lógica que favorece Push sobre Drive")
    
    print("\n💡 CORREÇÕES ESPERADAS:")
    print("• horizontal_amplitude: 0.3 → 0.15 (mais sensível)")
    print("• wrist_height_relative: 0.1 → 0.03 (menos restritivo)")
    print("• Favorecer Drive sobre Push quando em dúvida")

if __name__ == "__main__":
    examine_and_fix_classification()
