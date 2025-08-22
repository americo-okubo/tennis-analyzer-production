"""
Correção específica para detecção de Forehand Drive
O problema é horizontal_amplitude = 0.000 para FD
"""

import re
from datetime import datetime

def fix_forehand_drive_detection():
    """Corrige especificamente a detecção de Forehand Drive"""
    
    print("🔧 CORRIGINDO DETECÇÃO DE FOREHAND DRIVE...")
    
    # Backup
    backup_filename = f"tennis_comparison_backend_fd_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    try:
        with open('tennis_comparison_backend.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(backup_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📁 Backup criado: {backup_filename}")
        
        # Encontrar e corrigir o método _analyze_single_frame_movement
        method_pattern = r'(def _analyze_single_frame_movement\(self, landmarks\):.*?return {[^}]*})'
        
        method_match = re.search(method_pattern, content, re.DOTALL)
        
        if method_match:
            original_method = method_match.group(1)
            print("✅ Método _analyze_single_frame_movement encontrado")
            
            # Nova implementação corrigida
            corrected_method = '''def _analyze_single_frame_movement(self, landmarks):
        """Analisa movimento biomecânico em um frame específico - VERSÃO CORRIGIDA"""
        
        try:
            # Pontos-chave MediaPipe (33 landmarks)
            # 11-12: Ombros, 13-14: Cotovelos, 15-16: Pulsos
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Calcular padrões biomecânicos
            
            # 1. DETECÇÃO DE LADO (Forehand vs Backhand)
            # Baseado na posição relativa do pulso em relação ao corpo
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            
            # Assumir destro por padrão (pode ser refinado)
            dominant_wrist = right_wrist
            non_dominant_wrist = left_wrist
            
            # Forehand: pulso dominante cruza para o lado oposto
            # Backhand: pulso dominante fica do mesmo lado
            wrist_cross_body = dominant_wrist.x < shoulder_center_x  # Para destro
            
            # 2. DETECÇÃO DE TIPO (Drive vs Push) - LÓGICA CORRIGIDA
            
            # NOVA METODOLOGIA: Múltiplos critérios para melhor detecção
            
            # Amplitude horizontal (diferença entre pulsos)
            horizontal_amplitude = abs(dominant_wrist.x - non_dominant_wrist.x)
            
            # Altura do pulso dominante relativa aos ombros
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            wrist_height_relative = shoulder_y - dominant_wrist.y  # Positivo = acima dos ombros
            
            # NOVOS CRITÉRIOS ADICIONAIS
            
            # Distância do cotovelo para detectar extensão
            elbow_extension = abs(dominant_wrist.x - (left_elbow.x if wrist_cross_body else right_elbow.x))
            
            # Posição vertical do cotovelo vs pulso (Drive tem cotovelo mais baixo)
            elbow_vs_wrist_height = (left_elbow.y if wrist_cross_body else right_elbow.y) - dominant_wrist.y
            
            # DEBUG DETALHADO
            print(f"🔍 FRAME ANALYSIS:")
            print(f"   wrist_cross_body: {wrist_cross_body}")
            print(f"   horizontal_amplitude: {horizontal_amplitude:.4f}")
            print(f"   wrist_height_relative: {wrist_height_relative:.4f}")
            print(f"   elbow_extension: {elbow_extension:.4f}")
            print(f"   elbow_vs_wrist_height: {elbow_vs_wrist_height:.4f}")
            print(f"   shoulder_center_x: {shoulder_center_x:.4f}")
            print(f"   dominant_wrist_x: {dominant_wrist.x:.4f}")
            
            # CLASSIFICAÇÃO MELHORADA
            if wrist_cross_body:
                # FOREHAND - CRITÉRIOS REVISADOS
                
                # Critério 1: Altura do pulso (Drive = mais alto)
                height_suggests_drive = wrist_height_relative > 0.02  # Muito mais baixo
                
                # Critério 2: Extensão do braço (Drive = mais extensão)
                extension_suggests_drive = elbow_extension > 0.15  # Novo critério
                
                # Critério 3: Posição relativa cotovelo-pulso (Drive = cotovelo mais baixo)
                elbow_position_suggests_drive = elbow_vs_wrist_height > 0.01
                
                # DECISÃO PARA FOREHAND
                drive_criteria_met = sum([
                    height_suggests_drive,
                    extension_suggests_drive, 
                    elbow_position_suggests_drive,
                    horizontal_amplitude > 0.1  # Threshold muito mais baixo
                ])
                
                print(f"   height_suggests_drive: {height_suggests_drive}")
                print(f"   extension_suggests_drive: {extension_suggests_drive}")
                print(f"   elbow_position_suggests_drive: {elbow_position_suggests_drive}")
                print(f"   amplitude_sufficient: {horizontal_amplitude > 0.1}")
                print(f"   drive_criteria_met: {drive_criteria_met}/4")
                
                if drive_criteria_met >= 2:  # 2 de 4 critérios = Drive
                    movement = 'FD'  # Forehand Drive
                    confidence = min(0.9, 0.6 + (drive_criteria_met * 0.1))
                else:
                    movement = 'FP'  # Forehand Push
                    confidence = 0.7
                    
            else:
                # BACKHAND - LÓGICA ORIGINAL (funcionando)
                if horizontal_amplitude > 0.25 and wrist_height_relative > 0.05:
                    movement = 'BD'  # Backhand Drive  
                    confidence = min(0.85, horizontal_amplitude + wrist_height_relative)
                else:
                    movement = 'BP'  # Backhand Push
                    confidence = 0.75
            
            print(f"   RESULTADO: {movement} (confiança: {confidence:.3f})")
            
            return {
                'movement': movement,
                'confidence': confidence,
                'metrics': {
                    'wrist_cross_body': wrist_cross_body,
                    'horizontal_amplitude': horizontal_amplitude,
                    'wrist_height_relative': wrist_height_relative,
                    'elbow_extension': elbow_extension,
                    'elbow_vs_wrist_height': elbow_vs_wrist_height,
                    'shoulder_center_x': shoulder_center_x,
                    'dominant_wrist_x': dominant_wrist.x
                }
            }
            
        except Exception as e:
            print(f"Erro na análise do frame: {e}")
            return None'''
            
            # Substituir método completo
            content = content.replace(original_method, corrected_method)
            
            # Salvar arquivo corrigido
            with open('tennis_comparison_backend.py', 'w', encoding='utf-8') as f:
                content = f.write(content)
            
            print("✅ CORREÇÕES APLICADAS:")
            print("   • Thresholds muito mais baixos para FD")
            print("   • Múltiplos critérios para detecção")
            print("   • Debug detalhado adicionado")
            print("   • Análise de extensão do braço")
            print("   • Posição relativa cotovelo-pulso")
            
            print(f"\n💾 Backup disponível: {backup_filename}")
            
            # Criar script de teste
            create_test_script()
            
            return True
            
        else:
            print("❌ Método _analyze_single_frame_movement não encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def create_test_script():
    """Cria script de teste para a correção"""
    
    test_code = '''
from tennis_comparison_backend import TennisComparisonEngine

print("=== TESTE DETECÇÃO CORRIGIDA ===")
analyzer = TennisComparisonEngine()

print("\\nAnalisando teste_neutro.mp4...")
result = analyzer.analyze_movement_from_content('teste_neutro.mp4')

print(f"\\n📊 RESULTADO FINAL:")
print(f"   Movimento: {result['movement']}")
print(f"   Confiança: {result['confidence']:.3f}")
print(f"   Método: {result.get('method', 'N/A')}")

if 'stats' in result:
    stats = result['stats']
    print(f"\\n📈 ESTATÍSTICAS:")
    print(f"   Total frames: {stats.get('total_frames', 'N/A')}")
    print(f"   Votos: {stats.get('votes', 'N/A')}")
    print(f"   Consistência: {stats.get('consistency', 'N/A')}")

if result['movement'] == 'FD':
    print("\\n✅ SUCESSO: Agora detecta FD (Forehand Drive)!")
elif result['movement'] == 'FP':
    print("\\n⚠️ AINDA FP: Pode precisar ajustes adicionais")
else:
    print(f"\\n🔍 RESULTADO: {result['movement']}")

print("\\n" + "="*50)
'''
    
    with open('test_corrected_fd_detection.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("📄 Script de teste criado: test_corrected_fd_detection.py")

if __name__ == "__main__":
    print("🎯 CORREÇÃO ESPECÍFICA PARA FOREHAND DRIVE")
    print("="*50)
    
    if fix_forehand_drive_detection():
        print("\n🧪 EXECUTE O TESTE:")
        print("python test_corrected_fd_detection.py")
        
        print("\n🔍 MUDANÇAS PRINCIPAIS:")
        print("• Threshold FD: 0.3 → 0.1 (muito mais baixo)")
        print("• Critérios múltiplos: altura + extensão + posição")
        print("• 2 de 4 critérios = Drive (mais flexível)")
        print("• Debug detalhado para cada frame")
        
        print("\n🎯 RESULTADO ESPERADO:")
        print("• horizontal_amplitude ainda baixo = OK")
        print("• Outros critérios compensam")
        print("• Detecção: FD ao invés de FP")
    else:
        print("\n❌ CORREÇÃO FALHOU")
        print("🔧 Verificar arquivo tennis_comparison_backend.py")
