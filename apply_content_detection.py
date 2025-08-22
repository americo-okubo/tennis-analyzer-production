"""
Tennis Table Content Detection Modifier
Script para aplicar modificações automaticamente no tennis_comparison_backend.py
"""

import shutil
import re
from datetime import datetime

def apply_content_detection_modifications():
    """Aplica modificações para detecção de movimento por conteúdo"""
    
    print("APLICANDO MODIFICACOES PARA DETECCAO POR CONTEUDO...")
    
    # 1. Fazer backup
    backup_filename = f"tennis_comparison_backend_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    shutil.copy2('tennis_comparison_backend.py', backup_filename)
    print(f"Backup criado: {backup_filename}")
    
    # 2. Ler arquivo atual
    with open('tennis_comparison_backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 3. Adicionar imports necessários
    print("Adicionando imports...")
    import_addition = "import mediapipe as mp\nimport math\n"
    
    # Procurar local para inserir imports (após outros imports)
    if "import cv2" in content and "import mediapipe as mp" not in content:
        content = content.replace("import cv2", f"import cv2\n{import_addition}")
        print("Imports adicionados apos cv2")
    
    # 4. Adicionar métodos à classe TennisComparisonEngine
    print("Adicionando metodos de analise de conteudo...")
    
    new_methods = '''
    def analyze_movement_from_content(self, video_path: str, max_frames: int = 60) -> dict:
        """
        Analisa conteudo do video para detectar tipo de movimento biomecanico
        quando filename nao contem padroes identificaveis
        """
        print(f"ANALISANDO CONTEUDO: {video_path}")
        
        try:
            # Inicializar MediaPipe
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Balanceado para performance
                enable_segmentation=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # Abrir vídeo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'movement': 'unknown', 'confidence': 0.0, 'method': 'content_analysis_failed'}
            
            # Análise frame-by-frame
            frame_analyses = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip_frames = max(1, total_frames // max_frames)  # Pular frames para performance
            
            print(f"Total frames: {total_frames}, analisando a cada {skip_frames} frames")
            
            while frame_count < max_frames:
                # Pular frames para otimizar
                for _ in range(skip_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                if not ret:
                    break
                    
                frame_count += 1
                
                # Converter para RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detectar poses
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Analisar movimento deste frame
                    frame_analysis = self._analyze_single_frame_movement(landmarks)
                    if frame_analysis:
                        frame_analyses.append(frame_analysis)
            
            cap.release()
            pose.close()
            
            if not frame_analyses:
                return {'movement': 'unknown', 'confidence': 0.0, 'method': 'no_poses_detected'}
            
            # Consolidar análises dos frames
            movement_detection = self._consolidate_frame_analyses(frame_analyses)
            movement_detection['method'] = 'content_analysis'
            
            print(f"MOVIMENTO DETECTADO: {movement_detection['movement']} (confianca: {movement_detection['confidence']:.2f})")
            
            return movement_detection
            
        except Exception as e:
            print(f"Erro na analise de conteudo: {str(e)}")
            return {'movement': 'unknown', 'confidence': 0.0, 'method': 'analysis_error'}

    def _analyze_single_frame_movement(self, landmarks):
        """Analisa movimento biomecânico em um frame específico"""
        
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
            
            # 2. DETECÇÃO DE TIPO (Drive vs Push)
            # Drive: maior amplitude de movimento, pulso mais alto
            # Push: movimento mais compacto, pulso mais baixo
            
            # Amplitude horizontal (diferença entre pulsos)
            horizontal_amplitude = abs(dominant_wrist.x - non_dominant_wrist.x)
            
            # Altura do pulso dominante relativa aos ombros
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            wrist_height_relative = shoulder_y - dominant_wrist.y  # Positivo = acima dos ombros
            
            # CLASSIFICAÇÃO
            if wrist_cross_body:
                # Forehand detectado
                if horizontal_amplitude > 0.3 and wrist_height_relative > 0.1:
                    movement = 'FD'  # Forehand Drive
                    confidence = min(0.9, horizontal_amplitude + wrist_height_relative)
                else:
                    movement = 'FP'  # Forehand Push
                    confidence = 0.7
            else:
                # Backhand detectado
                if horizontal_amplitude > 0.25 and wrist_height_relative > 0.05:
                    movement = 'BD'  # Backhand Drive  
                    confidence = min(0.85, horizontal_amplitude + wrist_height_relative)
                else:
                    movement = 'BP'  # Backhand Push
                    confidence = 0.75
            
            return {
                'movement': movement,
                'confidence': confidence,
                'metrics': {
                    'wrist_cross_body': wrist_cross_body,
                    'horizontal_amplitude': horizontal_amplitude,
                    'wrist_height_relative': wrist_height_relative,
                    'shoulder_center_x': shoulder_center_x,
                    'dominant_wrist_x': dominant_wrist.x
                }
            }
            
        except Exception as e:
            return None

    def _consolidate_frame_analyses(self, frame_analyses: list) -> dict:
        """Consolida análises de múltiplos frames em detecção final"""
        
        # Contar votos por movimento
        movement_votes = {}
        confidence_sums = {}
        
        for analysis in frame_analyses:
            movement = analysis['movement']
            confidence = analysis['confidence']
            
            if movement not in movement_votes:
                movement_votes[movement] = 0
                confidence_sums[movement] = 0
                
            movement_votes[movement] += 1
            confidence_sums[movement] += confidence
        
        if not movement_votes:
            return {'movement': 'unknown', 'confidence': 0.0}
        
        # Encontrar movimento mais votado
        most_voted_movement = max(movement_votes, key=movement_votes.get)
        vote_count = movement_votes[most_voted_movement]
        avg_confidence = confidence_sums[most_voted_movement] / vote_count
        
        # Ajustar confiança baseada na consistência
        total_votes = sum(movement_votes.values())
        consistency_factor = vote_count / total_votes  # 0.0 - 1.0
        
        final_confidence = avg_confidence * consistency_factor
        
        print(f"ANALISE CONSOLIDADA:")
        print(f"   Votos: {movement_votes}")
        print(f"   Movimento escolhido: {most_voted_movement} ({vote_count}/{total_votes} votos)")
        print(f"   Confianca final: {final_confidence:.2f}")
        
        return {
            'movement': most_voted_movement,
            'confidence': final_confidence,
            'stats': {
                'total_frames': total_votes,
                'votes': movement_votes,
                'consistency': consistency_factor
            }
        }
'''
    
    # Encontrar local para inserir métodos (dentro da classe, antes do último método)
    class_match = re.search(r'class TennisComparisonEngine[^:]*:', content)
    if class_match:
        # Procurar local adequado para inserir (antes do último método da classe)
        insertion_point = content.rfind('\n    def ', class_match.end())
        if insertion_point != -1:
            content = content[:insertion_point] + new_methods + content[insertion_point:]
            print("Metodos de analise de conteudo adicionados")
    
    # 5. Modificar o método compare_techniques (linha 397)
    print("Modificando metodo compare_techniques...")
    
    # Localizar e substituir a linha 397 e adicionar lógica de análise de conteúdo
    original_line = "user_movement_from_file = self._extract_movement_from_professional({}, user_video_path)"
    
    replacement_code = '''user_movement_from_file = self._extract_movement_from_professional({}, user_video_path)
        
        # NOVA FUNCIONALIDADE - ANÁLISE DE CONTEÚDO QUANDO UNKNOWN
        if user_movement_from_file == 'unknown':
            print("Movimento nao detectado por filename, analisando conteudo do video...")
            content_analysis = self.analyze_movement_from_content(user_video_path)
            
            if content_analysis['confidence'] > 0.6:  # Threshold de confiança
                user_movement_from_file = content_analysis['movement']
                print(f"Movimento detectado por conteudo: {user_movement_from_file} (confianca: {content_analysis['confidence']:.2f})")
                
                # Log adicional para debug
                print(f"Metodo de deteccao: {content_analysis.get('method', 'content_analysis')}")
                if 'stats' in content_analysis:
                    print(f"Estatisticas: {content_analysis['stats']}")
            else:
                print(f"Baixa confianca na deteccao de conteudo: {content_analysis['confidence']:.2f}")
                print(f"Mantendo movimento como 'unknown' para permitir validacao manual")'''
    
    if original_line in content:
        content = content.replace(original_line, replacement_code)
        print("Metodo compare_techniques modificado")
    else:
        print("Linha original nao encontrada, pode precisar de ajuste manual")
    
    # 6. Salvar arquivo modificado
    with open('tennis_comparison_backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"MODIFICACOES APLICADAS COM SUCESSO!")
    print(f"Backup disponivel em: {backup_filename}")
    print("\nTESTE AGORA:")
    print("python -c \"")
    print("from tennis_comparison_backend import TennisComparisonEngine")
    print("analyzer = TennisComparisonEngine()")
    print("result = analyzer.compare_techniques(")
    print("    'teste_sem_metadados.mp4',")
    print("    'videos/Katsumi_BD_D_D.mp4',")
    print("    {'maoDominante': 'D', 'ladoRaquete': 'B', 'tipoMovimento': 'P'},")
    print("    {}")
    print(")")
    print("print('Resultado:', result.get('success'))")
    print("\"")

if __name__ == "__main__":
    apply_content_detection_modifications()
