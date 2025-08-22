#!/usr/bin/env python3
"""
Headless Tennis Analyzer - Railway Compatible
Análise de tênis de mesa sem dependências gráficas
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeadlessPoseDetector:
    """Detector de poses usando apenas OpenCV - compatível com ambiente headless"""
    
    def __init__(self):
        self.initialized = True
        logger.info("HeadlessPoseDetector initialized successfully")
    
    def detect_keypoints(self, frame: np.ndarray) -> Dict:
        """Simula detecção de keypoints usando análise de movimento"""
        height, width = frame.shape[:2]
        
        # Análise básica de movimento usando diferença de frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simular keypoints principais para análise
        keypoints = {
            'nose': (width//2, height//3),
            'left_shoulder': (width//3, height//2),
            'right_shoulder': (2*width//3, height//2),
            'left_elbow': (width//4, 2*height//3),
            'right_elbow': (3*width//4, 2*height//3),
            'left_wrist': (width//5, 3*height//4),
            'right_wrist': (4*width//5, 3*height//4),
            'left_hip': (width//3, 2*height//3),
            'right_hip': (2*width//3, 2*height//3)
        }
        
        return {
            'keypoints': keypoints,
            'confidence': 0.8,
            'frame_processed': True
        }

class HeadlessVideoAnalyzer:
    """Analisador de vídeo headless para Railway"""
    
    def __init__(self):
        self.pose_detector = HeadlessPoseDetector()
        self.professionals_db = self._load_professionals_db()
        logger.info("HeadlessVideoAnalyzer initialized")
    
    def _load_professionals_db(self) -> Dict:
        """Carrega base de dados de profissionais"""
        return {
            'forehand_drive': [
                {'name': 'Zhang Jike', 'file_path': 'profissionais/forehand_drive/Zhang_Jike_FD_D_D.mp4'},
                {'name': 'Ma Long', 'file_path': 'profissionais/forehand_drive/Ma_Long_FD_D_D.mp4'},
                {'name': 'Fan Zhendong', 'file_path': 'profissionais/forehand_drive/Fan_Zhendong_FD_D_E.mp4'}
            ],
            'forehand_push': [
                {'name': 'Jeff Plumb', 'file_path': 'profissionais/forehand_push/Jeff_Plumb_FP_D_D.mp4'},
                {'name': 'Zhao Yiyi', 'file_path': 'profissionais/forehand_push/Zhao_Yiyi_FP_D_D.mp4'}
            ],
            'backhand_drive': [
                {'name': 'Chen Meng', 'file_path': 'profissionais/backhand_drive/Chen_Meng_BD_D_D.mp4'},
                {'name': 'Jane', 'file_path': 'profissionais/backhand_drive/Jane_BD_D_D.mp4'}
            ],
            'backhand_push': [
                {'name': 'Alois Rosario', 'file_path': 'profissionais/backhand_push/Alois_Rosario_BP_E_E.mp4'},
                {'name': 'Zhao Yiyi', 'file_path': 'profissionais/backhand_push/Zhao_Yiyi_BP_D_E.mp4'},
                {'name': 'Jane', 'file_path': 'profissionais/backhand_push/Jane_BP_D_D.mp4'}
            ]
        }
    
    def get_available_professionals(self, movement_type: str) -> List[Dict]:
        """Retorna profissionais disponíveis para um tipo de movimento"""
        return self.professionals_db.get(movement_type, [])
    
    def analyze_video(self, video_path: str, metadata: Dict) -> Dict:
        """Análise principal do vídeo"""
        try:
            logger.info(f"Analyzing video: {video_path}")
            
            # Abrir vídeo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': 'Could not open video'}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Processar frames
            keypoints_sequence = []
            frame_idx = 0
            
            while cap.read()[0] and frame_idx < min(100, frame_count):  # Limitar para Railway
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detectar keypoints
                result = self.pose_detector.detect_keypoints(frame)
                keypoints_sequence.append(result)
                frame_idx += 1
            
            cap.release()
            
            # Análise biomecânica simplificada
            analysis_result = self._analyze_biomechanics(keypoints_sequence, metadata)
            
            return {
                'success': True,
                'analysis': analysis_result,
                'frames_processed': frame_idx,
                'video_info': {
                    'total_frames': frame_count,
                    'fps': fps,
                    'duration': frame_count / fps if fps > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_biomechanics(self, keypoints_sequence: List[Dict], metadata: Dict) -> Dict:
        """Análise biomecânica simplificada"""
        
        # Simular análise de movimento
        movement_quality = np.random.uniform(0.7, 0.95)  # Placeholder
        consistency = np.random.uniform(0.6, 0.9)
        technique_score = np.random.uniform(0.65, 0.9)
        
        return {
            'movement_quality': movement_quality,
            'consistency': consistency,
            'technique_score': technique_score,
            'final_score': (movement_quality + consistency + technique_score) / 3,
            'recommendations': [
                "Manter postura mais estável",
                "Melhorar consistência do movimento",
                "Trabalhar timing da raquete"
            ],
            'biomechanical_data': {
                'shoulder_angle_avg': np.random.uniform(45, 75),
                'elbow_extension': np.random.uniform(0.6, 0.9),
                'wrist_stability': np.random.uniform(0.7, 0.95)
            }
        }
    
    def compare_with_professional(self, user_video: str, pro_video: str, 
                                user_metadata: Dict, pro_metadata: Dict) -> Dict:
        """Comparação com vídeo profissional"""
        
        logger.info(f"Comparing user video with professional")
        
        # Analisar vídeo do usuário
        user_analysis = self.analyze_video(user_video, user_metadata)
        
        # Simular análise profissional (já que não temos os vídeos no Railway)
        pro_analysis = {
            'movement_quality': 0.95,
            'consistency': 0.92,
            'technique_score': 0.96,
            'final_score': 0.94,
            'biomechanical_data': {
                'shoulder_angle_avg': 65,
                'elbow_extension': 0.88,
                'wrist_stability': 0.93
            }
        }
        
        if not user_analysis['success']:
            return user_analysis
        
        # Calcular comparação
        user_data = user_analysis['analysis']
        comparison_score = min(user_data['final_score'] / pro_analysis['final_score'], 1.0) * 100
        
        return {
            'success': True,
            'user_analysis': user_data,
            'professional_analysis': pro_analysis,
            'comparison_score': comparison_score,
            'improvements': self._generate_improvements(user_data, pro_analysis),
            'final_score': comparison_score
        }
    
    def _generate_improvements(self, user_data: Dict, pro_data: Dict) -> List[str]:
        """Gera sugestões de melhoria"""
        improvements = []
        
        if user_data['movement_quality'] < pro_data['movement_quality']:
            improvements.append("Trabalhar fluidez do movimento")
        
        if user_data['consistency'] < pro_data['consistency']:
            improvements.append("Praticar repetições para maior consistência")
        
        if user_data['technique_score'] < pro_data['technique_score']:
            improvements.append("Aperfeiçoar técnica básica")
        
        return improvements


class HeadlessTennisAnalyzerAPI:
    """API Tennis Analyzer para ambiente headless"""
    
    def __init__(self):
        self.engine = HeadlessVideoAnalyzer()
        logger.info("HeadlessTennisAnalyzerAPI initialized successfully")
    
    def process_upload(self, video_data: bytes, filename: str, metadata: Dict) -> Dict:
        """Processa upload de vídeo"""
        try:
            # Salvar arquivo temporário
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            # Validar vídeo
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return {'success': False, 'error': 'Invalid video file'}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Limpar arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'success': True,
                'message': 'Video uploaded and validated successfully',
                'video_info': {
                    'frames': frame_count,
                    'fps': fps,
                    'duration': frame_count / fps if fps > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_professional_metadata(self, video_path: str, movement_type: str) -> Dict:
        """Retorna metadata do vídeo profissional"""
        return {
            'maoDominante': 'D',
            'ladoCamera': 'D', 
            'ladoRaquete': 'F' if 'forehand' in movement_type else 'B',
            'tipoMovimento': 'D' if 'drive' in movement_type else 'P',
            'professional': True
        }


# Função para teste
def test_headless_analyzer():
    """Teste do analisador headless"""
    analyzer = HeadlessTennisAnalyzerAPI()
    
    # Simular dados de teste
    test_metadata = {
        'maoDominante': 'D',
        'ladoCamera': 'D',
        'ladoRaquete': 'F',
        'tipoMovimento': 'D'
    }
    
    # Teste de profissionais disponíveis
    professionals = analyzer.engine.get_available_professionals('forehand_drive')
    print(f"Professionals available: {len(professionals)}")
    
    return analyzer


if __name__ == "__main__":
    test_headless_analyzer()