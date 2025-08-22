#!/usr/bin/env python3
"""
🏆 PROFESSIONAL DATA EXTRACTOR
Módulo para pré-extrair e armazenar dados biomecânicos de profissionais.
Integrado com TableTennisAnalyzer - evita reprocessamento desnecessário.
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict

# Importar componentes existentes do sistema
from cycle_detector_retracted_extended import CycleDetectorIntegration
from real_biomech_comparison import RealBiomechComparison

@dataclass
class ProfessionalProfile:
    """Estrutura padronizada para perfis de profissionais"""
    name: str
    movement: str
    dominant_hand: str
    camera_side: str
    video_path: str
    extraction_date: str
    cycles_data: Dict
    biomech_metrics: Dict
    quality_score: float
    frame_count: int
    fps: float

class ProfessionalDataExtractor:
    """
    🎾 Extrator de Dados Profissionais
    Pré-processa vídeos de profissionais para comparações instantâneas
    """
    
    def __init__(self, videos_directory: str = "videos/", database_directory: str = "professionals_database/"):
        self.videos_dir = Path(videos_directory)
        self.db_dir = Path(database_directory)
        self.db_dir.mkdir(exist_ok=True)
        
        # Inicializar componentes do sistema existente
        self.cycle_detector = CycleDetectorIntegration()
        self.biomech_analyzer = RealBiomechComparison()
        
        # Configuração de logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Profissionais conhecidos com metadados
        self.known_professionals = {
            'Ma_Long': {
                'full_name': 'Ma Long',
                'title': 'Campeão Mundial',
                'nationality': 'China',
                'ranking': '#1 Mundial (histórico)'
            },
            'Fan_Zhendong': {
                'full_name': 'Fan Zhendong', 
                'title': '#1 Mundial',
                'nationality': 'China',
                'ranking': '#1 Mundial Atual'
            },
            'Zhang_Jike': {
                'full_name': 'Zhang Jike',
                'title': 'Campeão Olímpico',
                'nationality': 'China',
                'ranking': 'Ex-#1 Mundial'
            },
            'Timo_Boll': {
                'full_name': 'Timo Boll',
                'title': 'Lenda Europeia',
                'nationality': 'Alemanha',
                'ranking': 'Top 10 Mundial'
            },
            'Ovtcharov': {
                'full_name': 'Dimitrij Ovtcharov',
                'title': 'Estrela Alemã',
                'nationality': 'Alemanha',
                'ranking': 'Top 10 Mundial'
            },
            'Liam_Pitchford': {
                'full_name': 'Liam Pitchford',
                'title': 'Jogador Inglês',
                'nationality': 'Inglaterra',
                'ranking': 'Top 50 Mundial'
            }
        }
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        📝 Parse do padrão: Nome_Movimento_MãoDominante_LadoCâmera.mp4
        """
        try:
            name_part = filename.replace('.mp4', '')
            parts = name_part.split('_')
            
            if len(parts) >= 4:
                name = '_'.join(parts[:-3])  # Pode ter underscores no nome
                movement = parts[-3]
                dominant_hand = parts[-2]
                camera_side = parts[-1]
                
                return {
                    'name': name,
                    'movement': movement,
                    'dominant_hand': dominant_hand,
                    'camera_side': camera_side
                }
        except Exception as e:
            self.logger.warning(f"Erro ao parsear filename {filename}: {e}")
        
        return None
        
    def extract_cycles_from_video(self, video_path: str) -> Dict[str, Any]:
        """
        🔬 Extrai ciclos biomecânicos de um vídeo profissional
        Usa o mesmo sistema TensorFlow do TableTennisAnalyzer
        """
        self.logger.info(f"Extraindo ciclos de: {video_path}")
        
        try:
            # Usar o detector de ciclos integrado existente
            cycles_result = self.cycle_detector.detect_cycles_from_video(str(video_path))
            
            if not cycles_result.get('success', False):
                self.logger.warning(f"Falha na detecção de ciclos: {video_path}")
                return {'success': False, 'error': 'Cycle detection failed'}
            
            # Extrair informações do vídeo
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Preparar dados estruturados
            extraction_data = {
                'success': True,
                'video_info': {
                    'frame_count': frame_count,
                    'fps': fps,
                    'duration': frame_count / fps if fps > 0 else 0
                },
                'cycles_raw': cycles_result.get('cycles', []),
                'cycles_count': len(cycles_result.get('cycles', [])),
                'extension_timeline': cycles_result.get('extension_values', []),
                'quality_metrics': {
                    'avg_cycle_duration': cycles_result.get('avg_cycle_duration', 0),
                    'cycle_consistency': cycles_result.get('cycle_consistency', 0),
                    'detection_confidence': cycles_result.get('confidence', 0)
                },
                'biomechanical_features': self._extract_biomech_features(cycles_result),
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Extraídos {extraction_data['cycles_count']} ciclos de {video_path}")
            return extraction_data
            
        except Exception as e:
            self.logger.error(f"Erro na extração de {video_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_biomech_features(self, cycles_result: Dict) -> Dict[str, float]:
        """
        📊 Extrai features biomecânicas específicas dos ciclos
        """
        cycles = cycles_result.get('cycles', [])
        if not cycles:
            return {}
        
        features = {}
        
        # Métricas de duração
        durations = [cycle.get('duration', 0) for cycle in cycles]
        if durations:
            features['avg_duration'] = np.mean(durations)
            features['std_duration'] = np.std(durations)
            features['min_duration'] = np.min(durations)
            features['max_duration'] = np.max(durations)
        
        # Métricas de amplitude
        amplitudes = [cycle.get('amplitude', 0) for cycle in cycles]
        if amplitudes:
            features['avg_amplitude'] = np.mean(amplitudes)
            features['std_amplitude'] = np.std(amplitudes)
            features['min_amplitude'] = np.min(amplitudes)
            features['max_amplitude'] = np.max(amplitudes)
        
        # Métricas de ritmo/timing
        start_frames = [cycle.get('start_frame', 0) for cycle in cycles]
        if len(start_frames) > 1:
            intervals = np.diff(start_frames)
            features['avg_interval'] = np.mean(intervals)
            features['rhythm_consistency'] = 1.0 / (1.0 + np.std(intervals))
        
        # Score de qualidade geral
        features['overall_quality'] = np.mean([
            features.get('rhythm_consistency', 0.5),
            1.0 / (1.0 + features.get('std_duration', 1.0)),
            features.get('avg_amplitude', 0.5)
        ])
        
        return features
    
    def save_professional_profile(self, name: str, movement: str, data: Dict, metadata: Dict) -> str:
        """
        💾 Salva perfil profissional em formato JSON otimizado
        """
        # Criar nome do arquivo padronizado
        profile_filename = f"{name.lower()}_{movement.lower()}_cycles.json"
        profile_path = self.db_dir / profile_filename
        
        # Criar perfil estruturado
        profile = ProfessionalProfile(
            name=name,
            movement=movement,
            dominant_hand=metadata.get('dominant_hand', 'D'),
            camera_side=metadata.get('camera_side', 'D'),
            video_path=metadata.get('video_path', ''),
            extraction_date=datetime.now().isoformat(),
            cycles_data=data.get('cycles_raw', []),
            biomech_metrics=data.get('biomechanical_features', {}),
            quality_score=data.get('biomechanical_features', {}).get('overall_quality', 0.0),
            frame_count=data.get('video_info', {}).get('frame_count', 0),
            fps=data.get('video_info', {}).get('fps', 30.0)
        )
        
        # Salvar como JSON
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(profile), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Perfil salvo: {profile_path}")
        return str(profile_path)
    
    def scan_professional_videos(self) -> List[Dict[str, str]]:
        """
        🔍 Escaneia diretório de vídeos buscando profissionais
        """
        professional_videos = []
        
        for video_file in self.videos_dir.glob("*.mp4"):
            parsed = self.parse_filename(video_file.name)
            if parsed and parsed['name'] in self.known_professionals:
                video_info = {
                    'file_path': str(video_file),
                    'filename': video_file.name,
                    **parsed,
                    **self.known_professionals[parsed['name']]
                }
                professional_videos.append(video_info)
        
        self.logger.info(f"🔍 Encontrados {len(professional_videos)} vídeos profissionais")
        return professional_videos
    
    def generate_professionals_database(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        🏗️ Gera base completa de dados de profissionais
        """
        self.logger.info("🏗️ Iniciando geração da base de profissionais...")
        
        professional_videos = self.scan_professional_videos()
        results = {
            'total_videos': len(professional_videos),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'profiles': []
        }
        
        for video_info in professional_videos:
            profile_name = f"{video_info['name'].lower()}_{video_info['movement'].lower()}_cycles.json"
            profile_path = self.db_dir / profile_name
            
            # Verificar se já existe (skip se não forçar reprocessamento)
            if profile_path.exists() and not force_reprocess:
                self.logger.info(f"⏭️ Pulando {video_info['filename']} (já processado)")
                results['skipped'] += 1
                continue
            
            # Extrair dados do vídeo
            extraction_data = self.extract_cycles_from_video(video_info['file_path'])
            
            if extraction_data.get('success', False):
                # Salvar perfil
                saved_path = self.save_professional_profile(
                    video_info['name'],
                    video_info['movement'],
                    extraction_data,
                    video_info
                )
                
                results['profiles'].append({
                    'name': video_info['name'],
                    'movement': video_info['movement'],
                    'file': profile_name,
                    'cycles_count': extraction_data.get('cycles_count', 0),
                    'quality_score': extraction_data.get('biomechanical_features', {}).get('overall_quality', 0.0)
                })
                results['processed'] += 1
                
            else:
                self.logger.error(f"❌ Falha ao processar {video_info['filename']}")
                results['failed'] += 1
        
        # Salvar índice geral
        self._save_profiles_index(results)
        
        self.logger.info(f"✅ Base gerada: {results['processed']} processados, {results['skipped']} pulados, {results['failed']} falharam")
        return results
    
    def _save_profiles_index(self, results: Dict) -> None:
        """
        📋 Salva índice geral dos perfis criados
        """
        index_data = {
            'generation_date': datetime.now().isoformat(),
            'system_version': 'TableTennisAnalyzer v2.0',
            'total_profiles': results['processed'],
            'profiles': results['profiles'],
            'available_professionals': list(self.known_professionals.keys()),
            'movements_detected': list(set(p['movement'] for p in results['profiles']))
        }
        
        index_path = self.db_dir / 'profiles_index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 Índice salvo: {index_path}")

def main():
    """
    🚀 Execução principal - geração da base de profissionais
    """
    print("🏆 PROFESSIONAL DATA EXTRACTOR")
    print("===============================")
    
    extractor = ProfessionalDataExtractor()
    
    # Escanear vídeos disponíveis
    videos = extractor.scan_professional_videos()
    print(f"\n🔍 Vídeos profissionais encontrados: {len(videos)}")
    for video in videos:
        print(f"   • {video['full_name']} - {video['movement']} ({video['filename']})")
    
    # Gerar base de dados
    print(f"\n🏗️ Gerando base de dados...")
    results = extractor.generate_professionals_database(force_reprocess=False)
    
    print(f"\n✅ CONCLUÍDO!")
    print(f"   📊 Processados: {results['processed']}")
    print(f"   ⏭️ Já existiam: {results['skipped']}")
    print(f"   ❌ Falharam: {results['failed']}")
    print(f"\n💾 Base salva em: professionals_database/")

if __name__ == "__main__":
    main()
