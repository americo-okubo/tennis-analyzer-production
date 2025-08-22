#!/usr/bin/env python3
"""
Professional Video Preprocessor - Tennis Analyzer

Script para pré-processar todos os vídeos profissionais com análise biomecânica
completa, salvando os dados para comparação rápida durante análises de usuários.

FUNCIONALIDADE:
- Analisa todos os vídeos profissionais com ImprovedBiomechClassifier2D
- Extrai dados biomecânicos completos (ângulos, velocidades, coordenação)
- Salva dados em formato JSON para carregamento rápido
- Inclui metadados do movimento e qualidade da análise
- Suporte para atualização incremental (só processa novos/modificados)

AUTOR: Sistema Tennis Analyzer + Claude
DATA: 2025-07-28
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalPreprocessor:
    """
    Pré-processador de vídeos profissionais para análise biomecânica
    """
    
    def __init__(self, professionals_dir: str = "profissionais", output_file: str = "professionals_biomech_data.json"):
        self.professionals_dir = Path(professionals_dir)
        self.output_file = Path(output_file)
        self.processed_data = {}
        
        # Load existing data if available
        self.load_existing_data()
        
        logger.info(f"[PREPROCESSOR] Inicializado")
        logger.info(f"  - Diretório de profissionais: {self.professionals_dir}")
        logger.info(f"  - Arquivo de saída: {self.output_file}")
        logger.info(f"  - Dados existentes: {len(self.processed_data)} profissionais")
    
    def load_existing_data(self):
        """Carrega dados existentes se disponível"""
        try:
            if self.output_file.exists():
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    self.processed_data = json.load(f)
                logger.info(f"[PREPROCESSOR] Carregados {len(self.processed_data)} profissionais existentes")
            else:
                self.processed_data = {}
                logger.info(f"[PREPROCESSOR] Nenhum dado existente encontrado, iniciando do zero")
        except Exception as e:
            logger.error(f"[PREPROCESSOR] Erro ao carregar dados existentes: {e}")
            self.processed_data = {}
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calcula hash do arquivo para detectar modificações"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"[PREPROCESSOR] Erro ao calcular hash de {file_path}: {e}")
            return ""
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extrai metadados do nome do arquivo
        Formato esperado: Nome_AB_C_D.mp4
        A: F(orehand)/B(ackhand), B: D(rive)/P(ush), C: mão dominante, D: lado da câmera
        """
        try:
            # Remove extensão
            base_name = filename.replace('.mp4', '').replace('.avi', '').replace('.mov', '')
            parts = base_name.split('_')
            
            if len(parts) >= 4:
                movement_code = parts[-3]  # AB (ex: FD, BP)
                dominant_hand = parts[-2]  # C (ex: D, E)
                camera_side = parts[-1]    # D (ex: D, E)
                
                # Mapear códigos
                racket_side = 'F' if movement_code[0] == 'F' else 'B'
                movement_type = 'D' if movement_code[1] == 'D' else 'P'
                
                metadata = {
                    'maoDominante': 'D' if dominant_hand == 'D' else 'E',
                    'ladoRaquete': racket_side,
                    'ladoCamera': 'D' if camera_side == 'D' else 'E',
                    'tipoMovimento': movement_type,
                    'movement_full_name': self._get_full_movement_name(racket_side, movement_type),
                    'player_name': '_'.join(parts[:-3]) if len(parts) > 4 else parts[0]
                }
                
                logger.debug(f"[METADATA] {filename} -> {metadata}")
                return metadata
            else:
                logger.warning(f"[METADATA] Formato de arquivo inválido: {filename}")
                return self._get_default_metadata()
                
        except Exception as e:
            logger.error(f"[METADATA] Erro ao extrair metadados de {filename}: {e}")
            return self._get_default_metadata()
    
    def _get_full_movement_name(self, racket_side: str, movement_type: str) -> str:
        """Retorna nome completo do movimento"""
        side = "forehand" if racket_side == 'F' else "backhand"
        type_mov = "drive" if movement_type == 'D' else "push"
        return f"{side}_{type_mov}"
    
    def _get_default_metadata(self) -> Dict[str, str]:
        """Retorna metadados padrão para casos de erro"""
        return {
            'maoDominante': 'D',
            'ladoRaquete': 'F',
            'ladoCamera': 'D',
            'tipoMovimento': 'D',
            'movement_full_name': 'forehand_drive',
            'player_name': 'unknown'
        }
    
    def analyze_professional_video(self, video_path: Path, metadata: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Analisa um vídeo profissional com análise biomecânica completa
        """
        try:
            from improved_biomech_classifier_2d import ImprovedBiomechClassifier2D
            from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
            
            logger.info(f"[ANALYSIS] Iniciando análise de {video_path.name}")
            
            # Initialize analyzers
            biomech_classifier = ImprovedBiomechClassifier2D()
            enhanced_analyzer = EnhancedSingleCycleAnalyzer()
            
            # Run biomechanical analysis
            logger.info(f"[ANALYSIS] Executando análise biomecânica...")
            biomech_result = biomech_classifier.process_video(str(video_path))
            
            if not biomech_result:
                logger.error(f"[ANALYSIS] Falha na análise biomecânica de {video_path.name}")
                return None
            
            # Skip enhanced single cycle analysis for now (causes JSON serialization issues)
            logger.info(f"[ANALYSIS] Pulando análise de ciclo específico para evitar problemas de serialização...")
            cycle_data = None
            
            # Compile comprehensive professional data
            professional_data = {
                # Basic info
                'player_name': metadata['player_name'],
                'video_file': video_path.name,
                'movement_type': biomech_result.movement_type.value,
                'full_movement_name': metadata['movement_full_name'],
                
                # Analysis metadata
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzer_version': '5.0_real_biomech_cycle_specific',
                'file_hash': self.get_file_hash(video_path),
                
                # Movement classification
                'classification': {
                    'detected_movement': biomech_result.movement_type.value,
                    'confidence': biomech_result.confidence,
                    'confidence_level': biomech_result.confidence_level,
                    'classification_zone': biomech_result.classification_zone,
                    'hierarchy_level': biomech_result.hierarchy_level
                },
                
                # Biomechanical metrics
                'biomechanics': {
                    'elbow_variation_degrees': biomech_result.elbow_variation_active,
                    'elbow_opening_trend': biomech_result.elbow_opening_trend_active,
                    'coordination_score': biomech_result.coordination_active,
                    'movement_signature': biomech_result.movement_signature,
                    'temporal_pattern': biomech_result.temporal_pattern
                },
                
                # Movement dynamics
                'dynamics': {
                    'amplitude_y': biomech_result.amplitude_y_active,
                    'max_velocity': biomech_result.max_velocity_active,
                    'racket_detection_score': biomech_result.racket_score_active,
                    'active_hand_side': biomech_result.active_hand_side
                },
                
                # Bilateral analysis
                'bilateral_analysis': {
                    'left_arm_amplitude': biomech_result.left_metrics.movement_amplitude_y,
                    'right_arm_amplitude': biomech_result.right_metrics.movement_amplitude_y,
                    'left_max_velocity': biomech_result.left_metrics.max_velocity,
                    'right_max_velocity': biomech_result.right_metrics.max_velocity,
                    'left_coordination': (biomech_result.left_metrics.shoulder_elbow_coordination + 
                                        biomech_result.left_metrics.elbow_wrist_coordination) / 2,
                    'right_coordination': (biomech_result.right_metrics.shoulder_elbow_coordination + 
                                         biomech_result.right_metrics.elbow_wrist_coordination) / 2
                },
                
                # Confidence breakdown
                'confidence_breakdown': {
                    'biomech_forehand_likelihood': biomech_result.biomech_forehand_likelihood,
                    'biomech_backhand_likelihood': biomech_result.biomech_backhand_likelihood,
                    'biomech_drive_likelihood': biomech_result.biomech_drive_likelihood,
                    'biomech_push_likelihood': biomech_result.biomech_push_likelihood,
                    'biomech_confidence': biomech_result.biomech_confidence
                },
                
                # Video metadata
                'video_metadata': metadata,
                
                # Cycle data (if available)
                'cycle_analysis': cycle_data,
                
                # Analysis quality metrics
                'quality_metrics': {
                    'analysis_success': True,
                    'has_cycle_data': cycle_data is not None,
                    'racket_detection_quality': biomech_result.racket_score_active,
                    'biomech_confidence_quality': biomech_result.biomech_confidence,
                    'overall_confidence': biomech_result.confidence
                }
            }
            
            logger.info(f"[ANALYSIS] ✅ Análise completa de {video_path.name}")
            logger.info(f"  - Movimento: {biomech_result.movement_type.value} ({biomech_result.confidence:.1%})")
            logger.info(f"  - Variação cotovelo: {biomech_result.elbow_variation_active:.1f}°")
            logger.info(f"  - Coordenação: {biomech_result.coordination_active:.1%}")
            logger.info(f"  - Amplitude: {biomech_result.amplitude_y_active:.3f}")
            
            return professional_data
            
        except ImportError as e:
            logger.error(f"[ANALYSIS] Erro de importação: {e}")
            return None
        except Exception as e:
            logger.error(f"[ANALYSIS] Erro na análise de {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def should_process_file(self, video_path: Path) -> bool:
        """
        Determina se um arquivo deve ser processado
        (novo ou modificado desde a última análise)
        """
        try:
            file_key = str(video_path.relative_to(self.professionals_dir))
            current_hash = self.get_file_hash(video_path)
            
            if file_key not in self.processed_data:
                logger.info(f"[CHECK] {video_path.name} - NOVO arquivo")
                return True
            
            stored_hash = self.processed_data[file_key].get('file_hash', '')
            if current_hash != stored_hash:
                logger.info(f"[CHECK] {video_path.name} - arquivo MODIFICADO")
                return True
            
            logger.info(f"[CHECK] {video_path.name} - já processado, PULANDO")
            return False
            
        except Exception as e:
            logger.error(f"[CHECK] Erro ao verificar {video_path.name}: {e}")
            return True  # Process if in doubt
    
    def process_all_professionals(self, force_reprocess: bool = False):
        """
        Processa todos os vídeos profissionais
        
        Args:
            force_reprocess: Se True, reprocessa todos os arquivos mesmo que já tenham sido analisados
        """
        logger.info(f"[PREPROCESSOR] ==========================================")
        logger.info(f"[PREPROCESSOR] INICIANDO PRÉ-PROCESSAMENTO DE PROFISSIONAIS")
        logger.info(f"[PREPROCESSOR] ==========================================")
        
        if not self.professionals_dir.exists():
            logger.error(f"[PREPROCESSOR] Diretório não encontrado: {self.professionals_dir}")
            return
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.professionals_dir.rglob(f'*{ext}'))
        
        logger.info(f"[PREPROCESSOR] Encontrados {len(video_files)} vídeos para processar")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for video_path in sorted(video_files):
            try:
                logger.info(f"\n[PREPROCESSOR] ========================================")
                logger.info(f"[PREPROCESSOR] Processando: {video_path.name}")
                
                # Check if processing is needed
                if not force_reprocess and not self.should_process_file(video_path):
                    skipped_count += 1
                    continue
                
                # Extract metadata from filename
                metadata = self.extract_metadata_from_filename(video_path.name)
                
                # Analyze video
                analysis_result = self.analyze_professional_video(video_path, metadata)
                
                if analysis_result:
                    # Store result
                    file_key = str(video_path.relative_to(self.professionals_dir))
                    self.processed_data[file_key] = analysis_result
                    processed_count += 1
                    
                    logger.info(f"[PREPROCESSOR] ✅ {video_path.name} processado com sucesso")
                else:
                    error_count += 1
                    logger.error(f"[PREPROCESSOR] ❌ Falha ao processar {video_path.name}")
                
                # Save progress periodically
                if (processed_count + error_count) % 3 == 0:
                    self.save_data()
                    logger.info(f"[PREPROCESSOR] Progresso salvo - Processados: {processed_count}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"[PREPROCESSOR] Erro ao processar {video_path.name}: {e}")
        
        # Final save
        self.save_data()
        
        # Summary
        logger.info(f"\n[PREPROCESSOR] ==========================================")
        logger.info(f"[PREPROCESSOR] PRÉ-PROCESSAMENTO CONCLUÍDO")
        logger.info(f"[PREPROCESSOR] ==========================================")
        logger.info(f"  ✅ Processados: {processed_count}")
        logger.info(f"  ⏭️  Pulados: {skipped_count}")
        logger.info(f"  ❌ Erros: {error_count}")
        logger.info(f"  📊 Total no database: {len(self.processed_data)}")
        logger.info(f"  💾 Dados salvos em: {self.output_file}")
        
    def save_data(self):
        """Salva dados processados em arquivo JSON com segurança"""
        try:
            # Create backup of existing file
            if self.output_file.exists():
                backup_file = self.output_file.with_suffix('.backup.json')
                import shutil
                shutil.copy2(self.output_file, backup_file)
            
            # Save to temporary file first, then rename (atomic operation)
            temp_file = self.output_file.with_suffix('.tmp.json')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.output_file)
            
            logger.info(f"[PREPROCESSOR] Dados salvos em {self.output_file}")
            
        except Exception as e:
            logger.error(f"[PREPROCESSOR] Erro ao salvar dados: {e}")
            # Clean up temp file if it exists
            temp_file = self.output_file.with_suffix('.tmp.json')
            if temp_file.exists():
                temp_file.unlink()
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo dos dados processados"""
        if not self.processed_data:
            return {"total": 0, "movements": {}, "players": []}
        
        movements = {}
        players = set()
        
        for file_key, data in self.processed_data.items():
            movement = data.get('movement_type', 'unknown')
            player = data.get('player_name', 'unknown')
            
            movements[movement] = movements.get(movement, 0) + 1
            players.add(player)
        
        return {
            "total": len(self.processed_data),
            "movements": movements,
            "players": sorted(list(players)),
            "last_updated": datetime.now().isoformat()
        }


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pré-processa vídeos profissionais para análise biomecânica")
    parser.add_argument('--force', action='store_true', help='Força reprocessamento de todos os arquivos')
    parser.add_argument('--summary', action='store_true', help='Mostra apenas resumo dos dados existentes')
    parser.add_argument('--professionals-dir', default='profissionais', help='Diretório dos vídeos profissionais')
    parser.add_argument('--output', default='professionals_biomech_data.json', help='Arquivo de saída')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ProfessionalPreprocessor(
        professionals_dir=args.professionals_dir,
        output_file=args.output
    )
    
    if args.summary:
        # Show summary only
        summary = preprocessor.get_summary()
        print(f"\n[RESUMO] DADOS PROFISSIONAIS:")
        print(f"  Total de profissionais: {summary['total']}")
        print(f"  Movimentos:")
        for movement, count in summary['movements'].items():
            print(f"    - {movement}: {count}")
        print(f"  Jogadores: {', '.join(summary['players'])}")
        print(f"  Ultima atualizacao: {summary.get('last_updated', 'N/A')}")
    else:
        # Process professionals
        preprocessor.process_all_professionals(force_reprocess=args.force)
        
        # Show final summary
        summary = preprocessor.get_summary()
        print(f"\n[RESUMO] FINAL:")
        print(f"  Total de profissionais processados: {summary['total']}")
        print(f"  Arquivo de dados: {args.output}")


if __name__ == "__main__":
    main()