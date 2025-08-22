#!/usr/bin/env python3

"""
Optimized Professional Comparison System - Tennis Analyzer

Sistema otimizado de comparação que usa dados pré-processados dos profissionais
para análises rápidas e comparações detalhadas com usuários.

FUNCIONALIDADE:
- Carrega dados biomecânicos pré-processados dos profissionais
- Realiza comparações diretas sem reprocessar vídeos profissionais
- Gera relatórios comparativos detalhados
- Suporta filtragem por tipo de movimento e características

AUTOR: Sistema Tennis Analyzer + Claude
DATA: 2025-07-28
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Resultado de comparação entre usuário e profissional"""
    professional_name: str
    professional_video: str
    similarity_score: float
    detailed_comparison: Dict[str, Any]
    recommendations: List[str]
    confidence: float

class OptimizedProfessionalComparator:
    """
    Comparador otimizado usando dados pré-processados dos profissionais
    """
    
    def __init__(self, professionals_data_file: str = "professionals_biomech_data.json"):
        self.professionals_data_file = Path(professionals_data_file)
        self.professionals_data = {}
        self.load_professional_data()
        
        logger.info(f"[COMPARATOR] Inicializado")
        logger.info(f"  - Arquivo de dados: {self.professionals_data_file}")
        logger.info(f"  - Profissionais carregados: {len(self.professionals_data)}")
    
    def load_professional_data(self):
        """Carrega dados pré-processados dos profissionais"""
        try:
            if self.professionals_data_file.exists():
                with open(self.professionals_data_file, 'r', encoding='utf-8') as f:
                    self.professionals_data = json.load(f)
                logger.info(f"[COMPARATOR] Carregados {len(self.professionals_data)} profissionais")
                
                # DEBUG: Log dos movement_types carregados
                movement_types = set()
                for file_key, prof_data in self.professionals_data.items():
                    movement_type = prof_data.get('movement_type')
                    player_name = prof_data.get('player_name')
                    movement_types.add(movement_type)
                    logger.debug(f"[DEBUG_LOAD] {player_name}: {movement_type}")
                
                logger.info(f"[DEBUG_LOAD] Movement types disponíveis: {sorted(movement_types)}")
            else:
                logger.error(f"[COMPARATOR] Arquivo não encontrado: {self.professionals_data_file}")
                self.professionals_data = {}
        except Exception as e:
            logger.error(f"[COMPARATOR] Erro ao carregar dados: {e}")
            self.professionals_data = {}
    
    def get_professionals_by_movement(self, movement_type: str) -> List[Dict[str, Any]]:
        """Retorna lista de profissionais filtrados por tipo de movimento"""
        logger.info(f"[DEBUG_FILTER] Filtrando profissionais para movimento: {movement_type}")
        logger.info(f"[DEBUG_FILTER] Total de profissionais na base: {len(self.professionals_data)}")
        
        matching_professionals = []
        
        for file_key, prof_data in self.professionals_data.items():
            prof_movement = prof_data.get('movement_type')
            logger.debug(f"[DEBUG_FILTER] {file_key}: movement_type = {prof_movement}")
            
            if prof_movement == movement_type:
                matching_professionals.append(prof_data)
                logger.info(f"[DEBUG_FILTER] Match encontrado: {prof_data.get('player_name', 'unknown')}")
        
        # Ordenar por confiança (maior primeiro)
        matching_professionals.sort(
            key=lambda x: x.get('classification', {}).get('confidence', 0), 
            reverse=True
        )
        
        logger.info(f"[FILTER] Encontrados {len(matching_professionals)} profissionais para {movement_type}")
        if not matching_professionals:
            # Log dos movement_types disponíveis para debug
            available_movements = set(prof_data.get('movement_type') for prof_data in self.professionals_data.values())
            logger.warning(f"[DEBUG_FILTER] Movimentos disponíveis na base: {available_movements}")
            
        return matching_professionals
    
    def calculate_biomechanical_similarity(self, user_data: Dict[str, Any], prof_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calcula similaridade biomecânica entre usuário e profissional"""
        similarities = {}
        weights = {
            'elbow_variation': 0.25,
            'coordination': 0.20,
            'amplitude': 0.20,
            'velocity': 0.15,
            'temporal_pattern': 0.10,
            'movement_signature': 0.10
        }
        
        # Extrair métricas do usuário
        user_biomech = user_data.get('detailed_analysis', {}).get('joint_angles', {})
        user_dynamics = user_data.get('detailed_analysis', {}).get('movement_dynamics', {})
        user_bilat = user_data.get('detailed_analysis', {}).get('biomechanical_metrics', {})
        
        # Extrair métricas do profissional
        prof_biomech = prof_data.get('biomechanics', {})
        prof_dynamics = prof_data.get('dynamics', {})
        prof_bilat = prof_data.get('bilateral_analysis', {})
        
        try:
            # 1. Similaridade de variação do cotovelo
            user_elbow = user_biomech.get('elbow_variation_degrees', 0)
            prof_elbow = prof_biomech.get('elbow_variation_degrees', 0)
            if user_elbow > 0 and prof_elbow > 0:
                elbow_diff = abs(user_elbow - prof_elbow) / max(user_elbow, prof_elbow)
                similarities['elbow_variation'] = max(0, 1 - elbow_diff)
            else:
                similarities['elbow_variation'] = 0.5
            
            # 2. Similaridade de coordenação
            user_coord = user_biomech.get('coordination_score', 0)
            prof_coord = prof_biomech.get('coordination_score', 0)
            if user_coord > 0 and prof_coord > 0:
                coord_diff = abs(user_coord - prof_coord) / max(user_coord, prof_coord)
                similarities['coordination'] = max(0, 1 - coord_diff)
            else:
                similarities['coordination'] = 0.5
            
            # 3. Similaridade de amplitude
            user_amp = user_dynamics.get('amplitude_y', 0)
            prof_amp = prof_dynamics.get('amplitude_y', 0)
            if user_amp > 0 and prof_amp > 0:
                amp_diff = abs(user_amp - prof_amp) / max(user_amp, prof_amp)
                similarities['amplitude'] = max(0, 1 - amp_diff)
            else:
                similarities['amplitude'] = 0.5
            
            # 4. Similaridade de velocidade
            user_vel = user_dynamics.get('max_velocity', 0)
            prof_vel = prof_dynamics.get('max_velocity', 0)
            if user_vel > 0 and prof_vel > 0:
                vel_diff = abs(user_vel - prof_vel) / max(user_vel, prof_vel)
                similarities['velocity'] = max(0, 1 - vel_diff)
            else:
                similarities['velocity'] = 0.5
            
            # 5. Padrão temporal (categórico)
            user_pattern = user_dynamics.get('temporal_pattern', '')
            prof_pattern = prof_biomech.get('temporal_pattern', '')
            similarities['temporal_pattern'] = 1.0 if user_pattern == prof_pattern else 0.3
            
            # 6. Assinatura do movimento
            user_sig = user_biomech.get('movement_signature', 0)
            prof_sig = prof_biomech.get('movement_signature', 0)
            if user_sig > 0 and prof_sig > 0:
                sig_diff = abs(user_sig - prof_sig) / max(user_sig, prof_sig)
                similarities['movement_signature'] = max(0, 1 - sig_diff)
            else:
                similarities['movement_signature'] = 0.5
            
        except Exception as e:
            logger.warning(f"[SIMILARITY] Erro no cálculo: {e}")
            # Valores padrão em caso de erro
            for key in weights.keys():
                similarities[key] = 0.5
        
        # Calcular score ponderado
        weighted_score = sum(similarities[key] * weights[key] for key in weights.keys())
        
        detailed_comparison = {
            'individual_similarities': similarities,
            'weights_used': weights,
            'user_metrics': {
                'elbow_variation': user_elbow,
                'coordination': user_coord,
                'amplitude': user_amp,
                'velocity': user_vel,
                'temporal_pattern': user_pattern,
                'movement_signature': user_sig
            },
            'professional_metrics': {
                'elbow_variation': prof_elbow,
                'coordination': prof_coord,
                'amplitude': prof_amp,
                'velocity': prof_vel,
                'temporal_pattern': prof_pattern,
                'movement_signature': prof_sig
            }
        }
        
        return weighted_score, detailed_comparison
    
    def generate_comparison_recommendations(self, similarity_data: Dict[str, Any], prof_data: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na comparação"""
        recommendations = []
        similarities = similarity_data['individual_similarities']
        user_metrics = similarity_data['user_metrics']
        prof_metrics = similarity_data['professional_metrics']
        
        # Recomendações baseadas nas maiores diferenças
        for metric, similarity in similarities.items():
            if similarity < 0.7:  # Baixa similaridade
                if metric == 'elbow_variation':
                    user_val = user_metrics['elbow_variation']
                    prof_val = prof_metrics['elbow_variation']
                    if user_val > prof_val:
                        recommendations.append(f"Reduza a variação do cotovelo: você tem {user_val:.1f}°, profissional tem {prof_val:.1f}°")
                    else:
                        recommendations.append(f"Aumente a variação do cotovelo: você tem {user_val:.1f}°, profissional tem {prof_val:.1f}°")
                
                elif metric == 'coordination':
                    user_val = user_metrics['coordination']
                    prof_val = prof_metrics['coordination']
                    if user_val < prof_val:
                        recommendations.append(f"Melhore a coordenação: você tem {user_val:.1%}, profissional tem {prof_val:.1%}")
                
                elif metric == 'amplitude':
                    user_val = user_metrics['amplitude']
                    prof_val = prof_metrics['amplitude']
                    if user_val < prof_val:
                        recommendations.append(f"Aumente a amplitude do movimento: você tem {user_val:.3f}, profissional tem {prof_val:.3f}")
                    else:
                        recommendations.append(f"Controle a amplitude do movimento: você tem {user_val:.3f}, profissional tem {prof_val:.3f}")
                
                elif metric == 'velocity':
                    user_val = user_metrics['velocity']
                    prof_val = prof_metrics['velocity']
                    if user_val < prof_val:
                        recommendations.append(f"Aumente a velocidade: você tem {user_val:.3f}, profissional tem {prof_val:.3f}")
                    else:
                        recommendations.append(f"Controle a velocidade: você tem {user_val:.3f}, profissional tem {prof_val:.3f}")
                
                elif metric == 'temporal_pattern':
                    user_pattern = user_metrics['temporal_pattern']
                    prof_pattern = prof_metrics['temporal_pattern']
                    recommendations.append(f"Ajuste o padrão temporal: você tem '{user_pattern}', profissional tem '{prof_pattern}'")
        
        # Adicionar pontos fortes
        strong_points = [metric for metric, sim in similarities.items() if sim > 0.8]
        if strong_points:
            recommendations.append(f"Pontos fortes: {', '.join(strong_points)}")
        
        return recommendations
    
    def compare_with_professional(self, user_analysis: Dict[str, Any], professional_key: str) -> Optional[ComparisonResult]:
        """Compara análise do usuário com um profissional específico"""
        if professional_key not in self.professionals_data:
            logger.error(f"[COMPARE] Profissional não encontrado: {professional_key}")
            return None
        
        prof_data = self.professionals_data[professional_key]
        
        # Calcular similaridade
        similarity_score, detailed_comparison = self.calculate_biomechanical_similarity(user_analysis, prof_data)
        
        # Gerar recomendações
        recommendations = self.generate_comparison_recommendations(detailed_comparison, prof_data)
        
        # Calcular confiança da comparação
        user_confidence = user_analysis.get('detailed_analysis', {}).get('movement_classification', {}).get('confidence', 0)
        prof_confidence = prof_data.get('classification', {}).get('confidence', 0)
        comparison_confidence = (user_confidence + prof_confidence) / 2
        
        return ComparisonResult(
            professional_name=prof_data.get('player_name', 'Unknown'),
            professional_video=prof_data.get('video_file', 'Unknown'),
            similarity_score=similarity_score,
            detailed_comparison=detailed_comparison,
            recommendations=recommendations,
            confidence=comparison_confidence
        )
    
    def find_best_matches(self, user_analysis: Dict[str, Any], movement_type: str, max_results: int = 3) -> List[ComparisonResult]:
        """Encontra os melhores matches profissionais para o usuário"""
        logger.info(f"[DEBUG_COMPARATOR] Iniciando busca por matches para movimento: {movement_type}")
        logger.info(f"[DEBUG_COMPARATOR] User analysis keys: {list(user_analysis.keys()) if user_analysis else 'None'}")
        
        # DEBUG: Escrever em arquivo
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"=== COMPARATOR DEBUG ===\n")
            f.write(f"Movement type solicitado: {movement_type}\n")
            f.write(f"Total profissionais na base: {len(self.professionals_data)}\n")
        
        professionals = self.get_professionals_by_movement(movement_type)
        
        if not professionals:
            logger.warning(f"[MATCH] Nenhum profissional encontrado para {movement_type}")
            logger.info(f"[DEBUG_COMPARATOR] Profissionais disponíveis: {list(self.professionals_data.keys())}")
            
            # DEBUG: Escrever resultado em arquivo
            with open("debug_comparison.log", "a", encoding="utf-8") as f:
                f.write(f"RESULTADO: Nenhum profissional encontrado!\n")
                f.write(f"Profissionais disponíveis:\n")
                for key, prof in self.professionals_data.items():
                    f.write(f"  - {prof.get('player_name', 'unknown')}: {prof.get('movement_type', 'unknown')}\n")
                f.write(f"=== FIM DEBUG ===\n\n")
            
            return []
        
        logger.info(f"[DEBUG_COMPARATOR] Encontrados {len(professionals)} profissionais para {movement_type}")
        
        comparisons = []
        
        for prof_data in professionals:
            # Encontrar a chave real nos dados
            matching_key = None
            for real_key, data in self.professionals_data.items():
                if data.get('video_file') == prof_data.get('video_file'):
                    matching_key = real_key
                    break
            
            if matching_key:
                logger.info(f"[MATCH] Comparando com {prof_data.get('player_name', 'unknown')}")
                comparison = self.compare_with_professional(user_analysis, matching_key)
                if comparison:
                    comparisons.append(comparison)
            else:
                logger.warning(f"[MATCH] Chave não encontrada para {prof_data.get('video_file', 'unknown')}")
        
        # Ordenar por score de similaridade
        comparisons.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return comparisons[:max_results]
    
    def get_movement_statistics(self, movement_type: str) -> Dict[str, Any]:
        """Retorna estatísticas dos profissionais para um tipo de movimento"""
        professionals = self.get_professionals_by_movement(movement_type)
        
        if not professionals:
            return {}
        
        # Coletar métricas
        elbow_variations = []
        coordinations = []
        amplitudes = []
        velocities = []
        
        for prof in professionals:
            biomech = prof.get('biomechanics', {})
            dynamics = prof.get('dynamics', {})
            
            if biomech.get('elbow_variation_degrees'):
                elbow_variations.append(biomech['elbow_variation_degrees'])
            if biomech.get('coordination_score'):
                coordinations.append(biomech['coordination_score'])
            if dynamics.get('amplitude_y'):
                amplitudes.append(dynamics['amplitude_y'])
            if dynamics.get('max_velocity'):
                velocities.append(dynamics['max_velocity'])
        
        statistics = {
            'count': len(professionals),
            'elbow_variation': {
                'mean': np.mean(elbow_variations) if elbow_variations else 0,
                'std': np.std(elbow_variations) if elbow_variations else 0,
                'min': np.min(elbow_variations) if elbow_variations else 0,
                'max': np.max(elbow_variations) if elbow_variations else 0
            },
            'coordination': {
                'mean': np.mean(coordinations) if coordinations else 0,
                'std': np.std(coordinations) if coordinations else 0,
                'min': np.min(coordinations) if coordinations else 0,
                'max': np.max(coordinations) if coordinations else 0
            },
            'amplitude': {
                'mean': np.mean(amplitudes) if amplitudes else 0,
                'std': np.std(amplitudes) if amplitudes else 0,
                'min': np.min(amplitudes) if amplitudes else 0,
                'max': np.max(amplitudes) if amplitudes else 0
            },
            'velocity': {
                'mean': np.mean(velocities) if velocities else 0,
                'std': np.std(velocities) if velocities else 0,
                'min': np.min(velocities) if velocities else 0,
                'max': np.max(velocities) if velocities else 0
            }
        }
        
        return statistics


def main():
    """Função de teste"""
    comparator = OptimizedProfessionalComparator()
    
    # Exemplo de uso
    movement_type = "backhand_push"
    professionals = comparator.get_professionals_by_movement(movement_type)
    
    print(f"\n[TESTE] Profissionais para {movement_type}:")
    for prof in professionals:
        print(f"  - {prof['player_name']}: {prof['classification']['confidence']:.1%} confiança")
    
    # Estatísticas
    stats = comparator.get_movement_statistics(movement_type)
    print(f"\n[ESTATÍSTICAS] {movement_type}:")
    print(f"  - Variação cotovelo: {stats.get('elbow_variation', {}).get('mean', 0):.1f}° ± {stats.get('elbow_variation', {}).get('std', 0):.1f}°")
    print(f"  - Coordenação: {stats.get('coordination', {}).get('mean', 0):.1%} ± {stats.get('coordination', {}).get('std', 0):.1%}")
    print(f"  - Amplitude: {stats.get('amplitude', {}).get('mean', 0):.3f} ± {stats.get('amplitude', {}).get('std', 0):.3f}")


if __name__ == "__main__":
    main()