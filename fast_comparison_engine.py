#!/usr/bin/env python3
"""
⚡ FAST COMPARISON ENGINE
Motor de comparação ultra-rápido usando dados pré-computados de profissionais.
Integrado com TableTennisAnalyzer - comparações instantâneas!
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass

# Importar componentes existentes
from cycle_detector_retracted_extended import CycleDetectorIntegration
from real_biomech_comparison import RealBiomechComparison

@dataclass
class ComparisonResult:
    """Resultado estruturado de comparação rápida"""
    user_name: str
    professional_name: str
    movement: str
    similarity_score: float
    confidence_level: str
    analysis_type: str
    data_estimated: bool
    cycles_user: int
    cycles_professional: int
    phase_scores: Dict[str, float]
    recommendations: List[str]
    executive_summary: Dict[str, str]

class FastComparisonEngine:
    """
    ⚡ Motor de Comparação Rápida
    Usa dados pré-computados para comparações instantâneas
    """
    
    def __init__(self, database_directory: str = "professionals_database/"):
        self.db_dir = Path(database_directory)
        self.logger = logging.getLogger(__name__)
        
        # Cache de perfis carregados
        self._profiles_cache = {}
        self._index_cache = None
        
        # Componentes para análise de usuário
        self.cycle_detector = CycleDetectorIntegration()
        self.biomech_analyzer = RealBiomechComparison()
        
        # Carregar índice de perfis
        self._load_profiles_index()
    
    def _load_profiles_index(self) -> None:
        """📋 Carrega índice de perfis disponíveis"""
        index_path = self.db_dir / 'profiles_index.json'
        
        if not index_path.exists():
            self.logger.warning("❌ Índice de perfis não encontrado. Execute o ProfessionalDataExtractor primeiro.")
            self._index_cache = {'profiles': []}
            return
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                self._index_cache = json.load(f)
            
            self.logger.info(f"📋 Índice carregado: {len(self._index_cache['profiles'])} perfis disponíveis")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar índice: {e}")
            self._index_cache = {'profiles': []}
    
    def load_professional_profile(self, name: str, movement: str) -> Optional[Dict]:
        """💾 Carrega perfil específico de profissional"""
        profile_key = f"{name.lower()}_{movement.lower()}"
        
        # Verificar cache
        if profile_key in self._profiles_cache:
            return self._profiles_cache[profile_key]
        
        # Carregar do arquivo
        profile_filename = f"{profile_key}_cycles.json"
        profile_path = self.db_dir / profile_filename
        
        if not profile_path.exists():
            self.logger.warning(f"⚠️ Perfil não encontrado: {profile_filename}")
            return None
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            # Adicionar ao cache
            self._profiles_cache[profile_key] = profile_data
            
            self.logger.info(f"✅ Perfil carregado: {name} - {movement}")
            return profile_data
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar perfil {profile_filename}: {e}")
            return None
    
    def get_available_professionals(self) -> List[Dict[str, str]]:
        """📋 Lista profissionais disponíveis na base"""
        if not self._index_cache:
            return []
        
        return self._index_cache.get('profiles', [])
    
    def find_best_professional_match(self, movement: str) -> Optional[str]:
        """🎯 Encontra o melhor profissional para um movimento específico"""
        available = self.get_available_professionals()
        movement_matches = [p for p in available if p['movement'].lower() == movement.lower()]
        
        if not movement_matches:
            return None
        
        # Ordenar por qualidade (maior score primeiro)
        movement_matches.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return movement_matches[0]['name']
    
    def analyze_user_video(self, user_video_path: str) -> Dict[str, Any]:
        """👤 Analisa vídeo do usuário usando sistema existente"""
        self.logger.info(f"👤 Analisando vídeo do usuário: {user_video_path}")
        
        try:
            # Usar detector de ciclos integrado
            user_cycles = self.cycle_detector.detect_cycles_from_video(user_video_path)
            
            if not user_cycles.get('success', False):
                return {'success': False, 'error': 'User cycle detection failed'}
            
            # Extrair features biomecânicas do usuário
            user_features = self._extract_user_biomech_features(user_cycles)
            
            return {
                'success': True,
                'cycles_data': user_cycles.get('cycles', []),
                'cycles_count': len(user_cycles.get('cycles', [])),
                'extension_timeline': user_cycles.get('extension_values', []),
                'biomech_features': user_features,
                'quality_metrics': {
                    'avg_cycle_duration': user_cycles.get('avg_cycle_duration', 0),
                    'cycle_consistency': user_cycles.get('cycle_consistency', 0),
                    'detection_confidence': user_cycles.get('confidence', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise do usuário: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_user_biomech_features(self, cycles_result: Dict) -> Dict[str, float]:
        """📊 Extrai features biomecânicas do usuário (igual ao extrator)"""
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
    
    def compare_user_vs_precomputed(self, user_video_path: str, professional_name: str, movement: str = None) -> Dict[str, Any]:
        """⚡ Comparação ultra-rápida usando dados pré-computados"""
        self.logger.info(f"⚡ Iniciando comparação rápida: usuário vs {professional_name}")
        
        # 1. Analisar usuário
        user_analysis = self.analyze_user_video(user_video_path)
        if not user_analysis.get('success', False):
            return {
                'success': False,
                'error': 'Falha na análise do usuário',
                'details': user_analysis.get('error', 'Unknown error')
            }
        
        # 2. Determinar movimento se não especificado
        if not movement:
            movement = 'FD'  # Default para forehand drive
            self.logger.info(f"Movimento não especificado, usando padrão: {movement}")
        
        # 3. Carregar perfil profissional pré-computado
        pro_profile = self.load_professional_profile(professional_name, movement)
        if not pro_profile:
            return {
                'success': False,
                'error': f'Perfil profissional não encontrado: {professional_name} - {movement}'
            }
        
        # 4. Realizar comparação biomecânica rápida
        comparison_result = self._fast_biomech_comparison(user_analysis, pro_profile)
        
        # 5. Gerar resultado estruturado
        result = self._generate_comparison_result(
            user_analysis, pro_profile, comparison_result, professional_name, movement
        )
        
        self.logger.info(f"✅ Comparação concluída: Score {result['final_score']:.1f}%")
        return result
    
    def _fast_biomech_comparison(self, user_data: Dict, pro_profile: Dict) -> Dict[str, float]:
        """🧮 Comparação biomecânica otimizada"""
        user_features = user_data.get('biomech_features', {})
        pro_features = pro_profile.get('biomech_metrics', {})
        
        # Comparações diretas entre features
        comparisons = {}
        
        # 1. Duração dos ciclos
        user_duration = user_features.get('avg_duration', 0)
        pro_duration = pro_features.get('avg_duration', 0)
        if pro_duration > 0:
            duration_similarity = 1.0 - abs(user_duration - pro_duration) / max(user_duration, pro_duration)
            comparisons['duration'] = max(0, min(1, duration_similarity))
        else:
            comparisons['duration'] = 0.5
        
        # 2. Amplitude dos movimentos
        user_amplitude = user_features.get('avg_amplitude', 0)
        pro_amplitude = pro_features.get('avg_amplitude', 0)
        if pro_amplitude > 0:
            amplitude_similarity = 1.0 - abs(user_amplitude - pro_amplitude) / max(user_amplitude, pro_amplitude)
            comparisons['amplitude'] = max(0, min(1, amplitude_similarity))
        else:
            comparisons['amplitude'] = 0.5
        
        # 3. Consistência/Ritmo
        user_consistency = user_features.get('rhythm_consistency', 0)
        pro_consistency = pro_features.get('rhythm_consistency', 0)
        if pro_consistency > 0:
            consistency_similarity = min(user_consistency / pro_consistency, 1.0)
            comparisons['consistency'] = max(0, min(1, consistency_similarity))
        else:
            comparisons['consistency'] = 0.5
        
        # 4. Qualidade geral
        user_quality = user_features.get('overall_quality', 0)
        pro_quality = pro_profile.get('quality_score', 0)
        if pro_quality > 0:
            quality_ratio = min(user_quality / pro_quality, 1.0)
            comparisons['quality'] = max(0, min(1, quality_ratio))
        else:
            comparisons['quality'] = 0.5
        
        return comparisons
    
    def _generate_comparison_result(self, user_data: Dict, pro_profile: Dict, 
                                  comparison: Dict, pro_name: str, movement: str) -> Dict[str, Any]:
        """📊 Gera resultado final estruturado"""
        
        # Calcular score final ponderado
        weights = {
            'duration': 0.25,
            'amplitude': 0.25,
            'consistency': 0.25,
            'quality': 0.25
        }
        
        final_score = sum(comparison.get(metric, 0.5) * weight 
                         for metric, weight in weights.items()) * 100
        
        # Determinar nível de confiança
        user_cycles_count = user_data.get('cycles_count', 0)
        pro_cycles_count = len(pro_profile.get('cycles_data', []))
        
        if user_cycles_count >= 3 and pro_cycles_count >= 5:
            confidence = 'high'
        elif user_cycles_count >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Phase scores (aproximação baseada nas comparações)
        phase_scores = {
            'preparation': comparison.get('duration', 0.5) * 100,
            'contact': comparison.get('amplitude', 0.5) * 100,
            'follow_through': comparison.get('consistency', 0.5) * 100,
            'overall_quality': comparison.get('quality', 0.5) * 100
        }
        
        # Gerar recomendações
        recommendations = self._generate_recommendations(comparison, user_data, pro_profile)
        
        # Executive summary
        if final_score >= 80:
            performance_level = 'Excelente'
            performance_emoji = '🏆'
            key_message = 'Técnica muito próxima ao profissional!'
        elif final_score >= 65:
            performance_level = 'Bom'
            performance_emoji = '👍'
            key_message = 'Boa técnica detectada'
        elif final_score >= 50:
            performance_level = 'Regular'
            performance_emoji = '🔄'
            key_message = 'Técnica regular detectada'
        else:
            performance_level = 'Precisa melhorar'
            performance_emoji = '📈'
            key_message = 'Há muito espaço para melhoria'
        
        return {
            'success': True,
            'final_score': round(final_score, 2),
            'analysis_type': 'fast_precomputed_comparison',
            'data_estimated': False,
            'confidence_level': confidence,
            'cycles_detected': {
                'user': user_cycles_count,
                'professional': pro_cycles_count
            },
            'phase_scores': phase_scores,
            'detailed_comparison': comparison,
            'professional_info': {
                'name': pro_name,
                'movement': movement,
                'profile_quality': pro_profile.get('quality_score', 0),
                'cycles_available': pro_cycles_count
            },
            'recommendations': recommendations,
            'executive_summary': {
                'performance_level': performance_level,
                'performance_emoji': performance_emoji,
                'key_message': key_message
            },
            'processing_info': {
                'method': 'Precomputed Professional Data',
                'speed': 'Ultra-fast',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _generate_recommendations(self, comparison: Dict, user_data: Dict, pro_profile: Dict) -> List[str]:
        """💡 Gera recomendações baseadas na comparação"""
        recommendations = []
        
        # Recomendações baseadas em métricas específicas
        if comparison.get('duration', 0.5) < 0.6:
            recommendations.append("Trabalhe a consistência do timing dos movimentos")
        
        if comparison.get('amplitude', 0.5) < 0.6:
            recommendations.append("Aumente a amplitude dos movimentos para melhor técnica")
        
        if comparison.get('consistency', 0.5) < 0.6:
            recommendations.append("Pratique mais para melhorar a consistência")
        
        if user_data.get('cycles_count', 0) < 4:
            recommendations.append("Realize movimentos mais longos para melhor análise")
        
        # Recomendação específica do profissional
        pro_quality = pro_profile.get('quality_score', 0)
        if pro_quality > 0.8:
            recommendations.append(f"Estude vídeos do {pro_profile.get('name', 'profissional')} para técnica de referência")
        
        # Recomendação padrão se não houver outras
        if not recommendations:
            recommendations.append("Continue praticando para manter a boa técnica")
        
        return recommendations
    
    def compare_multiple_professionals(self, user_video_path: str, movement: str = None) -> Dict[str, Any]:
        """🏆 Compara usuário com múltiplos profissionais"""
        self.logger.info(f"🏆 Comparação múltipla iniciada para movimento: {movement}")
        
        available_pros = self.get_available_professionals()
        if movement:
            available_pros = [p for p in available_pros if p['movement'].lower() == movement.lower()]
        
        if not available_pros:
            return {
                'success': False,
                'error': f'Nenhum profissional disponível para movimento: {movement}'
            }
        
        results = []
        for pro in available_pros:
            comparison = self.compare_user_vs_precomputed(
                user_video_path, pro['name'], pro['movement']
            )
            
            if comparison.get('success', False):
                results.append({
                    'professional': pro['name'],
                    'movement': pro['movement'],
                    'score': comparison['final_score'],
                    'confidence': comparison['confidence_level'],
                    'cycles_pro': comparison['cycles_detected']['professional'],
                    'quality_score': pro.get('quality_score', 0)
                })
        
        # Ordenar por score (maior primeiro)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'success': True,
            'total_comparisons': len(results),
            'best_match': results[0] if results else None,
            'all_comparisons': results,
            'user_video': user_video_path,
            'movement_filter': movement,
            'ranking_summary': [
                f"{i+1}. {r['professional']} - {r['score']:.1f}%" 
                for i, r in enumerate(results[:5])  # Top 5
            ]
        }

def main():
    """🚀 Teste do motor de comparação rápida"""
    print("⚡ FAST COMPARISON ENGINE - TESTE")
    print("==================================")
    
    engine = FastComparisonEngine()
    
    # Listar profissionais disponíveis
    professionals = engine.get_available_professionals()
    print(f"\n📋 Profissionais disponíveis: {len(professionals)}")
    for pro in professionals:
        print(f"   • {pro['name']} - {pro['movement']} (Score: {pro.get('quality_score', 0):.2f})")
    
    # Teste de comparação rápida (se houver vídeos)
    user_video = "videos/Americo_FD_D_E.mp4"
    if Path(user_video).exists() and professionals:
        print(f"\n⚡ Teste de comparação rápida:")
        print(f"   👤 Usuário: {user_video}")
        print(f"   🏆 Profissional: {professionals[0]['name']}")
        
        result = engine.compare_user_vs_precomputed(
            user_video, 
            professionals[0]['name'], 
            professionals[0]['movement']
        )
        
        if result.get('success', False):
            print(f"   📊 Score: {result['final_score']:.1f}%")
            print(f"   🎯 Confiança: {result['confidence_level']}")
            print(f"   ⚡ Método: {result['processing_info']['method']}")
        else:
            print(f"   ❌ Erro: {result.get('error', 'Unknown')}")
    
    print(f"\n✅ Teste concluído!")

if __name__ == "__main__":
    main()
