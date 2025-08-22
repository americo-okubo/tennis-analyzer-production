#!/usr/bin/env python3
"""
📈 PROGRESSION ANALYZER
Módulo para análise de progressão e tendências ao longo do tempo.
Rastreia evolução técnica, identifica padrões e fornece insights preditivos.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from scipy.interpolate import interp1d

# Imports do sistema existente
from fast_comparison_engine import FastComparisonEngine
from professional_data_extractor import ProfessionalDataExtractor

@dataclass
class ProgressionEntry:
    """Entrada individual de progressão"""
    timestamp: str
    user_video: str
    professional_reference: str
    movement: str
    score: float
    confidence: str
    cycles_detected: int
    phase_scores: Dict[str, float]
    technical_metrics: Dict[str, float]
    recommendations: List[str]
    session_notes: str = ""

@dataclass
class ProgressionSummary:
    """Resumo de progressão entre períodos"""
    period_start: str
    period_end: str
    total_sessions: int
    avg_score: float
    score_improvement: float
    best_score: float
    worst_score: float
    consistency_index: float
    primary_weaknesses: List[str]
    improvement_areas: List[str]
    trend_direction: str  # 'improving', 'stable', 'declining'

class ProgressionAnalyzer:
    """
    📈 Analisador de Progressão
    Rastreia evolução técnica e fornece insights baseados em dados históricos
    """
    
    def __init__(self, database_directory: str = "progression_database/"):
        self.db_dir = Path(database_directory)
        self.db_dir.mkdir(exist_ok=True)
        
        # Componentes do sistema
        self.fast_engine = FastComparisonEngine()
        
        # Configuração de logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cache de dados de progressão
        self._progression_cache = {}
        
        print("📈 Progression Analyzer inicializado")
    
    def record_session(self, user_video: str, professional_ref: str, movement: str, 
                      session_notes: str = "") -> Dict[str, Any]:
        """
        📝 Registra uma nova sessão de análise
        """
        self.logger.info(f"📝 Registrando sessão: {user_video} vs {professional_ref}")
        
        # Executar análise usando FastEngine
        analysis_result = self.fast_engine.compare_user_vs_precomputed(
            user_video, professional_ref, movement
        )
        
        if not analysis_result.get('success', False):
            return {
                'success': False,
                'error': 'Falha na análise da sessão',
                'details': analysis_result.get('error', 'Unknown')
            }
        
        # Extrair métricas técnicas adicionais
        technical_metrics = self._extract_technical_metrics(analysis_result)
        
        # Criar entrada de progressão
        entry = ProgressionEntry(
            timestamp=datetime.now().isoformat(),
            user_video=user_video,
            professional_reference=professional_ref,
            movement=movement,
            score=analysis_result['final_score'],
            confidence=analysis_result['confidence_level'],
            cycles_detected=analysis_result['cycles_detected']['user'],
            phase_scores=analysis_result['phase_scores'],
            technical_metrics=technical_metrics,
            recommendations=analysis_result['recommendations'],
            session_notes=session_notes
        )
        
        # Salvar no banco de dados
        user_id = self._extract_user_id(user_video)
        saved_path = self._save_progression_entry(user_id, entry)
        
        # Gerar insights da sessão
        session_insights = self._generate_session_insights(user_id, entry)
        
        return {
            'success': True,
            'entry_id': entry.timestamp,
            'score': entry.score,
            'saved_path': saved_path,
            'session_insights': session_insights,
            'technical_summary': {
                'score': entry.score,
                'confidence': entry.confidence,
                'cycles': entry.cycles_detected,
                'best_phase': max(entry.phase_scores.items(), key=lambda x: x[1]),
                'worst_phase': min(entry.phase_scores.items(), key=lambda x: x[1])
            }
        }
    
    def _extract_technical_metrics(self, analysis_result: Dict) -> Dict[str, float]:
        """🔬 Extrai métricas técnicas detalhadas"""
        metrics = {}
        
        # Métricas de fase
        phase_scores = analysis_result.get('phase_scores', {})
        if phase_scores:
            metrics['phase_consistency'] = np.std(list(phase_scores.values()))
            metrics['phase_average'] = np.mean(list(phase_scores.values()))
            metrics['preparation_score'] = phase_scores.get('preparation', 0)
            metrics['contact_score'] = phase_scores.get('contact', 0)
            metrics['follow_through_score'] = phase_scores.get('follow_through', 0)
        
        # Métricas de ciclos
        cycles_user = analysis_result.get('cycles_detected', {}).get('user', 0)
        cycles_pro = analysis_result.get('cycles_detected', {}).get('professional', 0)
        
        if cycles_pro > 0:
            metrics['cycle_ratio'] = cycles_user / cycles_pro
        else:
            metrics['cycle_ratio'] = 0
        
        metrics['cycles_detected'] = cycles_user
        
        # Métricas de confiança
        confidence_mapping = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        metrics['confidence_numeric'] = confidence_mapping.get(
            analysis_result.get('confidence_level', 'low'), 0.3
        )
        
        return metrics
    
    def _extract_user_id(self, video_path: str) -> str:
        """👤 Extrai ID do usuário do caminho do vídeo"""
        video_name = Path(video_path).stem
        # Extrair nome do usuário (parte antes do primeiro underscore)
        user_parts = video_name.split('_')
        return user_parts[0] if user_parts else 'unknown_user'
    
    def _save_progression_entry(self, user_id: str, entry: ProgressionEntry) -> str:
        """💾 Salva entrada de progressão no banco"""
        user_db_path = self.db_dir / f"{user_id}_progression.json"
        
        # Carregar progressão existente ou criar nova
        if user_db_path.exists():
            with open(user_db_path, 'r', encoding='utf-8') as f:
                progression_data = json.load(f)
        else:
            progression_data = {
                'user_id': user_id,
                'created_date': datetime.now().isoformat(),
                'total_sessions': 0,
                'entries': []
            }
        
        # Adicionar nova entrada
        progression_data['entries'].append(asdict(entry))
        progression_data['total_sessions'] += 1
        progression_data['last_updated'] = datetime.now().isoformat()
        
        # Salvar
        with open(user_db_path, 'w', encoding='utf-8') as f:
            json.dump(progression_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Progressão salva: {user_db_path}")
        return str(user_db_path)
    
    def _generate_session_insights(self, user_id: str, current_entry: ProgressionEntry) -> Dict[str, Any]:
        """🔍 Gera insights sobre a sessão atual"""
        progression_history = self.load_user_progression(user_id)
        
        if not progression_history or len(progression_history['entries']) < 2:
            return {
                'type': 'first_session',
                'message': 'Primeira sessão registrada! Continue praticando para ver progressão.',
                'baseline_score': current_entry.score
            }
        
        # Comparar com sessão anterior
        previous_entry = progression_history['entries'][-2]  # Penúltima entrada
        score_change = current_entry.score - previous_entry['score']
        
        insights = {
            'score_change': score_change,
            'previous_score': previous_entry['score'],
            'current_score': current_entry.score,
            'sessions_total': len(progression_history['entries'])
        }
        
        # Determinar tipo de insight
        if score_change > 2.0:
            insights['type'] = 'significant_improvement'
            insights['message'] = f"Excelente! Melhoria de {score_change:.1f} pontos desde a última sessão!"
        elif score_change > 0:
            insights['type'] = 'improvement'
            insights['message'] = f"Boa! Melhoria de {score_change:.1f} pontos."
        elif score_change > -2.0:
            insights['type'] = 'stable'
            insights['message'] = "Performance estável. Continue praticando."
        else:
            insights['type'] = 'decline'
            insights['message'] = f"Queda de {abs(score_change):.1f} pontos. Revise a técnica."
        
        # Análise de fases
        phase_analysis = self._analyze_phase_progression(progression_history, current_entry)
        insights['phase_analysis'] = phase_analysis
        
        return insights
    
    def load_user_progression(self, user_id: str) -> Optional[Dict]:
        """📊 Carrega progressão completa de um usuário"""
        user_db_path = self.db_dir / f"{user_id}_progression.json"
        
        if not user_db_path.exists():
            return None
        
        try:
            with open(user_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Erro ao carregar progressão de {user_id}: {e}")
            return None
    
    def analyze_progression_trends(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """📈 Analisa tendências de progressão"""
        progression_data = self.load_user_progression(user_id)
        
        if not progression_data:
            return {'success': False, 'error': 'Usuário não encontrado'}
        
        entries = progression_data['entries']
        if len(entries) < 3:
            return {'success': False, 'error': 'Dados insuficientes para análise de tendências'}
        
        # Filtrar por período
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_entries = [
            entry for entry in entries 
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
        ]
        
        if len(recent_entries) < 2:
            return {'success': False, 'error': 'Poucos dados no período especificado'}
        
        # Análise de tendência dos scores
        scores = [entry['score'] for entry in recent_entries]
        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in recent_entries]
        
        # Regressão linear para tendência
        x_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, scores)
        
        # Determinar direção da tendência
        if slope > 0.1:
            trend_direction = 'improving'
            trend_strength = min(abs(slope) * 10, 1.0)  # Normalizar
        elif slope < -0.1:
            trend_direction = 'declining'
            trend_strength = min(abs(slope) * 10, 1.0)
        else:
            trend_direction = 'stable'
            trend_strength = 1.0 - min(abs(slope) * 10, 1.0)
        
        # Calcular métricas de consistência
        consistency_index = 1.0 / (1.0 + np.std(scores))
        
        # Análise por fases
        phase_trends = self._analyze_phase_trends(recent_entries)
        
        # Projeção futura (próximos 7 dias)
        future_projection = self._project_future_performance(scores, timestamps)
        
        return {
            'success': True,
            'period_days': days_back,
            'sessions_analyzed': len(recent_entries),
            'score_trend': {
                'direction': trend_direction,
                'strength': trend_strength,
                'slope': slope,
                'correlation': r_value,
                'significance': p_value
            },
            'current_performance': {
                'latest_score': scores[-1],
                'average_score': np.mean(scores),
                'best_score': max(scores),
                'worst_score': min(scores),
                'consistency_index': consistency_index
            },
            'phase_trends': phase_trends,
            'future_projection': future_projection,
            'recommendations': self._generate_trend_recommendations(slope, consistency_index, phase_trends)
        }
    
    def _analyze_phase_trends(self, entries: List[Dict]) -> Dict[str, Dict]:
        """📊 Analisa tendências por fase do movimento"""
        phases = ['preparation', 'contact', 'follow_through', 'overall_quality']
        phase_trends = {}
        
        for phase in phases:
            phase_scores = [entry['phase_scores'].get(phase, 0) for entry in entries]
            
            if len(phase_scores) >= 3:
                x = list(range(len(phase_scores)))
                slope, _, r_value, _, _ = stats.linregress(x, phase_scores)
                
                phase_trends[phase] = {
                    'current_score': phase_scores[-1],
                    'average_score': np.mean(phase_scores),
                    'trend_slope': slope,
                    'correlation': r_value,
                    'improvement': slope > 0.1,
                    'stability': abs(slope) < 0.1
                }
        
        return phase_trends
    
    def _project_future_performance(self, scores: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """🔮 Projeta performance futura baseada em tendências"""
        if len(scores) < 3:
            return {'available': False, 'reason': 'Dados insuficientes'}
        
        # Usar últimos pontos para projeção
        recent_scores = scores[-5:] if len(scores) >= 5 else scores
        x = list(range(len(recent_scores)))
        
        # Ajuste polinomial para capturar não-linearidade
        if len(recent_scores) >= 3:
            poly_coeff = np.polyfit(x, recent_scores, min(2, len(recent_scores)-1))
            
            # Projetar próximos pontos
            future_x = [len(recent_scores) + i for i in range(1, 4)]  # 3 pontos futuros
            future_scores = [np.polyval(poly_coeff, fx) for fx in future_x]
            
            # Limitar scores a range realista (0-100)
            future_scores = [max(0, min(100, score)) for score in future_scores]
            
            return {
                'available': True,
                'projected_scores': future_scores,
                'confidence': min(1.0, len(recent_scores) / 10.0),  # Mais dados = mais confiança
                'trend_type': 'polynomial',
                'next_session_estimate': future_scores[0] if future_scores else None
            }
        
        return {'available': False, 'reason': 'Modelo de projeção não aplicável'}
    
    def _generate_trend_recommendations(self, slope: float, consistency: float, 
                                      phase_trends: Dict) -> List[str]:
        """💡 Gera recomendações baseadas em tendências"""
        recommendations = []
        
        # Recomendações baseadas na tendência geral
        if slope > 0.2:
            recommendations.append("Excelente progresso! Continue com a rotina atual de treinamento.")
        elif slope > 0:
            recommendations.append("Progresso positivo. Considere aumentar a frequência de treinos.")
        elif slope < -0.1:
            recommendations.append("Performance em declínio. Revise técnica e considere descanso.")
        else:
            recommendations.append("Performance estável. Experimente novas técnicas para evolução.")
        
        # Recomendações baseadas na consistência
        if consistency < 0.6:
            recommendations.append("Trabalhe na consistência. Pratique movimentos básicos repetidamente.")
        elif consistency > 0.8:
            recommendations.append("Excelente consistência! Foque em refinamentos técnicos avançados.")
        
        # Recomendações baseadas em fases específicas
        if phase_trends:
            weakest_phase = min(phase_trends.items(), key=lambda x: x[1]['current_score'])
            recommendations.append(f"Foque na melhoria da fase: {weakest_phase[0]}")
            
            improving_phases = [phase for phase, data in phase_trends.items() 
                              if data.get('improvement', False)]
            if improving_phases:
                recommendations.append(f"Continue desenvolvendo: {', '.join(improving_phases)}")
        
        return recommendations[:4]  # Limitar a 4 recomendações
    
    def generate_progression_report(self, user_id: str, format_type: str = 'detailed') -> Dict[str, Any]:
        """📄 Gera relatório completo de progressão"""
        self.logger.info(f"📄 Gerando relatório de progressão para {user_id}")
        
        progression_data = self.load_user_progression(user_id)
        if not progression_data:
            return {'success': False, 'error': 'Usuário não encontrado'}
        
        # Análise de tendências
        trend_analysis = self.analyze_progression_trends(user_id, days_back=90)
        
        # Estatísticas gerais
        entries = progression_data['entries']
        scores = [entry['score'] for entry in entries]
        
        report = {
            'user_id': user_id,
            'report_date': datetime.now().isoformat(),
            'total_sessions': len(entries),
            'date_range': {
                'first_session': entries[0]['timestamp'] if entries else None,
                'last_session': entries[-1]['timestamp'] if entries else None
            },
            'performance_summary': {
                'current_score': scores[-1] if scores else 0,
                'best_score': max(scores) if scores else 0,
                'average_score': np.mean(scores) if scores else 0,
                'improvement_total': scores[-1] - scores[0] if len(scores) >= 2 else 0,
                'consistency_rating': 1.0 / (1.0 + np.std(scores)) if scores else 0
            },
            'trend_analysis': trend_analysis if trend_analysis.get('success') else None,
            'milestone_achievements': self._identify_milestones(entries),
            'training_insights': self._generate_training_insights(entries),
            'next_goals': self._suggest_next_goals(entries, trend_analysis)
        }
        
        # Adicionar detalhes se solicitado
        if format_type == 'detailed':
            report['session_details'] = entries
            report['score_history'] = scores
            report['phase_analysis_detailed'] = self._detailed_phase_analysis(entries)
        
        return {'success': True, 'report': report}
    
    def _identify_milestones(self, entries: List[Dict]) -> List[Dict[str, Any]]:
        """🏆 Identifica marcos importantes na progressão"""
        milestones = []
        scores = [entry['score'] for entry in entries]
        
        if not scores:
            return milestones
        
        # Marco: Primeira sessão
        milestones.append({
            'type': 'first_session',
            'date': entries[0]['timestamp'],
            'description': f"Primeira análise com score {scores[0]:.1f}%",
            'significance': 'baseline'
        })
        
        # Marco: Melhor score
        best_score_idx = scores.index(max(scores))
        milestones.append({
            'type': 'best_performance',
            'date': entries[best_score_idx]['timestamp'],
            'description': f"Melhor performance: {max(scores):.1f}%",
            'significance': 'peak'
        })
        
        # Marco: Maior melhoria em uma sessão
        if len(scores) >= 2:
            improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
            best_improvement_idx = improvements.index(max(improvements)) + 1
            
            if max(improvements) > 5.0:  # Melhoria significativa
                milestones.append({
                    'type': 'breakthrough',
                    'date': entries[best_improvement_idx]['timestamp'],
                    'description': f"Grande salto: +{max(improvements):.1f} pontos",
                    'significance': 'breakthrough'
                })
        
        # Marco: Consistência (múltiplas sessões acima de threshold)
        high_scores = [score for score in scores if score >= 70]
        if len(high_scores) >= 3:
            first_high_idx = next(i for i, score in enumerate(scores) if score >= 70)
            milestones.append({
                'type': 'consistency_achieved',
                'date': entries[first_high_idx]['timestamp'],
                'description': "Começou a manter scores altos consistentemente",
                'significance': 'consistency'
            })
        
        return milestones
    
    def _generate_training_insights(self, entries: List[Dict]) -> Dict[str, Any]:
        """🎯 Gera insights sobre padrões de treinamento"""
        if len(entries) < 3:
            return {'available': False, 'reason': 'Dados insuficientes'}
        
        # Análise temporal dos treinos
        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in entries]
        intervals = [(timestamps[i] - timestamps[i-1]).days for i in range(1, len(timestamps))]
        
        # Análise de movimentos praticados
        movements = [entry['movement'] for entry in entries]
        movement_counts = {}
        for movement in movements:
            movement_counts[movement] = movement_counts.get(movement, 0) + 1
        
        # Análise de profissionais de referência
        professionals = [entry['professional_reference'] for entry in entries]
        pro_counts = {}
        for pro in professionals:
            pro_counts[pro] = pro_counts.get(pro, 0) + 1
        
        return {
            'available': True,
            'training_frequency': {
                'average_interval_days': np.mean(intervals) if intervals else 0,
                'most_frequent_interval': max(set(intervals), key=intervals.count) if intervals else 0,
                'consistency_rating': 1.0 / (1.0 + np.std(intervals)) if intervals else 0
            },
            'movement_focus': {
                'most_practiced': max(movement_counts.items(), key=lambda x: x[1]) if movement_counts else None,
                'movement_distribution': movement_counts,
                'variety_score': len(movement_counts) / max(1, len(entries))
            },
            'professional_preferences': {
                'most_compared': max(pro_counts.items(), key=lambda x: x[1]) if pro_counts else None,
                'professional_distribution': pro_counts
            }
        }
    
    def _suggest_next_goals(self, entries: List[Dict], trend_analysis: Dict) -> List[Dict[str, str]]:
        """🎯 Sugere próximos objetivos baseados no progresso"""
        goals = []
        
        if not entries:
            return [{'type': 'start', 'goal': 'Realize sua primeira análise técnica'}]
        
        current_score = entries[-1]['score']
        
        # Objetivos baseados no score atual
        if current_score < 50:
            goals.append({
                'type': 'improvement',
                'goal': 'Alcançar score de 50% (técnica básica)',
                'timeframe': '2-4 semanas'
            })
        elif current_score < 70:
            goals.append({
                'type': 'improvement', 
                'goal': 'Alcançar score de 70% (técnica boa)',
                'timeframe': '4-8 semanas'
            })
        elif current_score < 85:
            goals.append({
                'type': 'refinement',
                'goal': 'Alcançar score de 85% (técnica avançada)',
                'timeframe': '2-6 meses'
            })
        else:
            goals.append({
                'type': 'mastery',
                'goal': 'Manter consistência acima de 85%',
                'timeframe': 'Contínuo'
            })
        
        # Objetivos baseados em fases específicas
        latest_phases = entries[-1]['phase_scores']
        weakest_phase = min(latest_phases.items(), key=lambda x: x[1])
        
        if weakest_phase[1] < current_score - 10:
            goals.append({
                'type': 'phase_focus',
                'goal': f'Melhorar fase {weakest_phase[0]} para {weakest_phase[1] + 10:.0f}%',
                'timeframe': '2-4 semanas'
            })
        
        # Objetivos baseados em tendências
        if trend_analysis and trend_analysis.get('success'):
            trend_dir = trend_analysis['score_trend']['direction']
            if trend_dir == 'declining':
                goals.append({
                    'type': 'stabilization',
                    'goal': 'Estabilizar performance e reverter declínio',
                    'timeframe': '2-3 semanas'
                })
            elif trend_dir == 'stable':
                goals.append({
                    'type': 'breakthrough',
                    'goal': 'Quebrar platô atual com novas técnicas',
                    'timeframe': '4-6 semanas'
                })
        
        return goals[:3]  # Limitar a 3 objetivos
    
    def _detailed_phase_analysis(self, entries: List[Dict]) -> Dict[str, Any]:
        """📊 Análise detalhada por fase do movimento"""
        phases = ['preparation', 'contact', 'follow_through', 'overall_quality']
        phase_analysis = {}
        
        for phase in phases:
            phase_scores = [entry['phase_scores'].get(phase, 0) for entry in entries]
            
            if phase_scores:
                phase_analysis[phase] = {
                    'current_score': phase_scores[-1],
                    'best_score': max(phase_scores),
                    'average_score': np.mean(phase_scores),
                    'improvement_total': phase_scores[-1] - phase_scores[0] if len(phase_scores) >= 2 else 0,
                    'consistency': 1.0 / (1.0 + np.std(phase_scores)),
                    'score_history': phase_scores
                }
        
        return phase_analysis

def main():
    """🚀 Demonstração do Progression Analyzer"""
    print("📈 PROGRESSION ANALYZER - DEMONSTRAÇÃO")
    print("=" * 50)
    
    analyzer = ProgressionAnalyzer()
    
    # Simular algumas sessões para demonstração
    test_sessions = [
        {
            'video': 'videos/Americo_FD_D_E.mp4',
            'professional': 'Ma_Long',
            'movement': 'FD',
            'notes': 'Primeira sessão de análise'
        },
        # Adicionar mais sessões simuladas aqui
    ]
    
    for i, session in enumerate(test_sessions):
        print(f"\n📝 Simulando sessão {i+1}...")
        
        # Verificar se vídeo existe
        if Path(session['video']).exists():
            result = analyzer.record_session(
                session['video'],
                session['professional'], 
                session['movement'],
                session['notes']
            )
            
            if result.get('success'):
                print(f"✅ Sessão registrada: Score {result['score']:.1f}%")
                print(f"💡 Insight: {result['session_insights'].get('message', 'N/A')}")
            else:
                print(f"❌ Erro: {result.get('error')}")
        else:
            print(f"⚠️ Vídeo não encontrado: {session['video']}")
    
    # Demonstrar análise de tendências
    print(f"\n📊 Análise de tendências...")
    user_id = 'Americo'  # Extraído do nome do vídeo
    
    trends = analyzer.analyze_progression_trends(user_id)
    if trends.get('success'):
        print(f"✅ Tendência: {trends['score_trend']['direction']}")
        print(f"📈 Score atual: {trends['current_performance']['latest_score']:.1f}%")
    else:
        print(f"⚠️ {trends.get('error', 'Erro na análise de tendências')}")
    
    print(f"\n✅ Demonstração concluída!")

if __name__ == "__main__":
    main()
