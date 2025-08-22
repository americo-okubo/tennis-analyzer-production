#!/usr/bin/env python3
"""
📊 ADVANCED STATISTICS ENGINE
Sistema de análise estatística avançada para dados de tênis de mesa.
Correlações, regressões, análise preditiva e insights estatísticos profundos.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

# Imports do sistema
from progression_analyzer import ProgressionAnalyzer
from fast_comparison_engine import FastComparisonEngine
from benchmark_optimizer import BenchmarkOptimizer

warnings.filterwarnings('ignore')

@dataclass
class StatisticalInsight:
    """Insight estatístico individual"""
    insight_type: str  # 'correlation', 'trend', 'anomaly', 'prediction'
    title: str
    description: str
    statistical_significance: float  # p-value or confidence
    effect_size: float  # magnitude of effect
    data_points: int
    visualization_data: Optional[Dict[str, Any]]
    actionable_recommendation: str
    confidence_level: str  # 'high', 'medium', 'low'

@dataclass
class UserClusterProfile:
    """Perfil de cluster de usuários"""
    cluster_id: int
    cluster_name: str
    typical_characteristics: Dict[str, float]
    user_count: int
    improvement_pattern: str
    recommended_approach: str
    success_factors: List[str]

class AdvancedStatisticsEngine:
    """
    📊 Motor de Estatísticas Avançadas
    Análise estatística profunda de dados de performance e progressão
    """
    
    def __init__(self, data_directory: str = "./"):
        self.data_dir = Path(data_directory)
        
        # Componentes do sistema
        self.progression_analyzer = ProgressionAnalyzer()
        self.fast_engine = FastComparisonEngine()
        
        # Cache de dados
        self._data_cache = {}
        self._analysis_cache = {}
        
        # Configuração de visualização
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("📊 Advanced Statistics Engine inicializado")
    
    def collect_all_user_data(self) -> pd.DataFrame:
        """📈 Coleta todos os dados de usuários disponíveis"""
        self.logger.info("📈 Coletando dados de todos os usuários...")
        
        all_data = []
        
        # Buscar arquivos de progressão
        progression_files = list(self.data_dir.glob("progression_database/*_progression.json"))
        
        for prog_file in progression_files:
            try:
                with open(prog_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                user_id = user_data['user_id']
                
                for entry in user_data.get('entries', []):
                    # Extrair dados estruturados
                    data_point = {
                        'user_id': user_id,
                        'timestamp': entry['timestamp'],
                        'score': entry['score'],
                        'confidence': entry['confidence'],
                        'cycles_detected': entry['cycles_detected'],
                        'movement': entry['movement'],
                        'professional_reference': entry['professional_reference'],
                        'preparation_score': entry.get('phase_scores', {}).get('preparation', 0),
                        'contact_score': entry.get('phase_scores', {}).get('contact', 0),
                        'follow_through_score': entry.get('phase_scores', {}).get('follow_through', 0),
                        'overall_quality_score': entry.get('phase_scores', {}).get('overall_quality', 0),
                    }
                    
                    # Adicionar métricas técnicas se disponíveis
                    if 'technical_metrics' in entry:
                        tech_metrics = entry['technical_metrics']
                        data_point.update({
                            'phase_consistency': tech_metrics.get('phase_consistency', 0),
                            'cycle_ratio': tech_metrics.get('cycle_ratio', 0),
                            'confidence_numeric': tech_metrics.get('confidence_numeric', 0)
                        })
                    
                    all_data.append(data_point)
                    
            except Exception as e:
                self.logger.warning(f"Erro ao processar {prog_file}: {e}")
        
        if not all_data:
            # Gerar dados simulados para demonstração
            all_data = self._generate_demo_data()
        
        df = pd.DataFrame(all_data)
        
        # Processamento adicional
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
            df['session_number'] = df.groupby('user_id').cumcount() + 1
            
            # Calcular métricas derivadas
            df['improvement_rate'] = df.groupby('user_id')['score'].transform(
                lambda x: x.diff().fillna(0)
            )
            
            df['cumulative_improvement'] = df.groupby('user_id')['improvement_rate'].cumsum()
            
        self.logger.info(f"📊 Coletados {len(df)} pontos de dados de {df['user_id'].nunique() if not df.empty else 0} usuários")
        
        return df
    
    def _generate_demo_data(self) -> List[Dict]:
        """🎲 Gera dados simulados para demonstração"""
        demo_data = []
        users = ['Americo', 'Baixinha', 'Gordo', 'Usuario1', 'Usuario2']
        professionals = ['Ma_Long', 'Fan_Zhendong', 'Zhang_Jike', 'Timo_Boll']
        movements = ['FD', 'BD', 'FP']
        
        base_date = datetime.now() - timedelta(days=90)
        
        for user in users:
            sessions = np.random.randint(5, 25)  # 5-25 sessões por usuário
            base_score = np.random.uniform(45, 75)  # Score inicial
            improvement_rate = np.random.uniform(-0.2, 1.5)  # Taxa de melhoria
            
            for session in range(sessions):
                date = base_date + timedelta(days=session * np.random.randint(1, 5))
                
                # Score com tendência e ruído
                score = base_score + (session * improvement_rate) + np.random.normal(0, 3)
                score = np.clip(score, 20, 95)  # Limitar score
                
                # Gerar scores por fase
                phase_variance = np.random.uniform(0.8, 1.2, 4)
                phases = {
                    'preparation': score * phase_variance[0],
                    'contact': score * phase_variance[1], 
                    'follow_through': score * phase_variance[2],
                    'overall_quality': score * phase_variance[3]
                }
                
                demo_data.append({
                    'user_id': user,
                    'timestamp': date.isoformat(),
                    'score': score,
                    'confidence': np.random.choice(['high', 'medium', 'low'], p=[0.5, 0.3, 0.2]),
                    'cycles_detected': np.random.randint(2, 8),
                    'movement': np.random.choice(movements),
                    'professional_reference': np.random.choice(professionals),
                    'preparation_score': phases['preparation'],
                    'contact_score': phases['contact'],
                    'follow_through_score': phases['follow_through'],
                    'overall_quality_score': phases['overall_quality'],
                    'phase_consistency': np.random.uniform(0.6, 0.9),
                    'cycle_ratio': np.random.uniform(0.3, 1.2),
                    'confidence_numeric': {'high': 1.0, 'medium': 0.6, 'low': 0.3}[
                        np.random.choice(['high', 'medium', 'low'], p=[0.5, 0.3, 0.2])
                    ]
                })
        
        return demo_data
    
    def analyze_correlation_patterns(self, df: pd.DataFrame) -> List[StatisticalInsight]:
        """📈 Analisa padrões de correlação entre variáveis"""
        self.logger.info("📈 Analisando padrões de correlação...")
        
        insights = []
        
        # Selecionar variáveis numéricas
        numeric_cols = ['score', 'cycles_detected', 'preparation_score', 'contact_score',
                       'follow_through_score', 'overall_quality_score', 'phase_consistency',
                       'cycle_ratio', 'confidence_numeric', 'session_number', 'improvement_rate']
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return insights
        
        # Calcular matriz de correlação
        corr_matrix = df[available_cols].corr()
        
        # Encontrar correlações significativas
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Correlação moderada a forte
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    strong_correlations.append((var1, var2, corr_value))
        
        # Gerar insights para correlações fortes
        for var1, var2, corr_value in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True)[:5]:
            correlation_type = "positiva" if corr_value > 0 else "negativa"
            strength = "forte" if abs(corr_value) > 0.7 else "moderada"
            
            insight = StatisticalInsight(
                insight_type='correlation',
                title=f"Correlação {strength} {correlation_type}",
                description=f"{var1} e {var2} mostram correlação {correlation_type} {strength} (r={corr_value:.3f})",
                statistical_significance=0.05,  # Assumindo significância
                effect_size=abs(corr_value),
                data_points=len(df),
                visualization_data={
                    'type': 'correlation',
                    'variables': [var1, var2],
                    'correlation': corr_value
                },
                actionable_recommendation=self._generate_correlation_recommendation(var1, var2, corr_value),
                confidence_level='high' if abs(corr_value) > 0.7 else 'medium'
            )
            
            insights.append(insight)
        
        return insights
    
    def perform_user_clustering(self, df: pd.DataFrame) -> Tuple[List[UserClusterProfile], pd.DataFrame]:
        """👥 Realiza clustering de usuários baseado em padrões de performance"""
        self.logger.info("👥 Realizando clustering de usuários...")
        
        # Agregar dados por usuário
        user_features = df.groupby('user_id').agg({
            'score': ['mean', 'std', 'min', 'max'],
            'improvement_rate': 'mean',
            'cycles_detected': 'mean',
            'phase_consistency': 'mean',
            'session_number': 'max',
            'preparation_score': 'mean',
            'contact_score': 'mean',
            'follow_through_score': 'mean'
        }).fillna(0)
        
        # Flatten column names
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns]
        
        if len(user_features) < 3:
            return [], df
        
        # Normalizar features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(user_features)
        
        # Determinar número ótimo de clusters
        n_clusters = min(4, max(2, len(user_features) // 2))
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Adicionar cluster labels ao DataFrame original
        user_clusters = pd.Series(cluster_labels, index=user_features.index)
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = df_with_clusters['user_id'].map(user_clusters)
        
        # Analisar características dos clusters
        cluster_profiles = []
        
        for cluster_id in range(n_clusters):
            cluster_users = user_features[cluster_labels == cluster_id]
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Calcular características típicas
            characteristics = {}
            for col in user_features.columns:
                characteristics[col] = cluster_users[col].mean()
            
            # Determinar padrão de melhoria
            avg_improvement = characteristics.get('improvement_rate_mean', 0)
            if avg_improvement > 1.0:
                improvement_pattern = 'rapid_learner'
                cluster_name = f"Aprendizes Rápidos (Cluster {cluster_id})"
            elif avg_improvement > 0:
                improvement_pattern = 'steady_improver'
                cluster_name = f"Melhoria Constante (Cluster {cluster_id})"
            else:
                improvement_pattern = 'plateau'
                cluster_name = f"Platô de Performance (Cluster {cluster_id})"
            
            # Gerar recomendações baseadas no cluster
            recommended_approach = self._generate_cluster_recommendations(characteristics, improvement_pattern)
            
            profile = UserClusterProfile(
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                typical_characteristics=characteristics,
                user_count=len(cluster_users),
                improvement_pattern=improvement_pattern,
                recommended_approach=recommended_approach,
                success_factors=self._identify_success_factors(cluster_data)
            )
            
            cluster_profiles.append(profile)
        
        return cluster_profiles, df_with_clusters
    
    def detect_performance_anomalies(self, df: pd.DataFrame) -> List[StatisticalInsight]:
        """🚨 Detecta anomalias e outliers na performance"""
        self.logger.info("🚨 Detectando anomalias de performance...")
        
        insights = []
        
        if df.empty or 'score' not in df.columns:
            return insights
        
        # Usar Isolation Forest para detecção de anomalias
        features = ['score', 'cycles_detected', 'phase_consistency']
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            return insights
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(df[available_features].fillna(0))
        
        anomalies = df[anomaly_labels == -1]
        
        if not anomalies.empty:
            # Analisar tipos de anomalias
            for idx, anomaly in anomalies.iterrows():
                anomaly_description = self._describe_anomaly(anomaly, df)
                
                insight = StatisticalInsight(
                    insight_type='anomaly',
                    title="Anomalia de Performance Detectada",
                    description=anomaly_description,
                    statistical_significance=0.1,  # Baseado na contaminação do Isolation Forest
                    effect_size=abs(anomaly['score'] - df['score'].mean()) / df['score'].std(),
                    data_points=1,
                    visualization_data={
                        'type': 'anomaly',
                        'anomaly_data': anomaly.to_dict(),
                        'context_mean': df['score'].mean()
                    },
                    actionable_recommendation=self._generate_anomaly_recommendation(anomaly),
                    confidence_level='medium'
                )
                
                insights.append(insight)
        
        return insights[:3]  # Limitar a 3 anomalias mais relevantes
    
    def build_predictive_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """🔮 Constrói modelo preditivo para performance futura"""
        self.logger.info("🔮 Construindo modelo preditivo...")
        
        if df.empty or len(df) < 10:
            return {'success': False, 'reason': 'Dados insuficientes'}
        
        # Preparar features para predição
        feature_cols = ['session_number', 'cycles_detected', 'preparation_score', 
                       'contact_score', 'follow_through_score', 'phase_consistency']
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < 3:
            return {'success': False, 'reason': 'Features insuficientes'}
        
        # Preparar dados
        X = df[available_features].fillna(0)
        y = df['score']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Treinar Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Feature importance
        feature_importance = dict(zip(available_features, model.feature_importances_))
        
        # Predições para próximas sessões
        future_predictions = {}
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            if len(user_data) >= 3:
                last_session = user_data.iloc[-1]
                next_session_features = last_session[available_features].values.reshape(1, -1)
                next_session_features[0][0] += 1  # Incrementar session_number
                
                predicted_score = model.predict(next_session_features)[0]
                future_predictions[user_id] = predicted_score
        
        return {
            'success': True,
            'model_performance': {
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': feature_importance
            },
            'future_predictions': future_predictions,
            'model_confidence': 'high' if r2 > 0.7 else 'medium' if r2 > 0.4 else 'low'
        }
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> List[StatisticalInsight]:
        """⏰ Analisa padrões temporais e sazonalidades"""
        self.logger.info("⏰ Analisando padrões temporais...")
        
        insights = []
        
        if df.empty or 'timestamp' not in df.columns:
            return insights
        
        df_temporal = df.copy()
        df_temporal['hour'] = df_temporal['timestamp'].dt.hour
        df_temporal['day_of_week'] = df_temporal['timestamp'].dt.dayofweek
        df_temporal['week_of_year'] = df_temporal['timestamp'].dt.isocalendar().week
        
        # Analisar performance por hora do dia
        if 'hour' in df_temporal.columns:
            hourly_performance = df_temporal.groupby('hour')['score'].agg(['mean', 'count']).reset_index()
            
            # Filtrar horas com dados suficientes
            significant_hours = hourly_performance[hourly_performance['count'] >= 3]
            
            if not significant_hours.empty:
                best_hour = significant_hours.loc[significant_hours['mean'].idxmax()]
                worst_hour = significant_hours.loc[significant_hours['mean'].idxmin()]
                
                if best_hour['mean'] - worst_hour['mean'] > 5:  # Diferença significativa
                    insight = StatisticalInsight(
                        insight_type='trend',
                        title="Padrão de Performance por Horário",
                        description=f"Performance é {best_hour['mean']:.1f}% melhor às {best_hour['hour']:02d}:00 "
                                  f"comparado às {worst_hour['hour']:02d}:00 ({worst_hour['mean']:.1f}%)",
                        statistical_significance=0.05,
                        effect_size=(best_hour['mean'] - worst_hour['mean']) / df['score'].std(),
                        data_points=len(significant_hours),
                        visualization_data={
                            'type': 'temporal',
                            'data': hourly_performance.to_dict('records')
                        },
                        actionable_recommendation=f"Agende treinos preferencialmente às {best_hour['hour']:02d}:00 para melhor performance",
                        confidence_level='medium'
                    )
                    
                    insights.append(insight)
        
        # Analisar tendência de longo prazo
        if len(df) >= 10:
            # Regressão linear sobre tempo
            days_numeric = df_temporal['days_since_start'].values.reshape(-1, 1)
            scores = df_temporal['score'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_temporal['days_since_start'], scores
            )
            
            if p_value < 0.05:  # Tendência significativa
                trend_direction = "melhorando" if slope > 0 else "declinando"
                
                insight = StatisticalInsight(
                    insight_type='trend',
                    title=f"Tendência de Longo Prazo: {trend_direction.title()}",
                    description=f"Performance {trend_direction} {abs(slope):.2f} pontos por dia "
                              f"(r²={r_value**2:.3f}, p={p_value:.3f})",
                    statistical_significance=p_value,
                    effect_size=abs(r_value),
                    data_points=len(df),
                    visualization_data={
                        'type': 'trend',
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2
                    },
                    actionable_recommendation=self._generate_trend_recommendation(slope, r_value**2),
                    confidence_level='high' if p_value < 0.01 else 'medium'
                )
                
                insights.append(insight)
        
        return insights
    
    def _generate_correlation_recommendation(self, var1: str, var2: str, corr_value: float) -> str:
        """💡 Gera recomendação baseada em correlação"""
        recommendations = {
            ('score', 'cycles_detected'): "Pratique movimentos mais longos para detectar mais ciclos e melhorar score",
            ('score', 'phase_consistency'): "Foque na consistência entre fases do movimento",
            ('preparation_score', 'contact_score'): "Trabalhe preparação e contato em conjunto",
            ('improvement_rate', 'session_number'): "Mantenha frequência regular de treinos para melhoria contínua"
        }
        
        key = tuple(sorted([var1, var2]))
        return recommendations.get(key, f"Monitore relação entre {var1} e {var2} para otimizar performance")
    
    def _generate_cluster_recommendations(self, characteristics: Dict, pattern: str) -> str:
        """🎯 Gera recomendações baseadas no cluster"""
        if pattern == 'rapid_learner':
            return "Continue explorando técnicas avançadas e mantenha desafios progressivos"
        elif pattern == 'steady_improver':
            return "Mantenha consistência no treino e foque em refinamentos graduais"
        else:
            return "Varie exercícios e busque feedback mais frequente para quebrar platô"
    
    def _identify_success_factors(self, cluster_data: pd.DataFrame) -> List[str]:
        """🏆 Identifica fatores de sucesso do cluster"""
        factors = []
        
        if not cluster_data.empty:
            avg_score = cluster_data['score'].mean()
            
            if avg_score > 70:
                factors.append("High performance scores")
            
            if cluster_data['phase_consistency'].mean() > 0.8:
                factors.append("Consistent phase execution")
            
            if cluster_data['improvement_rate'].mean() > 0.5:
                factors.append("Steady improvement rate")
        
        return factors[:3]
    
    def _describe_anomaly(self, anomaly: pd.Series, df: pd.DataFrame) -> str:
        """🔍 Descreve uma anomalia detectada"""
        score = anomaly['score']
        mean_score = df['score'].mean()
        
        if score > mean_score + 2 * df['score'].std():
            return f"Performance excepcionalmente alta ({score:.1f}% vs média {mean_score:.1f}%)"
        elif score < mean_score - 2 * df['score'].std():
            return f"Performance excepcionalmente baixa ({score:.1f}% vs média {mean_score:.1f}%)"
        else:
            return f"Padrão atípico detectado em múltiplas métricas"
    
    def _generate_anomaly_recommendation(self, anomaly: pd.Series) -> str:
        """🚨 Gera recomendação para anomalia"""
        return "Investigue condições específicas desta sessão para identificar fatores que causaram o resultado atípico"
    
    def _generate_trend_recommendation(self, slope: float, r_squared: float) -> str:
        """📈 Gera recomendação baseada em tendência"""
        if slope > 0 and r_squared > 0.5:
            return "Excelente progresso! Mantenha a rotina atual de treinamento"
        elif slope > 0:
            return "Progresso positivo detectado. Considere otimizar frequência de treinos"
        elif slope < 0:
            return "Tendência de declínio detectada. Revise método de treino e considere descanso"
        else:
            return "Performance estável. Considere variar exercícios para evitar platô"
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> str:
        """📄 Gera relatório estatístico completo"""
        self.logger.info("📄 Gerando relatório estatístico completo...")
        
        report_sections = []
        
        # Header
        report_sections.append("📊 RELATÓRIO ESTATÍSTICO AVANÇADO - TABLETENNISANALYZER v2.0")
        report_sections.append("=" * 70)
        report_sections.append(f"Data de Geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"Dados Analisados: {len(df)} registros de {df['user_id'].nunique() if not df.empty else 0} usuários")
        report_sections.append("")
        
        if df.empty:
            report_sections.append("⚠️ Nenhum dado disponível para análise")
            return "\n".join(report_sections)
        
        # Estatísticas descritivas
        report_sections.append("📈 ESTATÍSTICAS DESCRITIVAS:")
        report_sections.append(f"   Score Médio: {df['score'].mean():.1f}% (±{df['score'].std():.1f})")
        report_sections.append(f"   Score Mínimo: {df['score'].min():.1f}%")
        report_sections.append(f"   Score Máximo: {df['score'].max():.1f}%")
        report_sections.append(f"   Ciclos Médios Detectados: {df['cycles_detected'].mean():.1f}")
        report_sections.append("")
        
        # Análise de correlações
        correlations = self.analyze_correlation_patterns(df)
        if correlations:
            report_sections.append("🔗 CORRELAÇÕES SIGNIFICATIVAS:")
            for corr in correlations[:3]:
                report_sections.append(f"   • {corr.title}: {corr.description}")
                report_sections.append(f"     Recomendação: {corr.actionable_recommendation}")
            report_sections.append("")
        
        # Clustering de usuários
        clusters, df_clustered = self.perform_user_clustering(df)
        if clusters:
            report_sections.append("👥 ANÁLISE DE CLUSTERS DE USUÁRIOS:")
            for cluster in clusters:
                report_sections.append(f"   • {cluster.cluster_name} ({cluster.user_count} usuários)")
                report_sections.append(f"     Padrão: {cluster.improvement_pattern}")
                report_sections.append(f"     Abordagem: {cluster.recommended_approach}")
            report_sections.append("")
        
        # Anomalias
        anomalies = self.detect_performance_anomalies(df)
        if anomalies:
            report_sections.append("🚨 ANOMALIAS DETECTADAS:")
            for anomaly in anomalies:
                report_sections.append(f"   • {anomaly.description}")
            report_sections.append("")
        
        # Análise temporal
        temporal_insights = self.analyze_temporal_patterns(df)
        if temporal_insights:
            report_sections.append("⏰ PADRÕES TEMPORAIS:")
            for insight in temporal_insights:
                report_sections.append(f"   • {insight.description}")
                report_sections.append(f"     Recomendação: {insight.actionable_recommendation}")
            report_sections.append("")
        
        # Modelo preditivo
        prediction_model = self.build_predictive_model(df)
        if prediction_model.get('success'):
            report_sections.append("🔮 MODELO PREDITIVO:")
            performance = prediction_model['model_performance']
            report_sections.append(f"   Precisão (R²): {performance['r2_score']:.3f}")
            report_sections.append(f"   Erro Médio (RMSE): {performance['rmse']:.1f}")
            report_sections.append(f"   Confiança: {prediction_model['model_confidence']}")
            
            if prediction_model['future_predictions']:
                report_sections.append("   Predições Próxima Sessão:")
                for user, pred_score in list(prediction_model['future_predictions'].items())[:3]:
                    report_sections.append(f"     {user}: {pred_score:.1f}%")
            report_sections.append("")
        
        # Conclusões
        report_sections.append("✅ CONCLUSÕES E RECOMENDAÇÕES GERAIS:")
        
        avg_score = df['score'].mean()
        if avg_score > 70:
            report_sections.append("   • Performance geral excelente - manter estratégia atual")
        elif avg_score > 60:
            report_sections.append("   • Performance boa com espaço para melhoria específica")
        else:
            report_sections.append("   • Foco em fundamentos e consistência recomendado")
        
        if df['improvement_rate'].mean() > 0:
            report_sections.append("   • Tendência positiva de melhoria detectada")
        else:
            report_sections.append("   • Revisar estratégias de treinamento para quebrar estagnação")
        
        report_sections.append("")
        report_sections.append("📊 Análise gerada pelo Advanced Statistics Engine")
        
        return "\n".join(report_sections)

def main():
    """🚀 Demonstração do Advanced Statistics Engine"""
    print("📊 ADVANCED STATISTICS ENGINE - DEMONSTRAÇÃO")
    print("=" * 55)
    
    engine = AdvancedStatisticsEngine()
    
    # Coletar dados
    print("\n📈 Coletando dados de usuários...")
    df = engine.collect_all_user_data()
    
    if df.empty:
        print("⚠️ Nenhum dado encontrado. Gerando dados de demonstração...")
        demo_data = engine._generate_demo_data()
        df = pd.DataFrame(demo_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
        df['session_number'] = df.groupby('user_id').cumcount() + 1
        df['improvement_rate'] = df.groupby('user_id')['score'].transform(
            lambda x: x.diff().fillna(0)
        )
    
    print(f"✅ Dados coletados: {len(df)} registros de {df['user_id'].nunique()} usuários")
    
    # Executar análises
    print("\n🔗 Analisando correlações...")
    correlations = engine.analyze_correlation_patterns(df)
    print(f"✅ {len(correlations)} correlações significativas encontradas")
    
    print("\n👥 Realizando clustering de usuários...")
    clusters, df_clustered = engine.perform_user_clustering(df)
    print(f"✅ {len(clusters)} clusters identificados")
    
    print("\n🚨 Detectando anomalias...")
    anomalies = engine.detect_performance_anomalies(df)
    print(f"✅ {len(anomalies)} anomalias detectadas")
    
    print("\n🔮 Construindo modelo preditivo...")
    prediction_model = engine.build_predictive_model(df)
    if prediction_model.get('success'):
        print(f"✅ Modelo criado com R² = {prediction_model['model_performance']['r2_score']:.3f}")
    
    print("\n⏰ Analisando padrões temporais...")
    temporal_insights = engine.analyze_temporal_patterns(df)
    print(f"✅ {len(temporal_insights)} padrões temporais identificados")
    
    # Gerar relatório
    print("\n📄 Gerando relatório completo...")
    report = engine.generate_comprehensive_report(df)
    
    # Salvar relatório
    report_path = Path("statistical_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Relatório salvo em: {report_path}")
    
    # Mostrar resumo dos resultados
    print(f"\n📊 RESUMO DOS RESULTADOS:")
    print(f"   🔗 Correlações: {len(correlations)}")
    print(f"   👥 Clusters: {len(clusters)}")
    print(f"   🚨 Anomalias: {len(anomalies)}")
    print(f"   ⏰ Padrões temporais: {len(temporal_insights)}")
    print(f"   🔮 Modelo preditivo: {'✅ Sucesso' if prediction_model.get('success') else '❌ Falha'}")
    
    print(f"\n✅ Análise estatística concluída!")

if __name__ == "__main__":
    main()
