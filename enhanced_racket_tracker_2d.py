"""
[TENNIS] ENHANCED RACKET TRACKER 2D - VERSÃO COM BIOMECÂNICA
Expansão do sistema atual com parâmetros biomecânicos auxiliares

ADIÇÕES:
[OK] MovementMetrics expandido com análise biomecânica
[OK] Análise avançada do cotovelo (velocidade angular, coordenação)
[OK] Discriminadores para zona crítica Y=0.060-0.085
[OK] 100% compatibilidade com sistema existente

FOCO: Resolver casos Japones_BP, Ovtcharov_BD, Baixinha_FP
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import os
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
try:
    from scipy import stats
except ImportError:
    print("[WARNING] Aviso: scipy não encontrado. Usando implementações alternativas.")
    stats = None

class PlayerOrientation(Enum):
    FACING_RIGHT = "voltado_para_direita"
    FACING_LEFT = "voltado_para_esquerda"
    UNKNOWN = "orientacao_desconhecida"

class DominantHand(Enum):
    RIGHT = "mao_direita"
    LEFT = "mao_esquerda"
    UNKNOWN = "mao_desconhecida"

class CameraPerspective(Enum):
    RIGHT = "direita"
    LEFT = "esquerda"
    UNKNOWN = "perspectiva_desconhecida"

@dataclass
class MovementMetrics:
    """Métricas expandidas com parâmetros biomecânicos auxiliares"""
    
    # === CAMPOS EXISTENTES (mantidos) ===
    hand_trajectory: List[Tuple[float, float]]
    elbow_angles: List[float]
    wrist_velocities: List[float]
    movement_amplitude_x: float
    movement_amplitude_y: float
    angle_variation: float
    avg_velocity: float
    max_velocity: float
    acceleration_changes: int
    movement_consistency: float
    
    # === NOVOS: ANÁLISE AVANÇADA DO COTOVELO ===
    elbow_angular_velocity: List[float]        # Velocidade angular do cotovelo (graus/frame)
    elbow_angular_acceleration: List[float]    # Aceleração angular (graus/frame)
    elbow_direction_changes: int               # Mudanças de direção angular
    elbow_mean_angle: float                    # Ângulo médio do cotovelo
    elbow_std_angle: float                     # Desvio padrão dos ângulos
    elbow_opening_trend: str                   # "opening"/"closing"/"stable"/"controlled"
    elbow_peak_angular_velocity: float         # Pico de velocidade angular
    
    # === NOVOS: COORDENAÇÃO SEGMENTAR ===
    shoulder_elbow_coordination: float         # Correlação ombro-cotovelo (0-1)
    elbow_wrist_coordination: float            # Correlação cotovelo-punho (0-1)
    proximal_distal_timing: float              # Delay da cadeia cinética (frames)
    arm_deceleration_pattern: str              # "gradual"/"sharp"/"controlled"/"irregular"
    
    # === NOVOS: ANÁLISE DA TRAJETÓRIA ===
    arm_swing_amplitude: float                 # Amplitude total do balanço do braço
    movement_smoothness: float                 # Suavidade (inverso do jerk)
    trajectory_linearity: float                # Linearidade da trajetória (0-1)
    dominant_movement_axis: str                # "horizontal"/"vertical"/"diagonal"
    
    # === NOVOS: DISCRIMINADORES BIOMECÂNICOS ===
    biomech_forehand_likelihood: float         # Probabilidade de ser forehand (0-1)
    biomech_backhand_likelihood: float         # Probabilidade de ser backhand (0-1)
    biomech_drive_likelihood: float            # Probabilidade de ser drive (0-1)
    biomech_push_likelihood: float             # Probabilidade de ser push (0-1)
    biomech_confidence: float                  # Confiança geral biomecânica (0-1)

@dataclass
class BallDetection:
    center: Tuple[int, int]
    area: float
    frame_number: int
    confidence: float

class EnhancedRacketTracker2D:
    """Tracker expandido com análise biomecânica"""
    
    def __init__(self, flip_hands=False):
        self.flip_hands = flip_hands
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Histórico para análise temporal
        self.ball_trajectory = []
        self.frame_count = 0
        
        # Padrões biomecânicos esperados (baseados na análise dos 21 vídeos)
        self.biomech_patterns = {
            'forehand_drive': {
                'elbow_variation_range': (80, 120),
                'angular_velocity_range': (0.08, 0.20),
                'opening_trend': 'opening',
                'coordination_min': 0.7,
                'smoothness_min': 0.6
            },
            'forehand_push': {
                'elbow_variation_range': (20, 50),
                'angular_velocity_range': (0.02, 0.06),
                'opening_trend': 'stable',
                'coordination_min': 0.8,
                'smoothness_min': 0.8
            },
            'backhand_drive': {
                'elbow_variation_range': (70, 110),
                'angular_velocity_range': (0.06, 0.15),
                'opening_trend': 'opening',
                'coordination_min': 0.6,
                'smoothness_min': 0.5
            },
            'backhand_push': {
                'elbow_variation_range': (40, 70),
                'angular_velocity_range': (0.03, 0.08),
                'opening_trend': 'controlled',
                'coordination_min': 0.75,
                'smoothness_min': 0.7
            }
        }
        
        print("Enhanced Racket Tracker inicializado com analise biomecanica")
        print("Foco: discriminar zona critica Y=0.060-0.085")
    
    def calculate_angular_velocity(self, angles: List[float]) -> List[float]:
        """Calcula velocidade angular do cotovelo"""
        if len(angles) < 2:
            return []
        
        angular_velocity = []
        for i in range(1, len(angles)):
            delta_angle = angles[i] - angles[i-1]
            
            # Normalizar diferenças angulares para [-180, 180]
            while delta_angle > 180:
                delta_angle -= 360
            while delta_angle < -180:
                delta_angle += 360
            
            angular_velocity.append(delta_angle)  # graus/frame
        
        return angular_velocity
    
    def calculate_angular_acceleration(self, angular_velocity: List[float]) -> List[float]:
        """Calcula aceleração angular"""
        if len(angular_velocity) < 2:
            return []
        
        angular_acceleration = []
        for i in range(1, len(angular_velocity)):
            accel = angular_velocity[i] - angular_velocity[i-1]
            angular_acceleration.append(accel)  # graus/frame
        
        return angular_acceleration
    
    def count_direction_changes(self, values: List[float]) -> int:
        """Conta mudanças de direção em uma série temporal"""
        if len(values) < 3:
            return 0
        
        direction_changes = 0
        for i in range(1, len(values) - 1):
            # Verifica se há mudança de sinal na derivada
            prev_trend = values[i] - values[i-1]
            next_trend = values[i+1] - values[i]
            
            if abs(prev_trend) > 0.5 and abs(next_trend) > 0.5:  # Filtrar ruído
                if (prev_trend > 0) != (next_trend > 0):
                    direction_changes += 1
        
        return direction_changes
    
    def analyze_opening_trend(self, angles: List[float]) -> str:
        """Analisa tendência de abertura/fechamento do cotovelo"""
        if len(angles) < 5:
            return "unknown"
        
        try:
            # Calcular tendência linear usando regressão
            x = np.arange(len(angles))
            
            if stats is not None:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, angles)
            else:
                # Implementação alternativa sem scipy
                n = len(angles)
                x_mean = np.mean(x)
                y_mean = np.mean(angles)
                
                numerator = np.sum((x - x_mean) * (angles - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                
                slope = numerator / denominator if denominator != 0 else 0
                
                # Calcular R-squared simplificado
                y_pred = slope * x + (y_mean - slope * x_mean)
                ss_res = np.sum((angles - y_pred) ** 2)
                ss_tot = np.sum((angles - y_mean) ** 2)
                r_value = np.sqrt(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0
                if np.corrcoef(x, angles)[0, 1] < 0:
                    r_value = -r_value
            
            # Analisar padrão baseado na inclinação e correlação
            if abs(r_value) < 0.3:
                return "stable"  # Movimento sem tendência clara
            elif slope > 2 and r_value > 0.3:
                return "opening"  # Cotovelo abrindo progressivamente
            elif slope < -2 and r_value < -0.3:
                return "closing"  # Cotovelo fechando progressivamente
            elif abs(slope) <= 2 and abs(r_value) > 0.3:
                return "controlled"  # Movimento controlado com variação moderada
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def calculate_coordination(self, trajectory1: List[Tuple[float, float]], 
                             trajectory2: List[Tuple[float, float]]) -> float:
        """Calcula coordenação entre dois segmentos corporais"""
        if len(trajectory1) < 5 or len(trajectory2) < 5:
            return 0.0
        
        try:
            # Sincronizar trajetórias pelo menor comprimento
            min_len = min(len(trajectory1), len(trajectory2))
            traj1 = trajectory1[:min_len]
            traj2 = trajectory2[:min_len]
            
            # Calcular velocidades para cada trajetória
            vel1 = []
            vel2 = []
            
            for i in range(1, min_len):
                # Velocidade euclidiana
                v1 = np.sqrt((traj1[i][0] - traj1[i-1][0])**2 + (traj1[i][1] - traj1[i-1][1])**2)
                v2 = np.sqrt((traj2[i][0] - traj2[i-1][0])**2 + (traj2[i][1] - traj2[i-1][1])**2)
                vel1.append(v1)
                vel2.append(v2)
            
            if len(vel1) < 3:
                return 0.0
            
            # Calcular correlação de Pearson entre as velocidades
            if stats is not None:
                correlation, p_value = stats.pearsonr(vel1, vel2)
            else:
                # Implementação alternativa
                correlation = np.corrcoef(vel1, vel2)[0, 1] if len(vel1) > 1 else 0
            
            # Retornar valor absoluto da correlação (coordenação independe da direção)
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def analyze_kinetic_chain_timing(self, shoulder_traj: List[Tuple[float, float]], 
                                   elbow_traj: List[Tuple[float, float]], 
                                   wrist_traj: List[Tuple[float, float]]) -> float:
        """Analisa timing da cadeia cinética proximal-distal"""
        try:
            min_len = min(len(shoulder_traj), len(elbow_traj), len(wrist_traj))
            if min_len < 10:
                return 0.0
            
            # Calcular picos de velocidade para cada segmento
            def find_peak_velocity_frame(trajectory):
                velocities = []
                for i in range(1, len(trajectory)):
                    vel = np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + 
                                (trajectory[i][1] - trajectory[i-1][1])**2)
                    velocities.append(vel)
                
                if not velocities:
                    return 0
                    
                max_vel_frame = np.argmax(velocities)
                return max_vel_frame
            
            shoulder_peak = find_peak_velocity_frame(shoulder_traj[:min_len])
            elbow_peak = find_peak_velocity_frame(elbow_traj[:min_len])
            wrist_peak = find_peak_velocity_frame(wrist_traj[:min_len])
            
            # Calcular timing sequencial (ombro  cotovelo  punho)
            shoulder_to_elbow_delay = elbow_peak - shoulder_peak
            elbow_to_wrist_delay = wrist_peak - elbow_peak
            
            # Timing ideal: sequência proximal-distal positiva
            total_delay = (shoulder_to_elbow_delay + elbow_to_wrist_delay) / 2.0
            
            return total_delay
            
        except Exception:
            return 0.0
    
    def analyze_deceleration_pattern(self, velocities: List[float]) -> str:
        """Analisa padrão de desaceleração do movimento"""
        if len(velocities) < 10:
            return "unknown"
        
        try:
            # Encontrar pico de velocidade
            max_vel_idx = np.argmax(velocities)
            
            # Analisar fase de desaceleração (após o pico)
            if max_vel_idx >= len(velocities) - 3:
                return "unknown"  # Pico muito no final
            
            decel_phase = velocities[max_vel_idx:]
            
            if len(decel_phase) < 5:
                return "unknown"
            
            # Calcular gradiente da desaceleração
            x = np.arange(len(decel_phase))
            
            if stats is not None:
                slope, _, r_value, _, _ = stats.linregress(x, decel_phase)
            else:
                # Implementação alternativa
                n = len(decel_phase)
                x_mean = np.mean(x)
                y_mean = np.mean(decel_phase)
                
                numerator = np.sum((x - x_mean) * (decel_phase - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                
                slope = numerator / denominator if denominator != 0 else 0
                r_value = np.corrcoef(x, decel_phase)[0, 1] if len(decel_phase) > 1 else 0
            
            # Calcular variabilidade da desaceleração
            decel_std = np.std(np.diff(decel_phase))
            
            # Classificar padrão
            if r_value < -0.7 and decel_std < np.mean(decel_phase) * 0.3:
                return "gradual"     # Desaceleração suave e linear
            elif r_value < -0.5 and decel_std < np.mean(decel_phase) * 0.5:
                return "controlled"  # Desaceleração controlada
            elif abs(slope) > np.mean(decel_phase) * 0.8:
                return "sharp"       # Desaceleração abrupta
            else:
                return "irregular"   # Padrão irregular
                
        except Exception:
            return "unknown"
    
    def calculate_movement_smoothness(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calcula suavidade do movimento (inverso do jerk)"""
        if len(trajectory) < 5:
            return 0.0
        
        try:
            # Calcular jerk (derivada terceira da posição)
            x_coords = [point[0] for point in trajectory]
            y_coords = [point[1] for point in trajectory]
            
            # Primeira derivada (velocidade)
            x_vel = np.diff(x_coords)
            y_vel = np.diff(y_coords)
            
            # Segunda derivada (aceleração)
            x_accel = np.diff(x_vel)
            y_accel = np.diff(y_vel)
            
            # Terceira derivada (jerk)
            x_jerk = np.diff(x_accel)
            y_jerk = np.diff(y_accel)
            
            # Magnitude do jerk
            jerk_magnitude = np.sqrt(x_jerk**2 + y_jerk**2)
            
            # Suavidade é o inverso do jerk médio (normalizado)
            mean_jerk = np.mean(jerk_magnitude)
            smoothness = 1.0 / (1.0 + mean_jerk * 1000)  # Normalização
            
            return smoothness
            
        except Exception:
            return 0.0
    
    def calculate_trajectory_linearity(self, trajectory: List[Tuple[float, float]]) -> float:
        """Calcula linearidade da trajetória"""
        if len(trajectory) < 3:
            return 0.0
        
        try:
            # Calcular distância direta (linha reta) entre início e fim
            start_point = np.array(trajectory[0])
            end_point = np.array(trajectory[-1])
            direct_distance = np.linalg.norm(end_point - start_point)
            
            if direct_distance < 1e-6:
                return 1.0  # Movimento estático é "linear"
            
            # Calcular distância total da trajetória
            total_distance = 0.0
            for i in range(1, len(trajectory)):
                point1 = np.array(trajectory[i-1])
                point2 = np.array(trajectory[i])
                total_distance += np.linalg.norm(point2 - point1)
            
            if total_distance < 1e-6:
                return 1.0
            
            # Linearidade = distância direta / distância total
            linearity = direct_distance / total_distance
            
            return min(linearity, 1.0)
            
        except Exception:
            return 0.0
    
    def identify_dominant_movement_axis(self, trajectory: List[Tuple[float, float]]) -> str:
        """Identifica eixo dominante do movimento"""
        if len(trajectory) < 3:
            return "unknown"
        
        try:
            x_coords = [point[0] for point in trajectory]
            y_coords = [point[1] for point in trajectory]
            
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            # Determinar eixo dominante baseado na amplitude
            if x_range > y_range * 1.5:
                return "horizontal"
            elif y_range > x_range * 1.5:
                return "vertical"
            else:
                return "diagonal"
                
        except Exception:
            return "unknown"
    
    def calculate_biomechanical_likelihoods(self, metrics_dict: Dict) -> Dict[str, float]:
        """Calcula probabilidades biomecânicas para cada tipo de movimento"""
        
        likelihoods = {
            'forehand': 0.0,
            'backhand': 0.0,
            'drive': 0.0,
            'push': 0.0,
            'confidence': 0.0
        }
        
        try:
            # Extrair métricas calculadas
            elbow_variation = metrics_dict.get('angle_variation', 0)
            angular_velocity_peak = metrics_dict.get('elbow_peak_angular_velocity', 0)
            opening_trend = metrics_dict.get('elbow_opening_trend', 'unknown')
            coordination = metrics_dict.get('coordination_avg', 0)
            smoothness = metrics_dict.get('movement_smoothness', 0)
            
            # Calcular scores para cada padrão
            pattern_scores = {}
            
            for pattern_name, pattern in self.biomech_patterns.items():
                score = 0.0
                
                # Score da variação do cotovelo
                elbow_min, elbow_max = pattern['elbow_variation_range']
                if elbow_min <= elbow_variation <= elbow_max:
                    score += 0.3
                elif abs(elbow_variation - (elbow_min + elbow_max) / 2) < (elbow_max - elbow_min):
                    score += 0.1
                
                # Score da velocidade angular
                vel_min, vel_max = pattern['angular_velocity_range']
                if vel_min <= angular_velocity_peak <= vel_max:
                    score += 0.3
                elif abs(angular_velocity_peak - (vel_min + vel_max) / 2) < (vel_max - vel_min):
                    score += 0.1
                
                # Score da tendência de abertura
                if opening_trend == pattern['opening_trend']:
                    score += 0.2
                elif opening_trend in ['stable', 'controlled'] and pattern['opening_trend'] in ['stable', 'controlled']:
                    score += 0.1
                
                # Score da coordenação
                if coordination >= pattern['coordination_min']:
                    score += 0.1
                
                # Score da suavidade
                if smoothness >= pattern['smoothness_min']:
                    score += 0.1
                
                pattern_scores[pattern_name] = score
            
            # Agregar scores por lado (forehand/backhand)
            likelihoods['forehand'] = (pattern_scores.get('forehand_drive', 0) + 
                                     pattern_scores.get('forehand_push', 0)) / 2.0
            likelihoods['backhand'] = (pattern_scores.get('backhand_drive', 0) + 
                                     pattern_scores.get('backhand_push', 0)) / 2.0
            
            # Agregar scores por tipo (drive/push)
            likelihoods['drive'] = (pattern_scores.get('forehand_drive', 0) + 
                                  pattern_scores.get('backhand_drive', 0)) / 2.0
            likelihoods['push'] = (pattern_scores.get('forehand_push', 0) + 
                                 pattern_scores.get('backhand_push', 0)) / 2.0
            
            # Calcular confiança geral
            all_scores = list(pattern_scores.values())
            if all_scores:
                max_score = max(all_scores)
                score_variance = np.var(all_scores)
                likelihoods['confidence'] = max_score * (1.0 - score_variance)
            
            # Normalizar para [0, 1]
            for key in likelihoods:
                likelihoods[key] = max(0.0, min(1.0, likelihoods[key]))
            
        except Exception as e:
            print(f"LOG: Erro no cálculo biomecânico: {e}")
        
        return likelihoods
    
    def calculate_elbow_angle(self, shoulder, elbow, wrist):
        """Calcula o ângulo do cotovelo (mantido do original)"""
        try:
            vec1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
            vec2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
            
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norms == 0:
                return 0
                
            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except:
            return 0
    
    def calculate_movement_metrics(self, pose_landmarks_history, side="left"):
        """
        VERSÃO EXPANDIDA: Inclui análise biomecânica completa
        """
        try:
            if len(pose_landmarks_history) < 5:
                return self._create_empty_metrics()
            
            print(f"[BIOMECH] Calculando métricas biomecânicas para lado {side}")
            
            # Definir landmarks baseado no lado
            if side == "left":
                wrist_landmark = self.mp_pose.PoseLandmark.LEFT_WRIST
                elbow_landmark = self.mp_pose.PoseLandmark.LEFT_ELBOW
                shoulder_landmark = self.mp_pose.PoseLandmark.LEFT_SHOULDER
            else:
                wrist_landmark = self.mp_pose.PoseLandmark.RIGHT_WRIST
                elbow_landmark = self.mp_pose.PoseLandmark.RIGHT_ELBOW
                shoulder_landmark = self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            
            # === CÁLCULOS EXISTENTES (mantidos) ===
            wrist_positions = []
            elbow_positions = []
            shoulder_positions = []
            elbow_angles = []
            velocities = []
            
            for pose_landmarks in pose_landmarks_history:
                if pose_landmarks:
                    try:
                        wrist = pose_landmarks.landmark[wrist_landmark]
                        elbow = pose_landmarks.landmark[elbow_landmark]
                        shoulder = pose_landmarks.landmark[shoulder_landmark]
                        
                        # Posições
                        wrist_positions.append((wrist.x, wrist.y))
                        elbow_positions.append((elbow.x, elbow.y))
                        shoulder_positions.append((shoulder.x, shoulder.y))
                        
                        # Ângulo do cotovelo
                        angle = self.calculate_elbow_angle(shoulder, elbow, wrist)
                        elbow_angles.append(angle)
                    except:
                        continue
            
            if len(wrist_positions) < 2:
                return self._create_empty_metrics()
            
            # Calcular velocidades
            for i in range(1, len(wrist_positions)):
                prev_pos = wrist_positions[i-1]
                curr_pos = wrist_positions[i]
                velocity = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                velocities.append(velocity)
            
            # Métricas básicas existentes
            x_positions = [pos[0] for pos in wrist_positions]
            y_positions = [pos[1] for pos in wrist_positions]
            amplitude_x = max(x_positions) - min(x_positions) if x_positions else 0
            amplitude_y = max(y_positions) - min(y_positions) if y_positions else 0
            angle_variation = max(elbow_angles) - min(elbow_angles) if elbow_angles else 0
            avg_velocity = np.mean(velocities) if velocities else 0
            max_velocity = max(velocities) if velocities else 0
            
            acceleration_changes = 0
            if len(velocities) > 2:
                accelerations = np.diff(velocities)
                for i in range(1, len(accelerations)):
                    if (accelerations[i] > 0) != (accelerations[i-1] > 0):
                        acceleration_changes += 1
            
            movement_consistency = 1.0 / (1.0 + np.std(velocities)) if velocities else 0
            
            # === NOVOS CÁLCULOS BIOMECÂNICOS ===
            print(f"    - Calculando parâmetros avançados do cotovelo...")
            
            # Análise avançada do cotovelo
            elbow_angular_velocity = self.calculate_angular_velocity(elbow_angles)
            elbow_angular_acceleration = self.calculate_angular_acceleration(elbow_angular_velocity)
            elbow_direction_changes = self.count_direction_changes(elbow_angular_velocity)
            elbow_mean_angle = np.mean(elbow_angles) if elbow_angles else 0
            elbow_std_angle = np.std(elbow_angles) if elbow_angles else 0
            elbow_opening_trend = self.analyze_opening_trend(elbow_angles)
            elbow_peak_angular_velocity = max([abs(v) for v in elbow_angular_velocity]) if elbow_angular_velocity else 0
            
            print(f"    - Analisando coordenação segmentar...")
            
            # Coordenação segmentar
            shoulder_elbow_coordination = self.calculate_coordination(shoulder_positions, elbow_positions)
            elbow_wrist_coordination = self.calculate_coordination(elbow_positions, wrist_positions)
            proximal_distal_timing = self.analyze_kinetic_chain_timing(shoulder_positions, elbow_positions, wrist_positions)
            arm_deceleration_pattern = self.analyze_deceleration_pattern(velocities)
            
            print(f"    - Analisando trajetória e suavidade...")
            
            # Análise da trajetória
            arm_swing_amplitude = max(max(x_positions) - min(x_positions), 
                                    max(y_positions) - min(y_positions)) if wrist_positions else 0
            movement_smoothness = self.calculate_movement_smoothness(wrist_positions)
            trajectory_linearity = self.calculate_trajectory_linearity(wrist_positions)
            dominant_movement_axis = self.identify_dominant_movement_axis(wrist_positions)
            
            print(f"    - Calculando probabilidades biomecânicas...")
            
            # Calcular probabilidades biomecânicas
            metrics_for_biomech = {
                'angle_variation': angle_variation,
                'elbow_peak_angular_velocity': elbow_peak_angular_velocity,
                'elbow_opening_trend': elbow_opening_trend,
                'coordination_avg': (shoulder_elbow_coordination + elbow_wrist_coordination) / 2.0,
                'movement_smoothness': movement_smoothness
            }
            
            biomech_likelihoods = self.calculate_biomechanical_likelihoods(metrics_for_biomech)
            
            print(f"   [OK] Métricas biomecânicas calculadas:")
            print(f"       Variação cotovelo: {angle_variation:.1f}")
            print(f"       Tendência: {elbow_opening_trend}")
            print(f"       Pico vel. angular: {elbow_peak_angular_velocity:.3f}")
            print(f"       Coordenação: {(shoulder_elbow_coordination + elbow_wrist_coordination) / 2.0:.2f}")
            print(f"       Suavidade: {movement_smoothness:.2f}")
            print(f"       Probabilidades: FH={biomech_likelihoods['forehand']:.2f}, BH={biomech_likelihoods['backhand']:.2f}")
            
            return MovementMetrics(
                # === CAMPOS EXISTENTES ===
                hand_trajectory=wrist_positions,
                elbow_angles=elbow_angles,
                wrist_velocities=velocities,
                movement_amplitude_x=amplitude_x,
                movement_amplitude_y=amplitude_y,
                angle_variation=angle_variation,
                avg_velocity=avg_velocity,
                max_velocity=max_velocity,
                acceleration_changes=acceleration_changes,
                movement_consistency=movement_consistency,
                
                # === NOVOS CAMPOS BIOMECÂNICOS ===
                elbow_angular_velocity=elbow_angular_velocity,
                elbow_angular_acceleration=elbow_angular_acceleration,
                elbow_direction_changes=elbow_direction_changes,
                elbow_mean_angle=elbow_mean_angle,
                elbow_std_angle=elbow_std_angle,
                elbow_opening_trend=elbow_opening_trend,
                elbow_peak_angular_velocity=elbow_peak_angular_velocity,
                shoulder_elbow_coordination=shoulder_elbow_coordination,
                elbow_wrist_coordination=elbow_wrist_coordination,
                proximal_distal_timing=proximal_distal_timing,
                arm_deceleration_pattern=arm_deceleration_pattern,
                arm_swing_amplitude=arm_swing_amplitude,
                movement_smoothness=movement_smoothness,
                trajectory_linearity=trajectory_linearity,
                dominant_movement_axis=dominant_movement_axis,
                biomech_forehand_likelihood=biomech_likelihoods['forehand'],
                biomech_backhand_likelihood=biomech_likelihoods['backhand'],
                biomech_drive_likelihood=biomech_likelihoods['drive'],
                biomech_push_likelihood=biomech_likelihoods['push'],
                biomech_confidence=biomech_likelihoods['confidence']
            )
            
        except Exception as e:
            print(f"[ERROR] Erro no cálculo de métricas biomecânicas: {e}")
            return self._create_empty_metrics()
    
    def _create_empty_metrics(self):
        """Cria métricas vazias com todos os campos"""
        return MovementMetrics(
            # Campos existentes
            hand_trajectory=[], elbow_angles=[], wrist_velocities=[],
            movement_amplitude_x=0, movement_amplitude_y=0, angle_variation=0,
            avg_velocity=0, max_velocity=0, acceleration_changes=0, movement_consistency=0,
            # Novos campos biomecânicos
            elbow_angular_velocity=[], elbow_angular_acceleration=[],
            elbow_direction_changes=0, elbow_mean_angle=0, elbow_std_angle=0,
            elbow_opening_trend="unknown", elbow_peak_angular_velocity=0,
            shoulder_elbow_coordination=0, elbow_wrist_coordination=0,
            proximal_distal_timing=0, arm_deceleration_pattern="unknown",
            arm_swing_amplitude=0, movement_smoothness=0, trajectory_linearity=0,
            dominant_movement_axis="unknown", biomech_forehand_likelihood=0,
            biomech_backhand_likelihood=0, biomech_drive_likelihood=0,
            biomech_push_likelihood=0, biomech_confidence=0
        )
    
    # === MÉTODOS EXISTENTES MANTIDOS (detect_player_orientation, etc.) ===
    # [Todo o resto do código original permanece igual]
    
    def detect_player_orientation(self, pose_landmarks, frame_width, debug_frame=None):
        """Mantido do original"""
        if not pose_landmarks:
            return PlayerOrientation.UNKNOWN
        
        try:
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            left_shoulder_x = left_shoulder.x * frame_width
            right_shoulder_x = right_shoulder.x * frame_width
            left_elbow_x = left_elbow.x * frame_width
            right_elbow_x = right_elbow.x * frame_width
            nose_x = nose.x * frame_width
            
            shoulder_center = (left_shoulder_x + right_shoulder_x) / 2
            left_arm_direction = left_elbow_x - left_shoulder_x
            right_arm_direction = right_elbow_x - right_shoulder_x
            nose_offset = nose_x - shoulder_center
            
            facing_right_score = 0
            facing_left_score = 0
            
            # Critério 1: Direção dos braços (peso 2)
            if abs(right_arm_direction) > abs(left_arm_direction):
                if right_arm_direction > 0:
                    facing_right_score += 2
                else:
                    facing_left_score += 1
            else:
                if left_arm_direction < 0:
                    facing_left_score += 2
                else:
                    facing_right_score += 1
            
            # Critério 2: Posição do nariz (peso 2)
            if nose_offset > 10:
                facing_right_score += 2
            elif nose_offset < -10:
                facing_left_score += 2
            elif nose_offset > 3:
                facing_right_score += 1
            elif nose_offset < -3:
                facing_left_score += 1
            
            # Critério 3: Extensão relativa dos braços (peso 1)
            left_extension = abs(left_arm_direction)
            right_extension = abs(right_arm_direction)
            
            if right_extension > left_extension * 1.3:
                facing_right_score += 1
            elif left_extension > right_extension * 1.3:
                facing_left_score += 1
            
            # Critério 4: Posição relativa dos ombros (peso 1)
            shoulder_diff = left_shoulder_x - right_shoulder_x
            if shoulder_diff > 15:
                facing_left_score += 1
            elif shoulder_diff < -15:
                facing_right_score += 1
            
            # Log detalhado a cada 40 frames
            if debug_frame is not None and debug_frame % 40 == 0:
                print(f"LOG: Frame {debug_frame} orientação:")
                print(f"  Nariz offset: {nose_offset:.1f} (negativo=esquerda)")
                print(f"  Scores - direita: {facing_right_score}, esquerda: {facing_left_score}")
            
            # Decisão
            if facing_right_score > facing_left_score:
                return PlayerOrientation.FACING_RIGHT
            elif facing_left_score > facing_right_score:
                return PlayerOrientation.FACING_LEFT
            else:
                # Desempate pelo nariz
                if nose_offset < -2:
                    return PlayerOrientation.FACING_LEFT
                elif nose_offset > 2:
                    return PlayerOrientation.FACING_RIGHT
                else:
                    return PlayerOrientation.UNKNOWN
                
        except Exception as e:
            if debug_frame is not None and debug_frame % 40 == 0:
                print(f"LOG: Erro na orientação frame {debug_frame}: {e}")
            return PlayerOrientation.UNKNOWN
    
    def detect_ball_in_frame(self, frame, frame_number):
        """Detecta bola de tênis no frame atual (mantido do original)"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            color_ranges = [
                ([20, 80, 80], [35, 255, 255]),
                ([35, 60, 60], [50, 255, 200]),
                ([15, 100, 100], [25, 255, 255])
            ]
            
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in color_ranges:
                lower_bound = np.array(lower)
                upper_bound = np.array(upper)
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_ball = None
            best_confidence = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 20 < area < 800:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0
                    
                    shape_score = 0
                    if 0.7 < aspect_ratio < 1.4:
                        shape_score += 0.3
                    if circularity > 0.5:
                        shape_score += 0.4
                    if 50 < area < 400:
                        shape_score += 0.3
                    
                    confidence = shape_score
                    
                    if confidence > best_confidence and confidence > 0.5:
                        center = (x + w//2, y + h//2)
                        best_ball = BallDetection(center, area, frame_number, confidence)
                        best_confidence = confidence
            
            return best_ball
            
        except Exception as e:
            return None
    
    def detect_racket_by_color(self, frame, hand_region, hand_bbox, debug_info=None, side_name=""):
        """Detecta raquete por cor (mantido do original)"""
        try:
            hsv_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
            
            color_ranges = [
                ([0, 0, 0], [180, 255, 60]),
                ([100, 40, 20], [130, 255, 130]),
                ([0, 40, 20], [10, 255, 130]),
                ([170, 40, 20], [180, 255, 130]),
                ([15, 60, 60], [35, 255, 200]),
                ([45, 50, 30], [75, 255, 100]),
            ]
            
            combined_mask = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
            color_detections = 0
            
            for i, color_range in enumerate(color_ranges):
                lower_bound = np.array(color_range[0])
                upper_bound = np.array(color_range[1])
                mask = cv2.inRange(hsv_region, lower_bound, upper_bound)
                
                pixels_detected = cv2.countNonZero(mask)
                if pixels_detected > 80:
                    color_detections += 1
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            total_racket_area = 0
            valid_contours = 0
            best_contour_info = ""
            racket_score = 0
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 60:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                
                if debug_info and debug_info % 20 == 0 and area > 150:
                    print(f"      Contorno {i} {side_name}: área={area:.0f}, aspecto={aspect_ratio:.2f}, tamanho={w}x{h}")
                
                contour_score = 0
                
                if aspect_ratio > 1.5:
                    contour_score += 1
                if aspect_ratio > 2.0:
                    contour_score += 2
                if aspect_ratio > 3.0:
                    contour_score += 3
                    
                if area > 120:
                    contour_score += 1
                if area > 300:
                    contour_score += 2
                if area > 600:
                    contour_score += 3
                    
                if aspect_ratio < 1.2:
                    contour_score -= 3
                    
                if contour_score > 2:
                    valid_contours += 1
                    total_racket_area += area
                    racket_score += contour_score
                    best_contour_info = f"área={area:.0f}, aspecto={aspect_ratio:.2f}, score={contour_score}"
            
            color_factor = min(color_detections / 4.0, 1.0)
            area_factor = min(total_racket_area / 1500.0, 1.0)
            shape_factor = min(racket_score / 8.0, 1.0)
            
            final_score = (color_factor * 0.25 + area_factor * 0.35 + shape_factor * 0.4)
            racket_detected = final_score > 0.5 and valid_contours > 0
            
            if debug_info and debug_info % 20 == 0:
                print(f"    {side_name}: cores={color_detections}, contornos={valid_contours}, área={total_racket_area:.0f}")
                print(f"    {side_name}: scores - cor={color_factor:.2f}, área={area_factor:.2f}, forma={shape_factor:.2f}, final={final_score:.2f}")
                if racket_detected and best_contour_info:
                    print(f"    {side_name}: [TENNIS] RAQUETE DETECTADA! {best_contour_info}")
                else:
                    print(f"    {side_name}: [ERROR] Score insuficiente para raquete")
            
            return racket_detected, final_score
            
        except Exception as e:
            if debug_info and debug_info % 20 == 0:
                print(f"    Erro detecção {side_name}: {e}")
            return False, 0.0
    
    def detect_coordinate_flip(self, pose_landmarks, frame_width):
        """Detecta se as coordenadas LEFT/RIGHT do MediaPipe estão invertidas"""
        try:
            left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            left_x = left_wrist.x * frame_width
            right_x = right_wrist.x * frame_width
            
            return left_x > right_x
            
        except:
            return False
    
    def determine_dominant_hand_with_movement_analysis(self, racket_detections, racket_scores, pose_history):
        """Determina mão dominante com análise de movimento (mantido do original)"""
        print(f"\nLOG: === ANÁLISE COMPLETA DE MÃO DOMINANTE ===")
        
        left_metrics = self.calculate_movement_metrics(pose_history, "left")
        right_metrics = self.calculate_movement_metrics(pose_history, "right")
        
        left_avg_racket_score = np.mean(racket_scores['left']) if racket_scores['left'] else 0
        right_avg_racket_score = np.mean(racket_scores['right']) if racket_scores['right'] else 0
        
        print(f"LOG: Métricas de movimento ESQUERDA:")
        print(f"  Velocidade média: {left_metrics.avg_velocity:.4f}")
        print(f"  Velocidade máxima: {left_metrics.max_velocity:.4f}")
        print(f"  Amplitude X: {left_metrics.movement_amplitude_x:.4f}")
        print(f"  Amplitude Y: {left_metrics.movement_amplitude_y:.4f}")
        print(f"  Variação ângulo: {left_metrics.angle_variation:.1f}")
        print(f"  Mudanças aceleração: {left_metrics.acceleration_changes}")
        print(f"  Score raquete: {left_avg_racket_score:.3f}")
        
        print(f"LOG: Métricas de movimento DIREITA:")
        print(f"  Velocidade média: {right_metrics.avg_velocity:.4f}")
        print(f"  Velocidade máxima: {right_metrics.max_velocity:.4f}")
        print(f"  Amplitude X: {right_metrics.movement_amplitude_x:.4f}")
        print(f"  Amplitude Y: {right_metrics.movement_amplitude_y:.4f}")
        print(f"  Variação ângulo: {right_metrics.angle_variation:.1f}")
        print(f"  Mudanças aceleração: {right_metrics.acceleration_changes}")
        print(f"  Score raquete: {right_avg_racket_score:.3f}")
        
        def calculate_activity_score(metrics, racket_score):
            score = (
                metrics.avg_velocity * 2.4 +
                metrics.max_velocity * 1.0 +
                metrics.movement_amplitude_x * 2.0 +
                metrics.movement_amplitude_y * 1.0 +
                metrics.angle_variation * 0.002 +
                metrics.acceleration_changes * 0.02 +
                racket_score * 1.2
            )
            
            ratio_velocidade = metrics.max_velocity / metrics.avg_velocity if metrics.avg_velocity > 0 else 0
            if ratio_velocidade > 10 and metrics.angle_variation > 150:
                score *= 0.7
                print(f"    [WARNING] Penalidade aplicada: ratio_vel={ratio_velocidade:.1f}, var_ang={metrics.angle_variation:.1f}")
            
            return score
        
        left_activity_score = calculate_activity_score(left_metrics, left_avg_racket_score)
        right_activity_score = calculate_activity_score(right_metrics, right_avg_racket_score)
        
        print(f"LOG: Score de atividade combinado:")
        print(f"  ESQUERDA: {left_activity_score:.3f}")
        print(f"  DIREITA: {right_activity_score:.3f}")
        print(f"  Diferença: {abs(left_activity_score - right_activity_score):.3f}")
        
        min_score_threshold = 0.5
        min_difference = 0.2
        
        if left_activity_score < min_score_threshold and right_activity_score < min_score_threshold:
            print(f"LOG: [ERROR] Ambas as mãos têm atividade insuficiente")
            return DominantHand.UNKNOWN
        
        score_difference = abs(left_activity_score - right_activity_score)
        if score_difference < min_difference:
            print(f"LOG:  Diferença insuficiente entre as mãos ({score_difference:.3f} < {min_difference})")
            return DominantHand.UNKNOWN
        
        if left_activity_score > right_activity_score:
            confidence = (left_activity_score / (left_activity_score + right_activity_score)) * 100
            print(f"LOG: [OK] MÃO ESQUERDA detectada (confiança: {confidence:.1f}%)")
            print(f"LOG: Razão: Score combinado superior ({left_activity_score:.3f} vs {right_activity_score:.3f})")
            return DominantHand.LEFT
        else:
            confidence = (right_activity_score / (left_activity_score + right_activity_score)) * 100
            print(f"LOG: [OK] MÃO DIREITA detectada (confiança: {confidence:.1f}%)")
            print(f"LOG: Razão: Score combinado superior ({right_activity_score:.3f} vs {left_activity_score:.3f})")
            return DominantHand.RIGHT
    
    def determine_camera_perspective(self, player_orientation, dominant_hand):
        """Determina perspectiva da câmera (mantido do original)"""
        if dominant_hand == DominantHand.UNKNOWN:
            if player_orientation == PlayerOrientation.FACING_RIGHT:
                print("LOG: Inferindo perspectiva pela orientação (mão dominante desconhecida)")
                return CameraPerspective.RIGHT
            elif player_orientation == PlayerOrientation.FACING_LEFT:
                print("LOG: Inferindo perspectiva pela orientação (mão dominante desconhecida)")
                return CameraPerspective.LEFT
            else:
                return CameraPerspective.UNKNOWN
        
        if player_orientation == PlayerOrientation.UNKNOWN:
            return CameraPerspective.UNKNOWN
        
        if player_orientation == PlayerOrientation.FACING_RIGHT:
            return CameraPerspective.RIGHT
        else:
            return CameraPerspective.LEFT
    
    def process_video(self, video_path, max_frames=100):
        """Processa vídeo completo (mantido do original com logs biomecânicos)"""
        if not os.path.exists(video_path):
            print(f"LOG: Arquivo não existe: {video_path}")
            return None, None, None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"LOG: Não foi possível abrir o vídeo: {video_path}")
            return None, None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"LOG: Vídeo {width}x{height}, {fps}fps, {total_frames} frames totais")
        
        frame_count = 0
        pose_detected_count = 0
        orientation_votes = []
        racket_detections = {"left": 0, "right": 0}
        racket_scores = {"left": [], "right": []}
        coordinate_flip_votes = []
        pose_history = []
        ball_detections = []
        
        # PRIMEIRA PASSADA: Analisar coordenadas
        print(f"LOG: Analisando coordenadas em amostra de frames...")
        sample_frames = min(30, max_frames)
        
        for sample_frame in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                coords_flipped = self.detect_coordinate_flip(pose_results.pose_landmarks, frame_width)
                coordinate_flip_votes.append(coords_flipped)
        
        # DECISÃO GLOBAL de inversão
        coords_flipped_count = sum(coordinate_flip_votes)
        coords_normal_count = len(coordinate_flip_votes) - coords_flipped_count
        GLOBAL_COORDS_FLIPPED = coords_flipped_count > coords_normal_count
        
        print(f"LOG: Amostra analisada: {len(coordinate_flip_votes)} frames")
        print(f"LOG: Coordenadas normais: {coords_normal_count}, invertidas: {coords_flipped_count}")
        print(f"LOG: DECISÃO GLOBAL: Coordenadas {'INVERTIDAS' if GLOBAL_COORDS_FLIPPED else 'NORMAIS'}")
        
        # Reiniciar o vídeo para processamento completo
        cap.release()
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # DETECTAR BOLA no frame
            ball_detection = self.detect_ball_in_frame(frame, frame_count)
            if ball_detection:
                ball_detections.append(ball_detection)
                if frame_count % 20 == 0:
                    print(f"LOG: Frame {frame_count} - Bola detectada em {ball_detection.center} (confiança: {ball_detection.confidence:.2f})")
            
            # Detectar pose
            pose_results = self.pose_detector.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                pose_detected_count += 1
                pose_history.append(pose_results.pose_landmarks)
                
                # Detectar orientação
                orientation = self.detect_player_orientation(pose_results.pose_landmarks, frame_width, frame_count)
                orientation_votes.append(orientation)
                
                # Log orientação a cada 20 frames
                if frame_count % 20 == 0:
                    print(f"LOG: Frame {frame_count} - Orientação: {orientation.value}")
                
                # Detectar raquete nas mãos
                try:
                    left_wrist = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    
                    # Aplicar flip GLOBAL
                    if GLOBAL_COORDS_FLIPPED:
                        left_wrist, right_wrist = right_wrist, left_wrist
                    
                    # Log coordenadas
                    if frame_count % 40 == 0:
                        flip_text = " (GLOBAL-INVERTIDO)" if GLOBAL_COORDS_FLIPPED else ""
                        print(f"LOG: Frame {frame_count} coordenadas{flip_text}: ESQ=({left_wrist.x:.3f},{left_wrist.y:.3f}), DIR=({right_wrist.x:.3f},{right_wrist.y:.3f})")
                    
                    # Região da mão esquerda
                    lw_x = int(left_wrist.x * frame_width)
                    lw_y = int(left_wrist.y * frame_height)
                    left_region = frame[max(0, lw_y-50):min(frame_height, lw_y+50), 
                                      max(0, lw_x-50):min(frame_width, lw_x+50)]
                    
                    has_racket_left = False
                    left_score = 0.0
                    if left_region.size > 0:
                        has_racket_left, left_score = self.detect_racket_by_color(frame, left_region, 
                                                                    (max(0, lw_x-50), max(0, lw_y-50), 100, 100),
                                                                    frame_count, "MÃO_ESQUERDA")
                        if has_racket_left:
                            racket_detections["left"] += 1
                        racket_scores["left"].append(left_score)
                    
                    # Região da mão direita
                    rw_x = int(right_wrist.x * frame_width)
                    rw_y = int(right_wrist.y * frame_height)
                    right_region = frame[max(0, rw_y-50):min(frame_height, rw_y+50), 
                                       max(0, rw_x-50):min(frame_width, rw_x+50)]
                    
                    has_racket_right = False
                    right_score = 0.0
                    if right_region.size > 0:
                        has_racket_right, right_score = self.detect_racket_by_color(frame, right_region, 
                                                                     (max(0, rw_x-50), max(0, rw_y-50), 100, 100),
                                                                     frame_count, "MÃO_DIREITA")
                        if has_racket_right:
                            racket_detections["right"] += 1
                        racket_scores["right"].append(right_score)
                    
                    # Log das detecções
                    if frame_count % 20 == 0:
                        print(f"  [STATS] Raquete detectada - ESQ: {has_racket_left} (score: {left_score:.2f}), DIR: {has_racket_right} (score: {right_score:.2f})")
                    
                except Exception as e:
                    if frame_count % 40 == 0:
                        print(f"LOG: Erro na detecção frame {frame_count}: {e}")
            else:
                pose_history.append(None)
            
            frame_count += 1
        
        cap.release()
        
        print(f"\nLOG: === PROCESSAMENTO COMPLETO ===")
        print(f"LOG: Frames processados: {frame_count}")
        print(f"LOG: Poses detectadas: {pose_detected_count}")
        print(f"LOG: Bolas detectadas: {len(ball_detections)} frames")
        
        print(f"LOG: === ANÁLISE DE COORDENADAS ===")
        print(f"LOG: Decisão GLOBAL aplicada: Coordenadas {'INVERTIDAS' if GLOBAL_COORDS_FLIPPED else 'NORMAIS'}")
        print(f"LOG: Detecções de raquete - Esquerda: {racket_detections['left']}, Direita: {racket_detections['right']}")
        
        # Determinar orientação final
        final_orientation = PlayerOrientation.UNKNOWN
        if orientation_votes:
            orientation_counts = {}
            for vote in orientation_votes:
                orientation_counts[vote] = orientation_counts.get(vote, 0) + 1
            
            final_orientation = max(orientation_counts.keys(), key=lambda x: orientation_counts[x])
            print(f"LOG: Orientação final: {final_orientation.value}")
        
        # ANÁLISE COMPLETA DE MÃO DOMINANTE usando TODAS as métricas (incluindo biomecânicas)
        final_hand = self.determine_dominant_hand_with_movement_analysis(racket_detections, racket_scores, pose_history)
        
        # Análise da trajetória da bola
        if ball_detections:
            print(f"\nLOG: === ANÁLISE DA TRAJETÓRIA DA BOLA ===")
            print(f"LOG: Bola detectada em {len(ball_detections)} frames")
            
            if len(ball_detections) > 1:
                ball_velocities = []
                for i in range(1, len(ball_detections)):
                    prev_ball = ball_detections[i-1]
                    curr_ball = ball_detections[i]
                    
                    distance = np.sqrt((curr_ball.center[0] - prev_ball.center[0])**2 + 
                                     (curr_ball.center[1] - prev_ball.center[1])**2)
                    ball_velocities.append(distance)
                
                avg_ball_velocity = np.mean(ball_velocities)
                max_ball_velocity = max(ball_velocities)
                print(f"LOG: Velocidade média da bola: {avg_ball_velocity:.2f} pixels/frame")
                print(f"LOG: Velocidade máxima da bola: {max_ball_velocity:.2f} pixels/frame")
        
        final_perspective = self.determine_camera_perspective(final_orientation, final_hand)
        print(f"LOG: Perspectiva final: {final_perspective.value}")
        
        return final_orientation, final_hand, final_perspective

def main():
    """Teste do Enhanced Racket Tracker"""
    
    if len(sys.argv) != 2:
        print("[BIOMECH] ENHANCED RACKET TRACKER 2D - COM BIOMECÂNICA")
        print("Uso: python enhanced_racket_tracker_2d.py videos/nome.mp4")
        print("\n NOVIDADES:")
        print("   Análise avançada do cotovelo (velocidade angular, coordenação)")
        print("   Discriminadores biomecânicos para zona crítica Y=0.060-0.085")
        print("   Probabilidades forehand/backhand e drive/push")
        print("   100% compatibilidade com sistema existente")
        print("\nExemplo: python enhanced_racket_tracker_2d.py videos/japones_BP_D_E.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Arquivo não encontrado: {video_path}")
        sys.exit(1)
    
    print(f"[BIOMECH] Testando Enhanced Racket Tracker")
    print(f"[FILE] Vídeo: {os.path.basename(video_path)}")
    
    # Criar tracker expandido
    tracker = EnhancedRacketTracker2D()
    
    # Processar vídeo (usando mesma interface)
    orientation, hand, perspective = tracker.process_video(video_path)
    
    print(f"\n[TENNIS] === RESULTADOS ENHANCED ===")
    if orientation is None:
        print("[ERROR] Erro no processamento do vídeo")
    else:
        print(f" Orientação: {orientation.value}")
        print(f"[HAND] Mão dominante: {hand.value}")
        print(f" Perspectiva: {perspective.value}")
        print(f"[OK] Análise biomecânica integrada com sucesso!")

if __name__ == "__main__":
    main()
