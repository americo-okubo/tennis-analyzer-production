"""
Racket Trajectory Analyzer for Tennis Biomechanics

Analyzes racket movement parameters that directly impact shot quality:
- Trajectory analysis and path optimization
- Velocity and acceleration patterns
- Impact point estimation
- Swing efficiency metrics
"""

import numpy as np
import scipy.signal
from scipy.interpolate import UnivariateSpline
from typing import Dict, List, Tuple, Optional
import logging

class RacketTrajectoryAnalyzer:
    """
    Specialized analysis of racket trajectory and movement patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Optimal racket parameters (based on tennis biomechanics)
        self.optimal_parameters = {
            'max_velocity': (8, 15),        # m/s (converted from pixels)
            'acceleration_peak': (50, 200), # m/s²
            'trajectory_efficiency': (0.7, 0.95),  # Straightness ratio
            'swing_smoothness': (0.6, 0.9),        # Smoothness score
        }
    
    def analyze(self, landmarks_data: Dict, kinematics_data: Dict) -> Dict:
        """
        Comprehensive racket trajectory analysis.
        """
        try:
            velocity_data = kinematics_data['velocities']
            acceleration_data = kinematics_data['accelerations']
            trajectory_data = kinematics_data['trajectories']
            
            # Analyze different aspects of racket movement
            velocity_analysis = self._analyze_racket_velocity(velocity_data)
            trajectory_analysis = self._analyze_racket_trajectory(trajectory_data)
            acceleration_analysis = self._analyze_racket_acceleration(acceleration_data)
            impact_analysis = self._estimate_impact_characteristics(
                velocity_data, acceleration_data, trajectory_data
            )
            
            return {
                'success': True,
                'velocity_analysis': velocity_analysis,
                'trajectory_analysis': trajectory_analysis,
                'acceleration_analysis': acceleration_analysis,
                'impact_analysis': impact_analysis,
                'overall_racket_score': self._calculate_overall_racket_score(
                    velocity_analysis, trajectory_analysis, acceleration_analysis, impact_analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"[RACKET_TRAJECTORY] Analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_racket_velocity(self, velocity_data: Dict) -> Dict:
        """
        Analyze racket velocity patterns throughout the stroke.
        """
        velocity_analysis = {}
        
        # Focus on dominant hand (right wrist as proxy for racket)
        wrist_key = 'right_wrist_velocity'
        
        if wrist_key in velocity_data and velocity_data[wrist_key]:
            velocities = velocity_data[wrist_key]
            magnitudes = [v['magnitude'] for v in velocities]
            times = [v['time'] for v in velocities]
            
            # Basic velocity statistics
            velocity_analysis['basic_stats'] = {
                'max_velocity': np.max(magnitudes),
                'mean_velocity': np.mean(magnitudes),
                'velocity_at_impact': self._find_impact_velocity(velocities),
                'acceleration_phase_velocity': self._analyze_velocity_phases(velocities)
            }
            
            # Velocity pattern analysis
            velocity_analysis['patterns'] = {
                'velocity_profile': self._analyze_velocity_profile(magnitudes, times),
                'consistency': self._calculate_velocity_consistency(magnitudes),
                'smoothness': self._calculate_velocity_smoothness(magnitudes)
            }
            
            # Performance assessment
            velocity_analysis['performance'] = {
                'velocity_efficiency': self._assess_velocity_efficiency(magnitudes),
                'optimal_velocity_score': self._score_optimal_velocity(
                    velocity_analysis['basic_stats']['max_velocity']
                )
            }
        
        return velocity_analysis
    
    def _analyze_racket_trajectory(self, trajectory_data: Dict) -> Dict:
        """
        Analyze racket trajectory characteristics.
        """
        trajectory_analysis = {}
        
        # Focus on dominant hand trajectory
        wrist_key = 'right_wrist_trajectory'
        
        if wrist_key in trajectory_data:
            traj = trajectory_data[wrist_key]
            
            # Basic trajectory metrics
            trajectory_analysis['basic_metrics'] = {
                'total_path_length': traj['total_distance'],
                'straight_line_distance': traj['straight_distance'],
                'trajectory_efficiency': traj['efficiency'],
                'movement_range': traj['movement_range']
            }
            
            # Path analysis
            if 'path_points' in traj:
                path_points = np.array(traj['path_points'])
                trajectory_analysis['path_analysis'] = self._analyze_trajectory_path(path_points)
            
            # Smoothness and control
            trajectory_analysis['control_metrics'] = {
                'trajectory_smoothness': traj['smoothness'],
                'path_consistency': self._assess_path_consistency(traj),
                'control_score': self._calculate_trajectory_control_score(traj)
            }
        
        return trajectory_analysis
    
    def _analyze_racket_acceleration(self, acceleration_data: Dict) -> Dict:
        """
        Analyze racket acceleration patterns.
        """
        acceleration_analysis = {}
        
        # Focus on dominant hand acceleration
        wrist_key = 'right_wrist_acceleration'
        
        if wrist_key in acceleration_data and acceleration_data[wrist_key]:
            accelerations = acceleration_data[wrist_key]
            magnitudes = [a['magnitude'] for a in accelerations]
            times = [a['time'] for a in accelerations]
            
            # Basic acceleration statistics
            acceleration_analysis['basic_stats'] = {
                'max_acceleration': np.max(magnitudes),
                'mean_acceleration': np.mean(magnitudes),
                'acceleration_variability': np.std(magnitudes)
            }
            
            # Acceleration phases
            acceleration_analysis['phases'] = self._identify_acceleration_phases(
                accelerations, magnitudes, times
            )
            
            # Performance metrics
            acceleration_analysis['performance'] = {
                'acceleration_efficiency': self._assess_acceleration_efficiency(magnitudes),
                'power_generation': self._estimate_power_generation(accelerations),
                'optimal_acceleration_score': self._score_optimal_acceleration(
                    acceleration_analysis['basic_stats']['max_acceleration']
                )
            }
        
        return acceleration_analysis
    
    def _estimate_impact_characteristics(self, velocity_data: Dict, acceleration_data: Dict, 
                                       trajectory_data: Dict) -> Dict:
        """
        Estimate characteristics at the point of impact.
        """
        impact_analysis = {}
        
        try:
            # Find impact point (typically at maximum velocity or zero acceleration)
            wrist_velocity_key = 'right_wrist_velocity'
            wrist_accel_key = 'right_wrist_acceleration'
            
            if (wrist_velocity_key in velocity_data and 
                wrist_accel_key in acceleration_data):
                
                velocities = velocity_data[wrist_velocity_key]
                accelerations = acceleration_data[wrist_accel_key]
                
                # Estimate impact timing
                impact_timing = self._estimate_impact_timing(velocities, accelerations)
                
                if impact_timing:
                    impact_analysis['timing'] = impact_timing
                    
                    # Impact velocity and direction
                    impact_analysis['velocity_at_impact'] = self._get_impact_velocity(
                        velocities, impact_timing['impact_frame']
                    )
                    
                    # Impact position
                    if 'right_wrist_trajectory' in trajectory_data:
                        traj = trajectory_data['right_wrist_trajectory']
                        impact_analysis['impact_position'] = self._estimate_impact_position(
                            traj, impact_timing
                        )
                    
                    # Power transfer estimation
                    impact_analysis['power_metrics'] = self._estimate_impact_power(
                        impact_analysis['velocity_at_impact'],
                        accelerations
                    )
            
        except Exception as e:
            self.logger.warning(f"[IMPACT] Could not estimate impact characteristics: {e}")
            impact_analysis['error'] = str(e)
        
        return impact_analysis
    
    def _find_impact_velocity(self, velocities: List[Dict]) -> float:
        """
        Find velocity at estimated impact point.
        """
        if not velocities:
            return 0.0
        
        # Impact typically occurs near maximum velocity
        magnitudes = [v['magnitude'] for v in velocities]
        max_vel_idx = np.argmax(magnitudes)
        
        return magnitudes[max_vel_idx]
    
    def _analyze_velocity_phases(self, velocities: List[Dict]) -> Dict:
        """
        Analyze different phases of velocity during the stroke.
        """
        if len(velocities) < 3:
            return {'error': 'Insufficient data for phase analysis'}
        
        magnitudes = [v['magnitude'] for v in velocities]
        times = [v['time'] for v in velocities]
        
        # Find phases
        max_vel_idx = np.argmax(magnitudes)
        
        # Acceleration phase (start to max velocity)
        accel_phase_velocities = magnitudes[:max_vel_idx+1]
        accel_phase_times = times[:max_vel_idx+1]
        
        # Deceleration phase (max velocity to end)
        decel_phase_velocities = magnitudes[max_vel_idx:]
        decel_phase_times = times[max_vel_idx:]
        
        return {
            'acceleration_phase': {
                'duration': accel_phase_times[-1] - accel_phase_times[0] if len(accel_phase_times) > 1 else 0,
                'velocity_change': accel_phase_velocities[-1] - accel_phase_velocities[0] if len(accel_phase_velocities) > 1 else 0,
                'average_acceleration': (accel_phase_velocities[-1] - accel_phase_velocities[0]) / 
                                      (accel_phase_times[-1] - accel_phase_times[0]) if len(accel_phase_times) > 1 and accel_phase_times[-1] > accel_phase_times[0] else 0
            },
            'deceleration_phase': {
                'duration': decel_phase_times[-1] - decel_phase_times[0] if len(decel_phase_times) > 1 else 0,
                'velocity_change': decel_phase_velocities[0] - decel_phase_velocities[-1] if len(decel_phase_velocities) > 1 else 0
            },
            'max_velocity_timing': times[max_vel_idx]
        }
    
    def _analyze_velocity_profile(self, magnitudes: List[float], times: List[float]) -> Dict:
        """
        Analyze the overall velocity profile shape.
        """
        if len(magnitudes) < 3:
            return {'error': 'Insufficient data for profile analysis'}
        
        # Smooth the velocity profile
        smoothed_velocities = scipy.signal.savgol_filter(magnitudes, 
                                                        min(5, len(magnitudes)), 2)
        
        # Find peaks and characteristics
        peaks, _ = scipy.signal.find_peaks(smoothed_velocities, height=np.max(smoothed_velocities) * 0.3)
        
        profile_analysis = {
            'num_peaks': len(peaks),
            'profile_smoothness': 1.0 - (np.var(np.diff(smoothed_velocities)) / np.var(smoothed_velocities)) if np.var(smoothed_velocities) > 0 else 0,
            'velocity_build_up_rate': self._calculate_buildup_rate(smoothed_velocities, times),
            'profile_symmetry': self._calculate_profile_symmetry(smoothed_velocities)
        }
        
        return profile_analysis
    
    def _calculate_velocity_consistency(self, magnitudes: List[float]) -> float:
        """
        Calculate consistency of velocity throughout the movement.
        """
        if len(magnitudes) < 2:
            return 0.0
        
        # Use coefficient of variation (inverted for consistency)
        cv = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 1
        consistency = 1.0 / (1.0 + cv)
        
        return consistency
    
    def _calculate_velocity_smoothness(self, magnitudes: List[float]) -> float:
        """
        Calculate smoothness of velocity profile.
        """
        if len(magnitudes) < 3:
            return 0.0
        
        # Calculate second derivative (acceleration of velocity)
        velocity_acceleration = np.diff(magnitudes, n=2)
        
        # Smoothness inversely related to variance in acceleration
        smoothness = 1.0 / (1.0 + np.var(velocity_acceleration))
        
        return smoothness
    
    def _assess_velocity_efficiency(self, magnitudes: List[float]) -> float:
        """
        Assess how efficiently velocity is used throughout the stroke.
        """
        if len(magnitudes) < 2:
            return 0.0
        
        max_velocity = np.max(magnitudes)
        mean_velocity = np.mean(magnitudes)
        
        # Efficiency based on how much of the stroke uses high velocity
        high_velocity_ratio = np.sum(np.array(magnitudes) > max_velocity * 0.7) / len(magnitudes)
        
        # Combined efficiency score
        efficiency = (high_velocity_ratio + mean_velocity / max_velocity) / 2
        
        return efficiency
    
    def _analyze_trajectory_path(self, path_points: np.ndarray) -> Dict:
        """
        Analyze the geometric properties of the trajectory path.
        """
        if len(path_points) < 3:
            return {'error': 'Insufficient points for path analysis'}
        
        # Calculate path curvature
        curvatures = self._calculate_path_curvature(path_points)
        
        # Analyze path direction changes
        direction_changes = self._calculate_direction_changes(path_points)
        
        # Path complexity
        complexity = self._calculate_path_complexity(path_points)
        
        return {
            'average_curvature': np.mean(curvatures) if len(curvatures) > 0 else 0,
            'max_curvature': np.max(curvatures) if len(curvatures) > 0 else 0,
            'direction_changes': direction_changes,
            'path_complexity': complexity,
            'geometric_efficiency': self._calculate_geometric_efficiency(path_points)
        }
    
    def _calculate_path_curvature(self, path_points: np.ndarray) -> List[float]:
        """
        Calculate curvature at each point along the path.
        """
        if len(path_points) < 3:
            return []
        
        curvatures = []
        
        for i in range(1, len(path_points) - 1):
            # Three consecutive points
            p1, p2, p3 = path_points[i-1], path_points[i], path_points[i+1]
            
            # Calculate curvature using the formula for discrete points
            area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            
            side_a = np.linalg.norm(p2 - p1)
            side_b = np.linalg.norm(p3 - p2)
            side_c = np.linalg.norm(p3 - p1)
            
            if side_a * side_b * side_c > 0:
                curvature = 4 * area / (side_a * side_b * side_c)
                curvatures.append(curvature)
        
        return curvatures
    
    def _estimate_impact_timing(self, velocities: List[Dict], accelerations: List[Dict]) -> Optional[Dict]:
        """
        Estimate when impact occurs during the stroke.
        """
        if not velocities or not accelerations:
            return None
        
        vel_magnitudes = [v['magnitude'] for v in velocities]
        accel_magnitudes = [a['magnitude'] for a in accelerations]
        
        # Impact typically occurs near maximum velocity
        max_vel_idx = np.argmax(vel_magnitudes)
        
        # Or when acceleration crosses zero (velocity peak)
        zero_crossings = []
        for i in range(1, len(accel_magnitudes)):
            if (accel_magnitudes[i-1] > 0 and accel_magnitudes[i] <= 0):
                zero_crossings.append(i)
        
        # Choose the most likely impact point
        impact_frame = max_vel_idx
        impact_confidence = 0.8
        
        if zero_crossings and abs(zero_crossings[0] - max_vel_idx) < 3:
            impact_frame = zero_crossings[0]
            impact_confidence = 0.9
        
        return {
            'impact_frame': impact_frame,
            'impact_time': velocities[impact_frame]['time'],
            'confidence': impact_confidence,
            'method': 'velocity_peak_detection'
        }
    
    def _score_optimal_velocity(self, max_velocity: float) -> float:
        """
        Score velocity against optimal range for tennis.
        """
        # Convert from pixels/frame to approximate m/s (rough conversion)
        # This is a simplified conversion - in real application, you'd need proper calibration
        velocity_ms = max_velocity * 0.01  # Rough conversion factor
        
        optimal_min, optimal_max = self.optimal_parameters['max_velocity']
        
        if optimal_min <= velocity_ms <= optimal_max:
            return 1.0
        elif velocity_ms < optimal_min:
            return max(0.0, velocity_ms / optimal_min)
        else:
            return max(0.0, 1.0 - (velocity_ms - optimal_max) / optimal_max)
    
    def _score_optimal_acceleration(self, max_acceleration: float) -> float:
        """
        Score acceleration against optimal range.
        """
        # Similar conversion as velocity
        accel_ms2 = max_acceleration * 0.01
        
        optimal_min, optimal_max = self.optimal_parameters['acceleration_peak']
        
        if optimal_min <= accel_ms2 <= optimal_max:
            return 1.0
        elif accel_ms2 < optimal_min:
            return max(0.0, accel_ms2 / optimal_min)
        else:
            return max(0.0, 1.0 - (accel_ms2 - optimal_max) / optimal_max)
    
    def _calculate_overall_racket_score(self, velocity_analysis: Dict, trajectory_analysis: Dict,
                                      acceleration_analysis: Dict, impact_analysis: Dict) -> float:
        """
        Calculate overall racket performance score.
        """
        scores = []
        
        # Velocity score
        if 'performance' in velocity_analysis and 'optimal_velocity_score' in velocity_analysis['performance']:
            scores.append(velocity_analysis['performance']['optimal_velocity_score'])
        
        # Trajectory efficiency
        if 'basic_metrics' in trajectory_analysis and 'trajectory_efficiency' in trajectory_analysis['basic_metrics']:
            scores.append(trajectory_analysis['basic_metrics']['trajectory_efficiency'])
        
        # Acceleration score
        if 'performance' in acceleration_analysis and 'optimal_acceleration_score' in acceleration_analysis['performance']:
            scores.append(acceleration_analysis['performance']['optimal_acceleration_score'])
        
        # Control metrics
        if 'control_metrics' in trajectory_analysis and 'control_score' in trajectory_analysis['control_metrics']:
            scores.append(trajectory_analysis['control_metrics']['control_score'])
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_path_consistency(self, trajectory_data: Dict) -> float:
        """Assess consistency of trajectory path."""
        return trajectory_data.get('smoothness', 0.5)
    
    def _calculate_trajectory_control_score(self, trajectory_data: Dict) -> float:
        """Calculate overall trajectory control score."""
        efficiency = trajectory_data.get('efficiency', 0.5)
        smoothness = trajectory_data.get('smoothness', 0.5)
        return (efficiency + smoothness) / 2
    
    def _identify_acceleration_phases(self, accelerations: List[Dict], magnitudes: List[float], times: List[float]) -> Dict:
        """Identify different phases of acceleration."""
        if len(magnitudes) < 3:
            return {'error': 'Insufficient data'}
        
        max_accel_idx = np.argmax(magnitudes)
        return {
            'peak_acceleration_time': times[max_accel_idx],
            'peak_acceleration_value': magnitudes[max_accel_idx]
        }
    
    def _assess_acceleration_efficiency(self, magnitudes: List[float]) -> float:
        """Assess acceleration efficiency."""
        if len(magnitudes) < 2:
            return 0.0
        return 1.0 - (np.std(magnitudes) / np.mean(magnitudes)) if np.mean(magnitudes) > 0 else 0
    
    def _estimate_power_generation(self, accelerations: List[Dict]) -> float:
        """Estimate power generation from acceleration data."""
        magnitudes = [a['magnitude'] for a in accelerations]
        return np.max(magnitudes) if magnitudes else 0.0
    
    def _get_impact_velocity(self, velocities: List[Dict], impact_frame: int) -> Dict:
        """Get velocity at impact frame."""
        if 0 <= impact_frame < len(velocities):
            return velocities[impact_frame]
        return {'magnitude': 0, 'vx': 0, 'vy': 0}
    
    def _estimate_impact_position(self, trajectory_data: Dict, impact_timing: Dict) -> Dict:
        """Estimate position at impact."""
        return {'x': 0, 'y': 0}  # Placeholder
    
    def _estimate_impact_power(self, impact_velocity: Dict, accelerations: List[Dict]) -> Dict:
        """Estimate power at impact."""
        return {'estimated_power': impact_velocity.get('magnitude', 0)}
    
    def _calculate_buildup_rate(self, velocities: List[float], times: List[float]) -> float:
        """Calculate velocity buildup rate."""
        if len(velocities) < 2:
            return 0.0
        max_vel = np.max(velocities)
        max_idx = np.argmax(velocities)
        if max_idx > 0:
            return max_vel / times[max_idx]
        return 0.0
    
    def _calculate_profile_symmetry(self, velocities: List[float]) -> float:
        """Calculate symmetry of velocity profile."""
        if len(velocities) < 3:
            return 0.0
        max_idx = np.argmax(velocities)
        left_part = velocities[:max_idx+1]
        right_part = velocities[max_idx:]
        if len(left_part) > 0 and len(right_part) > 0:
            return 1.0 - abs(len(left_part) - len(right_part)) / len(velocities)
        return 0.0
    
    def _calculate_direction_changes(self, path_points: np.ndarray) -> int:
        """Calculate number of significant direction changes."""
        if len(path_points) < 3:
            return 0
        
        directions = np.diff(path_points, axis=0)
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        angle_changes = np.abs(np.diff(angles))
        
        # Count significant direction changes (> 30 degrees)
        significant_changes = np.sum(angle_changes > np.pi/6)
        return int(significant_changes)
    
    def _calculate_path_complexity(self, path_points: np.ndarray) -> float:
        """Calculate overall path complexity."""
        if len(path_points) < 2:
            return 0.0
        
        total_distance = 0
        for i in range(1, len(path_points)):
            total_distance += np.linalg.norm(path_points[i] - path_points[i-1])
        
        straight_distance = np.linalg.norm(path_points[-1] - path_points[0])
        
        if straight_distance > 0:
            return total_distance / straight_distance
        return 1.0
    
    def _calculate_geometric_efficiency(self, path_points: np.ndarray) -> float:
        """Calculate geometric efficiency of the path."""
        if len(path_points) < 2:
            return 0.0
        
        complexity = self._calculate_path_complexity(path_points)
        return 1.0 / complexity if complexity > 0 else 1.0