"""
Joint Angles Analyzer for Tennis Biomechanics

Analyzes articular angles and their biomechanical significance:
- Arm joint angles (shoulder, elbow, wrist)
- Body alignment and posture
- Range of motion analysis
- Optimal angle assessment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class JointAnglesAnalyzer:
    """
    Specialized analysis of joint angles for tennis movements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Optimal angle ranges for tennis (based on biomechanics literature)
        self.optimal_ranges = {
            'forehand_elbow': (120, 160),      # Elbow extension during forehand
            'backhand_elbow': (90, 140),       # Elbow angle during backhand
            'shoulder_abduction': (45, 90),    # Shoulder elevation
            'torso_inclination': (5, 25),      # Forward lean
            'stance_width': (0.8, 1.5),        # Relative to body height
        }
    
    def analyze(self, landmarks_data: Dict, kinematics_data: Dict) -> Dict:
        """
        Comprehensive joint angle analysis.
        """
        try:
            angles_data = kinematics_data['joint_angles']
            
            # Analyze different aspects
            arm_analysis = self._analyze_arm_mechanics(angles_data)
            posture_analysis = self._analyze_posture(angles_data)
            range_analysis = self._analyze_range_of_motion(angles_data)
            optimal_analysis = self._analyze_optimal_angles(angles_data)
            
            return {
                'success': True,
                'arm_mechanics': arm_analysis,
                'posture_analysis': posture_analysis,
                'range_of_motion': range_analysis,
                'optimal_angle_assessment': optimal_analysis,
                'overall_joint_score': self._calculate_overall_score(
                    arm_analysis, posture_analysis, range_analysis, optimal_analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"[JOINT_ANGLES] Analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_arm_mechanics(self, angles_data: Dict) -> Dict:
        """
        Analyze arm joint mechanics throughout movement.
        """
        arm_analysis = {}
        
        # Right arm analysis (assuming dominant arm)
        if 'right_arm_angles' in angles_data and angles_data['right_arm_angles']:
            right_arm_values = [item['value'] for item in angles_data['right_arm_angles']]
            
            arm_analysis['dominant_arm'] = {
                'mean_angle': np.mean(right_arm_values),
                'angle_range': np.max(right_arm_values) - np.min(right_arm_values),
                'min_angle': np.min(right_arm_values),
                'max_angle': np.max(right_arm_values),
                'extension_phase': self._identify_extension_phase(angles_data['right_arm_angles']),
                'consistency': 1.0 - (np.std(right_arm_values) / np.mean(right_arm_values)) if np.mean(right_arm_values) > 0 else 0
            }
            
            # Evaluate arm mechanics quality
            arm_analysis['dominant_arm']['mechanics_score'] = self._evaluate_arm_mechanics(
                arm_analysis['dominant_arm']
            )
        
        # Left arm analysis
        if 'left_arm_angles' in angles_data and angles_data['left_arm_angles']:
            left_arm_values = [item['value'] for item in angles_data['left_arm_angles']]
            
            arm_analysis['non_dominant_arm'] = {
                'mean_angle': np.mean(left_arm_values),
                'angle_range': np.max(left_arm_values) - np.min(left_arm_values),
                'min_angle': np.min(left_arm_values),
                'max_angle': np.max(left_arm_values),
                'consistency': 1.0 - (np.std(left_arm_values) / np.mean(left_arm_values)) if np.mean(left_arm_values) > 0 else 0
            }
        
        # Bilateral coordination
        if 'dominant_arm' in arm_analysis and 'non_dominant_arm' in arm_analysis:
            arm_analysis['bilateral_coordination'] = self._analyze_bilateral_coordination(
                arm_analysis['dominant_arm'], arm_analysis['non_dominant_arm']
            )
        
        return arm_analysis
    
    def _analyze_posture(self, angles_data: Dict) -> Dict:
        """
        Analyze body posture and alignment.
        """
        posture_analysis = {}
        
        # Torso inclination analysis
        if 'torso_inclination' in angles_data and angles_data['torso_inclination']:
            inclination_values = [item['value'] for item in angles_data['torso_inclination']]
            
            posture_analysis['torso'] = {
                'mean_inclination': np.mean(inclination_values),
                'inclination_range': np.max(inclination_values) - np.min(inclination_values),
                'stability': 1.0 - (np.std(inclination_values) / 45.0),  # Normalized by max expected variation
                'optimal_inclination_score': self._score_optimal_range(
                    np.mean(inclination_values), self.optimal_ranges['torso_inclination']
                )
            }
        
        # Shoulder rotation analysis
        if 'shoulder_rotation' in angles_data and angles_data['shoulder_rotation']:
            shoulder_values = [item['value'] for item in angles_data['shoulder_rotation']]
            
            posture_analysis['shoulders'] = {
                'mean_rotation': np.mean(shoulder_values),
                'rotation_range': np.max(shoulder_values) - np.min(shoulder_values),
                'rotation_stability': 1.0 - (np.std(shoulder_values) / 90.0)
            }
        
        # Hip rotation analysis
        if 'hip_rotation' in angles_data and angles_data['hip_rotation']:
            hip_values = [item['value'] for item in angles_data['hip_rotation']]
            
            posture_analysis['hips'] = {
                'mean_rotation': np.mean(hip_values),
                'rotation_range': np.max(hip_values) - np.min(hip_values),
                'hip_stability': 1.0 - (np.std(hip_values) / 90.0)
            }
        
        # Overall posture score
        posture_analysis['overall_posture_score'] = self._calculate_posture_score(posture_analysis)
        
        return posture_analysis
    
    def _analyze_range_of_motion(self, angles_data: Dict) -> Dict:
        """
        Analyze range of motion for different joints.
        """
        rom_analysis = {}
        
        # Analyze ROM for each joint
        joint_mappings = {
            'elbow_rom': 'right_arm_angles',
            'shoulder_rom': 'shoulder_rotation', 
            'hip_rom': 'hip_rotation'
        }
        
        for rom_key, angle_key in joint_mappings.items():
            if angle_key in angles_data and angles_data[angle_key]:
                values = [item['value'] for item in angles_data[angle_key]]
                
                rom_analysis[rom_key] = {
                    'range': np.max(values) - np.min(values),
                    'min_value': np.min(values),
                    'max_value': np.max(values),
                    'rom_adequacy': self._assess_rom_adequacy(rom_key, np.max(values) - np.min(values))
                }
        
        # Stance width analysis
        if 'stance_width' in angles_data and angles_data['stance_width']:
            width_values = [item['value'] for item in angles_data['stance_width']]
            
            rom_analysis['stance_analysis'] = {
                'mean_width': np.mean(width_values),
                'width_variation': np.std(width_values),
                'stability_score': 1.0 - (np.std(width_values) / np.mean(width_values)) if np.mean(width_values) > 0 else 0
            }
        
        return rom_analysis
    
    def _analyze_optimal_angles(self, angles_data: Dict) -> Dict:
        """
        Compare actual angles with biomechanically optimal ranges.
        """
        optimal_analysis = {}
        
        # Analyze each joint against optimal ranges
        if 'right_arm_angles' in angles_data and angles_data['right_arm_angles']:
            arm_values = [item['value'] for item in angles_data['right_arm_angles']]
            mean_arm_angle = np.mean(arm_values)
            
            # Assess against forehand optimal range (assuming forehand movement)
            optimal_analysis['elbow_optimality'] = {
                'mean_angle': mean_arm_angle,
                'optimal_range': self.optimal_ranges['forehand_elbow'],
                'optimality_score': self._score_optimal_range(
                    mean_arm_angle, self.optimal_ranges['forehand_elbow']
                ),
                'recommendations': self._generate_angle_recommendations(
                    mean_arm_angle, self.optimal_ranges['forehand_elbow'], 'elbow'
                )
            }
        
        # Torso inclination optimality
        if 'torso_inclination' in angles_data and angles_data['torso_inclination']:
            torso_values = [item['value'] for item in angles_data['torso_inclination']]
            mean_torso = np.mean(torso_values)
            
            optimal_analysis['torso_optimality'] = {
                'mean_inclination': mean_torso,
                'optimal_range': self.optimal_ranges['torso_inclination'],
                'optimality_score': self._score_optimal_range(
                    mean_torso, self.optimal_ranges['torso_inclination']
                ),
                'recommendations': self._generate_angle_recommendations(
                    mean_torso, self.optimal_ranges['torso_inclination'], 'torso'
                )
            }
        
        return optimal_analysis
    
    def _identify_extension_phase(self, arm_angles: List[Dict]) -> Dict:
        """
        Identify the extension phase of arm movement.
        """
        if len(arm_angles) < 3:
            return {'success': False, 'error': 'Insufficient data'}
        
        values = [item['value'] for item in arm_angles]
        times = [item['time'] for item in arm_angles]
        
        # Find maximum extension (highest angle)
        max_idx = np.argmax(values)
        max_extension = values[max_idx]
        max_time = times[max_idx]
        
        # Find extension phase (increasing angles leading to max)
        extension_start_idx = 0
        for i in range(max_idx):
            if i > 0 and values[i] > values[i-1]:
                extension_start_idx = i
                break
        
        extension_phase = {
            'start_time': times[extension_start_idx],
            'peak_time': max_time,
            'duration': max_time - times[extension_start_idx],
            'angle_change': max_extension - values[extension_start_idx],
            'extension_rate': (max_extension - values[extension_start_idx]) / (max_time - times[extension_start_idx]) if max_time > times[extension_start_idx] else 0
        }
        
        return extension_phase
    
    def _evaluate_arm_mechanics(self, arm_data: Dict) -> float:
        """
        Evaluate overall arm mechanics quality (0-1 score).
        """
        scores = []
        
        # Range of motion score
        if 'angle_range' in arm_data:
            # Good range of motion for tennis: 60-120 degrees
            rom_score = self._score_optimal_range(arm_data['angle_range'], (60, 120))
            scores.append(rom_score)
        
        # Consistency score
        if 'consistency' in arm_data:
            scores.append(arm_data['consistency'])
        
        # Extension phase quality
        if 'extension_phase' in arm_data and 'extension_rate' in arm_data['extension_phase']:
            # Moderate extension rate is optimal
            ext_rate = arm_data['extension_phase']['extension_rate']
            ext_score = self._score_optimal_range(abs(ext_rate), (20, 80))
            scores.append(ext_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_bilateral_coordination(self, dominant_arm: Dict, non_dominant_arm: Dict) -> Dict:
        """
        Analyze coordination between both arms.
        """
        coordination = {}
        
        # Compare range of motion
        if 'angle_range' in dominant_arm and 'angle_range' in non_dominant_arm:
            rom_difference = abs(dominant_arm['angle_range'] - non_dominant_arm['angle_range'])
            coordination['rom_symmetry'] = 1.0 - min(rom_difference / 60.0, 1.0)  # Normalize by 60°
        
        # Compare consistency
        if 'consistency' in dominant_arm and 'consistency' in non_dominant_arm:
            consistency_diff = abs(dominant_arm['consistency'] - non_dominant_arm['consistency'])
            coordination['consistency_balance'] = 1.0 - consistency_diff
        
        # Overall coordination score
        coord_scores = [v for v in coordination.values() if isinstance(v, (int, float))]
        coordination['overall_coordination'] = np.mean(coord_scores) if coord_scores else 0.5
        
        return coordination
    
    def _calculate_posture_score(self, posture_data: Dict) -> float:
        """
        Calculate overall posture quality score.
        """
        scores = []
        
        if 'torso' in posture_data:
            torso = posture_data['torso']
            if 'stability' in torso:
                scores.append(torso['stability'])
            if 'optimal_inclination_score' in torso:
                scores.append(torso['optimal_inclination_score'])
        
        if 'shoulders' in posture_data and 'rotation_stability' in posture_data['shoulders']:
            scores.append(posture_data['shoulders']['rotation_stability'])
        
        if 'hips' in posture_data and 'hip_stability' in posture_data['hips']:
            scores.append(posture_data['hips']['hip_stability'])
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_rom_adequacy(self, joint_type: str, rom_value: float) -> float:
        """
        Assess if range of motion is adequate for tennis.
        """
        # Expected ROM ranges for tennis
        expected_ranges = {
            'elbow_rom': (60, 120),    # Degrees
            'shoulder_rom': (30, 90),  # Degrees
            'hip_rom': (20, 60)        # Degrees
        }
        
        if joint_type in expected_ranges:
            return self._score_optimal_range(rom_value, expected_ranges[joint_type])
        
        return 0.5  # Default neutral score
    
    def _score_optimal_range(self, value: float, optimal_range: Tuple[float, float]) -> float:
        """
        Score a value against an optimal range (0-1).
        """
        min_val, max_val = optimal_range
        
        if min_val <= value <= max_val:
            return 1.0  # Perfect score
        elif value < min_val:
            # Below optimal range
            return max(0.0, 1.0 - (min_val - value) / min_val)
        else:
            # Above optimal range
            return max(0.0, 1.0 - (value - max_val) / max_val)
    
    def _generate_angle_recommendations(self, actual_value: float, optimal_range: Tuple[float, float], 
                                      joint_name: str) -> List[str]:
        """
        Generate recommendations based on angle analysis.
        """
        recommendations = []
        min_val, max_val = optimal_range
        
        if actual_value < min_val:
            difference = min_val - actual_value
            if joint_name == 'elbow':
                recommendations.append(f"Increase elbow extension by {difference:.1f}° for better power transfer")
            elif joint_name == 'torso':
                recommendations.append(f"Increase forward lean by {difference:.1f}° for better balance")
        elif actual_value > max_val:
            difference = actual_value - max_val
            if joint_name == 'elbow':
                recommendations.append(f"Reduce elbow extension by {difference:.1f}° for better control")
            elif joint_name == 'torso':
                recommendations.append(f"Reduce forward lean by {difference:.1f}° to avoid over-reaching")
        else:
            recommendations.append(f"Optimal {joint_name} angle - maintain current technique")
        
        return recommendations
    
    def _calculate_overall_score(self, arm_analysis: Dict, posture_analysis: Dict, 
                               rom_analysis: Dict, optimal_analysis: Dict) -> float:
        """
        Calculate overall joint angle quality score.
        """
        scores = []
        
        # Arm mechanics score
        if 'dominant_arm' in arm_analysis and 'mechanics_score' in arm_analysis['dominant_arm']:
            scores.append(arm_analysis['dominant_arm']['mechanics_score'])
        
        # Posture score
        if 'overall_posture_score' in posture_analysis:
            scores.append(posture_analysis['overall_posture_score'])
        
        # Optimality scores
        for analysis_key in optimal_analysis:
            if 'optimality_score' in optimal_analysis[analysis_key]:
                scores.append(optimal_analysis[analysis_key]['optimality_score'])
        
        return np.mean(scores) if scores else 0.5