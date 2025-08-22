"""
Body Posture Analyzer for Tennis Biomechanics

Analyzes body posture and balance parameters:
- Stance and weight distribution
- Body alignment and symmetry
- Balance and stability metrics
- Core engagement indicators
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class BodyPostureAnalyzer:
    """
    Specialized analysis of body posture and balance for tennis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Optimal posture parameters for tennis
        self.optimal_parameters = {
            'stance_width_ratio': (0.8, 1.2),    # Relative to shoulder width
            'weight_distribution': (0.4, 0.6),   # Left-right balance ratio
            'forward_lean': (5, 20),              # Degrees from vertical
            'knee_bend': (10, 30),                # Degrees of knee flexion
        }
    
    def analyze(self, landmarks_data: Dict, kinematics_data: Dict) -> Dict:
        """
        Comprehensive body posture analysis.
        """
        try:
            landmarks_sequence = landmarks_data['landmarks_sequence']
            angles_data = kinematics_data['joint_angles']
            
            # Analyze different aspects of posture
            stance_analysis = self._analyze_stance(landmarks_sequence, angles_data)
            balance_analysis = self._analyze_balance(landmarks_sequence)
            alignment_analysis = self._analyze_body_alignment(landmarks_sequence, angles_data)
            stability_analysis = self._analyze_stability(landmarks_sequence)
            
            return {
                'success': True,
                'stance_analysis': stance_analysis,
                'balance_analysis': balance_analysis,
                'alignment_analysis': alignment_analysis,
                'stability_analysis': stability_analysis,
                'overall_posture_score': self._calculate_overall_posture_score(
                    stance_analysis, balance_analysis, alignment_analysis, stability_analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"[BODY_POSTURE] Analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_stance(self, landmarks_sequence: List[Dict], angles_data: Dict) -> Dict:
        """
        Analyze stance characteristics throughout the movement.
        """
        stance_analysis = {}
        
        # Collect stance width data
        stance_widths = []
        shoulder_widths = []
        
        for frame_data in landmarks_sequence:
            landmarks = frame_data['landmarks']
            
            # Calculate stance width (distance between feet)
            stance_width = self._calculate_distance(
                landmarks.get('left_ankle'),
                landmarks.get('right_ankle')
            )
            
            # Calculate shoulder width for normalization
            shoulder_width = self._calculate_distance(
                landmarks.get('left_shoulder'),
                landmarks.get('right_shoulder')
            )
            
            if stance_width is not None and shoulder_width is not None:
                stance_widths.append(stance_width)
                shoulder_widths.append(shoulder_width)
        
        if stance_widths and shoulder_widths:
            # Normalize stance width by shoulder width
            stance_ratios = [sw / shw for sw, shw in zip(stance_widths, shoulder_widths) if shw > 0]
            
            stance_analysis['width_metrics'] = {
                'mean_stance_ratio': np.mean(stance_ratios),
                'stance_consistency': 1.0 - (np.std(stance_ratios) / np.mean(stance_ratios)) if np.mean(stance_ratios) > 0 else 0,
                'optimal_width_score': self._score_optimal_stance_width(np.mean(stance_ratios))
            }
        
        # Analyze stance width from angles data
        if 'stance_width' in angles_data and angles_data['stance_width']:
            width_values = [item['value'] for item in angles_data['stance_width']]
            
            stance_analysis['stability_metrics'] = {
                'width_stability': 1.0 - (np.std(width_values) / np.mean(width_values)) if np.mean(width_values) > 0 else 0,
                'width_range': np.max(width_values) - np.min(width_values),
                'adaptive_width': self._analyze_adaptive_stance(angles_data['stance_width'])
            }
        
        return stance_analysis
    
    def _analyze_balance(self, landmarks_sequence: List[Dict]) -> Dict:
        """
        Analyze balance and weight distribution.
        """
        balance_analysis = {}
        
        # Calculate center of mass approximation
        com_positions = []
        left_right_shifts = []
        forward_back_shifts = []
        
        for frame_data in landmarks_sequence:
            landmarks = frame_data['landmarks']
            
            # Approximate center of mass using hip midpoint
            left_hip = landmarks.get('left_hip')
            right_hip = landmarks.get('right_hip')
            
            if left_hip and right_hip:
                com_x = (left_hip['x'] + right_hip['x']) / 2
                com_y = (left_hip['y'] + right_hip['y']) / 2
                
                com_positions.append({'x': com_x, 'y': com_y})
                
                # Calculate left-right balance
                hip_width = abs(right_hip['x'] - left_hip['x'])
                if hip_width > 0:
                    left_right_ratio = (com_x - left_hip['x']) / hip_width
                    left_right_shifts.append(left_right_ratio)
        
        if com_positions:
            # Analyze center of mass movement
            com_x_positions = [pos['x'] for pos in com_positions]
            com_y_positions = [pos['y'] for pos in com_positions]
            
            balance_analysis['com_analysis'] = {
                'lateral_stability': 1.0 - (np.std(com_x_positions) / np.mean(com_x_positions)) if np.mean(com_x_positions) > 0 else 0,
                'anterior_posterior_stability': 1.0 - (np.std(com_y_positions) / np.mean(com_y_positions)) if np.mean(com_y_positions) > 0 else 0,
                'com_range': {
                    'x_range': np.max(com_x_positions) - np.min(com_x_positions),
                    'y_range': np.max(com_y_positions) - np.min(com_y_positions)
                }
            }
        
        if left_right_shifts:
            balance_analysis['weight_distribution'] = {
                'mean_balance_ratio': np.mean(left_right_shifts),
                'balance_consistency': 1.0 - np.std(left_right_shifts),
                'optimal_balance_score': self._score_optimal_balance(np.mean(left_right_shifts))
            }
        
        return balance_analysis
    
    def _analyze_body_alignment(self, landmarks_sequence: List[Dict], angles_data: Dict) -> Dict:
        """
        Analyze body alignment and symmetry.
        """
        alignment_analysis = {}
        
        # Analyze spinal alignment approximation
        spinal_alignments = []
        shoulder_alignments = []
        hip_alignments = []
        
        for frame_data in landmarks_sequence:
            landmarks = frame_data['landmarks']
            
            # Shoulder alignment
            left_shoulder = landmarks.get('left_shoulder')
            right_shoulder = landmarks.get('right_shoulder')
            if left_shoulder and right_shoulder:
                shoulder_angle = self._calculate_line_angle(left_shoulder, right_shoulder)
                shoulder_alignments.append(shoulder_angle)
            
            # Hip alignment
            left_hip = landmarks.get('left_hip')
            right_hip = landmarks.get('right_hip')
            if left_hip and right_hip:
                hip_angle = self._calculate_line_angle(left_hip, right_hip)
                hip_alignments.append(hip_angle)
            
            # Spinal alignment (shoulder to hip midpoint)
            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_midpoint = {
                    'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                    'y': (left_shoulder['y'] + right_shoulder['y']) / 2
                }
                hip_midpoint = {
                    'x': (left_hip['x'] + right_hip['x']) / 2,
                    'y': (left_hip['y'] + right_hip['y']) / 2
                }
                spinal_angle = self._calculate_line_angle(hip_midpoint, shoulder_midpoint)
                spinal_alignments.append(spinal_angle)
        
        # Analyze alignment stability
        if shoulder_alignments:
            alignment_analysis['shoulder_alignment'] = {
                'mean_angle': np.mean(shoulder_alignments),
                'stability': 1.0 - (np.std(shoulder_alignments) / 45.0),  # Normalized by 45°
                'alignment_quality': self._assess_alignment_quality(shoulder_alignments)
            }
        
        if hip_alignments:
            alignment_analysis['hip_alignment'] = {
                'mean_angle': np.mean(hip_alignments),
                'stability': 1.0 - (np.std(hip_alignments) / 45.0),
                'alignment_quality': self._assess_alignment_quality(hip_alignments)
            }
        
        if spinal_alignments:
            alignment_analysis['spinal_alignment'] = {
                'mean_inclination': np.mean(spinal_alignments),
                'inclination_stability': 1.0 - (np.std(spinal_alignments) / 30.0),  # Normalized by 30°
                'optimal_inclination_score': self._score_optimal_inclination(np.mean(spinal_alignments))
            }
        
        # Analyze symmetry
        alignment_analysis['symmetry'] = self._analyze_body_symmetry(landmarks_sequence)
        
        return alignment_analysis
    
    def _analyze_stability(self, landmarks_sequence: List[Dict]) -> Dict:
        """
        Analyze overall stability throughout the movement.
        """
        stability_analysis = {}
        
        # Calculate stability metrics for key landmarks
        key_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        for landmark_name in key_landmarks:
            positions = []
            
            for frame_data in landmarks_sequence:
                landmark = frame_data['landmarks'].get(landmark_name)
                if landmark:
                    positions.append([landmark['x'], landmark['y']])
            
            if len(positions) > 1:
                positions = np.array(positions)
                
                # Calculate movement variance (lower = more stable)
                x_stability = 1.0 - (np.std(positions[:, 0]) / np.mean(positions[:, 0])) if np.mean(positions[:, 0]) > 0 else 0
                y_stability = 1.0 - (np.std(positions[:, 1]) / np.mean(positions[:, 1])) if np.mean(positions[:, 1]) > 0 else 0
                
                stability_analysis[f'{landmark_name}_stability'] = {
                    'x_stability': max(0, x_stability),
                    'y_stability': max(0, y_stability),
                    'overall_stability': (max(0, x_stability) + max(0, y_stability)) / 2
                }
        
        # Calculate overall stability score
        stability_scores = []
        for landmark_stability in stability_analysis.values():
            if isinstance(landmark_stability, dict) and 'overall_stability' in landmark_stability:
                stability_scores.append(landmark_stability['overall_stability'])
        
        stability_analysis['overall_stability_score'] = np.mean(stability_scores) if stability_scores else 0.5
        
        return stability_analysis
    
    def _analyze_body_symmetry(self, landmarks_sequence: List[Dict]) -> Dict:
        """
        Analyze bilateral symmetry of body positioning.
        """
        symmetry_analysis = {}
        
        # Analyze arm symmetry
        left_arm_positions = []
        right_arm_positions = []
        
        for frame_data in landmarks_sequence:
            landmarks = frame_data['landmarks']
            
            left_wrist = landmarks.get('left_wrist')
            right_wrist = landmarks.get('right_wrist')
            left_shoulder = landmarks.get('left_shoulder')
            right_shoulder = landmarks.get('right_shoulder')
            
            if all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
                # Calculate relative positions (wrist relative to shoulder)
                left_rel_x = left_wrist['x'] - left_shoulder['x']
                left_rel_y = left_wrist['y'] - left_shoulder['y']
                right_rel_x = right_wrist['x'] - right_shoulder['x']
                right_rel_y = right_wrist['y'] - right_shoulder['y']
                
                left_arm_positions.append([left_rel_x, left_rel_y])
                right_arm_positions.append([right_rel_x, right_rel_y])
        
        if left_arm_positions and right_arm_positions:
            left_arm_positions = np.array(left_arm_positions)
            right_arm_positions = np.array(right_arm_positions)
            
            # Calculate symmetry (mirror the right arm and compare)
            mirrored_right = right_arm_positions.copy()
            mirrored_right[:, 0] *= -1  # Mirror X coordinate
            
            # Calculate similarity between left arm and mirrored right arm
            differences = np.linalg.norm(left_arm_positions - mirrored_right, axis=1)
            mean_difference = np.mean(differences)
            
            # Normalize by average arm reach
            arm_reaches = np.linalg.norm(left_arm_positions, axis=1)
            avg_reach = np.mean(arm_reaches) if len(arm_reaches) > 0 else 1
            
            symmetry_score = 1.0 - min(mean_difference / avg_reach, 1.0) if avg_reach > 0 else 0
            
            symmetry_analysis['arm_symmetry'] = {
                'symmetry_score': max(0, symmetry_score),
                'mean_asymmetry': mean_difference,
                'relative_asymmetry': mean_difference / avg_reach if avg_reach > 0 else 0
            }
        
        return symmetry_analysis
    
    def _calculate_distance(self, p1: Optional[Dict], p2: Optional[Dict]) -> Optional[float]:
        """Calculate Euclidean distance between two points."""
        if not all([p1, p2]):
            return None
        
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        return np.sqrt(dx*dx + dy*dy)
    
    def _calculate_line_angle(self, p1: Optional[Dict], p2: Optional[Dict]) -> Optional[float]:
        """Calculate angle of line from horizontal."""
        if not all([p1, p2]):
            return None
        
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        angle = np.arctan2(dy, dx)
        return np.degrees(angle)
    
    def _score_optimal_stance_width(self, stance_ratio: float) -> float:
        """Score stance width against optimal range."""
        optimal_min, optimal_max = self.optimal_parameters['stance_width_ratio']
        
        if optimal_min <= stance_ratio <= optimal_max:
            return 1.0
        elif stance_ratio < optimal_min:
            return max(0.0, stance_ratio / optimal_min)
        else:
            return max(0.0, 1.0 - (stance_ratio - optimal_max) / optimal_max)
    
    def _score_optimal_balance(self, balance_ratio: float) -> float:
        """Score balance against optimal range."""
        optimal_min, optimal_max = self.optimal_parameters['weight_distribution']
        
        if optimal_min <= balance_ratio <= optimal_max:
            return 1.0
        elif balance_ratio < optimal_min:
            return max(0.0, balance_ratio / optimal_min)
        else:
            return max(0.0, 1.0 - (balance_ratio - optimal_max) / optimal_max)
    
    def _score_optimal_inclination(self, inclination: float) -> float:
        """Score spinal inclination against optimal range."""
        # Convert to absolute inclination from vertical
        abs_inclination = abs(90 - abs(inclination))
        
        optimal_min, optimal_max = self.optimal_parameters['forward_lean']
        
        if optimal_min <= abs_inclination <= optimal_max:
            return 1.0
        elif abs_inclination < optimal_min:
            return max(0.0, abs_inclination / optimal_min)
        else:
            return max(0.0, 1.0 - (abs_inclination - optimal_max) / optimal_max)
    
    def _analyze_adaptive_stance(self, stance_data: List[Dict]) -> Dict:
        """Analyze how stance adapts during movement."""
        if len(stance_data) < 3:
            return {'adaptability': 0.5}
        
        values = [item['value'] for item in stance_data]
        times = [item['time'] for item in stance_data]
        
        # Calculate rate of change
        changes = np.abs(np.diff(values))
        time_diffs = np.diff(times)
        
        rates = [change / dt for change, dt in zip(changes, time_diffs) if dt > 0]
        
        return {
            'adaptability': min(np.mean(rates) / 50.0, 1.0) if rates else 0.5,  # Normalized
            'max_adaptation_rate': np.max(rates) if rates else 0,
            'adaptation_consistency': 1.0 - (np.std(rates) / np.mean(rates)) if rates and np.mean(rates) > 0 else 0
        }
    
    def _assess_alignment_quality(self, angles: List[float]) -> float:
        """Assess quality of body alignment."""
        if len(angles) < 2:
            return 0.5
        
        # Good alignment should be consistent and near horizontal (0°)
        mean_angle = np.mean(angles)
        angle_consistency = 1.0 - (np.std(angles) / 45.0)  # Normalized by 45°
        
        # Penalty for being too far from horizontal
        horizontal_penalty = 1.0 - (abs(mean_angle) / 45.0)
        
        return (angle_consistency + horizontal_penalty) / 2
    
    def _calculate_overall_posture_score(self, stance_analysis: Dict, balance_analysis: Dict,
                                       alignment_analysis: Dict, stability_analysis: Dict) -> float:
        """Calculate overall posture quality score."""
        scores = []
        
        # Stance scores
        if 'width_metrics' in stance_analysis and 'optimal_width_score' in stance_analysis['width_metrics']:
            scores.append(stance_analysis['width_metrics']['optimal_width_score'])
        
        # Balance scores
        if 'weight_distribution' in balance_analysis and 'optimal_balance_score' in balance_analysis['weight_distribution']:
            scores.append(balance_analysis['weight_distribution']['optimal_balance_score'])
        
        # Alignment scores
        if 'spinal_alignment' in alignment_analysis and 'optimal_inclination_score' in alignment_analysis['spinal_alignment']:
            scores.append(alignment_analysis['spinal_alignment']['optimal_inclination_score'])
        
        # Stability score
        if 'overall_stability_score' in stability_analysis:
            scores.append(stability_analysis['overall_stability_score'])
        
        return np.mean(scores) if scores else 0.5