"""
Kinematics Calculator for Tennis Biomechanics

Calculates real biomechanical parameters from 2D landmark data:
- Angular measurements (joint angles, body rotations)
- Linear velocities and accelerations  
- Temporal analysis (timing, phases)
- Trajectory analysis

Focus on parameters that are meaningful in 2D analysis.
"""

import numpy as np
import scipy.signal
from typing import List, Dict, Tuple, Optional
import logging
import math

class KinematicsCalculator:
    """
    Calculate kinematic parameters from landmark sequences.
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps  # Time step between frames
        self.logger = logging.getLogger(__name__)
    
    def calculate_all(self, landmarks_data: Dict) -> Dict:
        """
        Calculate all kinematic parameters from landmarks sequence.
        """
        if not landmarks_data['success']:
            return {'success': False, 'error': 'Invalid landmarks data'}
        
        landmarks_sequence = landmarks_data['landmarks_sequence']
        self.fps = landmarks_data['fps']
        self.dt = 1.0 / self.fps
        
        try:
            # Calculate different types of kinematics
            angles = self._calculate_joint_angles(landmarks_sequence)
            velocities = self._calculate_velocities(landmarks_sequence)
            accelerations = self._calculate_accelerations(velocities)
            trajectories = self._calculate_trajectories(landmarks_sequence)
            
            return {
                'success': True,
                'fps': self.fps,
                'frame_count': len(landmarks_sequence),
                'duration': len(landmarks_sequence) / self.fps,
                'joint_angles': angles,
                'velocities': velocities,
                'accelerations': accelerations,
                'trajectories': trajectories
            }
            
        except Exception as e:
            self.logger.error(f"[KINEMATICS] Error calculating kinematics: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_joint_angles(self, landmarks_sequence: List[Dict]) -> Dict:
        """
        Calculate joint angles throughout the movement.
        """
        angles_data = {
            'right_arm_angles': [],  # Shoulder-elbow-wrist
            'left_arm_angles': [],
            'torso_inclination': [],  # Body inclination
            'shoulder_rotation': [],  # Shoulder plane rotation
            'hip_rotation': [],      # Hip plane rotation
            'stance_width': []       # Distance between feet
        }
        
        for frame_data in landmarks_sequence:
            landmarks = frame_data['landmarks']
            frame_angles = {}
            
            # Right arm angle (shoulder-elbow-wrist)
            right_arm_angle = self._calculate_three_point_angle(
                landmarks.get('right_shoulder'),
                landmarks.get('right_elbow'), 
                landmarks.get('right_wrist')
            )
            frame_angles['right_arm_angle'] = right_arm_angle
            
            # Left arm angle
            left_arm_angle = self._calculate_three_point_angle(
                landmarks.get('left_shoulder'),
                landmarks.get('left_elbow'),
                landmarks.get('left_wrist')
            )
            frame_angles['left_arm_angle'] = left_arm_angle
            
            # Torso inclination (shoulder line vs vertical)
            torso_inclination = self._calculate_torso_inclination(
                landmarks.get('left_shoulder'),
                landmarks.get('right_shoulder')
            )
            frame_angles['torso_inclination'] = torso_inclination
            
            # Shoulder rotation (shoulder line angle)
            shoulder_rotation = self._calculate_line_angle(
                landmarks.get('left_shoulder'),
                landmarks.get('right_shoulder')
            )
            frame_angles['shoulder_rotation'] = shoulder_rotation
            
            # Hip rotation
            hip_rotation = self._calculate_line_angle(
                landmarks.get('left_hip'),
                landmarks.get('right_hip')
            )
            frame_angles['hip_rotation'] = hip_rotation
            
            # Stance width
            stance_width = self._calculate_distance(
                landmarks.get('left_ankle'),
                landmarks.get('right_ankle')
            )
            frame_angles['stance_width'] = stance_width
            
            # Add to sequences
            for key, value in frame_angles.items():
                if value is not None:
                    angles_data[key.replace('_angle', '_angles').replace('width', 'width')].append({
                        'frame': frame_data['frame_idx'],
                        'value': value,
                        'time': frame_data['frame_idx'] / self.fps
                    })
        
        # Calculate angle statistics
        angle_types = list(angles_data.keys())  # Create a copy of keys
        for angle_type in angle_types:
            if angles_data[angle_type] and isinstance(angles_data[angle_type], list):
                values = [item['value'] for item in angles_data[angle_type]]
                if values:  # Only calculate stats if we have values
                    angles_data[f'{angle_type}_stats'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values)
                    }
        
        return angles_data
    
    def _calculate_velocities(self, landmarks_sequence: List[Dict]) -> Dict:
        """
        Calculate velocities for key landmarks.
        """
        velocity_data = {}
        
        # Key landmarks for velocity analysis
        key_landmarks = ['right_wrist', 'left_wrist', 'right_elbow', 'left_elbow',
                        'right_shoulder', 'left_shoulder']
        
        for landmark_name in key_landmarks:
            positions = []
            times = []
            
            # Extract position data
            for frame_data in landmarks_sequence:
                if landmark_name in frame_data['landmarks']:
                    landmark = frame_data['landmarks'][landmark_name]
                    positions.append([landmark['x'], landmark['y']])
                    times.append(frame_data['frame_idx'] / self.fps)
            
            if len(positions) < 2:
                continue
            
            positions = np.array(positions)
            times = np.array(times)
            
            # Calculate velocities using central difference
            velocities = []
            for i in range(len(positions)):
                if i == 0:
                    # Forward difference for first point
                    vel = (positions[i+1] - positions[i]) / (times[i+1] - times[i])
                elif i == len(positions) - 1:
                    # Backward difference for last point
                    vel = (positions[i] - positions[i-1]) / (times[i] - times[i-1])
                else:
                    # Central difference for middle points
                    vel = (positions[i+1] - positions[i-1]) / (times[i+1] - times[i-1])
                
                # Calculate magnitude
                vel_magnitude = np.linalg.norm(vel)
                velocities.append({
                    'frame': landmarks_sequence[i]['frame_idx'],
                    'time': times[i],
                    'vx': vel[0],
                    'vy': vel[1], 
                    'magnitude': vel_magnitude
                })
            
            velocity_data[f'{landmark_name}_velocity'] = velocities
            
            # Calculate velocity statistics
            magnitudes = [v['magnitude'] for v in velocities]
            velocity_data[f'{landmark_name}_velocity_stats'] = {
                'mean_speed': np.mean(magnitudes),
                'max_speed': np.max(magnitudes),
                'std_speed': np.std(magnitudes)
            }
        
        return velocity_data
    
    def _calculate_accelerations(self, velocity_data: Dict) -> Dict:
        """
        Calculate accelerations from velocity data.
        """
        acceleration_data = {}
        
        for vel_key in velocity_data:
            if not vel_key.endswith('_velocity') or vel_key.endswith('_stats'):
                continue
            
            velocities = velocity_data[vel_key]
            if len(velocities) < 2:
                continue
            
            accelerations = []
            landmark_name = vel_key.replace('_velocity', '')
            
            for i in range(len(velocities)):
                if i == 0:
                    # Forward difference
                    if i + 1 < len(velocities):
                        dt = velocities[i+1]['time'] - velocities[i]['time']
                        ax = (velocities[i+1]['vx'] - velocities[i]['vx']) / dt
                        ay = (velocities[i+1]['vy'] - velocities[i]['vy']) / dt
                    else:
                        ax = ay = 0
                elif i == len(velocities) - 1:
                    # Backward difference
                    dt = velocities[i]['time'] - velocities[i-1]['time']
                    ax = (velocities[i]['vx'] - velocities[i-1]['vx']) / dt
                    ay = (velocities[i]['vy'] - velocities[i-1]['vy']) / dt
                else:
                    # Central difference
                    dt = velocities[i+1]['time'] - velocities[i-1]['time']
                    ax = (velocities[i+1]['vx'] - velocities[i-1]['vx']) / dt
                    ay = (velocities[i+1]['vy'] - velocities[i-1]['vy']) / dt
                
                acc_magnitude = np.sqrt(ax*ax + ay*ay)
                accelerations.append({
                    'frame': velocities[i]['frame'],
                    'time': velocities[i]['time'],
                    'ax': ax,
                    'ay': ay,
                    'magnitude': acc_magnitude
                })
            
            acceleration_data[f'{landmark_name}_acceleration'] = accelerations
            
            # Calculate acceleration statistics
            magnitudes = [a['magnitude'] for a in accelerations]
            acceleration_data[f'{landmark_name}_acceleration_stats'] = {
                'mean_acceleration': np.mean(magnitudes),
                'max_acceleration': np.max(magnitudes),
                'std_acceleration': np.std(magnitudes)
            }
        
        return acceleration_data
    
    def _calculate_trajectories(self, landmarks_sequence: List[Dict]) -> Dict:
        """
        Calculate trajectory parameters for key landmarks.
        """
        trajectory_data = {}
        
        # Focus on hand trajectories (racket movement)
        key_landmarks = ['right_wrist', 'left_wrist']
        
        for landmark_name in key_landmarks:
            positions = []
            
            for frame_data in landmarks_sequence:
                if landmark_name in frame_data['landmarks']:
                    landmark = frame_data['landmarks'][landmark_name]
                    positions.append([landmark['x'], landmark['y']])
            
            if len(positions) < 3:
                continue
            
            positions = np.array(positions)
            
            # Calculate trajectory metrics
            total_distance = 0
            for i in range(1, len(positions)):
                total_distance += np.linalg.norm(positions[i] - positions[i-1])
            
            # Straight-line distance (start to end)
            straight_distance = np.linalg.norm(positions[-1] - positions[0])
            
            # Trajectory efficiency (straightness)
            efficiency = straight_distance / total_distance if total_distance > 0 else 0
            
            # Calculate trajectory smoothness (jerk analysis)
            smoothness = self._calculate_trajectory_smoothness(positions)
            
            # Bounding box (movement range)
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            movement_range = {
                'x_range': np.max(x_coords) - np.min(x_coords),
                'y_range': np.max(y_coords) - np.min(y_coords),
                'total_range': np.sqrt((np.max(x_coords) - np.min(x_coords))**2 + 
                                     (np.max(y_coords) - np.min(y_coords))**2)
            }
            
            trajectory_data[f'{landmark_name}_trajectory'] = {
                'total_distance': total_distance,
                'straight_distance': straight_distance,
                'efficiency': efficiency,
                'smoothness': smoothness,
                'movement_range': movement_range,
                'path_points': positions.tolist()
            }
        
        return trajectory_data
    
    # Utility functions
    def _calculate_three_point_angle(self, p1: Optional[Dict], p2: Optional[Dict], 
                                   p3: Optional[Dict]) -> Optional[float]:
        """Calculate angle formed by three points (p1-p2-p3)."""
        if not all([p1, p2, p3]):
            return None
        
        # Vectors from p2 to p1 and p2 to p3
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_line_angle(self, p1: Optional[Dict], p2: Optional[Dict]) -> Optional[float]:
        """Calculate angle of line from horizontal."""
        if not all([p1, p2]):
            return None
        
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        
        angle = np.arctan2(dy, dx)
        return np.degrees(angle)
    
    def _calculate_torso_inclination(self, left_shoulder: Optional[Dict], 
                                   right_shoulder: Optional[Dict]) -> Optional[float]:
        """Calculate torso inclination from vertical."""
        if not all([left_shoulder, right_shoulder]):
            return None
        
        # Shoulder line vector
        dx = right_shoulder['x'] - left_shoulder['x']
        dy = right_shoulder['y'] - left_shoulder['y']
        
        # Angle from horizontal
        horizontal_angle = np.arctan2(dy, dx)
        
        # Convert to inclination from vertical (90° - horizontal_angle)
        inclination = np.pi/2 - abs(horizontal_angle)
        
        return np.degrees(inclination)
    
    def _calculate_distance(self, p1: Optional[Dict], p2: Optional[Dict]) -> Optional[float]:
        """Calculate Euclidean distance between two points."""
        if not all([p1, p2]):
            return None
        
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        
        return np.sqrt(dx*dx + dy*dy)
    
    def _calculate_trajectory_smoothness(self, positions: np.ndarray) -> float:
        """
        Calculate trajectory smoothness using jerk analysis.
        Lower values indicate smoother movement.
        """
        if len(positions) < 3:
            return 0.0
        
        # Calculate first and second derivatives (velocity and acceleration)
        velocity = np.diff(positions, axis=0)
        acceleration = np.diff(velocity, axis=0)
        
        # Jerk is the derivative of acceleration
        if len(acceleration) < 2:
            return 0.0
        
        jerk = np.diff(acceleration, axis=0)
        
        # Calculate mean squared jerk as smoothness metric
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        mean_squared_jerk = np.mean(jerk_magnitude**2)
        
        # Convert to smoothness score (lower jerk = higher smoothness)
        smoothness = 1.0 / (1.0 + mean_squared_jerk / 1000.0)
        
        return smoothness