"""
Optimized MediaPipe Landmarks Extractor for Tennis Biomechanics

Extracts detailed pose landmarks with focus on tennis-relevant body parts:
- Hand/wrist (racket position)
- Shoulder, elbow (arm kinematics)
- Hip, torso (body rotation)
- Feet (stance and balance)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class LandmarksExtractor:
    """
    Optimized landmark extraction specifically for tennis analysis.
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher accuracy for biomechanics
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.logger = logging.getLogger(__name__)
        
        # Tennis-relevant landmark indices
        self.tennis_landmarks = {
            # Hand and wrist (racket control)
            'right_wrist': 16,
            'left_wrist': 15,
            'right_pinky': 18,
            'left_pinky': 17,
            'right_index': 20,
            'left_index': 19,
            'right_thumb': 22,
            'left_thumb': 21,
            
            # Arm kinematics
            'right_shoulder': 12,
            'left_shoulder': 11,
            'right_elbow': 14,
            'left_elbow': 13,
            
            # Torso and rotation
            'nose': 0,
            'right_eye': 2,
            'left_eye': 5,
            'right_hip': 24,
            'left_hip': 23,
            
            # Leg stability
            'right_knee': 26,
            'left_knee': 25,
            'right_ankle': 28,
            'left_ankle': 27,
            'right_foot_index': 32,
            'left_foot_index': 31
        }
    
    def extract_from_video(self, video_path: str) -> Dict:
        """
        Extract landmarks from entire video with quality validation.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"[LANDMARKS] Processing {total_frames} frames at {fps} FPS")
            
            landmarks_sequence = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract landmarks for this frame
                frame_landmarks = self._extract_frame_landmarks(frame, frame_count)
                if frame_landmarks is not None:
                    landmarks_sequence.append(frame_landmarks)
                
                frame_count += 1
            
            cap.release()
            
            if not landmarks_sequence:
                return {'success': False, 'error': 'No valid landmarks detected'}
            
            # Post-process and validate
            processed_landmarks = self._post_process_landmarks(landmarks_sequence)
            
            return {
                'success': True,
                'total_frames': total_frames,
                'valid_frames': len(landmarks_sequence),
                'fps': fps,
                'landmarks_sequence': processed_landmarks,
                'video_duration': total_frames / fps,
                'quality_score': len(landmarks_sequence) / total_frames
            }
            
        except Exception as e:
            self.logger.error(f"[LANDMARKS] Error processing video: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_frame_landmarks(self, frame: np.ndarray, frame_idx: int) -> Optional[Dict]:
        """
        Extract landmarks from a single frame with quality validation.
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Extract relevant landmarks
            landmarks = {}
            frame_height, frame_width = frame.shape[:2]
            
            for name, idx in self.tennis_landmarks.items():
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * frame_width
                    y = landmark.y * frame_height
                    z = landmark.z  # Relative depth (not absolute)
                    visibility = landmark.visibility
                    
                    landmarks[name] = {
                        'x': float(x),
                        'y': float(y), 
                        'z': float(z),
                        'visibility': float(visibility)
                    }
            
            # Validate landmark quality
            quality_score = self._calculate_frame_quality(landmarks)
            
            if quality_score < 0.5:  # Minimum quality threshold
                return None
            
            return {
                'frame_idx': frame_idx,
                'landmarks': landmarks,
                'quality_score': quality_score,
                'frame_size': (frame_width, frame_height)
            }
            
        except Exception as e:
            self.logger.warning(f"[LANDMARKS] Error in frame {frame_idx}: {e}")
            return None
    
    def _calculate_frame_quality(self, landmarks: Dict) -> float:
        """
        Calculate quality score for landmark detection in this frame.
        """
        if not landmarks:
            return 0.0
        
        # Key landmarks for tennis analysis
        key_landmarks = ['right_wrist', 'left_wrist', 'right_shoulder', 'left_shoulder', 
                        'right_elbow', 'left_elbow', 'right_hip', 'left_hip']
        
        total_visibility = 0.0
        valid_landmarks = 0
        
        for landmark_name in key_landmarks:
            if landmark_name in landmarks:
                visibility = landmarks[landmark_name]['visibility']
                if visibility > 0.5:  # Visible landmark
                    total_visibility += visibility
                    valid_landmarks += 1
        
        if valid_landmarks == 0:
            return 0.0
        
        # Quality based on visibility and completeness
        visibility_score = total_visibility / len(key_landmarks)
        completeness_score = valid_landmarks / len(key_landmarks)
        
        return (visibility_score + completeness_score) / 2
    
    def _post_process_landmarks(self, landmarks_sequence: List[Dict]) -> List[Dict]:
        """
        Post-process landmarks sequence for smoothing and interpolation.
        """
        if len(landmarks_sequence) < 3:
            return landmarks_sequence
        
        # Apply temporal smoothing to reduce noise
        smoothed_sequence = []
        
        for i, frame_data in enumerate(landmarks_sequence):
            if i == 0 or i == len(landmarks_sequence) - 1:
                # Keep first and last frames as-is
                smoothed_sequence.append(frame_data)
                continue
            
            # Apply 3-point moving average smoothing
            prev_frame = landmarks_sequence[i-1]
            curr_frame = frame_data
            next_frame = landmarks_sequence[i+1]
            
            smoothed_landmarks = {}
            
            for landmark_name in curr_frame['landmarks']:
                if (landmark_name in prev_frame['landmarks'] and 
                    landmark_name in next_frame['landmarks']):
                    
                    # Smooth x, y coordinates
                    prev_coords = prev_frame['landmarks'][landmark_name]
                    curr_coords = curr_frame['landmarks'][landmark_name]
                    next_coords = next_frame['landmarks'][landmark_name]
                    
                    smoothed_landmarks[landmark_name] = {
                        'x': (prev_coords['x'] + curr_coords['x'] + next_coords['x']) / 3,
                        'y': (prev_coords['y'] + curr_coords['y'] + next_coords['y']) / 3,
                        'z': curr_coords['z'],  # Keep original z
                        'visibility': curr_coords['visibility']
                    }
                else:
                    # Keep original if neighbors not available
                    smoothed_landmarks[landmark_name] = curr_frame['landmarks'][landmark_name]
            
            smoothed_frame = frame_data.copy()
            smoothed_frame['landmarks'] = smoothed_landmarks
            smoothed_sequence.append(smoothed_frame)
        
        self.logger.info(f"[LANDMARKS] Smoothed {len(smoothed_sequence)} frames")
        return smoothed_sequence
    
    def get_landmark_at_frame(self, landmarks_sequence: List[Dict], frame_idx: int, 
                             landmark_name: str) -> Optional[Dict]:
        """
        Get specific landmark at specific frame with interpolation if needed.
        """
        # Find exact frame
        for frame_data in landmarks_sequence:
            if frame_data['frame_idx'] == frame_idx:
                if landmark_name in frame_data['landmarks']:
                    return frame_data['landmarks'][landmark_name]
                return None
        
        # If exact frame not found, try interpolation
        return self._interpolate_landmark(landmarks_sequence, frame_idx, landmark_name)
    
    def _interpolate_landmark(self, landmarks_sequence: List[Dict], target_frame: int, 
                             landmark_name: str) -> Optional[Dict]:
        """
        Interpolate landmark position for missing frame.
        """
        # Find surrounding frames
        before_frame = None
        after_frame = None
        
        for frame_data in landmarks_sequence:
            frame_idx = frame_data['frame_idx']
            if frame_idx < target_frame and landmark_name in frame_data['landmarks']:
                before_frame = frame_data
            elif frame_idx > target_frame and landmark_name in frame_data['landmarks']:
                after_frame = frame_data
                break
        
        if before_frame is None or after_frame is None:
            return None
        
        # Linear interpolation
        before_landmark = before_frame['landmarks'][landmark_name]
        after_landmark = after_frame['landmarks'][landmark_name]
        
        t = (target_frame - before_frame['frame_idx']) / (after_frame['frame_idx'] - before_frame['frame_idx'])
        
        return {
            'x': before_landmark['x'] + t * (after_landmark['x'] - before_landmark['x']),
            'y': before_landmark['y'] + t * (after_landmark['y'] - before_landmark['y']),
            'z': before_landmark['z'] + t * (after_landmark['z'] - before_landmark['z']),
            'visibility': min(before_landmark['visibility'], after_landmark['visibility'])
        }