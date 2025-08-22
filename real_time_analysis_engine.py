#!/usr/bin/env python3
"""
Real-Time Analysis Engine for TableTennisAnalyzer
Provides real-time video capture and analysis capabilities
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
from queue import Queue

# Import our existing system
from tennis_comparison_backend import TableTennisAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeAnalysisEngine:
    """Real-time video analysis engine for table tennis techniques"""
    
    def __init__(self, camera_index: int = 0):
        """Initialize the real-time analysis engine"""
        self.camera_index = camera_index
        self.analyzer = TableTennisAnalyzer()
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize pose and hand detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis state
        self.is_recording = False
        self.frames_buffer = []
        self.analysis_results = {}
        self.stroke_detected = False
        self.stroke_start_time = None
        
        # Real-time metrics
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # Analysis queue for background processing
        self.analysis_queue = Queue()
        self.analysis_thread = None
        self.stop_analysis = False
        
    def start_camera(self) -> bool:
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera {self.camera_index} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera stopped")
    
    def detect_pose_and_hands(self, frame: np.ndarray) -> Dict:
        """Detect pose and hand landmarks in frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        return {
            'pose': pose_results,
            'hands': hand_results,
            'timestamp': time.time()
        }
    
    def analyze_stroke_motion(self, landmarks_data: List[Dict]) -> Dict:
        """Analyze stroke motion from landmark data"""
        if len(landmarks_data) < 10:
            return {'stroke_detected': False, 'reason': 'insufficient_data'}
        
        # Extract key points for analysis
        wrist_positions = []
        shoulder_positions = []
        
        for data in landmarks_data:
            if data['pose'] and data['pose'].pose_landmarks:
                landmarks = data['pose'].pose_landmarks.landmark
                
                # Right wrist (landmark 16) and shoulder (landmark 12)
                wrist = landmarks[16]
                shoulder = landmarks[12]
                
                wrist_positions.append([wrist.x, wrist.y, wrist.z])
                shoulder_positions.append([shoulder.x, shoulder.y, shoulder.z])
        
        if len(wrist_positions) < 5:
            return {'stroke_detected': False, 'reason': 'no_pose_data'}
        
        # Calculate motion metrics
        wrist_velocities = self._calculate_velocities(wrist_positions)
        motion_intensity = np.mean([np.linalg.norm(v) for v in wrist_velocities])
        
        # Stroke detection thresholds
        stroke_detected = motion_intensity > 0.1
        
        if stroke_detected:
            # Classify stroke type based on motion pattern
            stroke_type = self._classify_stroke_type(wrist_positions, shoulder_positions)
            
            return {
                'stroke_detected': True,
                'stroke_type': stroke_type,
                'motion_intensity': float(motion_intensity),
                'duration': len(landmarks_data) / 30.0,  # Assuming 30 FPS
                'quality_score': min(motion_intensity * 100, 100)
            }
        
        return {'stroke_detected': False, 'reason': 'low_motion_intensity'}
    
    def _calculate_velocities(self, positions: List[List[float]]) -> List[np.ndarray]:
        """Calculate velocities from position data"""
        velocities = []
        for i in range(1, len(positions)):
            vel = np.array(positions[i]) - np.array(positions[i-1])
            velocities.append(vel)
        return velocities
    
    def _classify_stroke_type(self, wrist_pos: List, shoulder_pos: List) -> str:
        """Classify stroke type based on motion pattern"""
        # Simple classification based on relative motion
        if len(wrist_pos) < 2:
            return 'unknown'
        
        start_wrist = np.array(wrist_pos[0])
        end_wrist = np.array(wrist_pos[-1])
        
        # Determine if it's forehand or backhand based on x-direction
        x_motion = end_wrist[0] - start_wrist[0]
        
        if x_motion > 0.1:
            return 'forehand_drive'
        elif x_motion < -0.1:
            return 'backhand_drive'
        else:
            return 'neutral'
    
    def draw_analysis_overlay(self, frame: np.ndarray, landmarks_data: Dict, analysis: Dict) -> np.ndarray:
        """Draw analysis overlay on frame"""
        overlay = frame.copy()
        
        # Draw pose landmarks
        if landmarks_data['pose'] and landmarks_data['pose'].pose_landmarks:
            self.mp_draw.draw_landmarks(
                overlay,
                landmarks_data['pose'].pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # Draw hand landmarks
        if landmarks_data['hands'] and landmarks_data['hands'].multi_hand_landmarks:
            for hand_landmarks in landmarks_data['hands'].multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    overlay,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        # Draw analysis info
        self._draw_info_panel(overlay, analysis)
        
        return overlay
    
    def _draw_info_panel(self, frame: np.ndarray, analysis: Dict):
        """Draw information panel on frame"""
        height, width = frame.shape[:2]
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "Table Tennis Analysis", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recording status
        status_color = (0, 0, 255) if self.is_recording else (128, 128, 128)
        status_text = "RECORDING" if self.is_recording else "READY"
        cv2.putText(frame, f"Status: {status_text}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Stroke detection
        if 'stroke_detected' in analysis and analysis['stroke_detected']:
            cv2.putText(frame, f"Stroke: {analysis.get('stroke_type', 'unknown')}", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Quality: {analysis.get('quality_score', 0):.1f}%", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Instructions
        cv2.putText(frame, "Controls:", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, "SPACE - Start/Stop Recording", (20, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "A - Analyze Last Recording", (20, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Q - Quit", (20, 195), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
    
    def start_recording(self):
        """Start recording frames for analysis"""
        self.is_recording = True
        self.frames_buffer = []
        self.stroke_start_time = time.time()
        logger.info("Started recording")
    
    def stop_recording(self):
        """Stop recording frames"""
        self.is_recording = False
        logger.info(f"Stopped recording - captured {len(self.frames_buffer)} frames")
    
    def analyze_recorded_sequence(self) -> Dict:
        """Analyze the recorded frame sequence"""
        if len(self.frames_buffer) < 10:
            return {'success': False, 'error': 'insufficient_frames'}
        
        logger.info(f"Analyzing {len(self.frames_buffer)} recorded frames...")
        
        # Extract landmarks from all frames
        landmarks_sequence = []
        for frame_data in self.frames_buffer:
            if 'landmarks' in frame_data:
                landmarks_sequence.append(frame_data['landmarks'])
        
        if not landmarks_sequence:
            return {'success': False, 'error': 'no_landmark_data'}
        
        # Analyze stroke motion
        stroke_analysis = self.analyze_stroke_motion(landmarks_sequence)
        
        # Create analysis result
        result = {
            'success': True,
            'timestamp': time.time(),
            'frames_analyzed': len(self.frames_buffer),
            'duration': len(self.frames_buffer) / 30.0,
            'stroke_analysis': stroke_analysis
        }
        
        # If stroke detected, add recommendations
        if stroke_analysis.get('stroke_detected'):
            result['recommendations'] = self._generate_recommendations(stroke_analysis)
        
        self.analysis_results = result
        return result
    
    def _generate_recommendations(self, stroke_analysis: Dict) -> List[str]:
        """Generate technique recommendations based on analysis"""
        recommendations = []
        
        quality_score = stroke_analysis.get('quality_score', 0)
        stroke_type = stroke_analysis.get('stroke_type', 'unknown')
        
        if quality_score < 70:
            recommendations.append("Focus on smoother racket motion")
            recommendations.append("Practice consistent follow-through")
        
        if stroke_type == 'forehand_drive':
            recommendations.append("Ensure full shoulder rotation in forehand")
        elif stroke_type == 'backhand_drive':
            recommendations.append("Keep elbow close to body in backhand")
        
        if not recommendations:
            recommendations.append("Good technique! Keep practicing for consistency")
        
        return recommendations
    
    def run_real_time_analysis(self):
        """Main real-time analysis loop"""
        if not self.start_camera():
            return False
        
        logger.info("Starting real-time analysis...")
        logger.info("Press SPACE to start/stop recording, A to analyze, Q to quit")
        
        landmarks_buffer = []
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Update FPS
                self.update_fps()
                
                # Detect landmarks
                landmarks_data = self.detect_pose_and_hands(frame)
                landmarks_buffer.append(landmarks_data)
                
                # Keep buffer size manageable
                if len(landmarks_buffer) > 60:  # 2 seconds at 30 FPS
                    landmarks_buffer.pop(0)
                
                # If recording, save frame data
                if self.is_recording:
                    self.frames_buffer.append({
                        'frame': frame.copy(),
                        'landmarks': landmarks_data,
                        'timestamp': time.time()
                    })
                
                # Analyze recent motion for real-time feedback
                if len(landmarks_buffer) >= 10:
                    recent_analysis = self.analyze_stroke_motion(landmarks_buffer[-10:])
                else:
                    recent_analysis = {'stroke_detected': False}
                
                # Draw overlay
                display_frame = self.draw_analysis_overlay(frame, landmarks_data, recent_analysis)
                
                # Show frame
                cv2.imshow('Table Tennis Real-Time Analysis', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space bar
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('a') and not self.is_recording:
                    if self.frames_buffer:
                        analysis_result = self.analyze_recorded_sequence()
                        self._display_analysis_result(analysis_result)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
        finally:
            self.stop_camera()
        
        return True
    
    def _display_analysis_result(self, result: Dict):
        """Display analysis result in console"""
        print("\n" + "="*60)
        print("ANALYSIS RESULT")
        print("="*60)
        
        if result['success']:
            print(f"Frames analyzed: {result['frames_analyzed']}")
            print(f"Duration: {result['duration']:.2f} seconds")
            
            stroke_analysis = result['stroke_analysis']
            if stroke_analysis['stroke_detected']:
                print(f"Stroke detected: {stroke_analysis['stroke_type']}")
                print(f"Quality score: {stroke_analysis.get('quality_score', 0):.1f}%")
                print(f"Motion intensity: {stroke_analysis.get('motion_intensity', 0):.3f}")
                
                if 'recommendations' in result:
                    print("\nRecommendations:")
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"  {i}. {rec}")
            else:
                print("No clear stroke detected")
                print(f"Reason: {stroke_analysis.get('reason', 'unknown')}")
        else:
            print(f"Analysis failed: {result.get('error', 'unknown_error')}")
        
        print("="*60)
    
    def save_analysis_session(self, filename: str = None):
        """Save analysis session data"""
        if not filename:
            timestamp = int(time.time())
            filename = f"analysis_session_{timestamp}.json"
        
        session_data = {
            'timestamp': time.time(),
            'analysis_results': self.analysis_results,
            'session_info': {
                'camera_index': self.camera_index,
                'total_frames': len(self.frames_buffer)
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving session: {e}")


def main():
    """Main function to run the real-time analysis engine"""
    print("Table Tennis Real-Time Analysis Engine")
    print("=====================================")
    
    # Initialize engine
    engine = RealTimeAnalysisEngine()
    
    # Run analysis
    success = engine.run_real_time_analysis()
    
    if success:
        print("Analysis session completed successfully")
        
        # Offer to save session
        if engine.analysis_results:
            save_session = input("Save analysis session? (y/n): ").lower().strip()
            if save_session == 'y':
                engine.save_analysis_session()
    else:
        print("Analysis session failed to start")


if __name__ == "__main__":
    main()