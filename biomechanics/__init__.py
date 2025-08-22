"""
Biomechanics Module - Modular 2D Tennis Analysis

Provides scientific biomechanical analysis for tennis movements using MediaPipe landmarks.
Focus on 2D intelligent analysis with real parameters instead of subjective estimates.

Main Components:
- landmarks_extractor: Optimized MediaPipe landmark extraction
- kinematics_calculator: Angles, velocities, accelerations
- joint_angles: Articular angle analysis  
- racket_trajectory: Racket movement analysis
- body_posture: Posture and balance analysis
"""

from .core.landmarks_extractor import LandmarksExtractor
from .core.kinematics_calculator import KinematicsCalculator
from .analyzers.joint_angles import JointAnglesAnalyzer
from .analyzers.racket_trajectory import RacketTrajectoryAnalyzer
from .analyzers.body_posture import BodyPostureAnalyzer
from .comparison import BiomechanicalComparison

class BiomechanicalAnalyzer2D:
    """
    Main biomechanical analysis interface.
    Combines all analyzers for comprehensive 2D movement analysis.
    """
    
    def __init__(self):
        self.landmarks_extractor = LandmarksExtractor()
        self.kinematics_calculator = KinematicsCalculator()
        self.joint_analyzer = JointAnglesAnalyzer()
        self.racket_analyzer = RacketTrajectoryAnalyzer()
        self.posture_analyzer = BodyPostureAnalyzer()
        self.comparison_engine = BiomechanicalComparison()
    
    def analyze_video_biomechanics(self, video_path: str, metadata: dict):
        """
        Complete biomechanical analysis of a tennis video.
        
        Returns real 2D biomechanical parameters instead of subjective estimates.
        """
        # Extract landmarks
        landmarks_data = self.landmarks_extractor.extract_from_video(video_path)
        
        # Calculate kinematics
        kinematics = self.kinematics_calculator.calculate_all(landmarks_data)
        
        # Analyze different aspects
        joint_analysis = self.joint_analyzer.analyze(landmarks_data, kinematics)
        racket_analysis = self.racket_analyzer.analyze(landmarks_data, kinematics)
        posture_analysis = self.posture_analyzer.analyze(landmarks_data, kinematics)
        
        return {
            'success': True,
            'analysis_type': 'biomechanical_2d',
            'kinematics': kinematics,
            'joint_analysis': joint_analysis,
            'racket_analysis': racket_analysis,
            'posture_analysis': posture_analysis,
            'landmarks_data': landmarks_data
        }
    
    def compare_biomechanics(self, user_analysis: dict, professional_analysis: dict):
        """
        Compare biomechanical data between user and professional.
        """
        return self.comparison_engine.compare(user_analysis, professional_analysis)

__version__ = "1.0.0"
__all__ = [
    'BiomechanicalAnalyzer2D',
    'LandmarksExtractor', 
    'KinematicsCalculator',
    'JointAnglesAnalyzer',
    'RacketTrajectoryAnalyzer', 
    'BodyPostureAnalyzer',
    'BiomechanicalComparison'
]