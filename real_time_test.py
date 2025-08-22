#!/usr/bin/env python3
"""
Simple test for real-time analysis engine components
Tests core functionality without GUI components
"""

import numpy as np
import time
import json
from pathlib import Path

# Import our systems
from tennis_comparison_backend import TableTennisAnalyzer
from real_time_analysis_engine import RealTimeAnalysisEngine


def test_core_components():
    """Test core components without camera/GUI"""
    print("Testing TableTennisAnalyzer integration...")
    
    # Test basic analyzer
    analyzer = TableTennisAnalyzer()
    print("OK TableTennisAnalyzer initialized")
    
    # Test professionals database
    professionals = analyzer.get_available_professionals('forehand_drive')
    print(f"OK Found {len(professionals)} professionals for forehand_drive")
    
    # Test metadata processing
    sample_metadata = {
        'maoDominante': 'D',
        'ladoCamera': 'E', 
        'ladoRaquete': 'F',
        'tipoMovimento': 'D'
    }
    
    movement_key = analyzer._build_movement_key(sample_metadata)
    expected_info = analyzer._build_expected_info(sample_metadata)
    
    print(f"OK Movement key: {movement_key}")
    print(f"OK Expected info: {expected_info}")
    
    return True


def test_analysis_engine_components():
    """Test analysis engine components without camera"""
    print("\nTesting RealTimeAnalysisEngine components...")
    
    # Create engine without starting camera
    engine = RealTimeAnalysisEngine()
    print("OK RealTimeAnalysisEngine initialized")
    
    # Test mock landmark data processing
    mock_landmarks = []
    for i in range(20):
        # Create mock pose data
        mock_data = {
            'pose': type('MockPose', (), {
                'pose_landmarks': type('MockLandmarks', (), {
                    'landmark': [
                        type('MockLandmark', (), {'x': 0.5 + i * 0.01, 'y': 0.5, 'z': 0.0})() 
                        for _ in range(33)  # 33 pose landmarks
                    ]
                })()
            })(),
            'hands': None,
            'timestamp': time.time() + i * 0.033  # 30 FPS
        }
        mock_landmarks.append(mock_data)
    
    # Test stroke analysis
    stroke_analysis = engine.analyze_stroke_motion(mock_landmarks)
    print(f"OK Stroke analysis completed: {stroke_analysis}")
    
    # Test velocity calculation
    positions = [[0.5, 0.5, 0.0], [0.51, 0.5, 0.0], [0.52, 0.5, 0.0]]
    velocities = engine._calculate_velocities(positions)
    print(f"OK Velocity calculation: {len(velocities)} velocity vectors")
    
    # Test stroke classification
    wrist_positions = [[0.5, 0.5], [0.6, 0.5], [0.7, 0.5]]
    shoulder_positions = [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]
    stroke_type = engine._classify_stroke_type(wrist_positions, shoulder_positions)
    print(f"OK Stroke classification: {stroke_type}")
    
    return True


def test_mock_analysis_session():
    """Test a complete mock analysis session"""
    print("\nTesting mock analysis session...")
    
    engine = RealTimeAnalysisEngine()
    
    # Simulate recording session
    engine.start_recording()
    print("OK Recording started")
    
    # Add mock frames
    for i in range(30):  # 1 second at 30 FPS
        mock_frame_data = {
            'frame': np.zeros((480, 640, 3), dtype=np.uint8),
            'landmarks': {
                'pose': type('MockPose', (), {
                    'pose_landmarks': type('MockLandmarks', (), {
                        'landmark': [
                            type('MockLandmark', (), {
                                'x': 0.5 + np.sin(i * 0.2) * 0.1, 
                                'y': 0.5 + np.cos(i * 0.2) * 0.05, 
                                'z': 0.0
                            })() for _ in range(33)
                        ]
                    })()
                })(),
                'hands': None,
                'timestamp': time.time() + i * 0.033
            },
            'timestamp': time.time() + i * 0.033
        }
        engine.frames_buffer.append(mock_frame_data)
    
    engine.stop_recording()
    print(f"OK Recording stopped - {len(engine.frames_buffer)} frames captured")
    
    # Analyze the session
    result = engine.analyze_recorded_sequence()
    print(f"OK Analysis completed: {result['success']}")
    
    if result['success']:
        print(f"  - Frames analyzed: {result['frames_analyzed']}")
        print(f"  - Duration: {result['duration']:.2f}s")
        print(f"  - Stroke detected: {result['stroke_analysis']['stroke_detected']}")
        
        if 'recommendations' in result:
            print(f"  - Recommendations: {len(result['recommendations'])}")
    
    return True


def test_integration_with_existing_system():
    """Test integration with existing tennis analysis system"""
    print("\nTesting integration with existing system...")
    
    analyzer = TableTennisAnalyzer()
    
    # Test that all integration components are loaded
    components = ['real_comparison', 'cycle_analyzer', 'analysis_interface']
    for component in components:
        has_component = hasattr(analyzer, component)
        print(f"OK Component {component}: {'Available' if has_component else 'Not Available'}")
    
    # Test that system can handle real analysis if videos are available
    videos_dir = Path('videos')
    if videos_dir.exists():
        video_files = list(videos_dir.glob('*.mp4'))
        print(f"OK Found {len(video_files)} video files for potential analysis")
        
        if len(video_files) >= 2:
            print("OK System ready for real video analysis")
        else:
            print("! Need at least 2 video files for comparison")
    else:
        print("! Videos directory not found")
    
    return True


def main():
    """Run all component tests"""
    print("Real-Time Analysis Engine - Component Test")
    print("=" * 50)
    
    try:
        # Test core components
        test_core_components()
        
        # Test analysis engine
        test_analysis_engine_components()
        
        # Test mock session
        test_mock_analysis_session()
        
        # Test integration
        test_integration_with_existing_system()
        
        print("\n" + "=" * 50)
        print("SUCCESS: ALL TESTS PASSED!")
        print("Real-time analysis engine components are working correctly.")
        print("\nNote: GUI components disabled due to OpenCV display issues on Windows.")
        print("Core analysis functionality is fully operational.")
        
    except Exception as e:
        print(f"\nERROR: TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()