#!/usr/bin/env python3
"""
Test script for the new Biomechanics module.

Tests the modular 2D biomechanical analysis system with real video data.
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the new biomechanics module
from biomechanics import BiomechanicalAnalyzer2D

def test_biomechanical_analysis():
    """
    Test the biomechanical analysis with real video files.
    """
    print("=== TESTING BIOMECHANICAL 2D ANALYSIS ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize the analyzer
    analyzer = BiomechanicalAnalyzer2D()
    
    # Test videos
    user_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    pro_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Fan_Zhendong_FD_D_E.mp4"
    
    # Metadata for analysis
    user_metadata = {
        'maoDominante': 'Destro',
        'ladoCamera': 'Direita',
        'ladoRaquete': 'F',
        'tipoMovimento': 'D'
    }
    
    prof_metadata = {
        'maoDominante': 'D',
        'ladoCamera': 'D',
        'ladoRaquete': 'F',
        'tipoMovimento': 'D'
    }
    
    print(f"User video: {os.path.basename(user_video)}")
    print(f"Professional video: {os.path.basename(pro_video)}")
    print()
    
    try:
        # Test 1: User analysis
        print("[1] Analyzing user video...")
        user_result = analyzer.analyze_video_biomechanics(user_video, user_metadata)
        
        if user_result['success']:
            print(f"[SUCCESS] User analysis successful!")
            print_biomechanical_summary(user_result, "USER")
        else:
            print(f"[ERROR] User analysis failed: {user_result.get('error')}")
            return False
        
        print("\n" + "="*60 + "\n")
        
        # Test 2: Professional analysis
        print("[2] Analyzing professional video...")
        pro_result = analyzer.analyze_video_biomechanics(pro_video, prof_metadata)
        
        if pro_result['success']:
            print(f" Professional analysis successful!")
            print_biomechanical_summary(pro_result, "PROFESSIONAL")
        else:
            print(f" Professional analysis failed: {pro_result.get('error')}")
            return False
        
        print("\n" + "="*60 + "\n")
        
        # Test 3: Comparison
        print("[3] Comparing biomechanical data...")
        comparison_result = analyzer.compare_biomechanics(user_result, pro_result)
        
        if comparison_result['success']:
            print(f" Comparison successful!")
            print_comparison_summary(comparison_result)
        else:
            print(f" Comparison failed: {comparison_result.get('error')}")
            return False
        
        print("\n" + "="*60 + "\n")
        print("[SUCCESS] ALL TESTS PASSED! Biomechanical module is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_biomechanical_summary(result: dict, label: str):
    """
    Print a summary of biomechanical analysis results.
    """
    print(f"\n=== {label} BIOMECHANICAL ANALYSIS ===")
    
    # Landmarks summary
    landmarks = result.get('landmarks_data', {})
    if landmarks.get('success'):
        print(f" Landmarks: {landmarks['valid_frames']}/{landmarks['total_frames']} frames")
        print(f" Quality: {landmarks['quality_score']:.1%}")
        print(f" Duration: {landmarks['video_duration']:.1f}s")
    
    # Kinematics summary
    kinematics = result.get('kinematics', {})
    if kinematics.get('success'):
        print(f" Kinematics: {kinematics['frame_count']} frames analyzed")
        print(f" FPS: {kinematics['fps']}")
        
        # Joint angles summary
        if 'joint_angles' in kinematics:
            angles = kinematics['joint_angles']
            if 'right_arm_angles_stats' in angles:
                arm_stats = angles['right_arm_angles_stats']
                print(f" Arm angle: {arm_stats['mean']:.1f}° ± {arm_stats['std']:.1f}° (range: {arm_stats['range']:.1f}°)")
    
    # Joint analysis summary
    joint_analysis = result.get('joint_analysis', {})
    if joint_analysis.get('success'):
        overall_score = joint_analysis.get('overall_joint_score', 0)
        print(f" Joint analysis score: {overall_score:.1%}")
        
        if 'arm_mechanics' in joint_analysis and 'dominant_arm' in joint_analysis['arm_mechanics']:
            arm_mech = joint_analysis['arm_mechanics']['dominant_arm']
            print(f" Arm mechanics score: {arm_mech.get('mechanics_score', 0):.1%}")
    
    # Racket analysis summary
    racket_analysis = result.get('racket_analysis', {})
    if racket_analysis.get('success'):
        overall_score = racket_analysis.get('overall_racket_score', 0)
        print(f" Racket analysis score: {overall_score:.1%}")
        
        if 'velocity_analysis' in racket_analysis:
            vel_analysis = racket_analysis['velocity_analysis']
            if 'basic_stats' in vel_analysis:
                vel_stats = vel_analysis['basic_stats']
                max_vel = vel_stats.get('max_velocity', 0)
                mean_vel = vel_stats.get('mean_velocity', 0)
                print(f" Max velocity: {max_vel:.1f} px/frame")
                print(f" Mean velocity: {mean_vel:.1f} px/frame")
    
    # Posture analysis summary
    posture_analysis = result.get('posture_analysis', {})
    if posture_analysis.get('success'):
        overall_score = posture_analysis.get('overall_posture_score', 0)
        print(f" Posture analysis score: {overall_score:.1%}")

def print_comparison_summary(comparison: dict):
    """
    Print a summary of biomechanical comparison results.
    """
    print("\n=== BIOMECHANICAL COMPARISON SUMMARY ===")
    
    # Overall comparison
    overall = comparison.get('overall_comparison', {})
    if overall:
        similarity = overall.get('similarity_score', 0)
        percentage = overall.get('similarity_percentage', 0)
        confidence = overall.get('confidence', 0)
        category = overall.get('performance_category', 'Unknown')
        
        print(f" Overall similarity: {similarity:.1%} ({percentage:.1f}%)")
        print(f" Performance category: {category}")
        print(f" Comparison confidence: {confidence:.1%}")
    
    # Component similarities
    components = {
        'joint_comparison': ' Joint angles',
        'racket_comparison': ' Racket trajectory',
        'posture_comparison': ' Body posture',
        'kinematics_comparison': ' Kinematics'
    }
    
    print("\n--- Component Similarities ---")
    for comp_key, comp_label in components.items():
        if comp_key in comparison:
            comp_data = comparison[comp_key]
            overall_key = comp_key.replace('_comparison', '_similarity').replace('joint_', 'joint_')
            
            # Try different possible keys for overall similarity
            similarity = (comp_data.get(f'overall_{comp_key.split("_")[0]}_similarity') or
                         comp_data.get('overall_similarity') or
                         0.5)
            
            print(f"{comp_label}: {similarity:.1%}")
    
    # Recommendations
    recommendations = comparison.get('recommendations', [])
    if recommendations:
        print("\n--- Recommendations ---")
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"{i}. {rec}")
    
    # Detailed metrics
    detailed = comparison.get('detailed_metrics', {})
    if detailed:
        print("\n--- Key Metrics ---")
        for metric_category, metrics in detailed.items():
            print(f"{metric_category.replace('_', ' ').title()}:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"  • {metric_name.replace('_', ' ').title()}: {metric_value:.1%}")

def test_individual_components():
    """
    Test individual components of the biomechanics module.
    """
    print("\n=== TESTING INDIVIDUAL COMPONENTS ===")
    
    from biomechanics.core.landmarks_extractor import LandmarksExtractor
    from biomechanics.core.kinematics_calculator import KinematicsCalculator
    
    # Test landmarks extractor
    print("\n[1] Testing Landmarks Extractor...")
    extractor = LandmarksExtractor()
    
    test_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    
    landmarks_result = extractor.extract_from_video(test_video)
    
    if landmarks_result['success']:
        print(f" Landmarks extraction successful!")
        print(f"   Valid frames: {landmarks_result['valid_frames']}/{landmarks_result['total_frames']}")
        print(f"   Quality score: {landmarks_result['quality_score']:.1%}")
    else:
        print(f" Landmarks extraction failed: {landmarks_result.get('error')}")
        return False
    
    # Test kinematics calculator
    print("\n[2] Testing Kinematics Calculator...")
    calculator = KinematicsCalculator()
    
    kinematics_result = calculator.calculate_all(landmarks_result)
    
    if kinematics_result['success']:
        print(f" Kinematics calculation successful!")
        print(f"   Frame count: {kinematics_result['frame_count']}")
        print(f"   Duration: {kinematics_result['duration']:.1f}s")
        
        # Show some statistics
        if 'joint_angles' in kinematics_result:
            angles = kinematics_result['joint_angles']
            print(f"   Joint angle sequences: {len([k for k in angles.keys() if not k.endswith('_stats')])}")
        
        if 'velocities' in kinematics_result:
            velocities = kinematics_result['velocities']
            vel_sequences = len([k for k in velocities.keys() if k.endswith('_velocity') and not k.endswith('_stats')])
            print(f"   Velocity sequences: {vel_sequences}")
    else:
        print(f" Kinematics calculation failed: {kinematics_result.get('error')}")
        return False
    
    print("\n All individual components working correctly!")
    return True

if __name__ == "__main__":
    print("Tennis Biomechanics Module Test Suite")
    print("="*50)
    
    # Test individual components first
    if not test_individual_components():
        print("\n[ERROR] Individual component tests failed!")
        sys.exit(1)
    
    print("\n" + "="*50)
    
    # Test full biomechanical analysis
    if test_biomechanical_analysis():
        print("\n[SUCCESS] ALL TESTS PASSED! The biomechanical module is ready for integration.")
        sys.exit(0)
    else:
        print("\n[ERROR] TESTS FAILED! Check the error messages above.")
        sys.exit(1)