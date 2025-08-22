#!/usr/bin/env python3
"""
Debug script to test tennis_cycle_integration directly
"""

import sys
import os
sys.path.append('.')

from tennis_cycle_integration import TennisAnalysisInterface

def test_cycle_integration():
    print("=== TESTING TENNIS CYCLE INTEGRATION DIRECTLY ===")
    
    # Create interface
    interface = TennisAnalysisInterface()
    print(f"Analysis interface created: {type(interface)}")
    
    # Test paths
    user_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    pro_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\profissionais\forehand_drive\Zhang_Jike_FD_D_D.mp4"
    
    # Test parameters
    validated_params = {
        'dominant_hand': 'right',
        'movement_type': 'forehand',
        'camera_side': 'right',
        'racket_side': 'forehand'
    }
    
    print(f"User video: {os.path.basename(user_video)}")
    print(f"Pro video: {os.path.basename(pro_video)}")
    print(f"Parameters: {validated_params}")
    
    # Test analysis
    try:
        result = interface.analyze_from_file_paths(
            user_video, pro_video,
            validated_params['dominant_hand'], 
            validated_params['movement_type'],
            validated_params['camera_side'], 
            validated_params['racket_side']
        )
        
        print(f"\\n=== ANALYSIS RESULT ===")
        print(f"Success: {result.get('success')}")
        print(f"Final Score: {result.get('final_score')}")
        print(f"Analysis Type: {result.get('analysis_type')}")
        
        # Check detailed fields
        user_analysis = result.get('user_analysis', {})
        pro_analysis = result.get('professional_analysis', {})
        comparison = result.get('comparison', {})
        
        print(f"\\n=== USER ANALYSIS ===")
        print(f"Keys: {list(user_analysis.keys())}")
        for key, value in user_analysis.items():
            if key != 'cycles_details':  # Skip detailed cycle data
                print(f"  {key}: {value}")
        
        print(f"\\n=== PROFESSIONAL ANALYSIS ===")
        print(f"Keys: {list(pro_analysis.keys())}")
        for key, value in pro_analysis.items():
            print(f"  {key}: {value}")
        
        print(f"\\n=== COMPARISON ===")
        print(f"Keys: {list(comparison.keys())}")
        for key, value in comparison.items():
            if key == 'detailed_similarities':
                print(f"  {key}: {list(value.keys()) if isinstance(value, dict) else value}")
            else:
                print(f"  {key}: {value}")
        
        # Check if we have the biomechanical metrics we need
        required_metrics = ['rhythm_variability', 'acceleration_smoothness', 'movement_efficiency', 'amplitude_consistency']
        print(f"\\n=== BIOMECHANICAL METRICS CHECK ===")
        for metric in required_metrics:
            user_value = user_analysis.get(metric, 'MISSING')
            pro_value = pro_analysis.get(metric, 'MISSING')
            print(f"  {metric}: User={user_value}, Pro={pro_value}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_cycle_integration()