#!/usr/bin/env python3
"""
Debug script to test tennis_comparison_backend directly to see what's wrong
"""

import sys
import os
sys.path.append('.')

from tennis_comparison_backend import TableTennisAnalyzer

def test_backend_directly():
    print("=== TESTING TENNIS COMPARISON BACKEND DIRECTLY ===")
    
    # Create analyzer
    analyzer = TableTennisAnalyzer()
    print(f"Analyzer created: {type(analyzer)}")
    print(f"Cycle analyzer available: {analyzer.cycle_analyzer is not None}")
    print(f"Analysis interface available: {analyzer.analysis_interface is not None}")
    
    # Test paths
    user_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    pro_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\profissionais\forehand_drive\Zhang_Jike_FD_D_D.mp4"
    
    # Test metadata (exactly what the API sends)
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
    
    print(f"User video: {os.path.basename(user_video)} (exists: {os.path.exists(user_video)})")
    print(f"Pro video: {os.path.basename(pro_video)} (exists: {os.path.exists(pro_video)})")
    print(f"User metadata: {user_metadata}")
    print(f"Prof metadata: {prof_metadata}")
    
    # Test the comparison directly
    try:
        result = analyzer.compare_techniques(
            user_video, pro_video, user_metadata, prof_metadata
        )
        
        print(f"\\n=== COMPARISON RESULT ===")
        print(f"Success: {result.get('success')}")
        print(f"Final Score: {result.get('final_score')}")
        print(f"Analysis Type: {result.get('analysis_type')}")
        print(f"Error: {result.get('error')}")
        
        # Check the detailed fields we care about
        user_analysis = result.get('user_analysis', {})
        pro_analysis = result.get('professional_analysis', {})
        comparison = result.get('comparison', {})
        
        print(f"\\n=== FIELD AVAILABILITY ===")
        print(f"user_analysis exists: {bool(user_analysis)}")
        print(f"professional_analysis exists: {bool(pro_analysis)}")
        print(f"comparison exists: {bool(comparison)}")
        
        if user_analysis:
            print(f"\\n=== USER ANALYSIS ===")
            print(f"Keys: {list(user_analysis.keys())}")
            for key, value in user_analysis.items():
                if key != 'cycles_details':
                    print(f"  {key}: {value}")
        
        if pro_analysis:
            print(f"\\n=== PROFESSIONAL ANALYSIS ===")
            print(f"Keys: {list(pro_analysis.keys())}")
            for key, value in pro_analysis.items():
                print(f"  {key}: {value}")
        
        if comparison:
            print(f"\\n=== COMPARISON ===")
            print(f"Keys: {list(comparison.keys())}")
            for key, value in comparison.items():
                print(f"  {key}: {value}")
        
        # Check detailed_analysis
        detailed = result.get('detailed_analysis', {})
        print(f"\\n=== DETAILED ANALYSIS ===")
        print(f"Keys: {list(detailed.keys())}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_backend_directly()