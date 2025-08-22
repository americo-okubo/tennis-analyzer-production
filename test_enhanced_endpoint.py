#!/usr/bin/env python3
"""
Test script to debug the enhanced analysis endpoint
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_analysis():
    """Test the enhanced analysis components directly"""
    try:
        print("[TEST] Starting enhanced analysis test...")
        
        # Test 1: Import check
        print("[TEST] Testing imports...")
        from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
        print("[TEST] Enhanced analyzer imported successfully")
        
        # Test 2: Initialize analyzer
        print("[TEST] Initializing analyzer...")
        analyzer = EnhancedSingleCycleAnalyzer()
        print("[TEST] Analyzer initialized successfully")
        
        # Test 3: Test with real videos
        user_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
        pro_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Fan_Zhendong_FD_D_E.mp4"
        
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
        
        print(f"[TEST] User video exists: {os.path.exists(user_video)}")
        print(f"[TEST] Professional video exists: {os.path.exists(pro_video)}")
        
        if not os.path.exists(user_video) or not os.path.exists(pro_video):
            print("[TEST] Video files not found")
            return False
        
        # Test 4: Run analysis
        print("[TEST] Running enhanced analysis...")
        result = analyzer.compare_enhanced_single_cycles(
            user_video, pro_video, user_metadata, prof_metadata, cycle_index=1
        )
        
        if result['success']:
            print(f"[TEST] Analysis successful: {result['final_score']:.1f}%")
            print(f"[TEST] Performance category: {result.get('combined_comparison', {}).get('performance_category', 'N/A')}")
            print(f"[TEST] Analysis confidence: {result.get('combined_comparison', {}).get('analysis_confidence', 0):.2f}")
            return True
        else:
            print(f"[TEST] Analysis failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"[TEST] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_analysis()
    if success:
        print("\n[TEST] All tests passed! Enhanced analysis is working.")
    else:
        print("\n[TEST] Tests failed. Check the error messages above.")