#!/usr/bin/env python3
"""
Test script to verify recommendations generation
"""

import sys
import json
from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer

def test_recommendations():
    print("Testing recommendations generation...")
    
    # Create analyzer
    analyzer = EnhancedSingleCycleAnalyzer()
    
    # Test metadata
    metadata_dict = {
        'maoDominante': 'D',
        'ladoCamera': 'D', 
        'ladoRaquete': 'B',
        'tipoMovimento': 'P'
    }
    
    # Test video
    video_path = "videos/Japones_BP_D_D.mp4"
    
    print(f"Testing video: {video_path}")
    print(f"Testing metadata: {metadata_dict}")
    
    # Call analyzer
    result = analyzer.compare_enhanced_single_cycles(video_path, None, metadata_dict, None, 1)
    
    print(f"\nAnalysis Result:")
    print(f"Success: {result.get('success', False)}")
    print(f"Final Score: {result.get('final_score', 0)}")
    print(f"Movement: {result.get('detected_movement', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0)}")
    
    # Check recommendations
    recommendations = result.get('recommendations', [])
    print(f"\nRecommendations ({len(recommendations)} total):")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Check professional comparisons
    prof_comparisons = result.get('professional_comparisons', [])
    print(f"\nProfessional Comparisons ({len(prof_comparisons)} total):")
    for i, comp in enumerate(prof_comparisons, 1):
        if hasattr(comp, '__dict__'):
            print(f"   {i}. {comp.professional_name} - Score: {comp.similarity_score:.1f}%")
        else:
            print(f"   {i}. {comp}")
    
    # Check detailed analysis
    detailed = result.get('detailed_analysis', {})
    print(f"\nDetailed Analysis Keys: {list(detailed.keys())}")
    
    return result

if __name__ == "__main__":
    try:
        result = test_recommendations()
        print(f"\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()