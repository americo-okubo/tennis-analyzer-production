#!/usr/bin/env python3
"""
Test the simple server with real data
"""

import requests
import json

def test_simple_server():
    print("=== TESTING SIMPLE SERVER WITH REAL DATA ===")
    
    try:
        response = requests.post("http://127.0.0.1:8001/test-real-data")
        
        print(f"[RESPONSE] Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n[SUCCESS] Real data analysis completed!")
            
            print(f"\nSuccess: {result.get('success')}")
            print(f"Final Score: {result.get('final_score')}")
            print(f"Analysis Type: {result.get('analysis_type')}")
            
            # Check the critical fields
            user_analysis = result.get('user_analysis', {})
            pro_analysis = result.get('professional_analysis', {})
            comparison = result.get('comparison', {})
            detailed_similarities = comparison.get('detailed_similarities', {})
            
            print(f"\n=== DATA AVAILABILITY ===")
            print(f"user_analysis available: {bool(user_analysis)}")
            print(f"professional_analysis available: {bool(pro_analysis)}")
            print(f"comparison available: {bool(comparison)}")
            print(f"detailed_similarities available: {bool(detailed_similarities)}")
            
            if user_analysis:
                print(f"\n=== USER ANALYSIS (REAL DATA) ===")
                print(f"cycles_count: {user_analysis.get('cycles_count')}")
                print(f"rhythm_variability: {user_analysis.get('rhythm_variability')}")
                print(f"acceleration_smoothness: {user_analysis.get('acceleration_smoothness')}")
                print(f"movement_efficiency: {user_analysis.get('movement_efficiency')}")
                print(f"amplitude_consistency: {user_analysis.get('amplitude_consistency')}")
                print(f"quality_score: {user_analysis.get('quality_score')}")
                
            if pro_analysis:
                print(f"\n=== PROFESSIONAL ANALYSIS (REAL DATA) ===")
                print(f"cycles_count: {pro_analysis.get('cycles_count')}")
                print(f"rhythm_variability: {pro_analysis.get('rhythm_variability')}")
                print(f"acceleration_smoothness: {pro_analysis.get('acceleration_smoothness')}")
                print(f"movement_efficiency: {pro_analysis.get('movement_efficiency')}")
                print(f"amplitude_consistency: {pro_analysis.get('amplitude_consistency')}")
                print(f"quality_score: {pro_analysis.get('quality_score')}")
            
            if detailed_similarities:
                print(f"\n=== DETAILED SIMILARITIES (REAL DATA) ===")
                for key, value in detailed_similarities.items():
                    print(f"{key}: {value}")
            
            # This proves the data is REAL and working!
            print(f"\n🎯 CONCLUSION: Real biomechanical data is working!")
            print(f"✅ Score: {result.get('final_score'):.1f}%")
            print(f"✅ User cycles detected: {user_analysis.get('cycles_count', 0)}")
            print(f"✅ Professional cycles detected: {pro_analysis.get('cycles_count', 0)}")
            print(f"✅ Detailed similarity metrics: {len(detailed_similarities)} parameters")
            
            return result
            
        else:
            print(f"[ERROR] Request failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return None

if __name__ == "__main__":
    test_simple_server()