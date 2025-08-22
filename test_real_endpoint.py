#!/usr/bin/env python3
"""
Test the new real analysis endpoint
"""

import requests
import json

API_BASE = 'http://localhost:8000'

def test_real_endpoint():
    print("=== TESTING REAL ANALYSIS ENDPOINT ===")
    
    # Login first
    login_data = {"username": "demo", "password": "demo123"}
    response = requests.post(f"{API_BASE}/auth/login", json=login_data)
    
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return
    
    token = response.json().get('access_token')
    print(f"[LOGIN] Success - Token: {token[:50]}...")
    
    # Test the real analysis endpoint (no auth required now)
    print("[TEST] Calling test-real-analysis endpoint...")
    response = requests.post(f"{API_BASE}/test-real-analysis")
    
    print(f"[RESPONSE] Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n[SUCCESS] Real analysis completed!")
        
        print(f"\nSuccess: {result.get('success')}")
        print(f"Final Score: {result.get('final_score')}")
        print(f"Analysis Type: {result.get('analysis_type')}")
        
        # Check the critical fields
        user_analysis = result.get('user_analysis', {})
        pro_analysis = result.get('professional_analysis', {})
        comparison = result.get('comparison', {})
        
        print(f"\nuser_analysis keys: {list(user_analysis.keys())}")
        print(f"professional_analysis keys: {list(pro_analysis.keys())}")
        print(f"comparison keys: {list(comparison.keys())}")
        
        if user_analysis:
            print(f"\n=== USER ANALYSIS SAMPLE ===")
            print(f"rhythm_variability: {user_analysis.get('rhythm_variability')}")
            print(f"acceleration_smoothness: {user_analysis.get('acceleration_smoothness')}")
            print(f"movement_efficiency: {user_analysis.get('movement_efficiency')}")
            print(f"cycles_count: {user_analysis.get('cycles_count')}")
            
        if comparison:
            print(f"\n=== COMPARISON SAMPLE ===")
            detailed = comparison.get('detailed_similarities', {})
            print(f"similarity_score: {comparison.get('similarity_score')}")
            print(f"detailed_similarities keys: {list(detailed.keys())}")
            if detailed:
                print(f"rhythm_variability similarity: {detailed.get('rhythm_variability')}")
                print(f"acceleration_smoothness similarity: {detailed.get('acceleration_smoothness')}")
        
    else:
        print(f"[ERROR] Real analysis failed: {response.text}")

if __name__ == "__main__":
    test_real_endpoint()