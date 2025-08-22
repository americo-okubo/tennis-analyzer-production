#!/usr/bin/env python3
"""
Debug script to check analysis response structure
"""

import requests
import json

API_BASE = 'http://localhost:8001'

def test_analysis_response():
    print("=== TESTING ANALYSIS RESPONSE STRUCTURE ===")
    
    # Login first
    login_data = {"username": "demo", "password": "demo123"}
    response = requests.post(f"{API_BASE}/auth/login", json=login_data)
    
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return
    
    token = response.json().get('access_token')
    print(f"[LOGIN] Success - Token: {token[:50]}...")
    
    # Create a dummy analysis request (simulate what frontend does)
    headers = {'Authorization': f'Bearer {token}'}
    
    # Get professional file for analysis
    video_path = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'user_video': ('test_video.mp4', video_file, 'video/mp4')}
            data = {
                'metadata': json.dumps({
                    "maoDominante": "Destro",
                    "ladoCamera": "Direita",  # Changed to match detection
                    "ladoRaquete": "F",
                    "tipoMovimento": "D"
                }),
                'professional_name': 'Zhang Jike',
                'analysis_type': 'full'
            }
            
            print("[ANALYSIS] Sending request...")
            response = requests.post(
                f"{API_BASE}/analyze",
                files=files,
                data=data,
                headers=headers,
                timeout=300
            )
            
            print(f"[RESPONSE] Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n[SUCCESS] Analysis completed!")
                
                # Print structure
                print("\n=== RESPONSE STRUCTURE ===")
                print_structure(result, max_depth=3)
                
                # Print full detailed_analysis
                print("\n=== DETAILED ANALYSIS CONTENT ===")
                detailed = result.get('detailed_analysis', {})
                print(f"detailed_analysis keys: {list(detailed.keys())}")
                if 'biomech_breakdown' in detailed:
                    print(f"biomech_breakdown: {detailed['biomech_breakdown']}")
                
                # Print full result for debugging
                print("\n=== FULL RESULT (for debugging) ===")
                print(json.dumps(result, indent=2, default=str))
                
                # Check specific fields we need
                print("\n=== CHECKING SPECIFIC FIELDS ===")
                user_analysis = result.get('user_analysis', {})
                pro_analysis = result.get('professional_analysis', {})
                comparison = result.get('comparison', {})
                
                print(f"user_analysis keys: {list(user_analysis.keys())}")
                print(f"professional_analysis keys: {list(pro_analysis.keys())}")
                print(f"comparison keys: {list(comparison.keys())}")
                
                detailed_similarities = comparison.get('detailed_similarities', {})
                print(f"detailed_similarities keys: {list(detailed_similarities.keys())}")
                
                # Print values
                print("\n=== VALUES ===")
                print(f"final_score: {result.get('final_score')}")
                print(f"user cycles_count: {user_analysis.get('cycles_count')}")
                print(f"user quality_score: {user_analysis.get('quality_score')}")
                print(f"user rhythm_variability: {user_analysis.get('rhythm_variability')}")
                print(f"similarity_score: {comparison.get('similarity_score')}")
                
            else:
                print(f"[ERROR] Analysis failed: {response.text}")
                
    except Exception as e:
        print(f"[ERROR] Exception: {e}")

def print_structure(obj, prefix="", max_depth=2, current_depth=0):
    if current_depth >= max_depth:
        return
        
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}: {type(value).__name__}")
                if current_depth < max_depth - 1:
                    print_structure(value, prefix + "  ", max_depth, current_depth + 1)
            else:
                print(f"{prefix}{key}: {value} ({type(value).__name__})")
    elif isinstance(obj, list) and obj:
        print(f"{prefix}[0]: {type(obj[0]).__name__}")
        if current_depth < max_depth - 1:
            print_structure(obj[0], prefix + "  ", max_depth, current_depth + 1)

if __name__ == "__main__":
    test_analysis_response()