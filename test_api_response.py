#!/usr/bin/env python3
"""
Test API response directly to see what's being returned
"""
import requests
import json

def test_api():
    print("Testing API response format...")
    
    # API endpoint
    url = "http://localhost:8000/analyze-selected-DEBUG-FIXED-v16"
    
    # Test data
    data = {
        'selected_video': 'Japones_BP_D_D.mp4',
        'metadata': json.dumps({
            'maoDominante': 'D',
            'ladoCamera': 'D', 
            'ladoRaquete': 'B',
            'tipoMovimento': 'P'
        }),
        'cycle_index': 1
    }
    
    try:
        print(f"Calling: {url}")
        print(f"Data: {data}")
        
        response = requests.post(url, data=data, timeout=60)
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse Keys: {list(result.keys())}")
            
            # Check recommendations specifically
            recommendations = result.get('recommendations', [])
            print(f"\nRecommendations ({len(recommendations)} items):")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
            
            # Check professional comparisons
            prof_comps = result.get('professional_comparisons', [])
            print(f"\nProfessional Comparisons ({len(prof_comps)} items):")
            for i, comp in enumerate(prof_comps, 1):
                if isinstance(comp, dict):
                    print(f"   {i}. {comp.get('professional_name', 'unknown')} - Score: {comp.get('similarity_score', 0):.1f}%")
                    comp_recs = comp.get('recommendations', [])
                    if comp_recs:
                        print(f"      Recommendations ({len(comp_recs)}): {comp_recs[:2]}")
                else:
                    print(f"   {i}. {comp}")
                    
        else:
            print(f"API Error: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()