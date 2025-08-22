#!/usr/bin/env python3
"""
Final test to verify API response and recommendations
"""
import requests
import json

def test_final():
    print("=== FINAL TEST: API Response Format ===")
    
    url = "http://localhost:8000/analyze-selected-DEBUG-FIXED-v16"
    
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
        response = requests.post(url, data=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"SUCCESS: {result.get('success', False)}")
            print(f"SCORE: {result.get('final_score', 0)}")
            
            # Extract recommendations
            recommendations = result.get('recommendations', [])
            print(f"\nRECOMMENDATIONS ({len(recommendations)} items):")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
            
            # Extract professional comparisons 
            prof_comparisons = result.get('professional_comparisons', [])
            print(f"\nPROFESSIONAL COMPARISONS ({len(prof_comparisons)} items):")
            for i, comp in enumerate(prof_comparisons, 1):
                if isinstance(comp, dict):
                    name = comp.get('professional_name', 'unknown')
                    score = comp.get('similarity_score', 0)
                    comp_recs = comp.get('recommendations', [])
                    print(f"  {i}. {name} - Score: {score:.1f}%")
                    if comp_recs:
                        print(f"     Has {len(comp_recs)} recommendations")
                else:
                    print(f"  {i}. {comp}")
            
            # Check what's in detailed_analysis
            detailed = result.get('detailed_analysis', {})
            print(f"\nDETAILED ANALYSIS KEYS: {list(detailed.keys())}")
            
            # Show full JSON structure (first 1000 chars)
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            print(f"\nFULL JSON (first 1000 chars):")
            print(json_str[:1000])
            
        else:
            print(f"ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_final()