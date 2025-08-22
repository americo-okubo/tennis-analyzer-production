#!/usr/bin/env python3
"""
Teste direto da API para verificar se as recomendações estão sendo retornadas
"""
import requests
import json

def test_api_direct():
    print("=== TESTE DIRETO DA API ===")
    
    url = "http://localhost:8000/analyze-selected-DEBUG-FIXED-v16"
    
    data = {
        'selected_video': 'Japones_BP_D_D.mp4',
        'metadata': json.dumps({
            'maoDominante': 'Destro',
            'ladoCamera': 'Direita', 
            'ladoRaquete': 'B',
            'tipoMovimento': 'P'
        }),
        'cycle_index': 1
    }
    
    try:
        print(f"Chamando: {url}")
        response = requests.post(url, data=data, timeout=120)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\nRESPOSTA RECEBIDA!")
            print(f"Keys da resposta: {list(result.keys())}")
            print(f"Success: {result.get('success', False)}")
            print(f"Final Score: {result.get('final_score', 0)}")
            print(f"Movement: {result.get('detected_movement', 'N/A')}")
            
            # TESTAR RECOMENDAÇÕES
            recommendations = result.get('recommendations', None)
            print(f"\nTESTE DAS RECOMENDACOES:")
            print(f"- recommendations existe? {recommendations is not None}")
            print(f"- Tipo: {type(recommendations)}")
            print(f"- Length: {len(recommendations) if recommendations else 'N/A'}")
            
            if recommendations:
                print(f"\nRECOMENDACOES ({len(recommendations)} itens):")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            else:
                print(f"\nNENHUMA RECOMENDACAO ENCONTRADA!")
                
            # Verificar comparações profissionais
            prof_comparisons = result.get('professional_comparisons', [])
            print(f"\nCOMPARACOES PROFISSIONAIS: {len(prof_comparisons)} itens")
            
            # Mostrar estrutura completa (primeiros 1000 chars)
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            print(f"\nJSON COMPLETO (primeiro 1000 chars):")
            print(json_str[:1000])
            print("...")
            
        else:
            print(f"ERRO: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"EXCECAO: {e}")

if __name__ == "__main__":
    test_api_direct()