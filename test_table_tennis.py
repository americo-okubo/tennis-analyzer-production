#!/usr/bin/env python3
"""
Simple test for TableTennisAnalyzer without Unicode issues
"""

import sys
import os

# Set encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

def test_table_tennis_analyzer():
    """Test TableTennisAnalyzer functionality"""
    try:
        from tennis_comparison_backend import TableTennisAnalyzer
        
        print("Iniciando teste do TableTennisAnalyzer...")
        
        # Create analyzer
        analyzer = TableTennisAnalyzer()
        print("Analyzer criado com sucesso")
        
        # Test parameters
        user_metadata = {
            'maoDominante': 'D', 
            'ladoCamera': 'E', 
            'ladoRaquete': 'F', 
            'tipoMovimento': 'D'
        }
        
        prof_metadata = {
            'maoDominante': 'D', 
            'ladoCamera': 'D', 
            'ladoRaquete': 'F', 
            'tipoMovimento': 'D'
        }
        
        print("Executando comparacao...")
        
        # Test with existing video files
        result = analyzer.compare_techniques(
            'videos/Americo_FD_D_E.mp4',
            'videos/Ma_Long_FD_D_D.mp4',
            user_metadata,
            prof_metadata
        )
        
        print("\n=== RESULTADO DA ANALISE ===")
        print(f"Sucesso: {result.get('success', False)}")
        print(f"Score Final: {result.get('final_score', 'N/A')}")
        print(f"Tipo de Analise: {result.get('analysis_type', 'N/A')}")
        
        # Detailed analysis
        detailed = result.get('detailed_analysis', {})
        if 'cycles_analysis' in detailed:
            cycles = detailed['cycles_analysis']
            print(f"Ciclos Usuario: {cycles.get('user_cycles', 0)}")
            print(f"Ciclos Profissional: {cycles.get('professional_cycles', 0)}")
            print(f"Fonte dos Dados: {cycles.get('data_source', 'N/A')}")
        
        # Phase scores
        phases = result.get('phase_scores', {})
        if phases:
            print("\nScores por Fase:")
            for phase, score in phases.items():
                print(f"  {phase}: {score:.1f}%")
        
        # Recommendations
        recommendations = detailed.get('recommendations', [])
        if recommendations:
            print("\nRecomendacoes:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. {rec}")
        
        if result.get('success'):
            print("\nTESTE CONCLUIDO COM SUCESSO!")
        else:
            print(f"\nERRO NO TESTE: {result.get('error', 'Erro desconhecido')}")
            
    except Exception as e:
        print(f"ERRO DURANTE TESTE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_table_tennis_analyzer()