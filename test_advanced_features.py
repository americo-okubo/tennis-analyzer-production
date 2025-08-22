#!/usr/bin/env python3
"""
🧪 TESTE DAS FUNCIONALIDADES AVANÇADAS
Script para testar todas as funcionalidades do TableTennisAnalyzer v2.0
"""

import sys
import time
from pathlib import Path

def print_header(title):
    """Imprime cabeçalho formatado"""
    print(f"\n{'='*60}")
    print(f"🎾 {title}")
    print(f"{'='*60}")

def print_step(step_num, description):
    """Imprime passo formatado"""
    print(f"\n🔸 PASSO {step_num}: {description}")
    print("-" * 50)

def test_system_basic():
    """Teste básico do sistema original"""
    print_step(1, "TESTANDO SISTEMA BÁSICO ORIGINAL")
    
    try:
        # Importar sistema original
        from tennis_comparison_backend import TableTennisAnalyzer
        
        print("✅ Importação do TableTennisAnalyzer: SUCESSO")
        
        # Criar instância
        analyzer = TableTennisAnalyzer()
        print("✅ Inicialização do sistema: SUCESSO")
        
        # Verificar método principal
        if hasattr(analyzer, 'compare_techniques'):
            print("✅ Método compare_techniques: DISPONÍVEL")
        else:
            print("❌ Método compare_techniques: NÃO ENCONTRADO")
            
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def test_videos_availability():
    """Verifica disponibilidade de vídeos"""
    print_step(2, "VERIFICANDO VÍDEOS DISPONÍVEIS")
    
    videos_dir = Path("videos")
    if not videos_dir.exists():
        print("❌ Diretório 'videos/' não encontrado")
        return False
    
    # Listar vídeos MP4
    mp4_files = list(videos_dir.glob("*.mp4"))
    print(f"📹 Total de vídeos encontrados: {len(mp4_files)}")
    
    # Separar por tipo
    user_videos = [v for v in mp4_files if any(name in v.name for name in ['Americo', 'Baixinha', 'Gordo'])]
    pro_videos = [v for v in mp4_files if any(name in v.name for name in ['Ma_Long', 'Fan_Zhendong', 'Zhang_Jike', 'Timo_Boll'])]
    
    print(f"👤 Vídeos de usuários: {len(user_videos)}")
    for video in user_videos[:3]:  # Mostrar até 3
        print(f"   • {video.name}")
    
    print(f"🏆 Vídeos de profissionais: {len(pro_videos)}")
    for video in pro_videos[:3]:  # Mostrar até 3
        print(f"   • {video.name}")
    
    if len(mp4_files) == 0:
        print("⚠️ Nenhum vídeo encontrado. Algumas funcionalidades não poderão ser testadas.")
        return False
    
    return True

def test_original_comparison():
    """Teste de comparação original (se vídeos disponíveis)"""
    print_step(3, "TESTE DE COMPARAÇÃO ORIGINAL")
    
    # Vídeos padrão para teste
    user_video = "videos/Americo_FD_D_E.mp4"
    pro_video = "videos/Ma_Long_FD_D_D.mp4"
    
    if not Path(user_video).exists() or not Path(pro_video).exists():
        print("⚠️ Vídeos de teste não encontrados. Pulando teste...")
        print(f"   Esperado: {user_video}")
        print(f"   Esperado: {pro_video}")
        return False
    
    try:
        from tennis_comparison_backend import TableTennisAnalyzer
        
        analyzer = TableTennisAnalyzer()
        
        print(f"👤 Usuário: {user_video}")
        print(f"🏆 Profissional: {pro_video}")
        print("🔄 Executando comparação...")
        
        start_time = time.time()
        result = analyzer.compare_techniques(
            user_video, pro_video,
            {'movement': 'forehand_drive'},
            {'movement': 'forehand_drive'}
        )
        elapsed = time.time() - start_time
        
        if result.get('success', False):
            print(f"✅ Comparação concluída em {elapsed:.2f}s")
            print(f"📊 Score: {result.get('final_score', 0):.1f}%")
            print(f"🔬 Tipo: {result.get('analysis_type', 'N/A')}")
            print(f"👤 Ciclos usuário: {result.get('cycles_detected', {}).get('user', 0)}")
            print(f"🏆 Ciclos profissional: {result.get('cycles_detected', {}).get('professional', 0)}")
            return True
        else:
            print(f"❌ Comparação falhou: {result.get('error', 'Erro desconhecido')}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def test_professional_extractor():
    """Teste do extrator de dados profissionais"""
    print_step(4, "TESTE DO PROFESSIONAL DATA EXTRACTOR")
    
    try:
        # Verificar se os novos módulos foram criados
        extractor_file = Path("professional_data_extractor.py")
        if not extractor_file.exists():
            print("❌ Arquivo professional_data_extractor.py não encontrado")
            print("💡 Certifique-se de que os novos módulos foram salvos corretamente")
            return False
        
        print("✅ Arquivo professional_data_extractor.py encontrado")
        
        # Tentar importar
        try:
            from professional_data_extractor import ProfessionalDataExtractor
            print("✅ Importação do ProfessionalDataExtractor: SUCESSO")
            
            extractor = ProfessionalDataExtractor()
            print("✅ Inicialização do extrator: SUCESSO")
            
            # Escanear vídeos profissionais
            pro_videos = extractor.scan_professional_videos()
            print(f"🔍 Vídeos profissionais encontrados: {len(pro_videos)}")
            
            return True
            
        except ImportError as e:
            print(f"❌ Erro de importação: {e}")
            print("💡 Verifique se todas as dependências estão disponíveis")
            return False
            
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def test_fast_engine():
    """Teste do motor de comparação rápida"""
    print_step(5, "TESTE DO FAST COMPARISON ENGINE")
    
    try:
        fast_engine_file = Path("fast_comparison_engine.py")
        if not fast_engine_file.exists():
            print("❌ Arquivo fast_comparison_engine.py não encontrado")
            return False
        
        print("✅ Arquivo fast_comparison_engine.py encontrado")
        
        try:
            from fast_comparison_engine import FastComparisonEngine
            print("✅ Importação do FastComparisonEngine: SUCESSO")
            
            engine = FastComparisonEngine()
            print("✅ Inicialização do motor rápido: SUCESSO")
            
            return True
            
        except ImportError as e:
            print(f"❌ Erro de importação: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def test_advanced_demo():
    """Teste do sistema de demonstração avançado"""
    print_step(6, "TESTE DO ADVANCED DEMO")
    
    try:
        demo_file = Path("advanced_features_demo.py")
        if not demo_file.exists():
            print("❌ Arquivo advanced_features_demo.py não encontrado")
            return False
        
        print("✅ Arquivo advanced_features_demo.py encontrado")
        
        try:
            from advanced_features_demo import AdvancedTennisAnalyzer
            print("✅ Importação do AdvancedTennisAnalyzer: SUCESSO")
            
            print("⚠️ Inicialização completa pode demorar (carregando todos os componentes)...")
            
            return True
            
        except ImportError as e:
            print(f"❌ Erro de importação: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def generate_test_report(results):
    """Gera relatório dos testes"""
    print_header("RELATÓRIO FINAL DOS TESTES")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"📊 RESUMO DOS TESTES:")
    print(f"   Total: {total_tests}")
    print(f"   ✅ Passou: {passed_tests}")
    print(f"   ❌ Falhou: {total_tests - passed_tests}")
    print(f"   📈 Taxa de sucesso: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n📋 DETALHES:")
    for test_name, result in results.items():
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"   {test_name}: {status}")
    
    # Recomendações baseadas nos resultados
    print(f"\n💡 RECOMENDAÇÕES:")
    
    if results.get('basic_system', False):
        print("   ✅ Sistema básico funcionando - pode usar funcionalidades originais")
    else:
        print("   ❌ Sistema básico com problemas - verificar instalação base")
    
    if results.get('videos', False):
        print("   ✅ Vídeos disponíveis - pode testar comparações reais")
    else:
        print("   ⚠️ Poucos/nenhum vídeo - funcionalidades limitadas")
    
    if results.get('extractor', False) and results.get('fast_engine', False):
        print("   ✅ Módulos avançados OK - pode usar funcionalidades rápidas")
        print("   🚀 Próximo passo: executar advanced_features_demo.py")
    else:
        print("   ⚠️ Módulos avançados com problemas - verificar dependências")
    
    if passed_tests == total_tests:
        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema totalmente operacional")
        print("🚀 Pronto para usar funcionalidades avançadas!")
    elif passed_tests >= total_tests * 0.7:
        print(f"\n👍 MAIORIA DOS TESTES PASSOU!")
        print("✅ Sistema majoritariamente funcional")
        print("🔧 Algumas funcionalidades podem precisar de ajustes")
    else:
        print(f"\n⚠️ VÁRIOS TESTES FALHARAM")
        print("🔧 Sistema precisa de correções antes do uso")
        print("📞 Verifique instalação e dependências")

def main():
    """Execução principal dos testes"""
    print_header("TABLETENNISANALYZER v2.0 - TESTE DE FUNCIONALIDADES")
    
    print("🧪 Executando bateria completa de testes...")
    print("⏱️ Tempo estimado: 1-2 minutos")
    
    # Executar todos os testes
    test_results = {}
    
    test_results['basic_system'] = test_system_basic()
    test_results['videos'] = test_videos_availability()
    test_results['original_comparison'] = test_original_comparison()
    test_results['extractor'] = test_professional_extractor()
    test_results['fast_engine'] = test_fast_engine()
    test_results['advanced_demo'] = test_advanced_demo()
    
    # Gerar relatório final
    generate_test_report(test_results)
    
    print(f"\n🏁 TESTES CONCLUÍDOS!")
    print("=" * 60)

if __name__ == "__main__":
    main()
