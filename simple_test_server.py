#!/usr/bin/env python3
"""
Simple test server to verify real data works properly
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from tennis_comparison_backend import TableTennisAnalyzer

app = FastAPI(title="Simple Test Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer
# analyzer = None

# @app.on_event("startup")
# async def startup():
#     global analyzer
#     analyzer = TableTennisAnalyzer()
#     print("[STARTUP] Tennis analyzer initialized")

@app.get("/")
async def root():
    return {"message": "Simple Test Server for Real Data", "status": "active"}

@app.post("/auth/login")
async def login():
    """Mock login for interface compatibility"""
    return {"access_token": "test_token", "token_type": "bearer", "expires_in": 3600}

@app.post("/get-professionals-by-movement")
async def get_professionals_by_movement(metadata: dict):
    """Get available professionals based on movement configuration"""
    try:
        print(f"[PROFESSIONALS] Buscando profissionais para: {metadata}")
        
        # Build movement key manually
        side = 'forehand' if metadata['ladoRaquete'] == 'F' else 'backhand'
        movement = 'drive' if metadata['tipoMovimento'] == 'D' else 'push'
        movement_key = f"{side}_{movement}"
        
        print(f"[PROFESSIONALS] Movement key: {movement_key}")
        
        # Search for professionals in the videos folder
        import os
        from pathlib import Path
        
        professionals = []
        base_path = Path(".")
        videos_folder = base_path / "videos"
        
        print(f"[PROFESSIONALS] Buscando em: {videos_folder}")
        
        if videos_folder.exists():
            # Look for videos that match the movement pattern
            pattern_parts = []
            if side == 'forehand':
                pattern_parts.append('FD')  # Forehand Drive
            else:
                pattern_parts.append('BD')  # Backhand Drive
                
            for video_file in videos_folder.glob("*.mp4"):
                filename = video_file.name
                print(f"[PROFESSIONALS] Verificando: {filename}")
                
                # Parse filename: Name_Movement_Hand_Camera.mp4
                # Ex: Zhang_Jike_FD_D_D.mp4
                parts = filename.replace('.mp4', '').split('_')
                if len(parts) >= 4:
                    name_parts = parts[:-3]  # All but last 3 parts
                    movement_part = parts[-3]  # FD, BD, etc
                    hand_part = parts[-2]     # D (destro), E (canhoto)
                    camera_part = parts[-1]   # D (direita), E (esquerda)
                    
                    prof_name = ' '.join(name_parts).replace('_', ' ')
                    
                    # Check if matches movement
                    if (side == 'forehand' and movement_part == 'FD') or (side == 'backhand' and movement_part == 'BD'):
                        professional = {
                            'name': prof_name,
                            'filename': filename,
                            'hand': hand_part,
                            'camera_side': camera_part,
                            'stats': {'tecnica': '95%'},
                            'video_exists': True,
                            'file_path': str(video_file)
                        }
                        professionals.append(professional)
                        print(f"[PROFESSIONALS] Adicionado: {prof_name}")
        
        print(f"[PROFESSIONALS] Total encontrados: {len(professionals)}")
        
        return {
            "success": True,
            "professionals": professionals,
            "count": len(professionals),
            "movement_key": movement_key,
            "message": f"Profissionais encontrados para {movement_key}"
        }
        
    except Exception as e:
        print(f"[PROFESSIONALS] Erro: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "professionals": [],
            "count": 0
        }

@app.get("/professionals")
async def get_professionals():
    """Get available professionals - requires movement configuration"""
    return {
        "success": False,
        "error": "Use POST /get-professionals-by-movement com configuração do movimento",
        "message": "Este endpoint requer configuração de movimento (lado da raquete e tipo de movimento)"
    }

@app.post("/validate-and-get-professionals")
async def validate_and_get_professionals(
    file: UploadFile = File(...),
    metadata: str = Form(...)
):
    """Mock professionals list for interface compatibility"""
    print(f"[VALIDATE] Received file: {file.filename}")
    print(f"[VALIDATE] Received metadata: {metadata}")
    
    return {
        "success": True,
        "professionals": [
            {"name": "Zhang Jike", "stats": {"tecnica": "95%"}},
            {"name": "Ma Long", "stats": {"tecnica": "97%"}},
            {"name": "Fan Zhendong", "stats": {"tecnica": "94%"}}
        ],
        "count": 3,
        "validation_passed": True,
        "message": "Video validated successfully"
    }

@app.post("/analyze")
async def analyze(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    professional_name: str = Form(None),
    analysis_type: str = Form("full")
):
    """Use enhanced analysis for /analyze endpoint - redirects to enhanced analysis"""
    print(f"[ANALYZE] Received file: {user_video.filename}")
    print(f"[ANALYZE] Received metadata: {metadata}")
    print(f"[ANALYZE] Professional: {professional_name}")
    print(f"[ANALYZE] Redirecting to enhanced analysis...")
    
    # Redirect to enhanced analysis with parameters
    return await enhanced_single_cycle_analysis(user_video, metadata, professional_name, analysis_type)

@app.post("/enhanced-single-cycle-analysis")
async def enhanced_single_cycle_analysis(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    professional_name: str = Form(...),
    analysis_type: str = Form("enhanced")
):
    """Enhanced single cycle analysis with biomechanics"""
    try:
        print(f"[ENHANCED] Starting enhanced analysis...")
        print(f"[ENHANCED] User video: {user_video.filename}")
        print(f"[ENHANCED] Professional: {professional_name}")
        print(f"[ENHANCED] Metadata: {metadata}")
        
        # Parse metadata
        import json
        user_metadata = json.loads(metadata)
        
        # Save user video temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await user_video.read()
            temp_file.write(content)
            user_video_path = temp_file.name
        
        print(f"[ENHANCED] User video saved to: {user_video_path}")
        
        # Find professional video
        from pathlib import Path
        videos_folder = Path("videos")
        pro_video_path = None
        
        for video_file in videos_folder.glob("*.mp4"):
            filename = video_file.name
            # Parse filename to get name
            parts = filename.replace('.mp4', '').split('_')
            if len(parts) >= 4:
                name_parts = parts[:-3]
                prof_name = ' '.join(name_parts).replace('_', ' ')
                if prof_name == professional_name:
                    pro_video_path = str(video_file)
                    break
        
        if not pro_video_path:
            return {"success": False, "error": f"Professional video not found for {professional_name}"}
        
        print(f"[ENHANCED] Professional video: {pro_video_path}")
        
        # Create professional metadata based on filename
        filename = Path(pro_video_path).name
        parts = filename.replace('.mp4', '').split('_')
        prof_metadata = {
            'maoDominante': 'Destro' if parts[-2] == 'D' else 'Canhoto',
            'ladoCamera': 'Direita' if parts[-1] == 'D' else 'Esquerda',
            'ladoRaquete': user_metadata['ladoRaquete'],  # Same movement type
            'tipoMovimento': user_metadata['tipoMovimento']
        }
        
        print(f"[ENHANCED] User metadata: {user_metadata}")
        print(f"[ENHANCED] Professional metadata: {prof_metadata}")
        
        # Import the enhanced analyzer
        from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
        analyzer = EnhancedSingleCycleAnalyzer()
        
        result = analyzer.compare_enhanced_single_cycles(
            user_video_path, pro_video_path, user_metadata, prof_metadata, cycle_index=1
        )
        
        # Clean up temporary file
        import os
        try:
            os.unlink(user_video_path)
        except:
            pass
        
        if result['success']:
            print(f"[ENHANCED] Analysis successful: {result['final_score']:.1f}%")
            
            # Convert to format expected by frontend  
            frontend_result = {
                'success': True,
                'final_score': result['final_score'],
                'analysis_type': 'enhanced_single_cycle_biomechanical',
                'cycle_index': result['cycle_index'],
                'performance_category': result['combined_comparison']['performance_category'],
                'analysis_confidence': result['combined_comparison']['analysis_confidence'],
                
                # User analysis
                'user_analysis': {
                    'cycles_count': result['user_analysis']['total_cycles_found'],
                    'cycle_index_analyzed': result['cycle_index'],
                    'average_duration': result['user_analysis']['cycle_data']['duration'],
                    'quality_score': result['user_analysis']['cycle_data']['quality'],
                    'amplitude': result['user_analysis']['cycle_data']['amplitude'],
                    'enhanced_metrics': result['user_analysis']['enhanced_metrics'],
                    'cycles_details': [result['user_analysis']['cycle_data']]
                },
                
                # Professional analysis  
                'professional_analysis': {
                    'cycles_count': result['professional_analysis']['total_cycles_found'],
                    'cycle_index_analyzed': result['cycle_index'],
                    'average_duration': result['professional_analysis']['cycle_data']['duration'],
                    'quality_score': result['professional_analysis']['cycle_data']['quality'],
                    'amplitude': result['professional_analysis']['cycle_data']['amplitude'],
                    'enhanced_metrics': result['professional_analysis']['enhanced_metrics'],
                },
                
                # Comparison results
                'comparison': {
                    'similarity_score': result['combined_comparison']['overall_similarity'],
                    'detailed_similarities': result['similarity_breakdown'],
                    'comparison_confidence': result['combined_comparison']['analysis_confidence']
                },
                
                # Detailed analysis
                'detailed_analysis': {
                    'overall_assessment': f"Enhanced Analysis - {result['combined_comparison']['performance_category']}",
                    'recommendations': result['recommendations'],
                    'detailed_metrics': result['detailed_metrics'],
                    'traditional_comparison': result.get('traditional_comparison', {}),
                    'biomechanical_comparison': result.get('biomechanical_comparison', {})
                },
                
                'timestamp': result.get('timestamp', '2025-07-26T22:35:00'),
                'comparison_id': f"enhanced_cycle_{result.get('cycle_index', 1)}"
            }
            
            return frontend_result
        else:
            print(f"[ENHANCED] Analysis failed: {result.get('error')}")
            return {'success': False, 'error': result.get('error')}
            
    except Exception as e:
        print(f"[ENHANCED] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

@app.post("/single-cycle-analysis")
async def single_cycle_analysis():
    """Test single cycle analysis with different videos"""
    try:
        print("[SINGLE_CYCLE] Starting single cycle analysis...")
        
        # Import the new analyzer
        from single_cycle_analysis import SingleCycleAnalyzer
        analyzer = SingleCycleAnalyzer()
        
        # Use different videos for real comparison
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
        
        print(f"[SINGLE_CYCLE] User video: {os.path.basename(user_video)}")
        print(f"[SINGLE_CYCLE] Professional video: {os.path.basename(pro_video)}")
        
        result = analyzer.compare_single_cycles(
            user_video, pro_video, user_metadata, prof_metadata, cycle_index=1
        )
        
        if result['success']:
            print(f"[SINGLE_CYCLE] Analysis successful: {result['final_score']:.1f}%")
            
            # Convert to format expected by frontend
            frontend_result = {
                'success': True,
                'final_score': result['final_score'],
                'analysis_type': 'single_cycle_biomechanical',
                'user_analysis': {
                    'cycles_count': result['user_analysis']['total_cycles'],
                    'cycle_index_analyzed': result['cycle_index'],
                    'average_duration': result['user_analysis']['cycle_data']['duration'],
                    'quality_score': result['user_analysis']['cycle_data']['quality'],
                    'rhythm_variability': result['user_analysis']['parameters'].get('frequency', 0.0),
                    'acceleration_smoothness': result['user_analysis']['parameters'].get('movement_smoothness', 0.0),
                    'movement_efficiency': result['user_analysis']['parameters'].get('movement_efficiency', 0.0),
                    'amplitude_consistency': result['user_analysis']['cycle_data']['amplitude'] / 50000,  # Normalize
                    'cycles_details': [result['user_analysis']['cycle_data']]
                },
                'professional_analysis': {
                    'cycles_count': result['professional_analysis']['total_cycles'],
                    'cycle_index_analyzed': result['cycle_index'],
                    'average_duration': result['professional_analysis']['cycle_data']['duration'],
                    'quality_score': result['professional_analysis']['cycle_data']['quality'],
                    'rhythm_variability': result['professional_analysis']['parameters'].get('frequency', 0.0),
                    'acceleration_smoothness': result['professional_analysis']['parameters'].get('movement_smoothness', 0.0),
                    'movement_efficiency': result['professional_analysis']['parameters'].get('movement_efficiency', 0.0),
                    'amplitude_consistency': result['professional_analysis']['cycle_data']['amplitude'] / 50000,  # Normalize
                },
                'comparison': {
                    'similarity_score': result['comparison']['similarity_score'],
                    'detailed_similarities': result['comparison']['detailed_similarities'],
                    'comparison_confidence': 1.0
                },
                'phase_scores': {
                    'preparation': result['final_score'] * 0.9,
                    'contact': result['final_score'] * 1.1,
                    'follow_through': result['final_score'] * 1.0
                },
                'detailed_analysis': {
                    'overall_assessment': 'Análise de Ciclo Único',
                    'recommendations': [f"Baseado no ciclo {result['cycle_index']} de cada vídeo"],
                    'key_metrics': result['comparison']['differences'],
                    'biomech_breakdown': result['comparison']['detailed_similarities']
                },
                'timestamp': result.get('timestamp', '2025-07-26T21:23:29'),
                'comparison_id': f"single_cycle_{result.get('cycle_index', 1)}"
            }
            
            return frontend_result
        else:
            print(f"[SINGLE_CYCLE] Analysis failed: {result.get('error')}")
            return {'success': False, 'error': result.get('error')}
            
    except Exception as e:
        print(f"[SINGLE_CYCLE] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

@app.post("/test-real-data")  
async def test_real_data():
    """Test with exact same parameters that worked in direct test"""
    try:
        print("[TEST] Starting real data test...")
        
        # Use DIFFERENT videos to test real differences
        user_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"  
        pro_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Fan_Zhendong_FD_D_E.mp4"  # Different video
        
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
        print(f"[TEST] Pro video exists: {os.path.exists(pro_video)}")
        
        # CRITICAL DEBUG: Let's verify what videos are actually being used
        print(f"[DEBUG_VIDEOS] User video: {user_video}")
        print(f"[DEBUG_VIDEOS] Professional video: {pro_video}")
        print(f"[DEBUG_VIDEOS] Are they the same file? {user_video == pro_video}")
        
        # Call tennis_comparison_backend directly
        result = analyzer.compare_techniques(
            user_video, pro_video, user_metadata, prof_metadata
        )
        
        print(f"[TEST] Result success: {result.get('success')}")
        print(f"[TEST] Result score: {result.get('final_score')}")
        
        # DETAILED ANALYSIS OF SUSPICIOUS VALUES
        user_analysis = result.get('user_analysis', {})
        pro_analysis = result.get('professional_analysis', {})
        
        print(f"\n[SUSPICIOUS_VALUES_CHECK]")
        print(f"User duration: {user_analysis.get('average_duration')}")
        print(f"Pro duration: {pro_analysis.get('average_duration')}")
        print(f"User quality: {user_analysis.get('quality_score')}")
        print(f"Pro quality: {pro_analysis.get('quality_score')}")
        print(f"User cycles_count: {user_analysis.get('cycles_count')}")
        print(f"Pro cycles_count: {pro_analysis.get('cycles_count')}")
        
        # Check if values are identical (which shouldn't happen with different videos)
        if user_analysis.get('average_duration') == pro_analysis.get('average_duration'):
            print(f"[WARNING] IDENTICAL DURATIONS - This suggests same video or wrong processing!")
        
        if user_analysis.get('quality_score') == pro_analysis.get('quality_score'):
            print(f"[WARNING] IDENTICAL QUALITY SCORES - This suggests same video!")
        
        # Log cycle details to understand what's happening
        user_cycles = user_analysis.get('cycles_details', [])
        print(f"\n[USER_CYCLES_DETAILS] Found {len(user_cycles)} cycles:")
        for i, cycle in enumerate(user_cycles):
            print(f"  Cycle {i}: duration={cycle.get('duration')}, amplitude={cycle.get('amplitude')}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8007)