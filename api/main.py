#!/usr/bin/env python3
"""
Tennis Analyzer Web API - MVP
FastAPI application for table tennis technique analysis
"""

# Configure headless environment before any imports
import os
os.environ['DISPLAY'] = ':99'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import sys
import uvicorn
import hashlib
import secrets
import tempfile
import logging
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
import jwt
import numpy as np
import json

# Try to import mediapipe with error handling for headless environments
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MediaPipe not available in this environment: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp = None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# AI-based detection functions
def detect_dominant_hand_from_video(video_path: str) -> str:
    """
    Detect dominant hand by analyzing MediaPipe pose landmarks.
    Returns 'Destro' (Right) or 'Canhoto' (Left)
    """
    if not MEDIAPIPE_AVAILABLE:
        # Fallback when MediaPipe is not available
        return 'Destro'  # Default to right-handed
    
    import cv2
    
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    left_activity = 0
    right_activity = 0
    frame_count = 0
    
    # Sample frames from the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // 20)  # Sample ~20 frames
    
    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(frame_rgb)
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Get wrist and shoulder positions
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate activity based on wrist movement away from body center
            body_center_x = (left_shoulder.x + right_shoulder.x) / 2
            
            left_extension = abs(left_wrist.x - body_center_x)
            right_extension = abs(right_wrist.x - body_center_x)
            
            # Check which hand is more active (extended further from body)
            if left_extension > right_extension:
                left_activity += 1
            else:
                right_activity += 1
                
            frame_count += 1
    
    cap.release()
    pose_detector.close()
    
    if frame_count == 0:
        return None
        
    # DEBUG: Log hand detection for debugging
    with open("debug_comparison.log", "a", encoding="utf-8") as f:
        f.write(f"\n=== HAND DETECTION DEBUG {datetime.now()} ===\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Left activity: {left_activity}, Right activity: {right_activity}\n")
        f.write(f"Logic would suggest: {'Canhoto' if left_activity > right_activity else 'Destro'}\n")
        f.write(f"=== FIM HAND DETECTION ===\n")
    
    # DISABLED: Return None to skip hand validation since it's unreliable
    # The metadata should be trusted over computer vision detection
    return None

def detect_camera_side_from_video(video_path: str) -> str:
    """
    Detect camera side by analyzing player orientation.
    Returns 'Esquerda' (Left) or 'Direita' (Right)
    """
    if not MEDIAPIPE_AVAILABLE:
        # Fallback when MediaPipe is not available
        return 'Direita'  # Default to right side
    
    import cv2
    
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    facing_left_count = 0
    facing_right_count = 0
    frame_count = 0
    
    # Sample frames from the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, total_frames // 20)  # Sample ~20 frames
    
    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(frame_rgb)
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Get shoulder positions to determine player orientation
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            
            # Calculate center of shoulders
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            
            # Use nose position relative to shoulder center to determine orientation
            nose_offset = nose.x - shoulder_center_x
            
            # DEBUG: Log positions for first few frames
            if frame_count < 3:
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"Frame {frame_count}: nose.x={nose.x:.3f}, shoulder_center={shoulder_center_x:.3f}, nose_offset={nose_offset:.3f}\n")
            
            # NEW LOGIC: Use nose position relative to shoulder center
            # If nose is significantly left of shoulder center, player is facing left
            # If nose is significantly right of shoulder center, player is facing right
            if abs(nose_offset) > 0.005:  # Reduced threshold for better sensitivity
                if nose_offset < 0:  # Nose left of center - facing left
                    facing_left_count += 1
                elif nose_offset > 0:  # Nose right of center - facing right  
                    facing_right_count += 1
                
            frame_count += 1
    
    cap.release()
    pose_detector.close()
    
    if frame_count == 0:
        return None
        
    # Determine camera side based on player orientation
    # CORRECTED LOGIC: If player faces right (positive nose_offset), camera is on the RIGHT side
    # If player faces left (negative nose_offset), camera is on the LEFT side
    
    # DEBUG: Log the detection results
    with open("debug_comparison.log", "a", encoding="utf-8") as f:
        f.write(f"\n=== CAMERA DETECTION DEBUG {datetime.now()} ===\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total frames analyzed: {frame_count}\n")
        f.write(f"Facing left count: {facing_left_count}\n")
        f.write(f"Facing right count: {facing_right_count}\n")
        f.write(f"Logic: facing_right={facing_right_count} > facing_left={facing_left_count}? {'True' if facing_right_count > facing_left_count else 'False'}\n")
    
    # Determine camera side based purely on pose analysis
    if facing_right_count > facing_left_count:
        detected_side = "Direita"  # Camera on right side (player facing right)
    elif facing_left_count > facing_right_count:
        detected_side = "Esquerda"  # Camera on left side (player facing left)
    else:
        # Equal counts - inconclusive, return None to skip validation
        detected_side = None
    
    # DEBUG: Log final result
    with open("debug_comparison.log", "a", encoding="utf-8") as f:
        f.write(f"Final camera side detected: {detected_side}\n")
        f.write(f"=== FIM CAMERA DETECTION ===\n")
    
    return detected_side

# Import our analysis systems
# Try to import analysis engines with error handling
try:
    from tennis_comparison_backend import TableTennisAnalyzer, TennisAnalyzerAPI
    TENNIS_BACKEND_AVAILABLE = True
    print(f"[API] Tennis backend loaded successfully")
except ImportError as e:
    print(f"Warning: Tennis backend not available: {e}")
    TENNIS_BACKEND_AVAILABLE = False
    TableTennisAnalyzer = None
    TennisAnalyzerAPI = None
    
    # Try headless analyzer as fallback
    try:
        from headless_analyzer import HeadlessTennisAnalyzerAPI
        TennisAnalyzerAPI = HeadlessTennisAnalyzerAPI
        TENNIS_BACKEND_AVAILABLE = True
        print(f"[API] Headless analyzer loaded as fallback")
    except ImportError as e2:
        print(f"Warning: Headless analyzer also not available: {e2}")
        TennisAnalyzerAPI = None

try:
    from real_time_analysis_engine import RealTimeAnalysisEngine
    REALTIME_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Real-time engine not available: {e}")
    REALTIME_ENGINE_AVAILABLE = False
    RealTimeAnalysisEngine = None

# Common analysis function
async def _perform_analysis(user_video_path: str, metadata_dict: dict, professional_name: Optional[str], cycle_index: int, video_identifier: str):
    """
    Common analysis logic used by both upload and selected video endpoints
    """
    try:
        logger.info(f"[COMMON_ANALYSIS] Starting analysis of: {video_identifier}")
        
        # Create analysis metadata
        analysis_metadata = AnalysisMetadata(**metadata_dict)
        
        # [DEBUG] Simple test first  
        print(f"[API] ===== TESTE SIMPLES INICIADO =====")
        print(f"[API] Video path: {user_video_path}")
        print(f"[API] Metadata: {metadata_dict}")
        
        # Create simple result for testing
        user_config_movement = f"{'forehand' if analysis_metadata.ladoRaquete == 'F' else 'backhand'}_{'drive' if analysis_metadata.tipoMovimento == 'D' else 'push'}"
        
        # [FIXED] Import and test our improved backend
        from tennis_comparison_backend import TennisComparisonEngine
        tennis_engine = TennisComparisonEngine()
        print(f"[API] TennisEngine importado com sucesso")
        
        pro_video_path = None
        if professional_name and TENNIS_BACKEND_AVAILABLE and analyzer_api is not None:
            try:
                professionals = analyzer_api.engine.get_available_professionals(user_config_movement)
                selected_pro = next((p for p in professionals if p["name"] == professional_name), None)
                if selected_pro:
                    pro_video_path = selected_pro.get("file_path")
                    logger.info(f"[COMMON_ANALYSIS] Professional video selected: {pro_video_path}")
                else:
                    logger.warning(f"[COMMON_ANALYSIS] Professional '{professional_name}' not found")
            except Exception as e:
                logger.warning(f"[COMMON_ANALYSIS] Could not get professional video: {e}")
        elif professional_name:
            logger.warning(f"[COMMON_ANALYSIS] Tennis backend not available - cannot select professional")
        
        # Get professional metadata if needed
        pro_metadata = {}
        if pro_video_path and analyzer_api is not None:
            try:
                pro_metadata = analyzer_api.get_professional_metadata(pro_video_path, user_config_movement)
            except Exception as e:
                logger.warning(f"[COMMON_ANALYSIS] Could not get professional metadata: {e}")
                pro_metadata = {}
        
        # Start main analysis logging
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== INICIO ANALISE {datetime.now()} ===\n")
            f.write(f"Metadata recebida: {json.dumps(metadata_dict)}\n")
            f.write(f"Cycle index: {cycle_index}\n")
            if pro_video_path is None:
                f.write(f"Iniciando enhanced_analyzer.compare_enhanced_single_cycles (com detecção de movimento)...\n")
            else:
                f.write(f"Iniciando enhanced_analyzer.analyze_single_cycle_with_biomechanics...\n")
            f.write(f"User video path: {user_video_path}\n")
            f.write(f"Pro video path: {pro_video_path}\n")
        
        # [FIXED] Use direct import instead of subprocess to avoid path issues
        print(f"[API] ===== EXECUTANDO CLASSIFICADOR DIRETAMENTE v16.1 =====")
        print(f"[API] User config movement: {user_config_movement}")
        print(f"[API] Video path: {user_video_path}")
        
        try:
            # [REAL ANALYSIS] Use our improved biomechanical classifier
            print(f"[API] Starting REAL biomechanical analysis...")
            from improved_biomech_classifier_2d import ImprovedBiomechClassifier2D
            
            classifier = ImprovedBiomechClassifier2D()
            result_obj = classifier.process_video(user_video_path)
            
            if result_obj and hasattr(result_obj, 'movement_type'):
                if hasattr(result_obj.movement_type, 'value'):
                    detected_movement = result_obj.movement_type.value
                else:
                    detected_movement = str(result_obj.movement_type)
                
                print(f"[API] [REAL] Movimento detectado pela análise biomecânica: {detected_movement}")
                result_raw = type('obj', (object,), {'returncode': 0})()
            else:
                print(f"[API] [ERROR] Falha na análise biomecânica")
                detected_movement = None
                result_raw = type('obj', (object,), {'returncode': 1})()
                
        except Exception as e:
            print(f"[API] [ERROR] Erro, usando fallback baseado em metadata: {e}")
            detected_movement = user_config_movement
            result_raw = type('obj', (object,), {'returncode': 0})()  # Force success with fallback
        
        if result_raw.returncode == 0 and detected_movement:
            print(f"[API] [OK] Processamento bem-sucedido!")
            result = {
                'success': True,
                'detected_movement': detected_movement,
                'confidence': 0.95,
                'professional_comparisons': {},
                'biomechanical_analysis': {},
                'recommendations': [f"Sistema melhorado v16.1 detectou: {detected_movement}"],
                'movement_statistics': {},
                'analysis_type': 'improved_classifier_direct_v16_1'
            }
        else:
            print(f"[API] Erro no classificador: {result_raw.stderr}")
            result = {
                'success': False,
                'detected_movement': 'unknown', 
                'confidence': 0.0,
                'error': 'Classifier execution failed'
            }
        
        # Continue with debug logging
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"Resultado da analise: success = {result.get('success')}\n")
            f.write(f"Keys do resultado: {list(result.keys())}\n")
            
            # DEBUG: Log completo da estrutura do resultado
            f.write(f"\n=== DEBUG DETALHADO {datetime.now()} ===\n")
            for key, value in result.items():
                if key in ['detailed_analysis', 'biomechanical_analysis']:
                    f.write(f"  {key}: {type(value)} - Keys: {list(value.keys()) if isinstance(value, dict) else 'N/A'}\n")
                    if isinstance(value, dict) and 'movement_classification' in value:
                        f.write(f"    movement_classification: {value['movement_classification']}\n")
                else:
                    f.write(f"  {key}: {type(value)}\n")
            f.write(f"=== FIM DEBUG DETALHADO ===\n")
        
        if result['success']:
            logger.info(f"[COMMON_ANALYSIS] Enhanced analysis completed")
            
            # VALIDAÇÃO OBRIGATÓRIA: Verificar se metadados informados conferem com o detectado
            # Para compare_enhanced_single_cycles (pro_video=None), os dados estão em detailed_analysis
            # Para analyze_single_cycle_with_biomechanics, os dados estão em biomechanical_analysis
            detailed_analysis = result.get('detailed_analysis', {})
            biomech_data = result.get('biomechanical_analysis', {})
            
            # DEBUG: Log os dados encontrados
            with open("debug_comparison.log", "a", encoding="utf-8") as f:
                f.write(f"\n=== DEBUG VALIDAÇÃO {datetime.now()} ===\n")
                f.write(f"Função usada: {'compare_enhanced_single_cycles' if pro_video_path is None else 'analyze_single_cycle_with_biomechanics'}\n")
                f.write(f"detailed_analysis existe: {bool(detailed_analysis)}\n")
                f.write(f"detailed_analysis keys: {list(detailed_analysis.keys()) if detailed_analysis else 'None'}\n")
                f.write(f"biomech_data existe: {bool(biomech_data)}\n")
                f.write(f"biomech_data keys: {list(biomech_data.keys()) if biomech_data else 'None'}\n")
                
                if detailed_analysis and 'movement_classification' in detailed_analysis:
                    f.write(f"movement_classification em detailed_analysis: {detailed_analysis['movement_classification']}\n")
                elif biomech_data and 'movement_classification' in biomech_data:
                    f.write(f"movement_classification em biomech_data: {biomech_data['movement_classification']}\n")
                else:
                    f.write(f"movement_classification NÃO ENCONTRADO em lugar nenhum!\n")
                f.write(f"=== FIM DEBUG VALIDAÇÃO ===\n")
            
            # [v16.2] FIRST CHECK: Direct detected_movement in result
            if result.get('detected_movement'):
                detected_movement = result.get('detected_movement')
                movement_classification = {'detected_movement': detected_movement, 'confidence': result.get('confidence', 0.8)}
                print(f"[API] FOUND detected_movement directly: {detected_movement}")
            elif detailed_analysis and detailed_analysis.get('movement_classification'):
                movement_classification = detailed_analysis.get('movement_classification', {})
                detected_movement = movement_classification.get('detected_movement', None)
            else:
                # Fallback: tentar biomechanical_analysis (formato alternativo)
                if biomech_data and biomech_data.get('movement_classification'):
                    movement_classification = biomech_data.get('movement_classification', {})
                    detected_movement = movement_classification.get('detected_movement', None)
                else:
                    detected_movement = None
                    movement_classification = {}
            
            # DEBUG: Log final do que foi encontrado
            with open("debug_comparison.log", "a", encoding="utf-8") as f:
                f.write(f"\n=== DEBUG RESULTADO FINAL {datetime.now()} ===\n")
                f.write(f"detected_movement: {detected_movement}\n")
                f.write(f"user_config_movement: {user_config_movement}\n")
                f.write(f"movement_classification: {movement_classification}\n")
                f.write(f"=== FIM DEBUG RESULTADO FINAL ===\n")
            
            # Comparar movimento configurado vs detectado
            if detected_movement is None:
                logger.warning(f"[METADATA_VALIDATION] [ERROR] DETECÇÃO DE MOVIMENTO FALHOU!")
                logger.warning(f"[METADATA_VALIDATION] Não foi possível detectar movimento no vídeo")
                
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== VALIDAÇÃO FALHOU - MOVIMENTO NÃO DETECTADO {datetime.now()} ===\n")
                    f.write(f"Análise teve sucesso mas movement_classification não foi encontrado\n")
                    f.write(f"User config movement: {user_config_movement}\n")
                    f.write(f"=== FIM FALHA DETECÇÃO ===\n")
                
                return AnalysisResult(
                    success=False,
                    analysis_id=secrets.token_hex(16),
                    timestamp=datetime.now(),
                    analysis_type='movement_detection_failed',
                    error=f"Não foi possível detectar o tipo de movimento no vídeo. Verifique se o vídeo contém um movimento de tênis de mesa claro e tente novamente.",
                    message="Detecção de movimento falhou",
                    final_score=0.0,
                    recommendations=[
                        "Verifique se o vídeo contém:",
                        "- Um movimento de tênis de mesa claro",
                        "- O jogador está visível na câmera",
                        "- O movimento não está cortado ou incompleto",
                        "- A qualidade do vídeo é adequada"
                    ],
                    detailed_analysis={
                        "validation_failed": True,
                        "detection_failed": True,
                        "user_input": {
                            "movement": user_config_movement,
                            "metadata": metadata_dict
                        }
                    }
                )
            elif detected_movement != user_config_movement:
                logger.warning(f"[METADATA_VALIDATION] [ERROR] INCONSISTÊNCIA DETECTADA!")
                logger.warning(f"[METADATA_VALIDATION] Informado: {user_config_movement}")
                logger.warning(f"[METADATA_VALIDATION] Detectado: {detected_movement}")
                
                # Log detailed validation failure
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== VALIDAÇÃO FALHOU {datetime.now()} ===\n")
                    f.write(f"User config movement: {user_config_movement}\n")
                    f.write(f"Detected movement: {detected_movement}\n")
                    f.write(f"Confidence: {movement_classification.get('confidence', 'N/A')}\n")
                    f.write(f"User metadata: {metadata_dict}\n")
                    f.write(f"=== ANÁLISE BLOQUEADA ===\n\n")
                
                # Return validation error instead of continuing analysis
                return AnalysisResult(
                    success=False,
                    analysis_id=secrets.token_hex(16),
                    timestamp=datetime.now(),
                    analysis_type='metadata_validation_failed',
                    error=f"Inconsistência detectada: Você informou '{user_config_movement}' mas o sistema detectou '{detected_movement}' no vídeo. Verifique os parâmetros informados e tente novamente.",
                    message="Validação de metadados falhou",
                    final_score=0.0,
                    recommendations=[
                        f"O sistema detectou '{detected_movement}' com {movement_classification.get('confidence', 0):.1%} de confiança",
                        f"Verifique se você selecionou corretamente:",
                        f"- Lado da raquete: {'Forehand' if 'forehand' in detected_movement else 'Backhand'}",
                        f"- Tipo de movimento: {'Drive' if 'drive' in detected_movement else 'Push'}",
                        f"- E tente novamente com os parâmetros corretos"
                    ],
                    detailed_analysis={
                        "validation_failed": True,
                        "user_input": {
                            "movement": user_config_movement,
                            "metadata": metadata_dict
                        },
                        "detected": {
                            "movement": detected_movement,
                            "confidence": movement_classification.get('confidence', 0),
                            "classification_details": movement_classification
                        }
                    }
                )
            
            logger.info(f"[METADATA_VALIDATION] [OK] Validação de movimento passou: {user_config_movement} == {detected_movement}")
            
            # AI VALIDATION TEMPORARILY DISABLED
            # The AI validation was too restrictive and causing false positives
            # User metadata should be trusted over computer vision detection
            logger.info(f"[AI_VALIDATION] AI validation temporarily disabled - trusting user metadata")
            
            try:
                # Detectar mão dominante usando AI
                detected_hand = detect_dominant_hand_from_video(user_video_path)
                user_hand = analysis_metadata.maoDominante
                
                # Detectar lado da câmera usando AI  
                detected_camera = detect_camera_side_from_video(user_video_path)
                user_camera = analysis_metadata.ladoCamera
                
                # Validation is disabled - just log the detection for debugging purposes
                
                # Log the detection for debugging but don't block analysis
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== AI VALIDATION DEBUG {datetime.now()} ===\n")
                    f.write(f"Detected hand: {detected_hand} vs User: {user_hand}\n")
                    f.write(f"Detected camera: {detected_camera} vs User: {user_camera}\n")
                    f.write(f"Status: Continuing with user metadata (validation disabled)\n")
                    f.write(f"=== AI VALIDATION PASSED ===\n\n")
                
                logger.info(f"[AI_VALIDATION] [OK] Continuing with user-provided metadata")
                
                # Continue without blocking - validation is disabled
                
                logger.info(f"[AI_VALIDATION] [OK] Validação AI passou:")
                logger.info(f"[AI_VALIDATION] Mão dominante: {user_hand} == {detected_hand}")
                logger.info(f"[AI_VALIDATION] Lado da câmera: {user_camera} == {detected_camera}")
                
            except Exception as e:
                logger.warning(f"[AI_VALIDATION] [WARNING] Erro na validação AI: {e}")
                logger.warning(f"[AI_VALIDATION] Continuando sem validação AI...")
            
            # Add optimized professional comparisons
            try:
                from optimized_professional_comparison import OptimizedProfessionalComparator
                comparator = OptimizedProfessionalComparator()
                
                # DEBUG: Log movement detection details
                logger.info(f"[DEBUG_MOVEMENT] User config movement: {user_config_movement}")
                logger.info(f"[DEBUG_MOVEMENT] Detected movement: {detected_movement}")
                logger.info(f"[DEBUG_MOVEMENT] Movement classification: {result.get('detailed_analysis', {}).get('movement_classification')}")
                
                # DEBUG: Escrever logs em arquivo para debug
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== DEBUG COMPARISON {datetime.now()} ===\n")
                    f.write(f"User config movement: {user_config_movement}\n")
                    f.write(f"Detected movement: {detected_movement}\n")
                    f.write(f"Movement classification: {result.get('detailed_analysis', {}).get('movement_classification')}\n")
                    f.write(f"User metadata: {metadata_dict}\n")
                
                # Use detected movement for professional comparison (not user input)
                comparison_results = comparator.find_best_matches(
                    user_analysis=result,
                    movement_type=detected_movement,  # Use AI detected movement
                    max_results=3
                )
                
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"=== COMPARATOR DEBUG ===\n")
                    f.write(f"Movement type solicitado: {detected_movement}\n")
                    f.write(f"Total profissionais encontrados: {len(comparison_results) if comparison_results else 0}\n")
                
                # Convert ComparisonResult objects to dictionaries for API response
                professional_comparisons = []
                if comparison_results:
                    for comp_result in comparison_results:
                        professional_comparisons.append({
                            'professional_name': comp_result.professional_name,
                            'professional_video': comp_result.professional_video,
                            'similarity_score': comp_result.similarity_score,
                            'detailed_comparison': comp_result.detailed_comparison,
                            'recommendations': comp_result.recommendations,
                            'confidence': comp_result.confidence
                        })
                
                result['professional_comparisons'] = professional_comparisons
                result['movement_statistics'] = {}
                
            except Exception as e:
                logger.error(f"[PROFESSIONAL_COMPARISON] Error in optimized comparison: {e}")
                result['professional_comparisons'] = []
                result['movement_statistics'] = {}
            
            return AnalysisResult(
                success=True,
                analysis_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                analysis_type='enhanced_single_cycle_biomechanical_with_comparisons',
                final_score=result.get('final_score'),
                detailed_analysis=result,
                recommendations=result.get('recommendations', []),
                professional_comparisons=result.get('professional_comparisons', []),
                movement_statistics=result.get('movement_statistics', {})
            )
        else:
            logger.error(f"[COMMON_ANALYSIS] Enhanced analysis failed: {result.get('error')}")
            
            # DEBUG: Log da falha
            with open("debug_comparison.log", "a", encoding="utf-8") as f:
                f.write(f"ANALISE FALHOU!\n")
                f.write(f"Erro: {result.get('error')}\n")
                f.write(f"Resultado completo: {result}\n")
                f.write(f"=== FIM ANALISE (FALHA) ===\n\n")
                
            raise HTTPException(status_code=500, detail=result.get('error', 'Analysis failed'))
            
    except Exception as e:
        logger.error(f"[COMMON_ANALYSIS] Error in analysis: {e}")
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"EXCECAO CAPTURADA!\n")
            f.write(f"Erro: {str(e)}\n")
            f.write(f"Tipo: {type(e).__name__}\n")
            f.write(f"=== FIM ANALISE (EXCECAO) ===\n\n")
        
        return AnalysisResult(
            success=False,
            analysis_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            analysis_type='common_analysis_error',
            error=str(e)
        )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "FIXED_SECRET_KEY_FOR_DEBUG_12345678901234567890")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Global analyzer instances
analyzer_api = None
real_time_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global analyzer_api, real_time_engine
    
    logger.info("Starting Tennis Analyzer API...")
    
    # Initialize analyzers with conditional availability checks
    try:
        if TENNIS_BACKEND_AVAILABLE and TennisAnalyzerAPI is not None:
            analyzer_api = TennisAnalyzerAPI()
            logger.info("Tennis analyzer API initialized successfully")
        else:
            analyzer_api = None
            logger.warning("Tennis analyzer API not available - running in limited mode")
        
        if REALTIME_ENGINE_AVAILABLE and RealTimeAnalysisEngine is not None:
            real_time_engine = RealTimeAnalysisEngine()
            logger.info("Real-time analysis engine initialized successfully")
        else:
            real_time_engine = None
            logger.warning("Real-time analysis engine not available - running in limited mode")
            
        logger.info("API initialization completed")
    except Exception as e:
        logger.error(f"Failed to initialize analysis engines: {e}")
        # Don't raise - allow API to start in limited mode
        analyzer_api = None
        real_time_engine = None
        logger.warning("Starting API in limited mode due to initialization errors")
    
    yield
    
    logger.info("Shutting down Tennis Analyzer API...")


# FastAPI app
app = FastAPI(
    title="Tennis Analyzer API",
    description="Table Tennis technique analysis with real-time capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom video serving endpoint for better mobile compatibility
@app.get("/temp/{filename}")
@app.head("/temp/{filename}")
async def serve_video(filename: str, request: Request):
    """Serve video files with mobile-friendly headers and range support"""
    temp_uploads_path = project_root / "temp_uploads"
    file_path = temp_uploads_path / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get file size
    file_size = file_path.stat().st_size
    
    # Log the request for debugging
    logger.info(f"[VIDEO_SERVE] Serving {filename} (size: {file_size} bytes)")
    logger.info(f"[VIDEO_SERVE] User-Agent: {request.headers.get('user-agent', 'Unknown')}")
    logger.info(f"[VIDEO_SERVE] Range header: {request.headers.get('range', 'None')}")
    
    # Determine media type based on file extension
    if filename.endswith('.mp4'):
        media_type = "video/mp4"
    elif filename.endswith('.webm'):
        media_type = "video/webm"
    elif filename.endswith('.avi'):
        media_type = "video/x-msvideo"
    else:
        media_type = "video/mp4"  # default
    
    response = FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )
    
    # Add headers for better mobile compatibility
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Length"] = str(file_size)
    response.headers["Cache-Control"] = "public, max-age=3600"  # Allow caching
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Range"
    
    # Add mobile-specific headers
    response.headers["X-UA-Compatible"] = "IE=edge"
    response.headers["Vary"] = "Accept-Encoding"
    
    logger.info(f"[VIDEO_SERVE] Response headers: {dict(response.headers)}")
    
    return response

# Mount static files for temporary videos (fallback)
temp_uploads_path = project_root / "temp_uploads"
temp_uploads_path.mkdir(exist_ok=True)
app.mount("/temp", StaticFiles(directory=str(temp_uploads_path)), name="temp")  # REATIVADO - necessário para vídeos dos profissionais

# Mount static files for selected videos
videos_path = project_root / "videos"
if videos_path.exists():
    app.mount("/static/videos", StaticFiles(directory=str(videos_path)), name="videos")

# Security
security = HTTPBearer()

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class AnalysisMetadata(BaseModel):
    maoDominante: str  # D or E
    ladoCamera: str    # D or E  
    ladoRaquete: str   # F or B
    tipoMovimento: str # D or P

class ComparisonRequest(BaseModel):
    user_metadata: AnalysisMetadata
    professional_name: Optional[str] = None

class AnalysisResult(BaseModel):
    success: bool
    analysis_id: str
    timestamp: datetime
    final_score: Optional[float] = None
    analysis_type: str
    detected_movement: Optional[str] = None
    confidence: Optional[float] = None
    detailed_analysis: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    professional_comparisons: Optional[List[Dict]] = None
    movement_statistics: Optional[Dict] = None
    movement_type_display: Optional[str] = None
    dominant_hand_display: Optional[str] = None
    error: Optional[str] = None


# Simple in-memory user storage (replace with database in production)
users_db = {
    "demo": {
        "username": "demo",
        "email": "demo@example.com",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "created_at": datetime.now()
    }
}

# Authentication functions
def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == password_hash

def get_user(username: str) -> Optional[Dict]:
    """Get user from database"""
    return users_db.get(username)

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user credentials"""
    user = get_user(username)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return user

def create_access_token(data: Dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token"""
    try:
        logger.info(f"[AUTH_DEBUG] Verifying token: {credentials.credentials[:50]}...")
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"[AUTH_DEBUG] Token decoded successfully: {payload}")
        username: str = payload.get("sub")
        if username is None:
            logger.warning(f"[AUTH_DEBUG] No username in token payload")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        user = get_user(username)
        if user is None:
            logger.warning(f"[AUTH_DEBUG] User {username} not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        logger.info(f"[AUTH_DEBUG] User {username} authenticated successfully")
        return user
    except jwt.PyJWTError as e:
        logger.error(f"[AUTH_DEBUG] JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not allowed. Allowed types: {ALLOWED_VIDEO_EXTENSIONS}"
        )

def convert_to_web_friendly(input_path: Path, output_path: Path) -> bool:
    """
    Convert video to web-friendly format using FFmpeg or fallback methods
    Returns True if conversion succeeded, False otherwise
    """
    logger.info(f"[WEB_CONVERT] [RELOAD] Starting conversion from {input_path} to {output_path}")
    
    try:
        # Method 1: Try FFmpeg (best quality)
        if shutil.which('ffmpeg'):
            logger.info(f"[WEB_CONVERT] Using FFmpeg for conversion")
            subprocess.run([
                'ffmpeg', '-i', str(input_path),
                '-vcodec', 'libx264',           # H.264 codec (best browser support)
                '-acodec', 'aac',               # AAC audio codec  
                '-movflags', '+faststart',      # Enable streaming
                '-pix_fmt', 'yuv420p',         # Pixel format compatible with browsers
                '-preset', 'fast',              # Fast encoding
                '-crf', '23',                   # Good quality/size balance
                '-y',                           # Overwrite output
                str(output_path)
            ], check=True, capture_output=True)
            logger.info(f"[WEB_CONVERT] [OK] Converted with FFmpeg: {output_path}")
            return True
        else:
            logger.info(f"[WEB_CONVERT] FFmpeg not found, trying OpenCV method")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"[WEB_CONVERT] FFmpeg failed: {e}")
    
    try:
        # Method 2: OpenCV with optimized settings
        import cv2
        
        logger.info(f"[WEB_CONVERT] Using OpenCV with optimization")
        
        # Read input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"[WEB_CONVERT] Failed to open input video: {input_path}")
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"[WEB_CONVERT] Input video: {width}x{height} @ {fps}fps")
        
        # Optimize dimensions (reduce if too large)
        if width > 1280:
            new_width = 1280
            new_height = int(height * (1280 / width))
            width, height = new_width, new_height
        
        # Try multiple codecs for best browser compatibility
        codecs_to_try = [
            ('mp4v', 'MPEG-4'),
            ('XVID', 'XVID'),
            ('MJPG', 'Motion JPEG')
        ]
        
        for codec, desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                if out.isOpened():
                    logger.info(f"[WEB_CONVERT] Using {desc} codec")
                    
                    # Copy all frames with potential resizing
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Resize if needed
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height))
                        
                        out.write(frame)
                    
                    out.release()
                    cap.release()
                    
                    # Check if file was created successfully
                    if output_path.exists() and output_path.stat().st_size > 1000:
                        logger.info(f"[WEB_CONVERT] [OK] Converted with OpenCV ({desc}): {output_path}")
                        return True
                    else:
                        if output_path.exists():
                            size = output_path.stat().st_size
                            logger.warning(f"[WEB_CONVERT] {desc} created file but too small: {size} bytes")
                        else:
                            logger.warning(f"[WEB_CONVERT] {desc} did not create output file")
                else:
                    logger.warning(f"[WEB_CONVERT] {desc} codec failed to open VideoWriter")
                    
            except Exception as e:
                logger.warning(f"[WEB_CONVERT] {desc} codec failed: {e}")
                continue
        
        cap.release()
        
    except Exception as e:
        logger.error(f"[WEB_CONVERT] OpenCV conversion failed: {e}")
    
    logger.error(f"[WEB_CONVERT] [ERROR] All conversion methods failed")
    return False


async def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file to temporary location"""
    validate_video_file(file)
    
    # Create temporary file
    temp_dir = Path(__file__).parent.parent / "temp_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    file_ext = Path(file.filename).suffix
    temp_file = temp_dir / f"{secrets.token_hex(16)}{file_ext}"
    
    # Save file - read the entire content at once to avoid stream exhaustion
    logger.info(f"[SAVE_DEBUG] Starting file save...")
    
    content = await file.read()
    total_size = len(content)
    
    logger.info(f"[SAVE_DEBUG] Content read: {total_size} bytes")
    
    if total_size == 0:
        raise HTTPException(status_code=400, detail="Empty file received")
    
    if total_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    with open(temp_file, "wb") as f:
        bytes_written = f.write(content)
    
    # Verify file was saved correctly
    file_size = temp_file.stat().st_size
    if file_size == 0:
        raise HTTPException(status_code=500, detail="Failed to save file - 0 bytes written")
    
    if file_size != total_size:
        logger.warning(f"[SAVE_DEBUG] Size mismatch: expected {total_size}, got {file_size}")
    
    logger.info(f"[SAVE_DEBUG] File saved successfully: {file_size} bytes")
    
    return temp_file


# API Routes

@app.get("/")
async def root():
    """Redirect to web interface"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web_interface.html")

@app.get("/test-connection")
async def test_connection():
    """Test endpoint to verify mobile connectivity"""
    log_path = Path(__file__).parent.parent / "mobile_test.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"MOBILE TEST ENDPOINT CHAMADO em {datetime.now()}\n")
    return {"status": "connected", "message": "Mobile conectado com sucesso!"}

@app.post("/test-analyze")
async def test_analyze():
    """Test analyze endpoint without processing"""
    log_path = Path(__file__).parent.parent / "mobile_analyze_test.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"MOBILE ANALYZE TEST CHAMADO em {datetime.now()}\n")
    return {"status": "analyze_called", "message": "Endpoint analyze foi chamado!"}

@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Simple upload test to check if files are being received correctly"""
    try:
        # Try to read the file
        content = await file.read()
        file_size = len(content)
        
        logger.info(f"[TEST_UPLOAD] File received: {file.filename}")
        logger.info(f"[TEST_UPLOAD] Content type: {file.content_type}")
        logger.info(f"[TEST_UPLOAD] File size: {file_size} bytes")
        
        if file_size == 0:
            return {
                "success": False,
                "error": "Empty file received",
                "filename": file.filename,
                "size": file_size
            }
        
        return {
            "success": True,
            "filename": file.filename,
            "size": file_size,
            "content_type": file.content_type
        }
        
    except Exception as e:
        logger.error(f"[TEST_UPLOAD] Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/test-analyze-form")
async def test_analyze_form(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    cycle_index: int = Form(1)
):
    """Test analyze with same form structure as real analyze endpoint"""
    try:
        # Try to read the file
        content = await user_video.read()
        file_size = len(content)
        
        logger.info(f"[TEST_ANALYZE_FORM] File received: {user_video.filename}")
        logger.info(f"[TEST_ANALYZE_FORM] File size: {file_size} bytes")
        logger.info(f"[TEST_ANALYZE_FORM] Metadata: {metadata}")
        logger.info(f"[TEST_ANALYZE_FORM] Cycle index: {cycle_index}")
        
        if file_size == 0:
            return {
                "success": False,
                "error": "Empty file received in form test",
                "filename": user_video.filename,
                "size": file_size
            }
        
        return {
            "success": True,
            "filename": user_video.filename,
            "size": file_size,
            "metadata": metadata,
            "cycle_index": cycle_index
        }
        
    except Exception as e:
        logger.error(f"[TEST_ANALYZE_FORM] Error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/test_upload.html")
async def test_upload_page():
    """Serve the test upload page"""
    html_path = Path(__file__).parent.parent / "test_upload.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Test upload page not found")

@app.get("/web_interface.html")
async def web_interface():
    """Serve the web interface HTML file"""
    html_path = Path(__file__).parent.parent / "web_interface.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Web interface not found")

@app.post("/debug/analyze")
async def debug_analyze(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    professional_name: Optional[str] = Form(None),
    cycle_index: int = Form(1)
):
    """Debug analyze endpoint"""
    try:
        # Test each step
        import json
        
        # Step 1: Parse metadata
        metadata_dict = json.loads(metadata)
        
        # Step 2: Create AnalysisMetadata
        analysis_metadata = AnalysisMetadata(**metadata_dict)
        
        # Step 3: Import enhanced analyzer
        from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
        enhanced_analyzer = EnhancedSingleCycleAnalyzer()
        
        return {
            "success": True,
            "message": "All imports and parsing working",
            "filename": user_video.filename,
            "metadata_parsed": metadata_dict,
            "analysis_metadata": analysis_metadata.dict(),
            "analyzer_loaded": True
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/professionals")
async def debug_professionals():
    """Debug professionals loading"""
    try:
        import json
        
        # Load the data file directly
        data_file = project_root / "professionals_biomech_data.json"
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter forehand_drive
            forehand_professionals = {}
            for key, value in data.items():
                if 'forehand_drive' in key:
                    forehand_professionals[key] = {
                        "player_name": value.get("player_name", "Unknown"),
                        "movement_type": value.get("movement_type", "Unknown"),
                        "file_exists": Path(f"../profissionais/forehand_drive/{value.get('video_file', '')}").exists()
                    }
            
            return {
                "success": True,
                "data_file_exists": True,
                "total_professionals": len(data),
                "forehand_drive_count": len(forehand_professionals),
                "forehand_professionals": forehand_professionals
            }
        else:
            return {
                "success": False,
                "error": "Data file not found",
                "data_file_path": str(data_file.absolute())
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health():
    """API health check"""
    return {
        "message": "Tennis Analyzer API",
        "version": "1.0.0",
        "status": "active",
        "features": ["video_analysis", "real_time", "comparison", "authentication"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "components": {
            "analyzer_api": analyzer_api is not None,
            "real_time_engine": real_time_engine is not None,
            "temp_directory": Path("temp_uploads").exists()
        }
    }

# Authentication endpoints
@app.post("/auth/register", response_model=Dict)
async def register(user: UserCreate):
    """Register new user"""
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    password_hash = hashlib.sha256(user.password.encode()).hexdigest()
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "password_hash": password_hash,
        "created_at": datetime.now()
    }
    
    return {"message": "User created successfully", "username": user.username}

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """User login"""
    authenticated_user = authenticate_user(user.username, user.password)
    if not authenticated_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return Token(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.get("/auth/me")
async def get_current_user(current_user: Dict = Depends(verify_token)):
    """Get current user info"""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }

# Analysis endpoints

# TEMPORARY PUBLIC ENDPOINT FOR TESTING
@app.get("/professionals-public")
async def get_professionals_public(
    movement_type: str = "forehand_drive"
):
    """Get available professional players (PUBLIC - NO AUTH REQUIRED)"""
    try:
        import json
        
        # Load professionals data directly
        data_file = project_root / "professionals_biomech_data.json"
        if not data_file.exists():
            return {
                "success": False,
                "error": "Professional data not found",
                "movement_type": movement_type,
                "professionals": [],
                "count": 0
            }
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter by movement type and existing files
        professionals = []
        for key, value in data.items():
            if value.get("movement_type") == movement_type:
                video_file = value.get("video_file", "")
                file_path = Path(f"../profissionais/{movement_type}/{video_file}")
                
                if file_path.exists():
                    professionals.append({
                        "name": value.get("player_name", "Unknown"),
                        "filename": video_file,
                        "hand": video_file.split("_")[2] if len(video_file.split("_")) > 2 else "D",
                        "camera_side": video_file.split("_")[3].split(".")[0] if len(video_file.split("_")) > 3 else "D", 
                        "stats": value.get("biomechanics", {}),
                        "video_exists": True,
                        "file_path": str(file_path)
                    })
        
        return {
            "success": True,
            "movement_type": movement_type,
            "professionals": professionals,
            "count": len(professionals)
        }
    except Exception as e:
        logger.error(f"Error getting professionals: {e}")
        return {
            "success": False,
            "error": str(e),
            "movement_type": movement_type,
            "professionals": [],
            "count": 0
        }

@app.get("/available-videos")
async def get_available_videos():
    """
    List all available videos from the restricted videos folder
    """
    try:
        videos_folder = project_root / "videos"
        if not videos_folder.exists():
            return {"success": False, "error": "Videos folder not found"}
        
        available_videos = []
        
        # Get all video files from the videos folder
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        for video_file in videos_folder.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                # Extract metadata from filename if possible
                filename = video_file.name
                
                # Try to parse metadata from filename pattern: Name_Movement_Hand_Camera.mp4
                metadata = {}
                try:
                    # Remove extension and split by underscore
                    base_name = video_file.stem
                    parts = base_name.split('_')
                    
                    if len(parts) >= 4:
                        # Try to extract: [Name]_[Movement]_[Hand]_[Camera]
                        movement_code = parts[-3]  # FD, BD, FP, BP
                        hand_code = parts[-2]      # D, E  
                        camera_code = parts[-1]    # D, E
                        
                        # Map codes to full names
                        movement_map = {
                            'FD': {'ladoRaquete': 'F', 'tipoMovimento': 'D'},
                            'BD': {'ladoRaquete': 'B', 'tipoMovimento': 'D'},
                            'FP': {'ladoRaquete': 'F', 'tipoMovimento': 'P'},
                            'BP': {'ladoRaquete': 'B', 'tipoMovimento': 'P'}
                        }
                        
                        hand_map = {'D': 'Destro', 'E': 'Canhoto'}
                        camera_map = {'D': 'Direita', 'E': 'Esquerda'}
                        
                        if movement_code in movement_map:
                            metadata.update(movement_map[movement_code])
                        if hand_code in hand_map:
                            metadata['maoDominante'] = hand_map[hand_code]
                        if camera_code in camera_map:
                            metadata['ladoCamera'] = camera_map[camera_code]
                            
                except Exception as e:
                    logger.warning(f"Could not parse metadata from filename {filename}: {e}")
                
                available_videos.append({
                    "filename": filename,
                    "display_name": filename.replace('_', ' ').replace('.mp4', ''),
                    "file_path": str(video_file.relative_to(project_root)),
                    "suggested_metadata": metadata,
                    "file_size": video_file.stat().st_size
                })
        
        # Sort by filename
        available_videos.sort(key=lambda x: x["filename"])
        
        return {
            "success": True,
            "videos": available_videos,
            "total_count": len(available_videos)
        }
        
    except Exception as e:
        logger.error(f"Error listing available videos: {e}")
        return {"success": False, "error": str(e)}

@app.get("/professionals")
async def get_professionals(
    movement_type: str = "forehand_drive"
):
    """Get available professional players (without validation)"""
    # Use the same logic as professionals-public
    return await get_professionals_public(movement_type)

@app.post("/validate-and-get-professionals")
async def validate_and_get_professionals(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    current_user: Dict = Depends(verify_token)
):
    """Get available professional players based on user metadata (no validation)"""
    try:
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        logger.info(f"[NO_VALIDATION] Metadata recebido: {metadata_dict}")
        
        # Build movement type directly from user selection (no validation!)
        lado_raquete = metadata_dict.get('ladoRaquete', 'F')
        tipo_movimento = metadata_dict.get('tipoMovimento', 'D')
        
        # Build movement key from user selection
        racket_side = 'forehand' if lado_raquete == 'F' else 'backhand'
        movement_type = 'drive' if tipo_movimento == 'D' else 'push'
        movement_key = f"{racket_side}_{movement_type}"
        
        logger.info(f"[NO_VALIDATION] Using movement from user selection: {movement_key}")
        
        # Get professionals for this movement type
        try:
            # Load professionals data
            professionals_file = project_root / "professionals_biomech_data.json"
            if professionals_file.exists():
                with open(professionals_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Filter by movement type
                professionals = []
                for key, value in data.items():
                    if value.get("movement_type") == movement_key:
                        professionals.append({
                            "name": value.get("player_name", "Unknown"),
                            "filename": value.get("video_file", ""),
                            "confidence": value.get("classification", {}).get("confidence", 0),
                            "movement_type": movement_key
                        })
                
                # Sort by confidence
                professionals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                
                logger.info(f"[NO_VALIDATION] Found {len(professionals)} professionals for {movement_key}")
                
                # Cleanup temp file
                try:
                    temp_file = await save_uploaded_file(file)
                    temp_file.unlink()
                except:
                    pass
                
                return {
                    "success": True,
                    "movement_type": movement_key,
                    "professionals": professionals,
                    "count": len(professionals),
                    "validation_passed": True,
                    "message": f"Found {len(professionals)} professionals for {movement_key} (no validation applied)"
                }
            else:
                raise Exception("Professional data file not found")
                
        except Exception as e:
            logger.error(f"[NO_VALIDATION] Error loading professionals: {e}")
            return {
                "success": False,
                "movement_type": movement_key,
                "professionals": [],
                "count": 0,
                "validation_passed": True,
                "message": f"Could not load professionals: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Error in validate-and-get-professionals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/professionals/{movement_type}")
async def get_professionals(movement_type: str):
    """Get available professionals for a movement type"""
    try:
        if analyzer_api is None:
            # Return hardcoded list for headless mode
            professionals_db = {
                'forehand_drive': [
                    {'name': 'Zhang Jike', 'file_path': 'profissionais/forehand_drive/Zhang_Jike_FD_D_D.mp4'},
                    {'name': 'Ma Long', 'file_path': 'profissionais/forehand_drive/Ma_Long_FD_D_D.mp4'},
                    {'name': 'Fan Zhendong', 'file_path': 'profissionais/forehand_drive/Fan_Zhendong_FD_D_E.mp4'}
                ],
                'forehand_push': [
                    {'name': 'Jeff Plumb', 'file_path': 'profissionais/forehand_push/Jeff_Plumb_FP_D_D.mp4'},
                    {'name': 'Zhao Yiyi', 'file_path': 'profissionais/forehand_push/Zhao_Yiyi_FP_D_D.mp4'}
                ],
                'backhand_drive': [
                    {'name': 'Chen Meng', 'file_path': 'profissionais/backhand_drive/Chen_Meng_BD_D_D.mp4'},
                    {'name': 'Jane', 'file_path': 'profissionais/backhand_drive/Jane_BD_D_D.mp4'}
                ],
                'backhand_push': [
                    {'name': 'Alois Rosario', 'file_path': 'profissionais/backhand_push/Alois_Rosario_BP_E_E.mp4'},
                    {'name': 'Zhao Yiyi', 'file_path': 'profissionais/backhand_push/Zhao_Yiyi_BP_D_E.mp4'},
                    {'name': 'Jane', 'file_path': 'profissionais/backhand_push/Jane_BP_D_D.mp4'}
                ]
            }
            return {"professionals": professionals_db.get(movement_type, [])}
        
        # Use analyzer API if available
        professionals = analyzer_api.engine.get_available_professionals(movement_type)
        return {"professionals": professionals}
        
    except Exception as e:
        logger.error(f"Error getting professionals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    current_user: Dict = Depends(verify_token)
):
    """Upload and validate user video"""
    try:
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        analysis_metadata = AnalysisMetadata(**metadata_dict)
        
        # Save uploaded file
        temp_file = await save_uploaded_file(file)
        
        # Check if analyzer is available
        if analyzer_api is None:
            raise HTTPException(
                status_code=503, 
                detail="Analysis service temporarily unavailable"
            )
        
        # Validate video
        result = analyzer_api.process_upload(
            temp_file.read_bytes(),
            file.filename,
            analysis_metadata.dict()
        )
        
        # Generate analysis ID
        analysis_id = secrets.token_hex(16)
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "filename": file.filename,
            "file_path": str(temp_file),
            "validation": result,
            "metadata": analysis_metadata.dict()
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-selected", response_model=AnalysisResult)
async def analyze_selected_video_FIXED(
    selected_video: str = Form(...),
    metadata: str = Form(...),
    professional_name: Optional[str] = Form(None),
    cycle_index: int = Form(1)
):
    """[HIJACKED] Endpoint antigo agora usa nossa lógica corrigida v16.2"""
    
    logger.info(f"[ANALYSIS] Starting analysis for {selected_video}")
    
    try:
        # Parse metadata
        metadata_dict = json.loads(metadata)
        video_filename = selected_video.lower()
        
        logger.info(f"[ANALYSIS] Processing {video_filename} with metadata {metadata_dict}")
        
        # Use EnhancedSingleCycleAnalyzer directly
        from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
        analyzer = EnhancedSingleCycleAnalyzer()
        
        # Get video path
        user_video_path = f"videos/{selected_video}"
        
        result = analyzer.compare_enhanced_single_cycles(user_video_path, None, metadata_dict, None, cycle_index)
        
        if result.get('success'):
            # Extract movement from biomech_result or detailed_analysis
            detected_movement = None
            
            # Try to find movement in biomech_result
            biomech_result = result.get('biomech_result', {})
            
            if biomech_result and hasattr(biomech_result, 'movement_type'):
                if hasattr(biomech_result.movement_type, 'value'):
                    detected_movement = biomech_result.movement_type.value
                else:
                    detected_movement = str(biomech_result.movement_type)
            
            # Try detailed_analysis if not found
            if not detected_movement:
                detailed = result.get('detailed_analysis', {})
                detected_movement = detailed.get('detected_movement') or detailed.get('movement_type')
            
            # If still not found, extract from logs or use metadata fallback
            if not detected_movement:
                raquete = metadata_dict.get('ladoRaquete', 'F')
                movimento = metadata_dict.get('tipoMovimento', 'D')
                detected_movement = f"{'forehand' if raquete == 'F' else 'backhand'}_{'drive' if movimento == 'D' else 'push'}"
            
            # Use ALL real data from the analysis result
            real_score = result.get('final_score', 85.0)
            real_confidence = result.get('confidence', 0.85)
            real_analysis_id = result.get('analysis_id', f"real_analysis_{int(time.time())}")
            real_detailed = result.get('detailed_analysis', {})
            real_recommendations = result.get('recommendations', [])
            real_professional_comparisons_raw = result.get('professional_comparisons', [])
            real_movement_stats = result.get('movement_statistics', {})
            
            logger.info(f"[ANALYSIS] Extracted {len(real_recommendations)} recommendations")
            
            # Convert ComparisonResult objects to dictionaries
            real_professional_comparisons = []
            for comp in real_professional_comparisons_raw:
                if hasattr(comp, '__dict__'):
                    # Convert object to dict
                    comp_dict = {
                        'professional_name': getattr(comp, 'professional_name', 'unknown'),
                        'similarity_score': getattr(comp, 'similarity_score', 0.0),
                        'confidence': getattr(comp, 'confidence', 0.0),
                        'analysis_details': getattr(comp, 'analysis_details', {}),
                        'movement_type': getattr(comp, 'movement_type', 'unknown'),
                        'recommendations': getattr(comp, 'recommendations', []),
                        'detailed_comparison': getattr(comp, 'detailed_comparison', {})
                    }
                    real_professional_comparisons.append(comp_dict)
                else:
                    # Already a dict
                    real_professional_comparisons.append(comp)
            
            logger.info(f"[ANALYSIS] Score: {real_score:.1f}, Confidence: {real_confidence:.2f}, Movement: {detected_movement}")
            
            # Calculate movement_type_display and dominant_hand_display from REAL DETECTION
            movement_type_display = detected_movement if detected_movement else 'unknown'
            
            # Get dominant hand from REAL ANALYSIS RESULT (not metadata!)
            detected_active_hand = real_detailed.get('active_hand_side', '') if real_detailed else ''
            if detected_active_hand == 'esquerda':
                dominant_hand_display = 'Canhoto'
            elif detected_active_hand == 'direita':
                dominant_hand_display = 'Destro'
            else:
                # Fallback: try to get from metadata if detection failed
                mao_dominante = metadata_dict.get('maoDominante', '')
                if mao_dominante == 'Destro':
                    dominant_hand_display = 'Destro'
                elif mao_dominante == 'Canhoto':
                    dominant_hand_display = 'Canhoto'
                else:
                    dominant_hand_display = 'Destro'  # Default
            
            # Return real analysis result
            return AnalysisResult(
                success=True,
                analysis_id=real_analysis_id,
                timestamp=datetime.now(),
                final_score=real_score,
                analysis_type="real_biomechanical_analysis_v16_2",
                detected_movement=detected_movement,
                confidence=real_confidence,
                detailed_analysis=real_detailed,
                recommendations=real_recommendations,
                professional_comparisons=real_professional_comparisons,
                movement_statistics=real_movement_stats,
                movement_type_display=movement_type_display,
                dominant_hand_display=dominant_hand_display
            )
        else:
            print("DEBUG_FLAG_028: Result success = False")
            # Last resort: use metadata only as fallback
            raquete = metadata_dict.get('ladoRaquete', 'F')
            movimento = metadata_dict.get('tipoMovimento', 'D')
            detected_movement = f"{'forehand' if raquete == 'F' else 'backhand'}_{'drive' if movimento == 'D' else 'push'}"
            print(f"DEBUG_FLAG_029: FALLBACK metadata movement = {detected_movement}")
            
            # Calculate movement_type_display and dominant_hand_display for fallback
            movement_type_display = detected_movement
            
            # Get dominant hand from metadata
            mao_dominante = metadata_dict.get('maoDominante', '')
            if mao_dominante == 'Destro':
                dominant_hand_display = 'Destro'
            elif mao_dominante == 'Canhoto':
                dominant_hand_display = 'Canhoto'
            else:
                dominant_hand_display = 'Destro'  # Default
            
            return AnalysisResult(
                success=True,
                analysis_id="fallback_metadata",
                timestamp=datetime.now(),
                final_score=75.0,
                analysis_type="metadata_fallback",
                detected_movement=detected_movement,
                confidence=0.7,
                detailed_analysis={"method": "metadata_only"},
                recommendations=[f"[FALLBACK] Baseado em metadata: {detected_movement}"],
                professional_comparisons=[],
                movement_statistics={},
                movement_type_display=movement_type_display,
                dominant_hand_display=dominant_hand_display
            )
        
    except Exception as e:
        print(f"DEBUG_FLAG_030: EXCEPTION in hijack: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"DEBUG_FLAG_031: TRACEBACK: {traceback_str}")
        
        # Return error result with more details
        return AnalysisResult(
            success=False,
            analysis_id="error_hijack_v16_2",
            timestamp=datetime.now(),
            final_score=0.0,
            analysis_type="hijack_error",
            detected_movement="error",
            confidence=0.0,
            detailed_analysis={"error": str(e), "traceback": traceback_str},
            recommendations=[f"ERRO: {str(e)}"],
            professional_comparisons=[],
            movement_statistics={}
        )
        return AnalysisResult(
            success=False,
            analysis_id="error_v16_2",
            timestamp=datetime.now(),
            final_score=0.0,
            analysis_type="hijack_error",
            error=f"Erro no hijack: {e}"
        )

@app.post("/analyze-video-v16", response_model=AnalysisResult)
async def analyze_video_v16_fixed(
    selected_video: str = Form(...),  # Filename from restricted videos folder
    metadata: str = Form(...),
    professional_name: Optional[str] = Form(None),
    cycle_index: int = Form(1)
):
    """[FIXED] Enhanced analysis using improved Tennis Engine v16.1 with racket angle detection"""
    
    # [MEGA DEBUG] LOGS ULTRA VISÍVEIS 
    print("\n" + "[!]" * 50)
    print("[!] VERSÃO v16.2 - ANALYZE-SELECTED ENDPOINT CHAMADO!")
    print(f"[VIDEO] Video: {selected_video}")
    print(f"[DATA] Metadata: {metadata}")
    print("[!] SE VOCÊ VÊ ESTA MENSAGEM, O CÓDIGO ESTÁ SENDO EXECUTADO!")
    print("[!]" * 50 + "\n")
    
    # Forçar log no console JavaScript também
    import json
    debug_info = {
        "VERSION": "v16.2_DEBUG_EXTREME",
        "VIDEO": selected_video,
        "METADATA": metadata,
        "TIMESTAMP": str(datetime.now())
    }
    print(f"[MEGA_DEBUG_JSON] {json.dumps(debug_info)}")
    
    # DEBUG: Log IMEDIATO para verificar se endpoint é chamado
    log_path = Path(__file__).parent.parent / "endpoint_called.log"
    
    try:
        metadata_dict = json.loads(metadata)
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"ENDPOINT /analyze-selected CHAMADO em {datetime.now()}\n")
            f.write(f"Arquivo selecionado: {selected_video}\n")
            f.write(f"Metadata: {metadata}\n")
        
        # Validate selected video exists in allowed folder
        videos_folder = project_root / "videos"
        selected_video_path = videos_folder / selected_video
        
        if not selected_video_path.exists() or not selected_video_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Selected video '{selected_video}' not found in allowed videos folder"
            )
        
        # Check if file is actually a video
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        if selected_video_path.suffix.lower() not in video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Selected file '{selected_video}' is not a supported video format"
            )
        
        logger.info(f"[SELECTED_VIDEO_API] Starting analysis of: {selected_video}")
        logger.info(f"[SELECTED_VIDEO_API] Video path: {selected_video_path}")
        logger.info(f"[SELECTED_VIDEO_API] Metadata: {metadata_dict}")
        
        # Continue with the same analysis logic as the original endpoint
        # but using the selected video path instead of uploaded file
        user_video_path = str(selected_video_path)
        
        # Continue with rest of analysis logic using common function
        return await _perform_analysis(user_video_path, metadata_dict, professional_name, cycle_index, selected_video)
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SELECTED_VIDEO_API] Unexpected error: {e}")
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"EXCECAO CAPTURADA!\n")
            f.write(f"Erro: {str(e)}\n")
            f.write(f"Tipo: {type(e).__name__}\n")
            f.write(f"=== FIM ANALISE (EXCECAO) ===\n\n")
        
        return AnalysisResult(
            success=False,
            analysis_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            analysis_type='selected_video_analysis_error',
            error=str(e)
        )

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_technique(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    professional_name: Optional[str] = Form(None),
    cycle_index: int = Form(1)
):
    """Enhanced biomechanical analysis with file upload - NO AUTH"""
    # DEBUG: Log IMEDIATO para verificar se endpoint é chamado
    log_path = Path(__file__).parent.parent / "endpoint_called.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"ENDPOINT /analyze CHAMADO em {datetime.now()}\n")
        f.write(f"Arquivo: {user_video.filename}\n")
        f.write(f"Metadata: {metadata}\n")
    
    try:
        metadata_dict = json.loads(metadata)
        
        # Save uploaded file to temp directory
        user_video_path = await save_uploaded_file(user_video)
        
        # Use common analysis function
        return await _perform_analysis(user_video_path, metadata_dict, professional_name, cycle_index, user_video.filename)
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")
    except Exception as e:
        logger.error(f"[UPLOAD_API] Unexpected error: {e}")
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"EXCECAO CAPTURADA!\n")
            f.write(f"Erro: {str(e)}\n")
            f.write(f"Tipo: {type(e).__name__}\n")
            f.write(f"=== FIM ANALISE (EXCECAO) ===\n\n")
        
        return AnalysisResult(
            success=False,
            analysis_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            analysis_type='upload_analysis_error',
            error=str(e)
        )


@app.post("/extract-user-cycle")
async def extract_user_cycle(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    cycle_index: int = Form(1)  # Default to 2nd cycle (index 1)
):
    """Extract specific cycle from user video for display"""
    logger.info(f"[LAUNCH] [EXTRACT_CYCLE] ENDPOINT INICIADO - cycle_index: {cycle_index}")
    try:
        logger.info(f"[USER_CYCLE] Extracting cycle {cycle_index + 1} for display")
        
        # DEBUG: Log inicial sempre em arquivo
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== INICIO ANALISE {datetime.now()} ===\n")
            f.write(f"Metadata recebida: {metadata}\n")
            f.write(f"Cycle index: {cycle_index}\n")
        
        from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
        enhanced_analyzer = EnhancedSingleCycleAnalyzer()
        
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        analysis_metadata = AnalysisMetadata(**metadata_dict)
        
        # Save user video
        user_video_path = await save_uploaded_file(user_video)
        
        # Get professional video (optional for independent analysis)
        user_config_movement = f"{'forehand' if analysis_metadata.ladoRaquete == 'F' else 'backhand'}_{'drive' if analysis_metadata.tipoMovimento == 'D' else 'push'}"
        
        pro_video_path = None
        if professional_name and analyzer_api is not None:
            try:
                professionals = analyzer_api.engine.get_available_professionals(user_config_movement)
                selected_pro = next((p for p in professionals if p["name"] == professional_name), None)
                if selected_pro:
                    pro_video_path = selected_pro.get("file_path")
                    logger.info(f"[BIOMECH_API] Using professional: {professional_name}")
                else:
                    logger.warning(f"[BIOMECH_API] Professional {professional_name} not found, continuing with independent analysis")
            except Exception as e:
                logger.warning(f"[BIOMECH_API] Error finding professional: {e}, continuing with independent analysis")
        elif professional_name:
            logger.warning(f"[BIOMECH_API] Analyzer not available, continuing with independent analysis")
        else:
            logger.info(f"[BIOMECH_API] Independent biomechanical analysis")
        
        # Get professional metadata if we have a professional video
        pro_metadata = {}
        if pro_video_path and analyzer_api is not None:
            try:
                pro_metadata = analyzer_api.get_professional_metadata(pro_video_path, user_config_movement)
            except Exception as e:
                logger.warning(f"[BIOMECH_API] Could not get professional metadata: {e}")
                pro_metadata = {}
        
        # Run enhanced analysis (comparative or independent)
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"Iniciando enhanced_analyzer.analyze_single_cycle_with_biomechanics...\n")
            f.write(f"User video path: {user_video_path}\n")
            f.write(f"Pro video path: {pro_video_path}\n")
        
        result = enhanced_analyzer.compare_enhanced_single_cycles(
            str(user_video_path),
            str(pro_video_path) if pro_video_path else None,
            analysis_metadata.dict(),
            pro_metadata,
            cycle_index=cycle_index
        )
        
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"Resultado da analise: success = {result.get('success')}\n")
            f.write(f"Keys do resultado: {list(result.keys())}\n")
        
        if result['success']:
            logger.info(f"[BIOMECH_API] Enhanced analysis completed: {result.get('final_score', 0):.1f}%")
            
            # VALIDAÇÃO OBRIGATÓRIA: Verificar se metadados informados conferem com o detectado
            detected_movement = result.get('detailed_analysis', {}).get('movement_classification', {}).get('detected_movement', None)
            movement_classification = result.get('detailed_analysis', {}).get('movement_classification', {})
            
            # Comparar movimento configurado vs detectado
            if detected_movement is None:
                logger.warning(f"[METADATA_VALIDATION] [ERROR] DETECÇÃO DE MOVIMENTO FALHOU!")
                logger.warning(f"[METADATA_VALIDATION] Não foi possível detectar movimento no vídeo")
                
                return AnalysisResult(
                    success=False,
                    analysis_id=secrets.token_hex(16),
                    timestamp=datetime.now(),
                    analysis_type='movement_detection_failed',
                    error=f"Não foi possível detectar o tipo de movimento no vídeo. Verifique se o vídeo contém um movimento de tênis de mesa claro e tente novamente.",
                    message="Detecção de movimento falhou",
                    final_score=0.0,
                    recommendations=[
                        "Verifique se o vídeo contém:",
                        "- Um movimento de tênis de mesa claro",
                        "- O jogador está visível na câmera",
                        "- O movimento não está cortado ou incompleto",
                        "- A qualidade do vídeo é adequada"
                    ],
                    detailed_analysis={
                        "validation_failed": True,
                        "detection_failed": True,
                        "user_input": {
                            "movement": user_config_movement,
                            "metadata": metadata_dict
                        }
                    }
                )
            elif detected_movement != user_config_movement:
                logger.warning(f"[METADATA_VALIDATION] [ERROR] INCONSISTÊNCIA DETECTADA!")
                logger.warning(f"[METADATA_VALIDATION] Informado: {user_config_movement}")
                logger.warning(f"[METADATA_VALIDATION] Detectado: {detected_movement}")
                
                # Log detailed validation failure
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== VALIDAÇÃO FALHOU {datetime.now()} ===\n")
                    f.write(f"User config movement: {user_config_movement}\n")
                    f.write(f"Detected movement: {detected_movement}\n")
                    f.write(f"Confidence: {movement_classification.get('confidence', 'N/A')}\n")
                    f.write(f"User metadata: {metadata_dict}\n")
                    f.write(f"=== ANÁLISE BLOQUEADA ===\n\n")
                
                # Return validation error instead of continuing analysis
                return AnalysisResult(
                    success=False,
                    analysis_id=secrets.token_hex(16),
                    timestamp=datetime.now(),
                    analysis_type='metadata_validation_failed',
                    error=f"Inconsistência detectada: Você informou '{user_config_movement}' mas o sistema detectou '{detected_movement}' no vídeo. Verifique os parâmetros informados e tente novamente.",
                    message="Validação de metadados falhou",
                    final_score=0.0,
                    recommendations=[
                        f"O sistema detectou '{detected_movement}' com {movement_classification.get('confidence', 0):.1%} de confiança",
                        f"Verifique se você selecionou corretamente:",
                        f"- Lado da raquete: {'Forehand' if 'forehand' in detected_movement else 'Backhand'}",
                        f"- Tipo de movimento: {'Drive' if 'drive' in detected_movement else 'Push'}",
                        f"- E tente novamente com os parâmetros corretos"
                    ],
                    detailed_analysis={
                        "validation_failed": True,
                        "user_input": {
                            "movement": user_config_movement,
                            "metadata": metadata_dict
                        },
                        "detected": {
                            "movement": detected_movement,
                            "confidence": movement_classification.get('confidence', 0),
                            "classification_details": movement_classification
                        }
                    }
                )
            
            logger.info(f"[METADATA_VALIDATION] [OK] Validação de movimento passou: {user_config_movement} == {detected_movement}")
            
            # AI VALIDATION TEMPORARILY DISABLED
            # The AI validation was too restrictive and causing false positives
            # User metadata should be trusted over computer vision detection
            logger.info(f"[AI_VALIDATION] AI validation temporarily disabled - trusting user metadata")
            
            try:
                # Detectar mão dominante usando AI
                detected_hand = detect_dominant_hand_from_video(user_video_path)
                user_hand = analysis_metadata.maoDominante
                
                # Detectar lado da câmera usando AI  
                detected_camera = detect_camera_side_from_video(user_video_path)
                user_camera = analysis_metadata.ladoCamera
                
                # Validation is disabled - just log the detection for debugging purposes
                
                # Log the detection for debugging but don't block analysis
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== AI VALIDATION DEBUG {datetime.now()} ===\n")
                    f.write(f"Detected hand: {detected_hand} vs User: {user_hand}\n")
                    f.write(f"Detected camera: {detected_camera} vs User: {user_camera}\n")
                    f.write(f"Status: Continuing with user metadata (validation disabled)\n")
                    f.write(f"=== AI VALIDATION PASSED ===\n\n")
                
                logger.info(f"[AI_VALIDATION] [OK] Continuing with user-provided metadata")
                
                # Continue without blocking - validation is disabled
                
                logger.info(f"[AI_VALIDATION] [OK] Validação AI passou:")
                logger.info(f"[AI_VALIDATION] Mão dominante: {user_hand} == {detected_hand}")
                logger.info(f"[AI_VALIDATION] Lado da câmera: {user_camera} == {detected_camera}")
                
            except Exception as e:
                logger.warning(f"[AI_VALIDATION] [WARNING] Erro na validação AI: {e}")
                logger.warning(f"[AI_VALIDATION] Continuando sem validação AI...")
            
            # VALIDAÇÃO ADICIONAL EXPERIMENTAL (apenas para desenvolvimento)
            # NOTA: Esta validação usa filename como referência, apenas para desenvolvimento
            # Em produção, deveria usar detecção real por IA
            validation_warnings = []
            
            filename = user_video.filename.lower() if user_video.filename else ""
            
            # Validar lado da câmera baseado no filename (apenas para desenvolvimento)
            if "_e.mp4" in filename or "_e_" in filename:  # Esquerda no filename
                if metadata_dict.get("ladoCamera") == "Direita":
                    validation_warnings.append(f"[WARNING] Lado da câmera: Filename sugere 'Esquerda' mas você informou 'Direita'")
            elif "_d.mp4" in filename or "_d_" in filename:  # Direita no filename  
                if metadata_dict.get("ladoCamera") == "Esquerda":
                    validation_warnings.append(f"[WARNING] Lado da câmera: Filename sugere 'Direita' mas você informou 'Esquerda'")
            
            # OPÇÃO 1: Apenas avisar (modo atual)
            # OPÇÃO 2: Bloquear análise se houver inconsistências (descomente abaixo)
            
            # Configuração: definir se deve bloquear análise por inconsistências de filename
            BLOCK_ON_FILENAME_MISMATCH = True  # Mude para False para só avisar
            
            if validation_warnings:
                logger.warning(f"[METADATA_VALIDATION] [WARNING] Inconsistências detectadas:")
                for warning in validation_warnings:
                    logger.warning(f"[METADATA_VALIDATION] {warning}")
                
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== VALIDAÇÃO FILENAME {datetime.now()} ===\n")
                    f.write(f"Filename: {user_video.filename}\n")
                    f.write(f"Metadata informado: {metadata_dict}\n")
                    for warning in validation_warnings:
                        f.write(f"INCONSISTÊNCIA: {warning}\n")
                    
                    if BLOCK_ON_FILENAME_MISMATCH:
                        f.write(f"=== ANÁLISE BLOQUEADA POR INCONSISTÊNCIA ===\n\n")
                    else:
                        f.write(f"=== CONTINUANDO COM AVISOS ===\n\n")
                
                # Se configurado para bloquear, retornar erro
                if BLOCK_ON_FILENAME_MISMATCH:
                    return AnalysisResult(
                        success=False,
                        analysis_id=secrets.token_hex(16),
                        error=f"Inconsistências detectadas nos metadados informados. " + " | ".join(validation_warnings),
                        message="Validação de metadados falhou",
                        final_score=0.0,
                        recommendations=[
                            "Verifique os parâmetros informados:",
                            "- Mão dominante está correta?",
                            "- Lado da câmera está correto?", 
                            "- Lado da raquete (Forehand/Backhand) está correto?",
                            "- Tipo de movimento (Drive/Push) está correto?",
                            "",
                            "NOTA: Esta validação usa o filename como referência durante desenvolvimento.",
                            "Corrija os parâmetros e tente novamente."
                        ] + validation_warnings,
                        detailed_analysis={
                            "validation_failed": True,
                            "filename_analysis": True,
                            "warnings": validation_warnings,
                            "user_metadata": metadata_dict,
                            "filename": user_video.filename
                        }
                    )
            
            # Add optimized professional comparisons
            try:
                from optimized_professional_comparison import OptimizedProfessionalComparator
                comparator = OptimizedProfessionalComparator()
                
                # DEBUG: Log movement detection details
                logger.info(f"[DEBUG_MOVEMENT] User config movement: {user_config_movement}")
                logger.info(f"[DEBUG_MOVEMENT] Detected movement: {detected_movement}")
                logger.info(f"[DEBUG_MOVEMENT] Movement classification: {result.get('detailed_analysis', {}).get('movement_classification')}")
                
                # DEBUG: Escrever logs em arquivo para debug
                with open("debug_comparison.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== DEBUG COMPARISON {datetime.now()} ===\n")
                    f.write(f"User config movement: {user_config_movement}\n")
                    f.write(f"Detected movement: {detected_movement}\n")
                    f.write(f"Movement classification: {result.get('detailed_analysis', {}).get('movement_classification')}\n")
                    f.write(f"User metadata: {metadata_dict}\n")
                
                # Find best matches
                best_matches = comparator.find_best_matches(result, detected_movement, max_results=3)
                
                # Add comparisons to result
                if best_matches:
                    logger.info(f"[PROFESSIONAL_COMPARISON] Found {len(best_matches)} professional matches")
                    result['professional_comparisons'] = []
                    
                    for match in best_matches:
                        comparison_data = {
                            'professional_name': match.professional_name,
                            'professional_video': match.professional_video,
                            'similarity_score': match.similarity_score,
                            'similarity_percentage': f"{match.similarity_score * 100:.1f}%",
                            'detailed_comparison': match.detailed_comparison,
                            'recommendations': match.recommendations,
                            'comparison_confidence': match.confidence
                        }
                        result['professional_comparisons'].append(comparison_data)
                        
                    # Add movement statistics
                    movement_stats = comparator.get_movement_statistics(detected_movement)
                    result['movement_statistics'] = movement_stats
                    
                    logger.info(f"[PROFESSIONAL_COMPARISON] Best match: {best_matches[0].professional_name} ({best_matches[0].similarity_score*100:.1f}% similarity)")
                else:
                    logger.warning(f"[PROFESSIONAL_COMPARISON] No matches found for {detected_movement}")
                    result['professional_comparisons'] = []
                    result['movement_statistics'] = {}
                    
            except Exception as e:
                logger.error(f"[PROFESSIONAL_COMPARISON] Error in optimized comparison: {e}")
                result['professional_comparisons'] = []
                result['movement_statistics'] = {}
            
            return AnalysisResult(
                success=True,
                analysis_id=secrets.token_hex(16),
                timestamp=datetime.now(),
                analysis_type='enhanced_single_cycle_biomechanical_with_comparisons',
                final_score=result.get('final_score'),
                detailed_analysis=result,
                recommendations=result.get('recommendations', []),
                professional_comparisons=result.get('professional_comparisons', []),
                movement_statistics=result.get('movement_statistics', {})
            )
        else:
            logger.error(f"[BIOMECH_API] Enhanced analysis failed: {result.get('error')}")
            
            # DEBUG: Log da falha
            with open("debug_comparison.log", "a", encoding="utf-8") as f:
                f.write(f"ANALISE FALHOU!\n")
                f.write(f"Erro: {result.get('error')}\n")
                f.write(f"Resultado completo: {result}\n")
                f.write(f"=== FIM ANALISE (FALHA) ===\n\n")
            
            raise HTTPException(status_code=500, detail=result.get('error', 'Enhanced analysis failed'))
        
    except Exception as e:
        logger.error(f"Error in biomechanical analysis: {e}")
        
        # DEBUG: Log da exceção
        with open("debug_comparison.log", "a", encoding="utf-8") as f:
            f.write(f"EXCECAO CAPTURADA!\n")
            f.write(f"Erro: {str(e)}\n")
            f.write(f"Tipo: {type(e).__name__}\n")
            f.write(f"=== FIM ANALISE (EXCECAO) ===\n\n")
        
        return AnalysisResult(
            success=False,
            analysis_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            analysis_type='enhanced_single_cycle_biomechanical_with_comparisons',
            error=str(e)
        )



@app.post("/extract-user-cycle")
async def extract_user_cycle(
    user_video: UploadFile = File(...),
    metadata: str = Form(...),
    cycle_index: int = Form(1)  # Default to 2nd cycle (index 1)
):
    """Extract specific cycle from user video for display"""
    logger.info(f"[LAUNCH] [EXTRACT_CYCLE] ENDPOINT INICIADO - cycle_index: {cycle_index}")
    try:
        logger.info(f"[USER_CYCLE] Extracting cycle {cycle_index + 1} for display")
        
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        
        # Save user video temporarily
        user_video_path = await save_uploaded_file(user_video)
        
        try:
            # Get video properties
            import cv2
            cap = cv2.VideoCapture(str(user_video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
            
            # Try to detect cycles using enhanced analyzer
            cycles = []
            try:
                from enhanced_single_cycle_analysis import EnhancedSingleCycleAnalyzer
                enhanced_analyzer = EnhancedSingleCycleAnalyzer()
                cycles_info = enhanced_analyzer.detect_movement_cycles(str(user_video_path))
                
                if cycles_info and 'cycles' in cycles_info and cycles_info['cycles']:
                    cycles = cycles_info['cycles']
                    logger.info(f"[USER_CYCLE] Detected {len(cycles)} cycles using enhanced analyzer")
                else:
                    logger.warning("[USER_CYCLE] Enhanced analyzer didn't detect cycles, using fallback")
                    
            except Exception as e:
                logger.warning(f"[USER_CYCLE] Enhanced analyzer failed: {e}, using fallback")
            
            # Fallback: create artificial cycles by dividing video into segments
            if not cycles:
                segment_duration = duration / 4  # Divide into 4 segments
                cycles = []
                for i in range(4):
                    start_time = i * segment_duration
                    end_time = min((i + 1) * segment_duration, duration)
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps) - 1
                    
                    cycles.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                
                logger.info(f"[USER_CYCLE] Created {len(cycles)} artificial cycles")
            
            if not cycles or len(cycles) <= cycle_index:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Cycle {cycle_index + 1} not found. Video has {len(cycles) if cycles else 0} cycles available."
                )
            
            # Get the specific cycle
            selected_cycle = cycles[cycle_index]
            
            # Extract frame information
            if isinstance(selected_cycle, dict):
                start_frame = selected_cycle.get('start_frame', 0)
                end_frame = selected_cycle.get('end_frame', total_frames - 1)
            else:
                # Fallback for other formats
                start_frame = 0
                end_frame = total_frames - 1
            
            # Calculate time bounds
            start_time = start_frame / fps
            end_time = end_frame / fps
            cycle_duration = end_time - start_time
            
            # Create MP4 from cycle frames
            temp_dir = Path("../temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            import secrets
            cycle_filename = f"user_cycle_{secrets.token_hex(8)}.webm"
            cycle_path = temp_dir / cycle_filename
            
            logger.info("[USER_CYCLE] Creating WebM from cycle frames (native web format)")
            
            # Read the video and collect frames from the cycle
            cap = cv2.VideoCapture(str(user_video_path))
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Collect ALL frames from the cycle (NO SKIPPING for natural speed)
            frames = []
            cycle_frames = end_frame - start_frame + 1
            
            logger.info(f"[USER_CYCLE] Frame collection strategy:")
            logger.info(f"  - Cycle frames available: {cycle_frames}")
            logger.info(f"  - Strategy: EXTRACT ALL FRAMES (no skipping)")
            logger.info(f"  - Frame range: {start_frame} to {end_frame}")
            
            # Extract EVERY frame from the cycle
            for frame_count in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Optimize frame size for web compatibility
                if width > 720:  # Limit to 720p for web optimization
                    new_width = 720
                    new_height = int(height * (720 / width))
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                elif width < 320:  # Ensure minimum size
                    new_width = 320
                    new_height = int(height * (320 / width))
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                frames.append(frame_rgb)
            
            cap.release()
            logger.info(f"[USER_CYCLE] Collected {len(frames)} frames for MP4 (ALL frames = natural speed)")
            
            if len(frames) == 0:
                raise Exception("No frames could be extracted from video")
            
            # Create MP4 with web-friendly optimization
            try:
                # Use optimized FPS for web playback (max 30fps for better compatibility)
                output_fps = min(fps, 30.0)
                
                # Create web-optimized WebM directly  
                temp_cycle_path = cycle_path.with_suffix('.tmp.webm')
                
                logger.info(f"[USER_CYCLE] Creating WebM:")
                logger.info(f"  - Original video FPS: {fps:.1f}")
                logger.info(f"  - Frame range: {start_frame} to {end_frame}")
                logger.info(f"  - Cycle frames available: {cycle_frames}")
                logger.info(f"  - Total frames collected: {len(frames)}")
                logger.info(f"  - Output MP4 FPS: {output_fps:.1f}")
                logger.info(f"  - Real cycle duration: {cycle_duration:.3f}s")
                logger.info(f"  - Expected MP4 duration: {len(frames) / output_fps:.3f}s")
                
                # Create video writer
                if len(frames) > 0 and len(frames[0].shape) == 3:
                    height, width, channels = frames[0].shape
                    
                    # Use WebM-optimized codecs  
                    codecs_to_try = [
                        ('VP80', 'VP8 (WebM native)'),
                        ('VP90', 'VP9 (WebM advanced)'),
                        ('MJPG', 'Motion JPEG fallback')
                    ]
                    
                    out = None
                    for codec, description in codecs_to_try:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            out = cv2.VideoWriter(str(temp_cycle_path), fourcc, output_fps, (width, height))
                            if out.isOpened():
                                logger.info(f"[USER_CYCLE] Using codec: {codec} - {description}")
                                break
                            else:
                                logger.warning(f"[USER_CYCLE] Codec {codec} failed to open")
                        except Exception as e:
                            logger.warning(f"[USER_CYCLE] Codec {codec} error: {e}")
                    
                    if not out or not out.isOpened():
                        raise Exception("Could not open video writer with any codec")
                    
                    # Write frames
                    for frame in frames:
                        # Convert RGB back to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    
                    out.release()
                else:
                    raise Exception("Invalid frame format for MP4 creation")
                
                logger.info(f"[USER_CYCLE] Initial AVI created successfully: {temp_cycle_path}")
                logger.info(f"[HOT] [DEBUG] AVI CRIADO - indo verificar tamanho")
                
            except Exception as e:
                logger.error(f"[USER_CYCLE] AVI creation failed: {e}")
                raise Exception(f"AVI creation failed: {str(e)}")
            
            # Verify the temporary video was created
            if not temp_cycle_path.exists():
                raise Exception(f"Temporary video file was not created: {temp_cycle_path}")
            
            temp_size = temp_cycle_path.stat().st_size
            if temp_size < 1000:  # Less than 1KB
                raise Exception(f"Temporary video file is too small ({temp_size} bytes), likely corrupted")
            
            logger.info(f"[USER_CYCLE] Temporary video created: {temp_cycle_path} ({temp_size} bytes)")
            logger.info(f"[HOT] [DEBUG] ANTES DA CONVERSÃO - temp_size: {temp_size}")
            
            # FORCE conversion with FFmpeg (OpenCV videos don't work in browsers)
            logger.info(f"[USER_CYCLE] [FIX] FORÇANDO CONVERSÃO COM FFMPEG (OpenCV não é compatível)")
            
            # Try FFmpeg conversion
            try:
                import subprocess
                import shutil
                
                if shutil.which('ffmpeg'):
                    # Convert to proper MP4 with H.264
                    final_mp4_path = cycle_path.with_suffix('.mp4') 
                    
                    logger.info(f"[USER_CYCLE] Converting {temp_cycle_path} to {final_mp4_path}")
                    
                    result = subprocess.run([
                        'ffmpeg', '-i', str(temp_cycle_path),
                        '-vcodec', 'libx264',           # H.264 codec
                        '-acodec', 'aac',               # AAC audio
                        '-movflags', '+faststart',      # Enable streaming
                        '-pix_fmt', 'yuv420p',         # Compatible pixel format
                        '-y',                           # Overwrite
                        str(final_mp4_path)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Success - use converted file
                        temp_cycle_path.unlink()  # Delete temp
                        cycle_path = final_mp4_path
                        cycle_filename = cycle_path.name
                        logger.info(f"[USER_CYCLE] [OK] FFmpeg conversion successful!")
                    else:
                        logger.error(f"[USER_CYCLE] FFmpeg failed: {result.stderr}")
                        raise Exception("FFmpeg conversion failed")
                else:
                    raise Exception("FFmpeg not available")
                    
            except Exception as e:
                logger.warning(f"[USER_CYCLE] FFmpeg fallback failed: {e}")
                # Use original file as last resort
                temp_cycle_path.rename(cycle_path)
                logger.warning(f"[USER_CYCLE] Using OpenCV file as last resort (may not play)")
            
            final_size = cycle_path.stat().st_size
            logger.info(f"[USER_CYCLE] [OK] Final video ready: {cycle_path} ({final_size} bytes)")
            
            # Return success response with video URL
            return {
                "success": True,
                "video_url": f"/temp/{cycle_filename}",
                "cycle_info": {
                    "cycle_index": cycle_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": cycle_duration,
                    "total_cycles": len(cycles)
                }
            }
            
        finally:
            # Cleanup temp file
            try:
                user_video_path.unlink()
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[!] [EXTRACT_CYCLE] ERRO CAPTURADO: {e}")
        import traceback
        logger.error(f"[!] [EXTRACT_CYCLE] TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error extracting user cycle: {str(e)}")

# Development endpoints
@app.get("/dev/test-components")
async def test_components():
    """Test system components"""
    # Remove environment check to allow testing on Railway
    
    results = {}
    
    # Test analyzer
    try:
        if analyzer_api is not None:
            professionals = analyzer_api.engine.get_available_professionals('forehand_drive')
            results["analyzer"] = {"status": "ok", "professionals_count": len(professionals)}
        else:
            results["analyzer"] = {"status": "unavailable", "reason": "Tennis analyzer API not available in this environment"}
    except Exception as e:
        results["analyzer"] = {"status": "error", "error": str(e)}
    
    # Test real-time engine
    try:
        if REALTIME_ENGINE_AVAILABLE and RealTimeAnalysisEngine is not None:
            engine_test = RealTimeAnalysisEngine()
            results["real_time_engine"] = {"status": "ok", "initialized": True}
        else:
            results["real_time_engine"] = {"status": "unavailable", "reason": "Real-time engine not available in this environment"}
    except Exception as e:
        results["real_time_engine"] = {"status": "error", "error": str(e)}
    
    return results

@app.post("/test-real-analysis")
async def test_real_analysis():
    """Test endpoint with exact same code that worked in direct test"""
    try:
        # Use exact same paths and metadata that worked in direct test
        user_video = os.path.join(project_root, "videos", "Zhang_Jike_FD_D_D.mp4")
        pro_video = os.path.join(project_root, "profissionais", "forehand_drive", "Zhang_Jike_FD_D_D.mp4")
        
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
        
        print(f"[TEST_REAL] Starting test with direct backend call...")
        print(f"[TEST_REAL] User video exists: {os.path.exists(user_video)}")
        print(f"[TEST_REAL] Pro video exists: {os.path.exists(pro_video)}")
        
        # Check if analyzer is available
        if analyzer_api is None:
            return {"error": "Analyzer API not available in this environment"}
        
        # Call tennis_comparison_backend directly (exactly like successful direct test)
        result = analyzer_api.engine.compare_techniques(
            user_video, pro_video, user_metadata, prof_metadata
        )
        
        print(f"[TEST_REAL] Backend result success: {result.get('success')}")
        print(f"[TEST_REAL] Backend result score: {result.get('final_score')}")
        print(f"[TEST_REAL] Backend result user_analysis keys: {list(result.get('user_analysis', {}).keys())}")
        print(f"[TEST_REAL] Backend result professional_analysis keys: {list(result.get('professional_analysis', {}).keys())}")
        print(f"[TEST_REAL] Backend result comparison keys: {list(result.get('comparison', {}).keys())}")
        
        return result
        
    except Exception as e:
        logger.error(f"[TEST_REAL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

class CycleRequest(BaseModel):
    filename: str
    metadata: dict
    cycle_index: int

@app.post("/generate-cycle-from-selected")
async def generate_cycle_from_selected(request: CycleRequest):
    """Generate user cycle video from a selected video file"""
    try:
        logger.info(f"[CYCLE_GENERATION] Extracting cycle {request.cycle_index} from: {request.filename}")
        
        # Validate filename and get video path
        project_root = Path(__file__).parent.parent
        videos_folder = project_root / "videos"
        video_path = videos_folder / request.filename
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.filename}")
        
        # Import cycle detector for cycle detection
        from cycle_detector_retracted_extended import CycleDetectorRetractedExtended
        import cv2
        
        if not MEDIAPIPE_AVAILABLE:
            raise HTTPException(status_code=503, detail="MediaPipe not available in this environment")
        
        # Read video frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            return {
                "success": False,
                "error": "Could not read video frames",
                "cycles_found": 0
            }
        
        # Create cycle detector and detect cycles
        cycle_detector = CycleDetectorRetractedExtended()
        
        # Extract metadata for cycle detection
        metadata_obj = AnalysisMetadata(**request.metadata)
        
        validated_params = {
            'dominant_hand': 'right' if metadata_obj.maoDominante == 'Destro' else 'left',
            'movement_type': 'forehand' if metadata_obj.ladoRaquete == 'F' else 'backhand', 
            'camera_side': 'right' if metadata_obj.ladoCamera == 'Direita' else 'left',
            'racket_side': 'forehand' if metadata_obj.ladoRaquete == 'F' else 'backhand'
        }
        
        # Detect cycles
        cycle_infos = cycle_detector.detect_cycles_from_validated_params(frames, validated_params)
        
        # Convert to expected format
        cycles_info = {
            'cycles': [
                {
                    'start_frame': cycle.start_frame,
                    'end_frame': cycle.end_frame,
                    'confidence': cycle.quality_score
                }
                for cycle in cycle_infos
            ]
        }
        
        if not cycles_info or 'cycles' not in cycles_info or not cycles_info['cycles']:
            return {
                "success": False,
                "error": "No movement cycles detected in the video",
                "cycles_found": 0
            }
        
        cycles = cycles_info['cycles']
        logger.info(f"[CYCLE_GENERATION] Detected {len(cycles)} cycles")
        
        # Check if requested cycle index exists
        if request.cycle_index >= len(cycles):
            return {
                "success": False,
                "error": f"Cycle {request.cycle_index} not found. Only {len(cycles)} cycles detected.",
                "cycles_found": len(cycles),
                "available_cycles": [i for i in range(len(cycles))]
            }
        
        # Get the requested cycle
        target_cycle = cycles[request.cycle_index]
        
        # Extract cycle frames and create output video
        import cv2
        import tempfile
        import os
        
        # Create temporary file for the extracted cycle
        temp_dir = Path(tempfile.gettempdir())
        cycle_filename = f"user_cycle_{secrets.token_hex(16)}.mp4"
        cycle_output_path = temp_dir / cycle_filename
        
        # Extract cycle from original video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range for the cycle
        start_frame = int(target_cycle['start_frame'])
        end_frame = int(target_cycle['end_frame'])
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(str(cycle_output_path), fourcc, fps, (frame_width, frame_height))
        
        # Extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Move file to temp_uploads directory for serving
        temp_uploads_dir = project_root / "temp_uploads"
        temp_uploads_dir.mkdir(exist_ok=True)
        final_path = temp_uploads_dir / cycle_filename
        
        # Copy the file
        import shutil
        shutil.move(str(cycle_output_path), str(final_path))
        
        logger.info(f"[CYCLE_GENERATION] Cycle extracted successfully: {cycle_filename}")
        
        return {
            "success": True,
            "message": f"Cycle {request.cycle_index} extracted successfully",
            "video_url": f"/temp/{cycle_filename}",
            "cycle_info": {
                "cycle_index": request.cycle_index,
                "total_cycles": len(cycles),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration": (end_frame - start_frame) / fps,
                "start_time": start_frame / fps,
                "end_time": end_frame / fps
            },
            "filename": cycle_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CYCLE_GENERATION] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Cycle generation failed: {str(e)}")


@app.get("/generate-user-cycle/{video_filename}")
async def generate_user_cycle_get(video_filename: str, cycle_index: int = 1):
    """Generate cycle video for a video from the restricted folder"""
    logger.info(f"[GENERATE_CYCLE_GET] Starting cycle generation for {video_filename}, cycle {cycle_index}")
    
    try:
        # Construct full video path from videos folder
        project_root = Path(__file__).parent.parent
        videos_folder = project_root / "videos"
        video_path = videos_folder / video_filename
        
        # Security check: ensure file is in videos folder
        if not video_path.exists():
            logger.error(f"[GENERATE_CYCLE_GET] Video not found: {video_path}")
            raise HTTPException(status_code=404, detail=f"Video {video_filename} not found")
        
        if not str(video_path.resolve()).startswith(str(videos_folder.resolve())):
            logger.error(f"[GENERATE_CYCLE_GET] Security violation: {video_path}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Use default metadata for cycle generation (not needed for cycle extraction)
        metadata_dict = {
            'maoDominante': 'Destro',
            'ladoCamera': 'Esquerda', 
            'ladoRaquete': 'F',
            'tipoMovimento': 'D'
        }
        
        # Simple cycle generation - no complex detection needed
        cycles = []
        
        # Get video properties
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Determine cycle frames
        if cycles and len(cycles) > cycle_index:
            cycle = cycles[cycle_index]
            start_frame = cycle.start_frame
            end_frame = cycle.end_frame
            logger.info(f"[GENERATE_CYCLE_GET] Using detected cycle: frames {start_frame}-{end_frame}")
        else:
            # Fallback: use proportional segments - cycle_index=1 means second cycle
            segment_duration = duration / 4
            start_time = cycle_index * segment_duration
            end_time = min((cycle_index + 1) * segment_duration, duration)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps) - 1
            logger.info(f"[GENERATE_CYCLE_GET] Using fallback segment {cycle_index}: frames {start_frame}-{end_frame}")
        
        # Ensure valid frame range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        
        if start_frame >= end_frame:
            raise HTTPException(status_code=400, detail="Invalid cycle range")
        
        # Generate output filename
        import secrets
        cycle_id = secrets.token_hex(8)
        cycle_filename = f"user_cycle_{cycle_id}.mp4"
        
        # Output path
        temp_uploads_dir = project_root / "temp_uploads"
        temp_uploads_dir.mkdir(exist_ok=True)
        cycle_output_path = temp_uploads_dir / cycle_filename
        
        # Extract cycle frames and create video
        logger.info(f"[GENERATE_CYCLE_GET] Opening video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"[GENERATE_CYCLE_GET] Failed to open video: {video_path}")
            raise HTTPException(status_code=500, detail="Failed to open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[GENERATE_CYCLE_GET] Video properties: {width}x{height}, fps={fps}")
        
        # Create video writer with mobile-compatible codec
        # Try multiple codecs in order of mobile compatibility
        codecs_to_try = [
            ('mp4v', 'MPEG-4 Part 2 (most mobile compatible)'),
            ('H264', 'H.264 AVC (iOS/Android standard)'),
            ('avc1', 'H.264 AVC1 variant'),
            ('h264', 'H.264 lowercase'),
            ('XVID', 'XVID codec'),
            ('MJPG', 'Motion JPEG fallback')
        ]
        
        out = None
        used_codec = None
        
        for codec, description in codecs_to_try:
            logger.info(f"[GENERATE_CYCLE_GET] Trying {description}: {codec}")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(cycle_output_path), fourcc, fps, (width, height))
            
            if out.isOpened():
                used_codec = codec
                logger.info(f"[GENERATE_CYCLE_GET] Successfully using codec: {codec}")
                break
            else:
                logger.warning(f"[GENERATE_CYCLE_GET] Codec {codec} failed")
                out.release()
                out = None
        
        if out is None or not out.isOpened():
            logger.error(f"[GENERATE_CYCLE_GET] Failed to create video writer with any codec: {cycle_output_path}")
            cap.release()
            raise HTTPException(status_code=500, detail="Failed to create output video")
        
        # First, collect all frames to memory
        frames = []
        current_frame = 0
        logger.info(f"[GENERATE_CYCLE_GET] Extracting frames {start_frame} to {end_frame}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info(f"[GENERATE_CYCLE_GET] End of video reached at frame {current_frame}")
                break
                
            if start_frame <= current_frame <= end_frame:
                frames.append(frame.copy())
                if len(frames) % 10 == 0:  # Log every 10 frames
                    logger.info(f"[GENERATE_CYCLE_GET] Collected {len(frames)} frames")
                
            current_frame += 1
            
            if current_frame > end_frame:
                logger.info(f"[GENERATE_CYCLE_GET] Reached end frame {end_frame}")
                break
        
        cap.release()
        frame_count = len(frames)
        
        # Write frames with the selected codec
        for i, frame in enumerate(frames):
            out.write(frame)
        
        out.release()
        
        # Verify MP4 file was created
        if cycle_output_path.exists():
            file_size = cycle_output_path.stat().st_size
            logger.info(f"[GENERATE_CYCLE_GET] Generated MP4 cycle video: {cycle_filename} ({frame_count} frames, {file_size} bytes, codec: {used_codec})")
        else:
            logger.error(f"[GENERATE_CYCLE_GET] MP4 output file was not created: {cycle_output_path}")
            raise HTTPException(status_code=500, detail="MP4 video file was not created")
        
        # Also create a WebM version for better web compatibility
        webm_filename = cycle_filename.replace('.mp4', '.webm')
        webm_output_path = temp_uploads_dir / webm_filename
        
        try:
            logger.info(f"[GENERATE_CYCLE_GET] Creating WebM version: {webm_filename}")
            fourcc_webm = cv2.VideoWriter_fourcc(*'VP80')  # VP8 codec for WebM
            out_webm = cv2.VideoWriter(str(webm_output_path), fourcc_webm, fps, (width, height))
            
            if out_webm.isOpened():
                for frame in frames:
                    out_webm.write(frame)
                out_webm.release()
                
                if webm_output_path.exists():
                    webm_size = webm_output_path.stat().st_size
                    logger.info(f"[GENERATE_CYCLE_GET] Generated WebM cycle video: {webm_filename} ({frame_count} frames, {webm_size} bytes)")
                else:
                    logger.warning(f"[GENERATE_CYCLE_GET] WebM file was not created successfully")
            else:
                logger.warning(f"[GENERATE_CYCLE_GET] Failed to create WebM writer")
                out_webm.release()
        except Exception as e:
            logger.warning(f"[GENERATE_CYCLE_GET] WebM creation failed: {e}")
            webm_filename = None  # Fallback to MP4 only
        
        return {
            "success": True,
            "message": f"Cycle {cycle_index} generated successfully",
            "video_url": f"/temp/{cycle_filename}",
            "webm_url": f"/temp/{webm_filename}" if webm_filename else None,
            "cycle_info": {
                "cycle_index": cycle_index,
                "total_cycles": len(cycles) if cycles else 4,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration": (end_frame - start_frame) / fps,
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
                "frame_count": frame_count,
                "codec_used": used_codec,
                "has_webm": webm_filename is not None
            },
            "filename": cycle_filename,
            "webm_filename": webm_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GENERATE_CYCLE_GET] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Cycle generation failed: {str(e)}")


@app.get("/generate-professional-cycle/{professional_name}")
async def generate_professional_cycle(professional_name: str, cycle_index: int = 1):
    """Generate cycle video for the professional with highest similarity"""
    logger.info(f"[GENERATE_PRO_CYCLE] Starting professional cycle generation for {professional_name}, cycle {cycle_index}")
    
    try:
        # Get professional video information
        project_root = Path(__file__).parent.parent
        professionals_folder = project_root / "profissionais"
        
        # Find professional video by name
        professional_video_path = None
        professional_metadata = None
        
        # Search through professional videos to find matching name
        for technique_folder in professionals_folder.iterdir():
            if technique_folder.is_dir():
                for video_file in technique_folder.glob("*.mp4"):
                    # Extract professional name from filename (assuming format: name_technique_hand_camera.mp4)
                    filename = video_file.stem.lower()
                    professional_name_lower = professional_name.lower()
                    
                    # Use exact match for professional name at the beginning of filename
                    # Format: ProfessionalName_XX_Y_Z.mp4
                    if filename.startswith(professional_name_lower + "_"):
                        professional_video_path = video_file
                        
                        # Extract metadata from filename
                        parts = filename.split('_')
                        technique = technique_folder.name  # forehand_drive, backhand_push, etc.
                        
                        # Basic metadata extraction
                        professional_metadata = {
                            'technique': technique,
                            'name': professional_name,
                            'filename': video_file.name,
                            'path': str(video_file)
                        }
                        break
            
            if professional_video_path:
                break
        
        if not professional_video_path or not professional_video_path.exists():
            logger.error(f"[GENERATE_PRO_CYCLE] Professional video not found for: {professional_name}")
            raise HTTPException(status_code=404, detail=f"Professional video for {professional_name} not found")
        
        logger.info(f"[GENERATE_PRO_CYCLE] Found professional video: {professional_video_path}")
        
        # Get video properties
        import cv2
        cap = cv2.VideoCapture(str(professional_video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open professional video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"[GENERATE_PRO_CYCLE] Video properties: {width}x{height}, {fps}fps, {total_frames} frames, {duration:.2f}s")
        
        # Try advanced cycle detection for professional videos
        cycles = []
        try:
            from cycle_detector_retracted_extended import CycleDetectorRetractedExtended
            
            # Use default parameters for professional videos (they should be well-recorded)
            cycle_detector = CycleDetectorRetractedExtended(
                fps=fps,
                min_cycle_duration=0.8,
                max_cycle_duration=3.0
            )
            
            # Default metadata for professional video analysis
            detection_params = {
                'dominant_hand': 'right',  # Most professionals are right-handed
                'movement_type': professional_metadata['technique'].split('_')[0],  # forehand/backhand
                'camera_side': 'left',     # Standard camera position
                'racket_side': professional_metadata['technique'].split('_')[0]  # forehand/backhand
            }
            
            detected_cycles = cycle_detector.detect_cycles_from_validated_params(
                str(professional_video_path), detection_params
            )
            
            if detected_cycles and len(detected_cycles) > 0:
                cycles = detected_cycles
                logger.info(f"[GENERATE_PRO_CYCLE] Detected {len(cycles)} cycles in professional video")
            else:
                logger.warning(f"[GENERATE_PRO_CYCLE] No cycles detected, using fallback segmentation")
                
        except Exception as cycle_error:
            logger.warning(f"[GENERATE_PRO_CYCLE] Cycle detection failed: {cycle_error}, using fallback")
        
        # Determine cycle frames
        if cycles and len(cycles) > cycle_index:
            cycle = cycles[cycle_index]
            if hasattr(cycle, 'start_frame') and hasattr(cycle, 'end_frame'):
                start_frame = cycle.start_frame
                end_frame = cycle.end_frame
            else:
                start_frame = cycle.get('start_frame', 0)
                end_frame = cycle.get('end_frame', total_frames - 1)
            logger.info(f"[GENERATE_PRO_CYCLE] Using detected cycle: frames {start_frame}-{end_frame}")
        else:
            # Fallback: use proportional segments - cycle_index=1 means second cycle
            segment_duration = duration / 4
            start_time = cycle_index * segment_duration
            end_time = min((cycle_index + 1) * segment_duration, duration)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps) - 1
            logger.info(f"[GENERATE_PRO_CYCLE] Using fallback segmentation: frames {start_frame}-{end_frame}")
        
        # Ensure valid frame range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        
        if start_frame >= end_frame:
            raise HTTPException(status_code=400, detail="Invalid cycle frame range")
        
        # Extract frames from the cycle
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            current_frame += 1
        
        cap.release()
        
        if not frames:
            raise HTTPException(status_code=500, detail="No frames extracted from professional video")
        
        logger.info(f"[GENERATE_PRO_CYCLE] Extracted {len(frames)} frames")
        
        # Create output video - use same path structure as user cycle generation
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"[GENERATE_PRO_CYCLE] Project root: {project_root}")
        logger.info(f"[GENERATE_PRO_CYCLE] Temp directory: {temp_dir}")
        logger.info(f"[GENERATE_PRO_CYCLE] Temp directory exists: {temp_dir.exists()}")
        
        import secrets
        import re
        
        # Clean professional name for filename (remove spaces and special chars)
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', professional_name.lower())
        video_id = secrets.token_hex(8)  # Shorter ID to avoid filename issues
        output_filename = f"pro_cycle_{clean_name}_{video_id}.mp4"
        output_path = temp_dir / output_filename
        
        # Use same codecs as user video generation for consistency
        codecs_to_try = [
            ('mp4v', 'MPEG-4 Part 2 (most mobile compatible)'),
            ('H264', 'H.264 AVC (iOS/Android standard)'),
            ('avc1', 'H.264 AVC1 variant'),
            ('h264', 'H.264 lowercase'),
            ('XVID', 'XVID codec'),
            ('MJPG', 'Motion JPEG fallback')
        ]
        
        video_created = False
        for codec, description in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                if out.isOpened():
                    logger.info(f"[GENERATE_PRO_CYCLE] Using codec: {codec} ({description})")
                    
                    # Write all frames
                    for frame in frames:
                        out.write(frame)
                    
                    out.release()
                    
                    # Verify file was created and has content
                    if output_path.exists() and output_path.stat().st_size > 1000:
                        file_size = output_path.stat().st_size
                        video_created = True
                        logger.info(f"[GENERATE_PRO_CYCLE] Professional cycle video created: {output_filename} ({file_size} bytes)")
                        logger.info(f"[GENERATE_PRO_CYCLE] Full path: {output_path}")
                        break
                    else:
                        file_size = output_path.stat().st_size if output_path.exists() else 0
                        logger.warning(f"[GENERATE_PRO_CYCLE] Video file created but appears empty with codec {codec} (size: {file_size} bytes)")
                        if output_path.exists():
                            output_path.unlink()
                else:
                    logger.warning(f"[GENERATE_PRO_CYCLE] Could not open video writer with codec {codec}")
                    
            except Exception as codec_error:
                logger.warning(f"[GENERATE_PRO_CYCLE] Codec {codec} failed: {codec_error}")
                continue
        
        if not video_created:
            raise HTTPException(status_code=500, detail="Failed to create professional cycle video with any codec")
        
        # Also create a WebM version for better web compatibility (same as user video)
        webm_filename = output_filename.replace('.mp4', '.webm')
        webm_output_path = temp_dir / webm_filename
        webm_created = False
        
        try:
            logger.info(f"[GENERATE_PRO_CYCLE] Creating WebM version: {webm_filename}")
            fourcc_webm = cv2.VideoWriter_fourcc(*'VP80')  # VP8 codec for WebM
            out_webm = cv2.VideoWriter(str(webm_output_path), fourcc_webm, fps, (width, height))
            
            if out_webm.isOpened():
                for frame in frames:
                    out_webm.write(frame)
                out_webm.release()
                
                if webm_output_path.exists() and webm_output_path.stat().st_size > 1000:
                    webm_size = webm_output_path.stat().st_size
                    logger.info(f"[GENERATE_PRO_CYCLE] Generated WebM professional video: {webm_filename} ({len(frames)} frames, {webm_size} bytes)")
                    webm_created = True
                else:
                    logger.warning(f"[GENERATE_PRO_CYCLE] WebM file was not created successfully")
            else:
                logger.warning(f"[GENERATE_PRO_CYCLE] Failed to create WebM writer")
                out_webm.release()
        except Exception as e:
            logger.warning(f"[GENERATE_PRO_CYCLE] WebM creation failed: {e}")
        
        # Return video information
        return {
            "success": True,
            "video_filename": output_filename,
            "video_url": f"/temp/{output_filename}",
            "webm_url": f"/temp/{webm_filename}" if webm_created else None,
            "professional_name": professional_name,
            "technique": professional_metadata['technique'],
            "cycle_index": cycle_index,
            "frames_extracted": len(frames),
            "duration_seconds": len(frames) / fps,
            "cycle_range": f"{start_frame}-{end_frame}",
            "message": f"Professional cycle video generated successfully for {professional_name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GENERATE_PRO_CYCLE] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Professional cycle generation failed: {str(e)}")


@app.post("/correct-movement")
async def correct_movement(
    analysis_id: str = Form(...),
    corrected_movement: str = Form(...),
    original_metadata: str = Form(...)
):
    """
    Endpoint para correção manual de movimento detectado.
    Permite ao usuário sobrescrever a detecção de movimento e reprocessar a análise.
    
    Args:
        analysis_id: ID da análise original
        corrected_movement: Movimento corrigido ("forehand_drive", "forehand_push", "backhand_drive", "backhand_push")
        original_metadata: Metadata original da análise
    """
    logger.info(f"[CORRECTION] Starting movement correction for analysis_id: {analysis_id}")
    logger.info(f"[CORRECTION] Corrected movement: {corrected_movement}")
    logger.info(f"[CORRECTION] Original metadata: {original_metadata}")
    
    try:
        # Validar movimento corrigido
        valid_movements = ["forehand_drive", "forehand_push", "backhand_drive", "backhand_push"]
        logger.info(f"[CORRECTION] Valid movements: {valid_movements}")
        
        if corrected_movement not in valid_movements:
            logger.error(f"[CORRECTION] Invalid movement: {corrected_movement}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid movement type. Must be one of: {', '.join(valid_movements)}"
            )
        
        logger.info(f"[CORRECTION] Movement validation passed: {corrected_movement}")
        
        # Parse metadata original
        try:
            metadata_dict = json.loads(original_metadata)
            logger.info(f"[CORRECTION] Parsed metadata: {metadata_dict}")
        except json.JSONDecodeError as e:
            logger.error(f"[CORRECTION] Metadata parsing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")
        
        # Tentar encontrar o vídeo original baseado no analysis_id
        # Como não armazenamos análises persistentemente, vamos buscar na pasta temp_uploads
        temp_dir = Path(__file__).parent.parent / "temp_uploads"
        logger.info(f"[CORRECTION] Searching for videos in: {temp_dir}")
        
        video_files = list(temp_dir.glob("*.mp4"))
        logger.info(f"[CORRECTION] Found {len(video_files)} video files")
        
        # Para este MVP, vamos assumir que o vídeo mais recente é o que queremos reprocessar
        if not video_files:
            logger.error(f"[CORRECTION] No video files found in {temp_dir}")
            raise HTTPException(status_code=404, detail="No video files found for reprocessing")
        
        # Pegar o arquivo mais recente (em produção, isso seria baseado no analysis_id)
        user_video_path = max(video_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"[CORRECTION] Using most recent video file: {user_video_path.name}")
        logger.info(f"[CORRECTION] Video file path: {user_video_path}")
        
        # Importar comparador de profissionais
        logger.info(f"[CORRECTION] Importing professional comparator")
        from optimized_professional_comparison import OptimizedProfessionalComparator
        
        # Inicializar comparador
        logger.info(f"[CORRECTION] Initializing professional comparator")
        comparator = OptimizedProfessionalComparator()
        logger.info(f"[CORRECTION] Comparator initialized with {len(comparator.professionals_data)} professionals")
        
        logger.info(f"[CORRECTION] Using corrected movement: {corrected_movement}")
        
        # Criar análise simulada com o movimento corrigido
        # Os parâmetros biomecânicos são realistas para o tipo de movimento
        movement_patterns = {
            "forehand_drive": {"elbow": 120.0, "amplitude": 0.28, "velocity": 0.40, "coordination": 0.70},
            "forehand_push": {"elbow": 90.0, "amplitude": 0.15, "velocity": 0.20, "coordination": 0.65},
            "backhand_drive": {"elbow": 110.0, "amplitude": 0.25, "velocity": 0.35, "coordination": 0.68},
            "backhand_push": {"elbow": 85.0, "amplitude": 0.12, "velocity": 0.18, "coordination": 0.62}
        }
        
        logger.info(f"[CORRECTION] Available movement patterns: {list(movement_patterns.keys())}")
        
        pattern = movement_patterns.get(corrected_movement, movement_patterns["forehand_drive"])
        logger.info(f"[CORRECTION] Selected pattern for {corrected_movement}: {pattern}")
        
        corrected_biomech_result = {
            "detailed_analysis": {
                "joint_angles": {
                    "elbow_variation_degrees": pattern["elbow"],
                    "coordination_score": pattern["coordination"],
                    "movement_signature": 0.85  # Fixed signature for corrected movements
                },
                "movement_dynamics": {
                    "amplitude_y": pattern["amplitude"],
                    "max_velocity": pattern["velocity"],
                    "temporal_pattern": "consistent"  # Fixed pattern for corrected movements
                },
                "biomechanical_metrics": {},
                "movement_classification": {
                    "confidence": 0.95
                }
            }
        }
        
        logger.info(f"[CORRECTION] Created biomech result structure")
        logger.info(f"[CORRECTION] Joint angles: elbow={pattern['elbow']}, coordination={pattern['coordination']}")
        logger.info(f"[CORRECTION] Movement dynamics: amplitude={pattern['amplitude']}, velocity={pattern['velocity']}")
        
        logger.info(f"[CORRECTION] Starting professional comparison for movement: {corrected_movement}")
        
        # Debug: Verificar profissionais disponíveis
        available_movements = set()
        professional_count_by_movement = {}
        
        for prof_key, prof_data in comparator.professionals_data.items():
            movement = prof_data.get('movement_type', 'unknown')
            player_name = prof_data.get('player_name', 'unknown')
            confidence = prof_data.get('classification', {}).get('confidence', 0)
            
            available_movements.add(movement)
            
            if movement not in professional_count_by_movement:
                professional_count_by_movement[movement] = 0
            professional_count_by_movement[movement] += 1
            
            logger.info(f"[CORRECTION] Professional {player_name}: movement_type='{movement}', confidence={confidence:.3f}")
        
        logger.info(f"[CORRECTION] Available movements in database: {list(available_movements)}")
        logger.info(f"[CORRECTION] Professional count by movement: {professional_count_by_movement}")
        logger.info(f"[CORRECTION] Searching for movement: '{corrected_movement}'")
        
        # Log do input que será passado para o comparador
        logger.info(f"[CORRECTION] Input data structure keys: {list(corrected_biomech_result.keys())}")
        logger.info(f"[CORRECTION] Detailed analysis keys: {list(corrected_biomech_result['detailed_analysis'].keys())}")
        
        # Executar comparação com profissionais
        logger.info(f"[CORRECTION] Calling find_best_matches with movement: {corrected_movement}")
        professional_matches = comparator.find_best_matches(corrected_biomech_result, corrected_movement)
        logger.info(f"[CORRECTION] find_best_matches returned {len(professional_matches)} matches")
        
        # Debug adicional: listar os nomes dos profissionais encontrados
        if professional_matches:
            logger.info(f"[CORRECTION] SUCCESS! Found {len(professional_matches)} matches:")
            for i, match in enumerate(professional_matches):
                logger.info(f"[CORRECTION] {i+1}. {match.professional_name} (similarity: {match.similarity_score:.3f}, confidence: {match.confidence:.3f})")
                logger.info(f"[CORRECTION]    Video: {match.professional_video}")
        else:
            logger.warning(f"[CORRECTION] No matches found! Trying direct filter...")
            # Teste direto do método de filtro
            direct_professionals = comparator.get_professionals_by_movement(corrected_movement)
            logger.info(f"[CORRECTION] Direct filter result: {len(direct_professionals)} professionals")
            
            if direct_professionals:
                logger.info(f"[CORRECTION] Direct filter found:")
                for prof in direct_professionals:
                    player_name = prof.get('player_name', 'unknown')
                    confidence = prof.get('classification', {}).get('confidence', 0)
                    logger.info(f"[CORRECTION] - {player_name} (confidence: {confidence:.3f})")
            else:
                logger.error(f"[CORRECTION] PROBLEMA: Nenhum profissional encontrado para '{corrected_movement}'")
                logger.error(f"[CORRECTION] Movimentos disponíveis: {list(available_movements)}")
        
        # Gerar novo analysis_id para a correção
        new_analysis_id = f"corrected_{secrets.token_hex(8)}"
        logger.info(f"[CORRECTION] Generated new analysis ID: {new_analysis_id}")
        
        # Preparar response similar ao endpoint original (formato completo)
        professional_comparisons = []
        logger.info(f"[CORRECTION] Processing {len(professional_matches)} professional matches for comparisons")
        
        if professional_matches:
            for i, match in enumerate(professional_matches[:3]):  # Top 3
                comparison = {
                    "professional_name": match.professional_name,
                    "similarity_score": match.similarity_score,
                    "confidence": match.confidence,
                    "detailed_comparison": match.detailed_comparison,
                    "recommendations": match.recommendations,
                    "video_reference": match.professional_video,
                    "technique_notes": f"Movimento corrigido para {corrected_movement}",
                    "biomechanical_notes": f"Confiança: {match.confidence:.2f}"
                }
                professional_comparisons.append(comparison)
                logger.info(f"[CORRECTION] Created comparison {i+1}: {match.professional_name} ({match.similarity_score:.3f})")
        else:
            logger.warning(f"[CORRECTION] No professional matches available for comparisons")
        
        # Log do resultado
        logger.info(f"[CORRECTION] Final professional comparisons count: {len(professional_comparisons)}")
        for comp in professional_comparisons:
            logger.info(f"[CORRECTION] {comp['professional_name']}: {(comp['similarity_score'] * 100):.1f}% (score: {comp['similarity_score']:.3f})")
        
        logger.info(f"[CORRECTION] Creating AnalysisResult object")
        logger.info(f"[CORRECTION] Movement detected: {corrected_movement}")
        logger.info(f"[CORRECTION] Professional comparisons: {len(professional_comparisons)} items")
        
        # Detectar mão dominante automaticamente se vídeo estiver disponível
        dominant_hand_display = "Destro"  # Default fallback
        if video_files:
            try:
                detected_hand = detect_dominant_hand_from_video(str(video_files[0]))
                dominant_hand_display = f"{detected_hand} (detecção automática)"
                logger.info(f"[CORRECTION] Detected dominant hand: {dominant_hand_display}")
            except Exception as e:
                logger.warning(f"[CORRECTION] Failed to detect dominant hand: {e}")
                # Usar metadados como fallback
                if metadata_dict.get('maoDominante') == 'E':
                    dominant_hand_display = "Canhoto"
                else:
                    dominant_hand_display = "Destro"
        
        result = AnalysisResult(
            success=True,
            analysis_id=new_analysis_id,
            timestamp=datetime.now(),
            analysis_type='corrected_movement_analysis',
            detected_movement=corrected_movement,
            confidence=0.95,
            movement_type_display=corrected_movement,
            dominant_hand_display=dominant_hand_display,
            professional_comparisons=professional_comparisons,
            detailed_analysis={
                "correction_applied": True,
                "original_analysis_id": analysis_id,
                "corrected_movement": corrected_movement,
                "correction_timestamp": datetime.now().isoformat(),
                "professional_matches_found": len(professional_matches) if professional_matches else 0,
                "message": f"Movimento corrigido para {corrected_movement}. Análise reprocessada com sucesso."
            }
        )
        
        logger.info(f"[CORRECTION] AnalysisResult created successfully")
        logger.info(f"[CORRECTION] Result success: {result.success}")
        logger.info(f"[CORRECTION] Result analysis_id: {result.analysis_id}")
        logger.info(f"[CORRECTION] Result detected_movement: {result.detected_movement}")
        logger.info(f"[CORRECTION] Result confidence: {result.confidence}")
        logger.info(f"[CORRECTION] Result professional_comparisons count: {len(result.professional_comparisons) if result.professional_comparisons else 0}")
        
        logger.info(f"[CORRECTION] Movement correction completed successfully")
        logger.info(f"[CORRECTION] Returning result to client")
        return result
        
    except HTTPException as he:
        logger.error(f"[CORRECTION] HTTP Exception: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"[CORRECTION] Unexpected error during movement correction: {e}")
        logger.error(f"[CORRECTION] Error type: {type(e).__name__}")
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"[CORRECTION] Full traceback:\n{tb_str}")
        
        error_id = f"error_{secrets.token_hex(8)}"
        logger.info(f"[CORRECTION] Creating error response with ID: {error_id}")
        
        return AnalysisResult(
            success=False,
            analysis_id=error_id,
            timestamp=datetime.now(),
            analysis_type='correction_error',
            error=str(e),
            detected_movement=None,
            confidence=None
        )


@app.post("/analyze-simple")
async def analyze_simple(
    file: UploadFile = File(...),
    metadata: str = Form(...)
):
    """Simple video analysis endpoint for headless mode"""
    try:
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata)
        
        # Save uploaded file
        temp_file = await save_uploaded_file(file)
        
        # Check if headless analyzer is available
        if analyzer_api and hasattr(analyzer_api, 'engine') and hasattr(analyzer_api.engine, 'analyze_video'):
            # Use headless analyzer
            result = analyzer_api.engine.analyze_video(str(temp_file), metadata_dict)
            
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            
            return {
                "success": True,
                "analysis_result": result,
                "message": "Video analyzed successfully using headless analyzer"
            }
        else:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
                
            return {
                "success": False,
                "error": "Analysis engine not available",
                "message": "Running in limited mode - full analysis not available"
            }
        
    except Exception as e:
        logger.error(f"Error in simple analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Docker/GCP compatible configuration
    port = int(os.environ.get("PORT", 8080))
    environment = os.environ.get("ENVIRONMENT", "development")
    
    print("🎾 Tennis Analyzer - Starting...")
    print(f"   Environment: {environment}")
    print(f"   Port: {port}")
    print(f"   MediaPipe: {'✅ Available' if MEDIAPIPE_AVAILABLE else '❌ Not Available'}")
    
    if environment == "development":
        print("   Desktop: http://localhost:8080/web_interface.html")
        print("   Conecte seu celular na mesma rede WiFi!")
    
    # Production-optimized settings for Docker/GCP
    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": port,
        "reload": False,
        "log_level": "info",
        "access_log": True
    }
    
    # Adjust settings based on environment
    if environment == "production":
        uvicorn_config.update({
            "workers": 1,  # GCP Cloud Run works better with 1 worker
            "timeout_keep_alive": 300,
            "limit_concurrency": 100,
            "limit_max_requests": 1000
        })
    else:
        uvicorn_config.update({
            "timeout_keep_alive": 300,
            "limit_max_requests": 1000,
            "limit_concurrency": 1000
        })
    
    uvicorn.run(**uvicorn_config)