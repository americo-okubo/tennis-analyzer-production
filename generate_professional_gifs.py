#!/usr/bin/env python3
"""
Script para gerar GIFs dos vídeos profissionais
Cria uma biblioteca de GIFs pré-processados para comparação visual
"""

import os
import cv2
import sys
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_gif_from_video(video_path, output_path, max_width=320, target_fps=12, max_duration=3.0):
    """
    Cria um GIF a partir de um vídeo profissional
    
    Args:
        video_path: Caminho para o vídeo original
        output_path: Caminho para salvar o GIF
        max_width: Largura máxima do GIF (default: 320px para economizar espaço)
        target_fps: FPS alvo para o GIF (default: 15 FPS)
        max_duration: Duração máxima em segundos (default: 3s)
    """
    try:
        logger.info(f"🎬 Processando: {video_path.name}")
        
        # Abrir vídeo
        cap = cv2.VideoCapture(str(video_path))
        
        # Propriedades do vídeo
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps
        
        logger.info(f"   📊 Original: {width}x{height}, {original_fps:.1f} FPS, {duration:.1f}s")
        
        # Calcular dimensões do GIF
        if width > max_width:
            new_width = max_width
            new_height = int(height * (max_width / width))
        else:
            new_width = width
            new_height = height
        
        # Determinar frames a extrair
        max_frames = int(target_fps * max_duration)
        frame_step = max(1, total_frames // max_frames)
        
        # Coletar frames
        frames = []
        frame_count = 0
        
        while len(frames) < max_frames and frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Converter BGR para RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar se necessário
            if new_width != width or new_height != height:
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            frames.append(frame_rgb)
            frame_count += frame_step
        
        cap.release()
        
        if len(frames) == 0:
            logger.error(f"   ❌ Nenhum frame extraído de {video_path.name}")
            return False
        
        logger.info(f"   📷 Coletados: {len(frames)} frames para GIF {new_width}x{new_height}")
        
        # Criar GIF
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # COMPLETE LOG2.TXT APPROACH: Natural movement preservation
        # Key insight: "Para movimentos naturais, tente manter o mesmo FPS do vídeo original"
        # Insight 2: "Certifique-se de que o tempo total do GIF seja igual ao do vídeo"
        
        # Use original FPS (typically 24-30) but cap at reasonable limit
        if original_fps >= 24:
            gif_fps = min(original_fps, 30)  # Keep original up to 30 FPS
        else:
            gif_fps = max(original_fps, 15)  # Ensure minimum 15 FPS for smooth playback
            
        # Calculate frame duration based on EXACT video timing
        actual_video_duration = len(frames) / original_fps  # Real duration of collected frames
        frame_duration = int((actual_video_duration * 1000) / len(frames))  # Exact timing per frame
        
        # Ensure minimum frame duration for smooth playback
        frame_duration = max(frame_duration, 33)  # No faster than 30 FPS (33ms per frame)
        
        # Debug logging for professional GIF creation
        logger.info(f"[PROFESSIONAL_GIF] DEBUG for {video_path.name}:")
        logger.info(f"  - Original video FPS: {original_fps:.1f}")
        logger.info(f"  - Original full duration: {duration:.2f}s")
        logger.info(f"  - Total original frames: {total_frames}")
        logger.info(f"  - Frame step: {frame_step}")
        logger.info(f"  - Frames collected: {len(frames)}")
        logger.info(f"  - GIF FPS (natural): {gif_fps:.1f}")
        logger.info(f"  - Actual video duration: {actual_video_duration:.3f}s")
        logger.info(f"  - Frame duration (exact timing): {frame_duration}ms")
        logger.info(f"  - Expected GIF duration: {(len(frames) * frame_duration / 1000):.3f}s")
        logger.info(f"  - Speed preservation: NATURAL (matches original timing)")
        
        # Salvar GIF
        pil_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration,
            loop=0,  # Loop infinito
            optimize=True,
            quality=85
        )
        
        # Verificar arquivo criado
        if output_path.exists():
            file_size = output_path.stat().st_size
            logger.info(f"   ✅ GIF criado: {file_size / 1024:.1f} KB")
            return True
        else:
            logger.error(f"   ❌ Falha ao criar GIF: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Erro ao processar {video_path.name}: {e}")
        return False

def main():
    """Gera GIFs para todos os vídeos profissionais"""
    
    project_root = Path(__file__).parent
    professionals_dir = project_root / "profissionais"
    
    if not professionals_dir.exists():
        logger.error(f"❌ Diretório de profissionais não encontrado: {professionals_dir}")
        return
    
    # Estatísticas
    total_videos = 0
    successful_gifs = 0
    failed_gifs = 0
    
    logger.info("🚀 Iniciando geração de GIFs dos profissionais...")
    
    # Processar cada categoria de movimento
    for movement_dir in professionals_dir.iterdir():
        if not movement_dir.is_dir():
            continue
            
        logger.info(f"\n📂 Processando categoria: {movement_dir.name}")
        
        # Criar diretório de GIFs se não existir
        gifs_dir = movement_dir / "gifs"
        gifs_dir.mkdir(exist_ok=True)
        
        # Processar cada vídeo na categoria
        for video_file in movement_dir.glob("*.mp4"):
            total_videos += 1
            
            # Nome do GIF (mesmo nome base do vídeo)
            gif_name = video_file.stem + ".gif"
            gif_path = gifs_dir / gif_name
            
            # Verificar se GIF já existe
            if gif_path.exists():
                logger.info(f"   ⏭️  GIF já existe: {gif_name}")
                successful_gifs += 1
                continue
            
            # Gerar GIF
            if create_gif_from_video(video_file, gif_path):
                successful_gifs += 1
            else:
                failed_gifs += 1
    
    # Relatório final
    logger.info(f"\n📊 RELATÓRIO FINAL:")
    logger.info(f"   📹 Vídeos processados: {total_videos}")
    logger.info(f"   ✅ GIFs criados com sucesso: {successful_gifs}")
    logger.info(f"   ❌ Falhas: {failed_gifs}")
    logger.info(f"   📈 Taxa de sucesso: {(successful_gifs / total_videos * 100):.1f}%")
    
    if failed_gifs == 0:
        logger.info("🎉 Todos os GIFs foram gerados com sucesso!")
    else:
        logger.warning(f"⚠️  {failed_gifs} GIFs falharam na geração")

if __name__ == "__main__":
    main()