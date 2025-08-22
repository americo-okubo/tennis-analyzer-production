"""
Enhanced Production Deployment System - Tennis Analyzer
Sistema de produção aprimorado com correção do erro 422
"""

import os
import json
import uvicorn
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

# Importar o backend de integração que criamos
try:
    from tennis_comparison_backend import TennisAnalyzerAPI, TennisComparisonEngine
    BACKEND_AVAILABLE = True
except ImportError:
    print("⚠️ Backend de comparação não encontrado. Usando modo simulação.")
    BACKEND_AVAILABLE = False


class UserMetadata(BaseModel):
    """Modelo para metadata do usuário"""
    maoDominante: str  # 'D' ou 'E'
    ladoCamera: str    # 'D' ou 'E' 
    ladoRaquete: str   # 'F' ou 'B'
    tipoMovimento: str # 'D' ou 'P'


class ComparisonRequest(BaseModel):
    """Modelo para requisição de comparação"""
    userFilePath: str
    professionalName: str
    userMetadata: UserMetadata


class MockTennisAPI:
    """API simulada para quando o backend real não está disponível"""
    
    def process_upload(self, file_data, filename, metadata):
        """Simula processamento de upload com cenários realistas"""
        import random
        
        # Simular detecção de movimento
        movement_map = {
            ('F', 'D'): 'forehand_drive',
            ('F', 'P'): 'forehand_push', 
            ('B', 'D'): 'backhand_drive',
            ('B', 'P'): 'backhand_push'
        }
        
        expected_movement = movement_map.get(
            (metadata['ladoRaquete'], metadata['tipoMovimento']), 
            'unknown_movement'
        )
        
        # Simular diferentes cenários de confiança
        confidence_scenarios = [
            (85, True),   # Alta confiança - movimento correto
            (75, True),   # Confiança moderada - movimento correto 
            (65, False),  # Baixa confiança - possível divergência
            (90, True),   # Muito alta confiança
        ]
        
        # Escolher cenário baseado em probabilidade
        rand = random.random()
        if rand < 0.6:  # 60% - alta confiança
            confidence, validation_passed = confidence_scenarios[0]
            detected_movement = expected_movement
        elif rand < 0.8:  # 20% - confiança moderada
            confidence, validation_passed = confidence_scenarios[1] 
            detected_movement = expected_movement
        elif rand < 0.95:  # 15% - baixa confiança
            confidence, validation_passed = confidence_scenarios[2]
            # Simular detecção incorreta ocasional
            if random.random() < 0.3:
                alternative_movements = [k for k in movement_map.values() if k != expected_movement]
                detected_movement = random.choice(alternative_movements) if alternative_movements else expected_movement
            else:
                detected_movement = expected_movement
        else:  # 5% - muito alta confiança
            confidence, validation_passed = confidence_scenarios[3]
            detected_movement = expected_movement
        
        # Adicionar variação realista
        confidence += random.uniform(-5, 5)
        confidence = max(60, min(98, confidence))  # Limitar entre 60-98%
        
        # Determinar se passou na validação
        validation_passed = confidence >= 70 and detected_movement == expected_movement
        
        return {
            'success': True,
            'validation_passed': validation_passed,
            'detected_movement': detected_movement.replace('_', ' ').title(),
            'expected_movement': expected_movement.replace('_', ' ').title(),
            'confidence': confidence,
            'temp_file_path': f'/temp/{filename}',
            'movement_key': detected_movement,
            'professionals': self.get_mock_professionals(detected_movement),
            'message': f'Movimento detectado com {confidence:.1f}% de confiança',
            'details': self._get_detection_details(detected_movement, expected_movement, confidence)
        }
    
    def _get_detection_details(self, detected, expected, confidence):
        """Gera detalhes da detecção para ajudar o usuário"""
        if detected == expected:
            if confidence >= 85:
                return "Movimento detectado com alta precisão. Análise biomecânica será muito confiável."
            elif confidence >= 75:
                return "Movimento detectado corretamente. Boa qualidade para análise comparativa."
            else:
                return "Movimento detectado, mas com algumas incertezas. Resultados podem variar."
        else:
            return f"Detectado '{detected.replace('_', ' ').title()}' mas esperado '{expected.replace('_', ' ').title()}'. Verifique configurações e qualidade do vídeo."
    
    def get_mock_professionals(self, movement_key):
        """Retorna profissionais simulados"""
        professionals_db = {
            'forehand_drive': [
                {'name': 'Ma Long', 'filename': 'ma_long_FD_D_E.mp4', 'rating': 98},
                {'name': 'Fan Zhendong', 'filename': 'fan_zhendong_FD_E_D.mp4', 'rating': 97},
                {'name': 'Ovtcharov', 'filename': 'ovtcharov_FD_D_E.mp4', 'rating': 95}
            ],
            'forehand_push': [
                {'name': 'Xu Xin', 'filename': 'xu_xin_FP_D_E.mp4', 'rating': 98},
                {'name': 'Liu Shiwen', 'filename': 'liu_shiwen_FP_E_D.mp4', 'rating': 97},
                {'name': 'Ito Mima', 'filename': 'ito_mima_FP_D_E.mp4', 'rating': 96}
            ],
            'backhand_drive': [
                {'name': 'Harimoto', 'filename': 'harimoto_BD_E_D.mp4', 'rating': 97},
                {'name': 'Zhang Jike', 'filename': 'zhang_jike_BD_D_E.mp4', 'rating': 95},
                {'name': 'Lin Gaoyuan', 'filename': 'lin_gaoyuan_BD_E_D.mp4', 'rating': 96}
            ],
            'backhand_push': [
                {'name': 'Chen Meng', 'filename': 'chen_meng_BP_D_E.mp4', 'rating': 99},
                {'name': 'Wang Chuqin', 'filename': 'wang_chuqin_BP_E_D.mp4', 'rating': 96},
                {'name': 'Sun Yingsha', 'filename': 'sun_yingsha_BP_D_E.mp4', 'rating': 98}
            ]
        }
        
        return professionals_db.get(movement_key, [])
    
    def start_comparison(self, user_file_path, professional_name, metadata):
        """Simula comparação"""
        import random
        
        base_score = 70 + random.random() * 25  # 70-95
        
        return {
            'success': True,
            'final_score': base_score,
            'professional_name': professional_name,
            'phase_scores': {
                'preparation': base_score + (random.random() - 0.5) * 10,
                'contact': base_score + (random.random() - 0.5) * 15, 
                'follow_through': base_score + (random.random() - 0.5) * 8
            },
            'detailed_analysis': {
                'similarity_components': {
                    'amplitude_similarity': 0.75 + random.random() * 0.2,
                    'velocity_similarity': 0.7 + random.random() * 0.25,
                    'angle_similarity': 0.8 + random.random() * 0.15,
                    'temporal_matching': 0.85 + random.random() * 0.12
                },
                'recommendations': [
                    'Melhore a consistência do movimento de preparação',
                    'Trabalhe na velocidade da raquete durante o contato',
                    'Ajuste o timing da finalização do movimento',
                    'Pratique exercícios de coordenação para maior fluidez'
                ]
            }
        }


class EnhancedTennisAnalyzerApp:
    """Aplicação FastAPI aprimorada do Tennis Analyzer"""
    
    def __init__(self, base_path: str = "."):
        self.app = FastAPI(
            title="Tennis Analyzer - Comparação Técnica",
            description="Sistema avançado de análise biomecânica e comparação técnica",
            version="2.0.1"
        )
        
        self.base_path = Path(base_path)
        self.static_path = self.base_path / "static"
        self.upload_path = self.base_path / "uploads"
        
        # Criar diretórios necessários
        self.static_path.mkdir(exist_ok=True)
        self.upload_path.mkdir(exist_ok=True)
        
        # Inicializar API backend (real ou simulada)
        if BACKEND_AVAILABLE:
            self.tennis_api = TennisAnalyzerAPI()
            print("✅ Backend real carregado")
        else:
            self.tennis_api = MockTennisAPI()
            print("⚠️ Usando backend simulado")
        
        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Configurar rotas
        self._setup_routes()
        
        # Servir arquivos estáticos
        if self.static_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.static_path)), name="static")
    
    def _setup_routes(self):
        """Configura todas as rotas da aplicação"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_main_interface():
            """Retorna a interface principal de comparação"""
            try:
                # Tentar ler arquivo da interface
                interface_file = Path("tennis_comparison_interface.html")
                if interface_file.exists():
                    return interface_file.read_text(encoding='utf-8')
                else:
                    return self._get_embedded_interface_html()
            except Exception as e:
                print(f"Erro ao carregar interface: {e}")
                return self._get_embedded_interface_html()
        
        @self.app.post("/api/upload-and-validate")
        async def upload_and_validate(
            file: UploadFile = File(...),
            maoDominante: str = Form(...),
            ladoCamera: str = Form(...),
            ladoRaquete: str = Form(...),
            tipoMovimento: str = Form(...)
        ):
            """Endpoint corrigido para upload e validação do vídeo do usuário"""
            try:
                print(f"📁 Upload recebido: {file.filename}")
                print(f"📊 Metadata: mão={maoDominante}, camera={ladoCamera}, raquete={ladoRaquete}, movimento={tipoMovimento}")
                
                # Validar arquivo
                if not file.filename:
                    raise HTTPException(status_code=400, detail="Nome do arquivo não fornecido")
                    
                if not file.filename.lower().endswith('.mp4'):
                    raise HTTPException(status_code=400, detail="Apenas arquivos MP4 são aceitos")
                
                # Validar metadata
                valid_values = {
                    'maoDominante': ['D', 'E'],
                    'ladoCamera': ['D', 'E'],
                    'ladoRaquete': ['F', 'B'],
                    'tipoMovimento': ['D', 'P']
                }
                
                for field, value in [
                    ('maoDominante', maoDominante),
                    ('ladoCamera', ladoCamera), 
                    ('ladoRaquete', ladoRaquete),
                    ('tipoMovimento', tipoMovimento)
                ]:
                    if value not in valid_values[field]:
                        raise HTTPException(
                            status_code=422, 
                            detail=f"Valor inválido para {field}: {value}. Valores aceitos: {valid_values[field]}"
                        )
                
                # Ler dados do arquivo
                file_data = await file.read()
                print(f"📏 Arquivo lido: {len(file_data)} bytes")
                
                # Preparar metadata
                metadata = {
                    'maoDominante': maoDominante,
                    'ladoCamera': ladoCamera,
                    'ladoRaquete': ladoRaquete,
                    'tipoMovimento': tipoMovimento
                }
                
                # Processar upload via API backend
                result = self.tennis_api.process_upload(file_data, file.filename, metadata)
                print(f"✅ Processamento concluído: {result.get('success', False)}")
                
                return JSONResponse(content=result)
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"❌ Erro no processamento: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
        
        @self.app.post("/api/start-comparison")
        async def start_comparison(request: ComparisonRequest):
            """Endpoint para iniciar comparação com profissional"""
            try:
                print(f"🔄 Iniciando comparação com {request.professionalName}")
                
                # Converter metadata Pydantic para dict
                metadata_dict = {
                    'maoDominante': request.userMetadata.maoDominante,
                    'ladoCamera': request.userMetadata.ladoCamera,
                    'ladoRaquete': request.userMetadata.ladoRaquete,
                    'tipoMovimento': request.userMetadata.tipoMovimento
                }
                
                # Executar comparação via API backend
                result = self.tennis_api.start_comparison(
                    request.userFilePath,
                    request.professionalName,
                    metadata_dict
                )
                
                print(f"✅ Comparação concluída: score={result.get('final_score', 'N/A')}")
                return JSONResponse(content=result)
                
            except Exception as e:
                print(f"❌ Erro na comparação: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Erro na comparação: {str(e)}")
        
        @self.app.get("/api/professionals/{movement_key}")
        async def get_professionals(movement_key: str):
            """Retorna lista de profissionais para um tipo de movimento"""
            try:
                if BACKEND_AVAILABLE:
                    professionals = self.tennis_api.engine.get_available_professionals(movement_key)
                else:
                    professionals = self.tennis_api.get_mock_professionals(movement_key)
                    
                return JSONResponse(content={"professionals": professionals})
            except Exception as e:
                print(f"❌ Erro ao buscar profissionais: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Erro ao buscar profissionais: {str(e)}")
        
        @self.app.get("/api/health")
        async def health_check():
            """Endpoint de health check"""
            return {
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "backend_mode": "real" if BACKEND_AVAILABLE else "simulation"
            }
        
        @self.app.get("/api/debug")
        async def debug_info():
            """Endpoint de debug para desenvolvimento"""
            return {
                "backend_available": BACKEND_AVAILABLE,
                "upload_path": str(self.upload_path),
                "static_path": str(self.static_path),
                "base_path": str(self.base_path)
            }
    
    def _get_embedded_interface_html(self) -> str:
        """Interface HTML básica incorporada"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tennis Analyzer</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { background: white; padding: 30px; border-radius: 10px; max-width: 800px; margin: 0 auto; }
                .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
                .form-group { margin: 15px 0; }
                .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; }
                .btn:disabled { opacity: 0.5; }
                .error { color: red; margin-top: 10px; }
                .success { color: green; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎾 Tennis Analyzer</h1>
                <p>Interface básica para teste</p>
                
                <div class="upload-area">
                    <input type="file" id="videoFile" accept=".mp4">
                    <p>Selecione um arquivo MP4</p>
                </div>
                
                <div class="form-group">
                    <label>Mão Dominante:</label>
                    <select id="maoDominante">
                        <option value="D">Direita</option>
                        <option value="E">Esquerda</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Lado da Câmera:</label>
                    <select id="ladoCamera">
                        <option value="D">Direita</option>
                        <option value="E">Esquerda</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Lado da Raquete:</label>
                    <select id="ladoRaquete">
                        <option value="F">Forehand</option>
                        <option value="B">Backhand</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Tipo de Movimento:</label>
                    <select id="tipoMovimento">
                        <option value="D">Drive</option>
                        <option value="P">Push</option>
                    </select>
                </div>
                
                <button class="btn" onclick="testUpload()" id="uploadBtn">Testar Upload</button>
                <div id="result"></div>
                
                <script>
                    async function testUpload() {
                        const fileInput = document.getElementById('videoFile');
                        const resultDiv = document.getElementById('result');
                        
                        if (!fileInput.files[0]) {
                            resultDiv.innerHTML = '<div class="error">Selecione um arquivo primeiro</div>';
                            return;
                        }
                        
                        const formData = new FormData();
                        formData.append('file', fileInput.files[0]);
                        formData.append('maoDominante', document.getElementById('maoDominante').value);
                        formData.append('ladoCamera', document.getElementById('ladoCamera').value);
                        formData.append('ladoRaquete', document.getElementById('ladoRaquete').value);
                        formData.append('tipoMovimento', document.getElementById('tipoMovimento').value);
                        
                        try {
                            resultDiv.innerHTML = 'Enviando...';
                            
                            const response = await fetch('/api/upload-and-validate', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const result = await response.json();
                            
                            if (response.ok) {
                                resultDiv.innerHTML = `<div class="success">✅ Sucesso: ${JSON.stringify(result, null, 2)}</div>`;
                            } else {
                                resultDiv.innerHTML = `<div class="error">❌ Erro ${response.status}: ${result.detail}</div>`;
                            }
                        } catch (error) {
                            resultDiv.innerHTML = `<div class="error">❌ Erro de conexão: ${error.message}</div>`;
                        }
                    }
                </script>
            </div>
        </body>
        </html>
        """
    
    def run(self, host: str = "localhost", port: int = 8000, reload: bool = False):
        """Executa o servidor"""
        print(f"🎾 Tennis Analyzer - Servidor Enhanced (v2.0.1)")
        print(f"🌐 Interface Principal: http://{host}:{port}/")
        print(f"📖 Documentação API: http://{host}:{port}/docs")
        print(f"🔧 Health Check: http://{host}:{port}/api/health")
        print(f"🐛 Debug Info: http://{host}:{port}/api/debug")
        print(f"🤖 Backend Mode: {'Real' if BACKEND_AVAILABLE else 'Simulation'}")
        
        uvicorn.run(self.app, host=host, port=port, reload=reload)


def main():
    """Função principal para executar o sistema"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tennis Analyzer Enhanced Production System")
    parser.add_argument("--host", default="localhost", help="Host para o servidor")
    parser.add_argument("--port", type=int, default=8000, help="Porta para o servidor")
    parser.add_argument("--reload", action="store_true", help="Ativar auto-reload")
    parser.add_argument("--base-path", default=".", help="Caminho base do projeto")
    
    args = parser.parse_args()
    
    # Inicializar aplicação
    app = EnhancedTennisAnalyzerApp(base_path=args.base_path)
    
    # Executar servidor
    app.run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
