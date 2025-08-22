"""
SOLUÇÃO DEFINITIVA: Sistema analisa conteúdo real do vídeo e compara com configuração do usuário
"""

import re
from datetime import datetime

def create_complete_content_validation_system():
    """Cria sistema completo de validação por conteúdo"""
    
    print("🔥 CRIANDO SISTEMA COMPLETO DE VALIDAÇÃO POR CONTEÚDO...")
    
    # 1. CRIAR ENDPOINT NA API
    api_endpoint = '''
# ADICIONAR NO ARQUIVO api/main.py

@app.post("/validate-content")
async def validate_video_content(
    video: UploadFile = File(...),
    user_config: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Valida conteúdo do vídeo vs configuração do usuário"""
    import tempfile
    import json
    from tennis_comparison_backend import TennisComparisonEngine
    
    try:
        # Parse configuração
        config = json.loads(user_config)
        
        # Salvar vídeo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        try:
            # Analisar conteúdo
            analyzer = TennisComparisonEngine()
            content_result = analyzer.analyze_movement_from_content(temp_path)
            
            detected = content_result.get('movement', 'unknown')
            confidence = content_result.get('confidence', 0.0)
            
            # Movimento esperado da configuração
            lado = 'F' if config.get('ladoRaquete') == 'Forehand' else 'B'
            tipo = 'D' if config.get('tipoMovimento', '').startswith('Drive') else 'P'
            expected = f"{lado}{tipo}"
            
            # Validação
            is_compatible = detected == expected and confidence > 0.6
            
            return {
                "success": True,
                "detected_movement": detected,
                "expected_movement": expected,
                "confidence": confidence,
                "is_compatible": is_compatible,
                "message": f"Detectado: {detected} | Esperado: {expected} | Confiança: {confidence:.1%}"
            }
            
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return {"success": False, "error": str(e)}
'''
    
    # 2. BACKUP DA INTERFACE
    backup_filename = f"web_interface_final_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup criado: {backup_filename}")
    
    # 3. JAVASCRIPT COMPLETO COM VALIDAÇÃO REAL
    complete_js = '''
// VALIDAÇÃO COMPLETA POR CONTEÚDO
async function validateFileAndConfiguration() {
    const fileInput = document.getElementById('user-video');
    const fileName = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    
    if (!fileName) {
        hideMessage();
        return { isValid: false, canLoadProfessionals: false };
    }
    
    console.log(`🔍 VALIDANDO CONTEÚDO: ${fileName}`);
    showMessage('🔍 Analisando conteúdo do vídeo com IA...', 'info');
    
    try {
        // Preparar dados
        const userConfig = getUserConfiguration();
        const formData = new FormData();
        formData.append('video', fileInput.files[0]);
        formData.append('user_config', JSON.stringify(userConfig));
        
        // Chamar API de validação
        const response = await fetch(`${API_BASE}/validate-content`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        console.log('📊 Resultado validação:', result);
        
        if (!result.success) {
            showMessage(`❌ Erro na análise: ${result.error}`, 'error');
            return { isValid: false, canLoadProfessionals: false };
        }
        
        const { detected_movement, expected_movement, confidence, is_compatible } = result;
        
        if (is_compatible) {
            showMessage(`✅ VALIDAÇÃO APROVADA
🎯 Movimento detectado: ${detected_movement}
⚙️ Configuração: ${expected_movement}
📊 Confiança: ${(confidence * 100).toFixed(1)}%
🎾 Compatibilidade confirmada!`, 'success');
            
            return { isValid: true, canLoadProfessionals: true };
            
        } else {
            const reason = confidence < 0.6 ? 
                `Baixa confiança (${(confidence * 100).toFixed(1)}%)` :
                `Movimentos diferentes (${detected_movement} ≠ ${expected_movement})`;
                
            showMessage(`❌ VALIDAÇÃO REJEITADA
🎯 Detectado: ${detected_movement}
⚙️ Configurado: ${expected_movement}  
📊 Confiança: ${(confidence * 100).toFixed(1)}%
❌ Motivo: ${reason}

🔧 CORRIJA A CONFIGURAÇÃO para corresponder ao movimento real do vídeo.`, 'error');
            
            return { isValid: false, canLoadProfessionals: false };
        }
        
    } catch (error) {
        console.error('❌ Erro validação:', error);
        showMessage(`❌ Erro na validação de conteúdo: ${error.message}`, 'error');
        return { isValid: false, canLoadProfessionals: false };
    }
}

function getUserConfiguration() {
    return {
        maoDominante: document.getElementById('mao-dominante').value,
        ladoRaquete: document.getElementById('lado-raquete').value,
        ladoCamera: document.getElementById('lado-camera').value,
        tipoMovimento: document.getElementById('tipo-movimento').value
    };
}

function showMessage(message, type) {
    let messageDiv = document.getElementById('validation-message');
    if (!messageDiv) {
        messageDiv = document.createElement('div');
        messageDiv.id = 'validation-message';
        messageDiv.style.cssText = `
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-line;
            font-weight: 500;
        `;
        
        const fileInput = document.getElementById('user-video');
        fileInput.parentNode.insertBefore(messageDiv, fileInput.nextSibling);
    }
    
    if (type === 'error') {
        messageDiv.style.cssText += 'background: #ffebee; color: #c62828; border: 2px solid #e57373;';
    } else if (type === 'success') {
        messageDiv.style.cssText += 'background: #e8f5e8; color: #2e7d32; border: 2px solid #81c784;';
    } else if (type === 'info') {
        messageDiv.style.cssText += 'background: #e3f2fd; color: #1565c0; border: 2px solid #64b5f6;';
    }
    
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';
}

function hideMessage() {
    const messageDiv = document.getElementById('validation-message');
    if (messageDiv) messageDiv.style.display = 'none';
}

async function validateAndLoad() {
    const validation = await validateFileAndConfiguration();
    
    const loadButton = document.getElementById('load-professionals-btn');
    const analyzeButton = document.getElementById('analyze-btn');
    
    if (validation.canLoadProfessionals) {
        loadButton.disabled = false;
        loadButton.style.background = '#4a7c59';
        loadButton.style.cursor = 'pointer';
        setTimeout(loadProfessionals, 1000);
    } else {
        loadButton.disabled = true;
        loadButton.style.background = '#ccc';
        loadButton.style.cursor = 'not-allowed';
        analyzeButton.disabled = true;
        analyzeButton.style.background = '#ccc';
        analyzeButton.style.cursor = 'not-allowed';
        
        const profList = document.getElementById('professionals-list');
        if (profList) {
            profList.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">Corrija a validação para carregar profissionais</div>';
        }
    }
}'''

    # Substituir JavaScript
    patterns = [
        r'async function validateFileAndConfiguration\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function getUserConfiguration\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function showMessage\([^)]*\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function hideMessage\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'async function validateAndLoad\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}'
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Adicionar novo JavaScript
    script_end = content.rfind('</script>')
    if script_end != -1:
        content = content[:script_end] + "\n" + complete_js + "\n" + content[script_end:]
    
    # Salvar interface atualizada
    with open('web_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Interface atualizada!")
    
    # 4. ARQUIVO DE INSTRUÇÕES
    with open('INSTRUCOES_FINAIS.txt', 'w', encoding='utf-8') as f:
        f.write(f"""
🔥 INSTRUÇÕES PARA SISTEMA COMPLETO DE VALIDAÇÃO POR CONTEÚDO

1. ADICIONAR ENDPOINT NA API:
   - Abra: api/main.py
   - Adicione no final (antes da última linha):
   
{api_endpoint}

   - Adicione imports no topo:
     import tempfile
     import json
     from tennis_comparison_backend import TennisComparisonEngine

2. REINICIAR API:
   python start_api.py

3. TESTAR SISTEMA:
   - Recarregue interface (Ctrl+F5)
   - Login: demo/demo123
   - Upload: teste_neutro.mp4
   - Configure: Backhand + Drive (ERRADO)
   - Deve REJEITAR com análise de IA!
   
   - Configure: Forehand + Push (CORRETO)  
   - Deve APROVAR e carregar profissionais!

4. BACKUP CRIADO: {backup_filename}

🎯 AGORA O SISTEMA VAI:
✅ Analisar CONTEÚDO real do vídeo com MediaPipe + TensorFlow
✅ Comparar com configuração do usuário
✅ REJEITAR se incompatível
✅ APROVAR se compatível
✅ Funcionar sem depender de metadados no filename!
""")
    
    print(f"""
🎯 SISTEMA COMPLETO CRIADO!

📋 PRÓXIMOS PASSOS:
1. Adicione o endpoint /validate-content na API (veja INSTRUCOES_FINAIS.txt)
2. Reinicie API: python start_api.py  
3. Teste com configuração ERRADA: deve REJEITAR
4. Teste com configuração CORRETA: deve APROVAR

✅ Interface atualizada: web_interface.html
📄 Instruções completas: INSTRUCOES_FINAIS.txt
💾 Backup: {backup_filename}

🔥 AGORA VAI FUNCIONAR EXATAMENTE COMO VOCÊ QUER!
""")

if __name__ == "__main__":
    create_complete_content_validation_system()
