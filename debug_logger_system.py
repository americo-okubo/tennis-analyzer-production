"""
Sistema de Debug Visual para Interface Web
Adiciona janela de logs em tempo real na tela para acompanhar validação
"""

import re
from datetime import datetime

def create_debug_logger_system():
    """Cria sistema completo de debug visual na interface web"""
    
    print("🔍 CRIANDO SISTEMA DE DEBUG VISUAL...")
    
    # Backup
    backup_filename = f"web_interface_debug_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup criado: {backup_filename}")
    
    # CSS para janela de debug
    debug_css = '''
<style>
#debug-panel {
    position: fixed;
    top: 10px;
    right: 10px;
    width: 400px;
    max-height: 500px;
    background: #1e1e1e;
    color: #00ff00;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    border: 2px solid #333;
    border-radius: 8px;
    z-index: 9999;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.8);
}

#debug-header {
    background: #333;
    color: #fff;
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: move;
}

#debug-content {
    height: 400px;
    overflow-y: auto;
    padding: 10px;
    background: #1e1e1e;
}

.debug-log {
    margin: 2px 0;
    padding: 2px 0;
    border-left: 3px solid transparent;
}

.debug-info { border-left-color: #00bfff; color: #00bfff; }
.debug-success { border-left-color: #00ff00; color: #00ff00; }
.debug-warning { border-left-color: #ffaa00; color: #ffaa00; }
.debug-error { border-left-color: #ff4444; color: #ff4444; }
.debug-step { border-left-color: #ff00ff; color: #ff00ff; font-weight: bold; }

#debug-toggle {
    background: #555;
    border: none;
    color: white;
    padding: 4px 8px;
    border-radius: 3px;
    cursor: pointer;
}

#debug-clear {
    background: #ff4444;
    border: none;
    color: white;
    padding: 4px 8px;
    border-radius: 3px;
    cursor: pointer;
    margin-left: 5px;
}
</style>'''
    
    # HTML da janela de debug
    debug_html = '''
<div id="debug-panel">
    <div id="debug-header">
        <span>🔍 DEBUG CONSOLE</span>
        <div>
            <button id="debug-toggle" onclick="toggleDebugContent()">Minimizar</button>
            <button id="debug-clear" onclick="clearDebugLog()">Limpar</button>
        </div>
    </div>
    <div id="debug-content"></div>
</div>'''
    
    # JavaScript com sistema de debug completo
    debug_js = '''
// SISTEMA DE DEBUG VISUAL
let debugVisible = true;

function debugLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const content = document.getElementById('debug-content');
    if (!content) return;
    
    const logEntry = document.createElement('div');
    logEntry.className = `debug-log debug-${type}`;
    logEntry.innerHTML = `[${timestamp}] ${message}`;
    
    content.appendChild(logEntry);
    content.scrollTop = content.scrollHeight;
    
    // Console também
    console.log(`[DEBUG-${type.toUpperCase()}] ${message}`);
}

function toggleDebugContent() {
    const content = document.getElementById('debug-content');
    const button = document.getElementById('debug-toggle');
    
    if (debugVisible) {
        content.style.display = 'none';
        button.textContent = 'Mostrar';
        debugVisible = false;
    } else {
        content.style.display = 'block';
        button.textContent = 'Minimizar';
        debugVisible = true;
    }
}

function clearDebugLog() {
    const content = document.getElementById('debug-content');
    if (content) {
        content.innerHTML = '';
    }
}

// VALIDAÇÃO COM DEBUG COMPLETO
async function validateFileAndConfiguration() {
    debugLog('🔍 INICIANDO VALIDAÇÃO DE ARQUIVO', 'step');
    
    const fileInput = document.getElementById('user-video');
    const fileName = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    
    if (!fileName) {
        debugLog('❌ Nenhum arquivo selecionado', 'error');
        hideMessage();
        return { isValid: false, canLoadProfessionals: false };
    }
    
    debugLog(`📁 Arquivo selecionado: ${fileName}`, 'info');
    
    // Extrair movimento do filename
    debugLog('🔍 Extraindo movimento do filename...', 'step');
    const fileMovement = extractMovementFromFilename(fileName);
    debugLog(`📊 Movimento do filename: ${fileMovement}`, 'info');
    
    // Obter configuração do usuário
    debugLog('⚙️ Obtendo configuração do usuário...', 'step');
    const userConfig = getUserConfiguration();
    const userMovement = `${userConfig.ladoRaquete}${userConfig.tipoMovimento}`;
    
    debugLog(`👤 Configuração usuário: ${JSON.stringify(userConfig)}`, 'info');
    debugLog(`🎯 Movimento configurado: ${userMovement}`, 'info');
    
    // Se arquivo tem metadados, validar diretamente
    if (fileMovement !== 'unknown') {
        debugLog('📋 Arquivo COM metadados - validação direta', 'step');
        const isCompatible = fileMovement === userMovement;
        
        debugLog(`🔍 Comparação: ${fileMovement} === ${userMovement} = ${isCompatible}`, 'info');
        
        if (!isCompatible) {
            debugLog(`❌ INCOMPATIBILIDADE: ${fileMovement} ≠ ${userMovement}`, 'error');
            showMessage(`❌ INCOMPATIBILIDADE: arquivo "${fileName}" contém "${fileMovement}" mas configuração é "${userMovement}"`, 'error');
            return { isValid: false, canLoadProfessionals: false };
        }
        
        debugLog('✅ Validação direta APROVADA', 'success');
        showMessage('✅ Arquivo e configuração compatíveis', 'success');
        return { isValid: true, canLoadProfessionals: true };
    }
    
    // Para arquivos sem metadados, fazer análise de conteúdo
    debugLog('🎥 Arquivo SEM metadados - iniciando análise de conteúdo', 'step');
    
    showMessage('🔍 Analisando conteúdo do vídeo com IA...', 'info');
    
    try {
        debugLog('📤 Preparando requisição para API...', 'info');
        
        // Verificar se token existe
        if (!authToken) {
            debugLog('❌ Token de autenticação não encontrado', 'error');
            showMessage('❌ Erro: faça login novamente', 'error');
            return { isValid: false, canLoadProfessionals: false };
        }
        
        debugLog('🔑 Token de autenticação OK', 'info');
        
        // Preparar dados
        const formData = new FormData();
        formData.append('video', fileInput.files[0]);
        formData.append('user_config', JSON.stringify(userConfig));
        
        debugLog(`📦 FormData preparado - arquivo: ${fileInput.files[0].name} (${fileInput.files[0].size} bytes)`, 'info');
        
        // Chamar API de validação
        debugLog('🌐 Chamando API /validate-content...', 'step');
        
        const response = await fetch(`${API_BASE}/validate-content`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
            body: formData
        });
        
        debugLog(`📡 Resposta da API: HTTP ${response.status}`, 'info');
        
        if (!response.ok) {
            if (response.status === 404) {
                debugLog('❌ Endpoint /validate-content não existe na API', 'error');
                showMessage('⚠️ Endpoint de validação não disponível. Sistema permitirá prosseguir.', 'warning');
                return { isValid: true, canLoadProfessionals: true, requiresManualValidation: true };
            }
            
            debugLog(`❌ Erro HTTP: ${response.status} ${response.statusText}`, 'error');
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        debugLog(`📊 Resultado da API: ${JSON.stringify(result)}`, 'info');
        
        if (!result.success) {
            debugLog(`❌ API retornou erro: ${result.error}`, 'error');
            showMessage(`❌ Erro na análise: ${result.error}`, 'error');
            return { isValid: false, canLoadProfessionals: false };
        }
        
        const { detected_movement, expected_movement, confidence, is_compatible } = result;
        
        debugLog(`🎯 Movimento detectado: ${detected_movement}`, 'info');
        debugLog(`⚙️ Movimento esperado: ${expected_movement}`, 'info');
        debugLog(`📊 Confiança: ${(confidence * 100).toFixed(1)}%`, 'info');
        debugLog(`✅ Compatível: ${is_compatible}`, 'info');
        
        if (is_compatible) {
            debugLog('✅ VALIDAÇÃO POR CONTEÚDO APROVADA', 'success');
            showMessage(`✅ VALIDAÇÃO APROVADA\\n🎯 Movimento detectado: ${detected_movement}\\n⚙️ Configuração: ${expected_movement}\\n📊 Confiança: ${(confidence * 100).toFixed(1)}%\\n🎾 Compatibilidade confirmada!`, 'success');
            
            return { isValid: true, canLoadProfessionals: true };
            
        } else {
            const reason = confidence < 0.6 ? 
                `Baixa confiança (${(confidence * 100).toFixed(1)}%)` :
                `Movimentos diferentes (${detected_movement} ≠ ${expected_movement})`;
                
            debugLog(`❌ VALIDAÇÃO POR CONTEÚDO REJEITADA: ${reason}`, 'error');
            showMessage(`❌ VALIDAÇÃO REJEITADA\\n🎯 Detectado: ${detected_movement}\\n⚙️ Configurado: ${expected_movement}\\n📊 Confiança: ${(confidence * 100).toFixed(1)}%\\n❌ Motivo: ${reason}\\n\\n🔧 CORRIJA A CONFIGURAÇÃO para corresponder ao movimento real do vídeo.`, 'error');
            
            return { isValid: false, canLoadProfessionals: false };
        }
        
    } catch (error) {
        debugLog(`❌ ERRO na validação de conteúdo: ${error.message}`, 'error');
        console.error('Erro validação:', error);
        
        // Fallback: permitir prosseguir se API não está disponível
        debugLog('⚠️ Fallback: permitindo prosseguir com validação manual', 'warning');
        showMessage(`⚠️ Erro na validação automática: ${error.message}\\n\\n🔧 Sistema permitirá prosseguir. Configure corretamente o movimento.`, 'warning');
        return { isValid: true, canLoadProfessionals: true, requiresManualValidation: true };
    }
}

function extractMovementFromFilename(filename) {
    debugLog(`🔍 Analisando filename: ${filename}`, 'info');
    
    const patterns = {
        'FD': /_FD_/i,
        'FP': /_FP_/i, 
        'BD': /_BD_/i,
        'BP': /_BP_/i
    };
    
    for (const [movement, pattern] of Object.entries(patterns)) {
        if (pattern.test(filename)) {
            debugLog(`✅ Padrão encontrado: ${movement}`, 'success');
            return movement;
        }
    }
    
    debugLog('❌ Nenhum padrão encontrado - movimento unknown', 'warning');
    return 'unknown';
}

function getUserConfiguration() {
    const config = {
        maoDominante: document.getElementById('mao-dominante').value,
        ladoRaquete: document.getElementById('lado-raquete').value,
        ladoCamera: document.getElementById('lado-camera').value,
        tipoMovimento: document.getElementById('tipo-movimento').value
    };
    
    // Converter para formato do backend
    config.ladoRaquete = config.ladoRaquete === 'Forehand' ? 'F' : 'B';
    config.tipoMovimento = config.tipoMovimento.includes('Drive') ? 'D' : 'P';
    
    debugLog(`⚙️ Configuração convertida: ${JSON.stringify(config)}`, 'info');
    
    return config;
}

async function validateAndLoad() {
    debugLog('🚀 INICIANDO validateAndLoad()', 'step');
    
    const validation = await validateFileAndConfiguration();
    
    debugLog(`📋 Resultado validação: ${JSON.stringify(validation)}`, 'info');
    
    const loadButton = document.getElementById('load-professionals-btn');
    const analyzeButton = document.getElementById('analyze-btn');
    
    if (validation.canLoadProfessionals) {
        debugLog('✅ Habilitando botão de carregar profissionais', 'success');
        loadButton.disabled = false;
        loadButton.style.background = '#4a7c59';
        loadButton.style.cursor = 'pointer';
        
        // Auto-carregar profissionais
        debugLog('⏳ Auto-carregando profissionais em 1 segundo...', 'info');
        setTimeout(() => {
            debugLog('🔄 Executando loadProfessionals()', 'step');
            loadProfessionals();
        }, 1000);
    } else {
        debugLog('❌ Desabilitando botões - validação falhou', 'error');
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
}

// Event listeners para debug
document.addEventListener('DOMContentLoaded', function() {
    debugLog('🌐 DOM carregado - sistema de debug ativo', 'step');
    
    // Debug nos event listeners
    const fileInput = document.getElementById('user-video');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            debugLog('📁 Arquivo alterado - triggering validateAndLoad()', 'info');
            validateAndLoad();
        });
    }
    
    // Debug nos selects
    ['mao-dominante', 'lado-raquete', 'lado-camera', 'tipo-movimento'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', function() {
                debugLog(`⚙️ Configuração alterada: ${id} = ${this.value}`, 'info');
                validateAndLoad();
            });
        }
    });
});'''

    # Inserir CSS no head
    head_end = content.find('</head>')
    if head_end != -1:
        content = content[:head_end] + debug_css + '\n' + content[head_end:]
    
    # Inserir HTML do debug panel no body
    body_start = content.find('<body>') + len('<body>')
    if body_start > len('<body>') - 1:
        content = content[:body_start] + '\n' + debug_html + '\n' + content[body_start:]
    
    # Substituir JavaScript existente
    patterns = [
        r'async function validateFileAndConfiguration\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function extractMovementFromFilename\([^)]*\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function getUserConfiguration\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'async function validateAndLoad\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}'
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Adicionar novo JavaScript com debug
    script_end = content.rfind('</script>')
    if script_end != -1:
        content = content[:script_end] + '\n' + debug_js + '\n' + content[script_end:]
    
    # Salvar arquivo atualizado
    with open('web_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Sistema de debug visual criado!")
    print(f"💾 Backup: {backup_filename}")
    print("\n🔍 AGORA A INTERFACE TERÁ:")
    print("- Janela de debug visual no canto superior direito")
    print("- Logs detalhados de cada etapa da validação")
    print("- Timestamps de cada operação")
    print("- Cores diferentes para cada tipo de log")
    print("- Possibilidade de minimizar/maximizar")
    print("- Botão para limpar logs")
    print("\n🧪 TESTE:")
    print("1. Recarregue interface (Ctrl+F5)")
    print("2. Veja janela de debug no canto direito")
    print("3. Faça upload do arquivo")
    print("4. Acompanhe logs em tempo real")
    print("5. Identifique onde a validação está falhando")

if __name__ == "__main__":
    create_debug_logger_system()
