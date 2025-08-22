"""
Script para corrigir erros JavaScript na interface web
Resolve problemas de sintaxe e função login undefined
"""

import re
from datetime import datetime

def fix_javascript_errors():
    """Corrige erros JavaScript específicos encontrados no console"""
    
    print("Corrigindo erros JavaScript da interface...")
    
    # Backup
    backup_filename = f"web_interface_error_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup de erro criado: {backup_filename}")
    
    # JavaScript corrigido para login e interface principal
    corrected_javascript = '''
<script>
// Configuração da API
const API_BASE = 'http://localhost:8000';
let authToken = null;

// Função de login corrigida
async function login() {
    console.log('Função login chamada');
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        alert('Por favor, preencha usuário e senha');
        return;
    }
    
    try {
        console.log('Tentando login...');
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        console.log('Resposta recebida:', response.status);
        const data = await response.json();
        
        if (data.access_token) {
            authToken = data.access_token;
            localStorage.setItem('token', data.access_token);
            console.log('Login realizado com sucesso');
            showMainInterface();
        } else {
            console.error('Login falhou:', data);
            alert('Login falhou: ' + (data.detail || 'Erro desconhecido'));
        }
    } catch (error) {
        console.error('Erro de conexão:', error);
        alert('Erro de conexão com a API. Verifique se o servidor está rodando.');
    }
}

// Função para mostrar interface principal
function showMainInterface() {
    console.log('Mostrando interface principal');
    document.body.innerHTML = `
        <div style="max-width: 1200px; margin: 0 auto; padding: 20px;">
            <header style="background: #4a7c59; color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
                <h1>🎾 Tennis Analyzer</h1>
                <p>Análise Avançada de Técnica de Tênis de Mesa</p>
                <button onclick="logout()" style="float: right; background: #fff; color: #4a7c59; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">Logout</button>
            </header>
            
            <div style="background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h2>📹 Upload do Vídeo</h2>
                <input type="file" id="user-video" accept="video/*" style="width: 100%; padding: 10px; border: 2px dashed #ccc; border-radius: 5px; margin-bottom: 20px;">
                <div id="validation-message" style="display: none;"></div>
                
                <h3>⚙️ Configurações da Análise</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                    <div>
                        <label>Mão Dominante:</label>
                        <select id="mao-dominante" style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                            <option value="Destro">Destro</option>
                            <option value="Esquerdo">Esquerdo</option>
                        </select>
                    </div>
                    <div>
                        <label>Lado da Câmera:</label>
                        <select id="lado-camera" style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                            <option value="Direita">Direita</option>
                            <option value="Esquerda">Esquerda</option>
                        </select>
                    </div>
                    <div>
                        <label>Lado da Raquete:</label>
                        <select id="lado-raquete" onchange="validateAndLoad()" style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                            <option value="Forehand">Forehand</option>
                            <option value="Backhand">Backhand</option>
                        </select>
                    </div>
                    <div>
                        <label>Tipo de Movimento:</label>
                        <select id="tipo-movimento" onchange="validateAndLoad()" style="width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                            <option value="Drive (Ataque)">Drive (Ataque)</option>
                            <option value="Push (Defesa)">Push (Defesa)</option>
                        </select>
                    </div>
                </div>
                
                <button onclick="loadProfessionals()" id="load-professionals-btn" style="width: 100%; background: #4a7c59; color: white; padding: 15px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin: 20px 0;">
                    CARREGAR PROFISSIONAIS
                </button>
                
                <h3>🏆 Selecionar Profissional</h3>
                <div id="professionals-list" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
                    <!-- Profissionais serão carregados aqui -->
                </div>
                
                <button onclick="startAnalysis()" id="analyze-btn" disabled style="width: 100%; background: #ccc; color: white; padding: 15px; border: none; border-radius: 5px; font-size: 16px; cursor: not-allowed; margin: 20px 0;">
                    INICIAR ANÁLISE
                </button>
                
                <div id="results" style="margin-top: 30px;"></div>
            </div>
        </div>
    `;
    
    // Adicionar event listeners
    document.getElementById('user-video').addEventListener('change', validateAndLoad);
}

// Função de logout
function logout() {
    authToken = null;
    localStorage.removeItem('token');
    location.reload();
}

// Função de validação corrigida
function validateFileAndConfiguration() {
    const fileInput = document.getElementById('user-video');
    const fileName = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    
    if (!fileName) {
        console.log('Nenhum arquivo selecionado');
        return { isValid: false, canLoadProfessionals: false };
    }
    
    console.log(`Validando arquivo: ${fileName}`);
    
    // Extrair movimento do filename
    const fileMovement = extractMovementFromFilename(fileName);
    
    // Obter configuração do usuário
    const userConfig = getUserConfiguration();
    const userMovement = `${userConfig.ladoRaquete}${userConfig.tipoMovimento}`;
    
    console.log(`Movimento do arquivo: ${fileMovement}`);
    console.log(`Movimento configurado: ${userMovement}`);
    
    // Validação flexível para arquivos sem metadados
    if (fileMovement === 'unknown') {
        console.log('Arquivo sem metadados - permitindo análise por conteúdo');
        showMessage('Arquivo sem metadados detectado. Sistema usará análise de conteúdo.', 'info');
        return { 
            isValid: true, 
            canLoadProfessionals: true,
            requiresContentAnalysis: true 
        };
    }
    
    // Validação normal para arquivos com metadados
    const isCompatible = fileMovement === userMovement;
    
    if (!isCompatible) {
        const errorMsg = `Incompatibilidade: arquivo contém "${fileMovement}" mas configuração é "${userMovement}"`;
        console.log(errorMsg);
        showMessage(errorMsg, 'error');
        return { isValid: false, canLoadProfessionals: false };
    }
    
    console.log('Validação passou - movimentos compatíveis');
    return { isValid: true, canLoadProfessionals: true };
}

function extractMovementFromFilename(filename) {
    const patterns = {
        'FD': /_FD_/i,
        'FP': /_FP_/i, 
        'BD': /_BD_/i,
        'BP': /_BP_/i
    };
    
    for (const [movement, pattern] of Object.entries(patterns)) {
        if (pattern.test(filename)) {
            return movement;
        }
    }
    
    return 'unknown';
}

function getUserConfiguration() {
    const ladoRaquete = document.getElementById('lado-raquete').value === 'Forehand' ? 'F' : 'B';
    const tipoMovimento = document.getElementById('tipo-movimento').value.includes('Drive') ? 'D' : 'P';
    
    return {
        maoDominante: document.getElementById('mao-dominante').value === 'Destro' ? 'D' : 'E',
        ladoRaquete: ladoRaquete,
        ladoCamera: document.getElementById('lado-camera').value === 'Direita' ? 'D' : 'E', 
        tipoMovimento: tipoMovimento
    };
}

function showMessage(message, type) {
    let messageDiv = document.getElementById('validation-message');
    if (!messageDiv) return;
    
    if (type === 'error') {
        messageDiv.style.backgroundColor = '#ffebee';
        messageDiv.style.color = '#c62828';
        messageDiv.style.border = '1px solid #e57373';
    } else if (type === 'info') {
        messageDiv.style.backgroundColor = '#e3f2fd';
        messageDiv.style.color = '#1565c0';
        messageDiv.style.border = '1px solid #64b5f6';
    } else {
        messageDiv.style.backgroundColor = '#e8f5e8';
        messageDiv.style.color = '#2e7d32';
        messageDiv.style.border = '1px solid #81c784';
    }
    
    messageDiv.style.padding = '10px';
    messageDiv.style.borderRadius = '5px';
    messageDiv.style.margin = '10px 0';
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';
    
    if (type === 'info') {
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, 5000);
    }
}

function validateAndLoad() {
    console.log('validateAndLoad chamado');
    const validation = validateFileAndConfiguration();
    
    const loadButton = document.getElementById('load-professionals-btn');
    const analyzeButton = document.getElementById('analyze-btn');
    
    if (validation.canLoadProfessionals) {
        loadButton.disabled = false;
        loadButton.style.background = '#4a7c59';
        loadButton.style.cursor = 'pointer';
    } else {
        loadButton.disabled = true;
        loadButton.style.background = '#ccc';
        loadButton.style.cursor = 'not-allowed';
        analyzeButton.disabled = true;
        analyzeButton.style.background = '#ccc';
        analyzeButton.style.cursor = 'not-allowed';
    }
}

async function loadProfessionals() {
    console.log('Carregando profissionais...');
    const profList = document.getElementById('professionals-list');
    profList.innerHTML = '<p>Carregando profissionais...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/professionals`, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        const professionals = await response.json();
        
        profList.innerHTML = '';
        professionals.forEach(prof => {
            const profCard = document.createElement('div');
            profCard.style.cssText = 'border: 2px solid #ddd; padding: 15px; border-radius: 8px; cursor: pointer; text-align: center;';
            profCard.innerHTML = `
                <h4>${prof.name}</h4>
                <p>Mão: ${prof.hand}</p>
                <p>Câmera: ${prof.camera}</p>
            `;
            profCard.onclick = () => selectProfessional(prof, profCard);
            profList.appendChild(profCard);
        });
        
    } catch (error) {
        console.error('Erro ao carregar profissionais:', error);
        profList.innerHTML = '<p>Erro ao carregar profissionais</p>';
    }
}

let selectedProfessional = null;

function selectProfessional(prof, element) {
    // Remover seleção anterior
    document.querySelectorAll('#professionals-list > div').forEach(div => {
        div.style.border = '2px solid #ddd';
    });
    
    // Selecionar atual
    element.style.border = '2px solid #4a7c59';
    selectedProfessional = prof;
    
    // Ativar botão de análise
    const analyzeButton = document.getElementById('analyze-btn');
    analyzeButton.disabled = false;
    analyzeButton.style.background = '#4a7c59';
    analyzeButton.style.cursor = 'pointer';
}

async function startAnalysis() {
    if (!selectedProfessional) {
        alert('Selecione um profissional primeiro');
        return;
    }
    
    const fileInput = document.getElementById('user-video');
    if (!fileInput.files[0]) {
        alert('Selecione um vídeo primeiro');
        return;
    }
    
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<div style="text-align: center; padding: 20px;"><h3>Analisando...</h3><p>Por favor, aguarde. Isso pode levar alguns minutos.</p></div>';
    
    const formData = new FormData();
    formData.append('user_video', fileInput.files[0]);
    formData.append('professional_video', `videos/${selectedProfessional.video}`);
    formData.append('user_metadata', JSON.stringify(getUserConfiguration()));
    
    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            resultsDiv.innerHTML = `
                <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; border: 1px solid #81c784;">
                    <h3>🎾 ANÁLISE BIOMECÂNICA COMPLETA</h3>
                    <p><strong>Score de Similaridade:</strong> ${result.score}/100 (${result.assessment || 'N/A'})</p>
                    <p><strong>Método:</strong> ${result.analysis_type || 'N/A'}</p>
                    <p><strong>ID da Análise:</strong> ${result.analysis_id}</p>
                    ${result.recommendations ? `<p><strong>Recomendações:</strong> ${result.recommendations}</p>` : ''}
                </div>
            `;
        } else {
            resultsDiv.innerHTML = `
                <div style="background: #ffebee; padding: 20px; border-radius: 10px; border: 1px solid #e57373;">
                    <h3>❌ Erro na Análise</h3>
                    <p>${result.error || 'Erro desconhecido'}</p>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Erro na análise:', error);
        resultsDiv.innerHTML = `
            <div style="background: #ffebee; padding: 20px; border-radius: 10px; border: 1px solid #e57373;">
                <h3>❌ Erro de Conexão</h3>
                <p>Erro ao conectar com a API: ${error.message}</p>
            </div>
        `;
    }
}

// Verificar token ao carregar
window.onload = function() {
    const token = localStorage.getItem('token');
    if (token) {
        authToken = token;
        showMainInterface();
    }
};
</script>
'''
    
    # Procurar e substituir todo o JavaScript
    script_pattern = r'<script>.*?</script>'
    
    if re.search(script_pattern, content, re.DOTALL):
        content = re.sub(script_pattern, corrected_javascript, content, flags=re.DOTALL)
        print("JavaScript substituído completamente")
    else:
        # Se não encontrou, adicionar antes do </body>
        body_end = content.rfind('</body>')
        if body_end != -1:
            content = content[:body_end] + corrected_javascript + "\n" + content[body_end:]
            print("JavaScript adicionado antes do </body>")
    
    # Salvar arquivo corrigido
    with open('web_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Erros JavaScript corrigidos!")
    print(f"Backup disponível em: {backup_filename}")
    print("\nAgora:")
    print("1. Recarregue a página (F5)")
    print("2. Faça login: demo/demo123")
    print("3. Interface deve funcionar perfeitamente")

if __name__ == "__main__":
    fix_javascript_errors()
