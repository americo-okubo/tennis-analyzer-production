"""
Correção simples: para arquivos 'unknown', avisar usuário que precisa configurar manualmente
e validar apenas no final da análise (onde a detecção de conteúdo já funciona)
"""

import re
from datetime import datetime

def fix_simple_validation():
    """Correção simples da validação para arquivos sem metadados"""
    
    print("Aplicando correção simples da validação...")
    
    # Backup
    backup_filename = f"web_interface_simple_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup criado: {backup_filename}")
    
    # Função de validação simples e clara
    new_validation_function = '''
function validateFileAndConfiguration() {
    const fileInput = document.getElementById('user-video');
    const fileName = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    
    if (!fileName) {
        console.log('Nenhum arquivo selecionado');
        hideMessage();
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
    
    // Para arquivos sem metadados, dar instruções claras
    if (fileMovement === 'unknown') {
        console.log('Arquivo sem metadados - exigindo configuração manual precisa');
        showMessage(`
            ⚠️ ARQUIVO SEM METADADOS DETECTADO
            
            Configure exatamente o movimento do vídeo:
            • Se vídeo mostra FOREHAND: configure "Forehand"
            • Se vídeo mostra BACKHAND: configure "Backhand"  
            • Se movimento é ATAQUE: configure "Drive"
            • Se movimento é DEFESA: configure "Push"
            
            ⚡ O sistema validará durante a análise usando IA.
        `, 'warning');
        
        return { 
            isValid: true, 
            canLoadProfessionals: true,
            requiresManualConfiguration: true 
        };
    }
    
    // Validação rigorosa para arquivos com metadados
    const isCompatible = fileMovement === userMovement;
    
    if (!isCompatible) {
        const errorMsg = `❌ INCOMPATIBILIDADE: arquivo "${fileName}" contém movimento "${fileMovement}" mas configuração é "${userMovement}"`;
        console.log(errorMsg);
        showMessage(errorMsg, 'error');
        return { isValid: false, canLoadProfessionals: false };
    }
    
    console.log('✅ Validação passou - movimentos compatíveis');
    showMessage('✅ Arquivo e configuração compatíveis', 'success');
    return { isValid: true, canLoadProfessionals: true };
}

function showMessage(message, type) {
    let messageDiv = document.getElementById('validation-message');
    if (!messageDiv) {
        messageDiv = document.createElement('div');
        messageDiv.id = 'validation-message';
        messageDiv.style.padding = '15px';
        messageDiv.style.margin = '15px 0';
        messageDiv.style.borderRadius = '8px';
        messageDiv.style.fontSize = '14px';
        messageDiv.style.lineHeight = '1.5';
        messageDiv.style.whiteSpace = 'pre-line';
        
        // Inserir após o upload de arquivo
        const fileInput = document.getElementById('user-video');
        fileInput.parentNode.insertBefore(messageDiv, fileInput.nextSibling);
    }
    
    // Configurar cor baseada no tipo
    if (type === 'error') {
        messageDiv.style.backgroundColor = '#ffebee';
        messageDiv.style.color = '#c62828';
        messageDiv.style.border = '2px solid #e57373';
    } else if (type === 'warning') {
        messageDiv.style.backgroundColor = '#fff3e0';
        messageDiv.style.color = '#f57c00';
        messageDiv.style.border = '2px solid #ffb74d';
    } else if (type === 'success') {
        messageDiv.style.backgroundColor = '#e8f5e8';
        messageDiv.style.color = '#2e7d32';
        messageDiv.style.border = '2px solid #81c784';
    } else { // info
        messageDiv.style.backgroundColor = '#e3f2fd';
        messageDiv.style.color = '#1565c0';
        messageDiv.style.border = '2px solid #64b5f6';
    }
    
    messageDiv.innerHTML = message.replace(/\\n/g, '<br>');
    messageDiv.style.display = 'block';
    
    // Auto-hide success messages
    if (type === 'success') {
        setTimeout(() => {
            hideMessage();
        }, 3000);
    }
}

function hideMessage() {
    const messageDiv = document.getElementById('validation-message');
    if (messageDiv) {
        messageDiv.style.display = 'none';
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
        
        // Limpar lista de profissionais
        const profList = document.getElementById('professionals-list');
        if (profList) {
            profList.innerHTML = '';
        }
    }
}'''

    # Substituir funções de validação
    patterns = [
        r'function validateFileAndConfiguration\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function showMessage\([^)]*\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function hideMessage\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}',
        r'function validateAndLoad\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}'
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Adicionar novas funções
    script_end = content.rfind('</script>')
    if script_end != -1:
        content = content[:script_end] + "\n" + new_validation_function + "\n" + content[script_end:]
        print("✅ Validação simples aplicada")
    
    # Salvar arquivo corrigido
    with open('web_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Correção simples aplicada!")
    print(f"Backup disponível em: {backup_filename}")
    print("\n🧪 AGORA O COMPORTAMENTO SERÁ:")
    print("📁 Arquivos COM metadados: validação rigorosa")
    print("📁 Arquivos SEM metadados: aviso para configurar corretamente")
    print("🔬 Validação real: acontece durante análise (backend já funciona)")
    print("\n🧪 TESTE:")
    print("1. Recarregue página (Ctrl+F5)")
    print("2. Upload: teste_neutro.mp4") 
    print("3. Configure: Backhand + Drive")
    print("4. Deve mostrar AVISO mas permitir carregar profissionais")
    print("5. Na análise final: sistema detectará incompatibilidade")

if __name__ == "__main__":
    fix_simple_validation()
