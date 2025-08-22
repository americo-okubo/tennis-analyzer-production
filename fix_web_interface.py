"""
Script para corrigir validacao da interface web
Permite arquivos sem metadados e ativa analise por conteudo
"""

import re
from datetime import datetime

def fix_web_interface_validation():
    """Corrige validacao no web_interface.html para aceitar arquivos sem metadados"""
    
    print("Corrigindo validacao da interface web...")
    
    # Backup
    backup_filename = f"web_interface_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup criado: {backup_filename}")
    
    # Nova função de validação que aceita arquivos sem metadados
    new_validation_function = '''
function validateFileAndConfiguration() {
    const fileInput = document.getElementById('user-video');
    const fileName = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    
    if (!fileName) {
        console.log('Nenhum arquivo selecionado');
        return { isValid: false, canLoadProfessionals: false };
    }
    
    console.log(`Validando arquivo: ${fileName}`);
    
    // NOVA LÓGICA: Tentar extrair movimento do filename
    const fileMovement = extractMovementFromFilename(fileName);
    
    // Obter configuração do usuário
    const userConfig = getUserConfiguration();
    const userMovement = `${userConfig.ladoRaquete}${userConfig.tipoMovimento}`;
    
    console.log(`Movimento do arquivo: ${fileMovement}`);
    console.log(`Movimento configurado: ${userMovement}`);
    
    // VALIDAÇÃO FLEXÍVEL PARA ARQUIVOS SEM METADADOS
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
    // Extrair padrões de movimento do filename
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
    
    // Se não encontrou padrões, retorna unknown
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
    // Criar elemento de mensagem se não existir
    let messageDiv = document.getElementById('validation-message');
    if (!messageDiv) {
        messageDiv = document.createElement('div');
        messageDiv.id = 'validation-message';
        messageDiv.style.padding = '10px';
        messageDiv.style.margin = '10px 0';
        messageDiv.style.borderRadius = '5px';
        
        // Inserir após o upload de arquivo
        const fileInput = document.getElementById('user-video');
        fileInput.parentNode.insertBefore(messageDiv, fileInput.nextSibling);
    }
    
    // Configurar cor baseada no tipo
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
    
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';
    
    // Esconder após 5 segundos se for mensagem de info
    if (type === 'info') {
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, 5000);
    }
}

function validateAndLoad() {
    const validation = validateFileAndConfiguration();
    
    const loadButton = document.getElementById('load-professionals-btn');
    const analyzeButton = document.getElementById('analyze-btn');
    
    if (validation.canLoadProfessionals) {
        loadButton.disabled = false;
        loadButton.classList.remove('disabled');
        
        // Carregar profissionais automaticamente se válido
        loadProfessionals();
    } else {
        loadButton.disabled = true;
        loadButton.classList.add('disabled');
        analyzeButton.disabled = true;
        analyzeButton.classList.add('disabled');
        
        // Limpar lista de profissionais
        const profList = document.getElementById('professionals-list');
        if (profList) {
            profList.innerHTML = '';
        }
    }
}'''
    
    # Procurar e substituir a função validateFileAndConfiguration existente
    pattern = r'function validateFileAndConfiguration\(\)[^}]*(?:{[^{}]*}[^{}]*)*}'
    
    # Verificar se encontrou a função
    if re.search(pattern, content):
        # Substituir a função existente
        content = re.sub(pattern, new_validation_function.strip(), content, flags=re.DOTALL)
        print("Funcao validateFileAndConfiguration substituida")
    else:
        # Se não encontrou, adicionar no final do script
        script_end = content.rfind('</script>')
        if script_end != -1:
            content = content[:script_end] + "\n" + new_validation_function + "\n" + content[script_end:]
            print("Funcoes de validacao adicionadas no final do script")
    
    # Salvar arquivo modificado
    with open('web_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Interface web corrigida com sucesso!")
    print(f"Backup disponivel em: {backup_filename}")
    print("\nAgora teste novamente:")
    print("1. Recarregue a pagina web (F5)")
    print("2. Faca upload do teste_neutro.mp4")
    print("3. Configure como quiser")
    print("4. Deve mostrar mensagem de 'analise por conteudo'")

if __name__ == "__main__":
    fix_web_interface_validation()
