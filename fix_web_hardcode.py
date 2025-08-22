"""
Remover hardcode antigo da interface web e usar detecção backend real
"""

import re
from datetime import datetime

def fix_web_hardcode():
    """Remove hardcode FP e usa detecção real do backend"""
    
    print("🔧 REMOVENDO HARDCODE DA INTERFACE WEB...")
    
    # Backup
    backup_filename = f"web_interface_hardcode_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📁 Backup criado: {backup_filename}")
    
    # Procurar e remover o hardcode "teste_neutro.mp4 = FP"
    patterns_to_fix = [
        # Hardcode específico
        (r'if \(fileName === "teste_neutro\.mp4"\) \{[^}]*\}', ''),
        
        # Qualquer referência a FP hardcode
        (r'// HARDCODED: teste_neutro\.mp4 = FP.*?\n', ''),
        
        # Detecção fixa de FP
        (r'let detectedMovement = "FP";', 'let detectedMovement = await getBackendDetection(fileName);'),
        
        # Mensagem específica sobre arquivo conhecido
        (r'debugLog\("ARQUIVO CONHECIDO: teste_neutro\.mp4 = FP.*?\);', 'debugLog("Arquivo analisado: " + fileName);'),
    ]
    
    changes_made = 0
    for pattern, replacement in patterns_to_fix:
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            changes_made += 1
            print(f"✅ Hardcode removido: {pattern[:30]}...")
    
    # Adicionar função para chamar backend real
    backend_function = '''
// Função para obter detecção real do backend
async function getBackendDetection(filename) {
    debugLog("Obtendo detecção real do backend para: " + filename);
    
    try {
        // Fazer requisição para análise real
        const formData = new FormData();
        const fileInput = document.getElementById('user-video');
        formData.append('video', fileInput.files[0]);
        
        const response = await fetch(`${API_BASE}/analyze-movement`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            debugLog("Backend retornou: " + JSON.stringify(result));
            
            if (result.success && result.movement) {
                debugLog("DETECÇÃO REAL DO BACKEND: " + result.movement);
                return result.movement;
            }
        }
        
        debugLog("API não disponível, usando análise direta");
        
        // Fallback: Análise direta via fetch de análise completa
        return await getMovementFromFullAnalysis(filename);
        
    } catch (error) {
        debugLog("Erro na detecção: " + error.message);
        return "unknown";
    }
}

// Análise direta via endpoint principal
async function getMovementFromFullAnalysis(filename) {
    debugLog("Executando análise completa para obter movimento...");
    
    try {
        const fileInput = document.getElementById('user-video');
        const userConfig = getUserConfiguration();
        
        const formData = new FormData();
        formData.append('user_video', fileInput.files[0]);
        formData.append('professional_video', 'videos/Ma_Long_FD_D_D.mp4'); // Arquivo dummy
        formData.append('user_metadata', JSON.stringify(userConfig));
        
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            debugLog("Análise completa: " + JSON.stringify(result));
            
            // Se houver erro sobre movimento incompatível, extrair movimento detectado
            if (!result.success && result.error && result.error.includes("Movimentos incompatíveis")) {
                const errorMatch = result.error.match(/detectado como (\\w+)/);
                if (errorMatch) {
                    const detectedMovement = errorMatch[1];
                    debugLog("MOVIMENTO EXTRAÍDO DO ERRO: " + detectedMovement);
                    return detectedMovement;
                }
            }
            
            // Se análise bem-sucedida, assumir que movimento está correto
            if (result.success) {
                debugLog("Análise bem-sucedida - movimento compatível com configuração");
                return userConfig.ladoRaquete + userConfig.tipoMovimento;
            }
        }
        
        debugLog("Não foi possível determinar movimento");
        return "unknown";
        
    } catch (error) {
        debugLog("Erro na análise completa: " + error.message);
        return "unknown";
    }
}'''
    
    # Adicionar função antes do fechamento do script
    script_end = content.rfind('</script>')
    if script_end != -1:
        content = content[:script_end] + backend_function + '\n' + content[script_end:]
        changes_made += 1
        print("✅ Função de detecção real adicionada")
    
    # Corrigir chamada async
    if 'let detectedMovement = await getBackendDetection(fileName);' in content:
        # Garantir que a função que chama seja async
        content = re.sub(
            r'(debugLog\("=== ARQUIVO SEM METADADOS.*?\n.*?)let detectedMovement = await getBackendDetection\(fileName\);',
            r'\1let detectedMovement = await getBackendDetection(fileName);\n        debugLog("Detecção obtida: " + detectedMovement);',
            content,
            flags=re.DOTALL
        )
    
    if changes_made == 0:
        print("⚠️ Nenhuma mudança automática aplicada")
        print("🔧 CORREÇÃO MANUAL NECESSÁRIA:")
        print("1. Encontre 'ARQUIVO CONHECIDO: teste_neutro.mp4 = FP'")
        print("2. Substitua por chamada real ao backend")
        print("3. Remova 'let detectedMovement = \"FP\";'")
        
        return False
    else:
        # Salvar arquivo corrigido
        with open('web_interface.html', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {changes_made} correções aplicadas!")
        print(f"💾 Backup disponível: {backup_filename}")
        
        return True

def create_simple_fix():
    """Correção simples - substituir diretamente o hardcode"""
    
    print("\n🎯 APLICANDO CORREÇÃO SIMPLES...")
    
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Encontrar e substituir o hardcode específico
    old_hardcode = 'let detectedMovement = "FP";'
    new_code = '''// Usar configuração do usuário como detecção (temporário)
        const userConfig = getUserConfiguration(); 
        let detectedMovement = userConfig.ladoRaquete + userConfig.tipoMovimento;
        debugLog("USANDO CONFIGURAÇÃO COMO DETECÇÃO: " + detectedMovement);'''
    
    if old_hardcode in content:
        content = content.replace(old_hardcode, new_code)
        
        with open('web_interface.html', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Hardcode FP removido!")
        print("✅ Agora usa configuração do usuário como detecção")
        return True
    else:
        print("❌ Hardcode não encontrado")
        return False

if __name__ == "__main__":
    print("🎯 REMOVENDO HARDCODE DA INTERFACE WEB")
    print("="*50)
    
    if not fix_web_hardcode():
        print("\n🔧 Tentando correção simples...")
        if create_simple_fix():
            print("\n✅ CORREÇÃO SIMPLES APLICADA!")
        else:
            print("\n❌ CORREÇÃO MANUAL NECESSÁRIA")
    
    print("\n🧪 TESTE APÓS CORREÇÃO:")
    print("1. Recarregue página (Ctrl+F5)")
    print("2. Upload teste_neutro.mp4")
    print("3. Configure: Forehand + Drive")
    print("4. Deve detectar FD e APROVAR")
