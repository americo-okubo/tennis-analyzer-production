"""
Debug da função loadProfessionals que está falhando
"""

def debug_professionals_loading():
    """Adiciona debug detalhado na função loadProfessionals"""
    
    print("🔍 ADICIONANDO DEBUG NA FUNÇÃO loadProfessionals...")
    
    # Backup
    from datetime import datetime
    backup_filename = f"web_interface_professionals_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📁 Backup criado: {backup_filename}")
    
    # Nova função loadProfessionals com debug completo
    debug_load_professionals = '''
async function loadProfessionals() {
    debugLog("🔄 INICIANDO loadProfessionals()");
    
    const profList = document.getElementById('professionals-list');
    if (!profList) {
        debugLog("❌ Elemento professionals-list não encontrado");
        return;
    }
    
    profList.innerHTML = '<p style="text-align: center; padding: 20px;">Carregando profissionais...</p>';
    
    try {
        // Verificar token
        if (!authToken) {
            debugLog("❌ Token não encontrado");
            profList.innerHTML = '<p style="color: red;">Erro: Token de autenticação não encontrado. Faça login novamente.</p>';
            return;
        }
        
        debugLog("✅ Token OK: " + authToken.substring(0, 20) + "...");
        
        // Obter configuração do usuário para filtro
        const userConfig = getUserConfiguration();
        debugLog("⚙️ Configuração para filtro: " + JSON.stringify(userConfig));
        
        // Determinar filtro de movimento
        let movementFilter = "backhand_drive"; // Padrão baseado na config atual
        
        if (userConfig.ladoRaquete === 'F' && userConfig.tipoMovimento === 'D') {
            movementFilter = "forehand_drive";
        } else if (userConfig.ladoRaquete === 'F' && userConfig.tipoMovimento === 'P') {
            movementFilter = "forehand_push";
        } else if (userConfig.ladoRaquete === 'B' && userConfig.tipoMovimento === 'D') {
            movementFilter = "backhand_drive";
        } else if (userConfig.ladoRaquete === 'B' && userConfig.tipoMovimento === 'P') {
            movementFilter = "backhand_push";
        }
        
        debugLog("🎯 Filtro de movimento: " + movementFilter);
        
        // Construir URL com filtro
        const url = `${API_BASE}/professionals?movement_type=${movementFilter}`;
        debugLog("🌐 URL da requisição: " + url);
        
        // Fazer requisição
        debugLog("📡 Fazendo requisição para API...");
        
        const response = await fetch(url, {
            method: 'GET',
            headers: { 
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            }
        });
        
        debugLog("📊 Status da resposta: " + response.status);
        debugLog("📊 Headers da resposta: " + JSON.stringify([...response.headers.entries()]));
        
        if (!response.ok) {
            debugLog("❌ Resposta não OK: " + response.status + " " + response.statusText);
            
            const errorText = await response.text();
            debugLog("❌ Texto do erro: " + errorText);
            
            profList.innerHTML = `
                <div style="text-align: center; padding: 20px; background: #ffebee; border: 1px solid #e57373; border-radius: 5px;">
                    <h4 style="color: #c62828;">❌ Erro ao carregar profissionais</h4>
                    <p><strong>Status:</strong> ${response.status}</p>
                    <p><strong>Erro:</strong> ${errorText}</p>
                    <button onclick="loadProfessionals()" style="margin-top: 10px; padding: 8px 16px; background: #4a7c59; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        Tentar Novamente
                    </button>
                </div>
            `;
            return;
        }
        
        // Processar resposta
        debugLog("✅ Resposta OK, processando dados...");
        
        const data = await response.json();
        debugLog("📦 Dados recebidos: " + JSON.stringify(data));
        
        // Extrair profissionais (tratando diferentes formatos)
        let professionals = [];
        
        if (data.success && Array.isArray(data.professionals)) {
            professionals = data.professionals;
            debugLog("✅ Formato com success: " + professionals.length + " profissionais");
        } else if (Array.isArray(data)) {
            professionals = data;
            debugLog("✅ Formato array direto: " + professionals.length + " profissionais");
        } else if (data.professionals && Array.isArray(data.professionals)) {
            professionals = data.professionals;
            debugLog("✅ Formato com professionals: " + professionals.length + " profissionais");
        } else {
            debugLog("❌ Formato de dados não reconhecido: " + typeof data);
            debugLog("❌ Estrutura dos dados: " + Object.keys(data));
        }
        
        // Limpar lista
        profList.innerHTML = '';
        
        if (professionals.length === 0) {
            debugLog("⚠️ Nenhum profissional encontrado para filtro: " + movementFilter);
            profList.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 20px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
                    <h4>⚠️ Nenhum profissional encontrado</h4>
                    <p><strong>Filtro usado:</strong> ${movementFilter}</p>
                    <p><strong>Configuração:</strong> ${userConfig.ladoRaquete} + ${userConfig.tipoMovimento}</p>
                    <p>Tente alterar as configurações de movimento.</p>
                    <button onclick="loadProfessionals()" style="margin-top: 10px; padding: 8px 16px; background: #4a7c59; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        Recarregar
                    </button>
                </div>
            `;
            return;
        }
        
        // Criar cards dos profissionais
        debugLog("🎨 Criando " + professionals.length + " cards de profissionais...");
        
        professionals.forEach((prof, index) => {
            debugLog("👤 Profissional " + index + ": " + JSON.stringify(prof));
            
            const profCard = document.createElement('div');
            profCard.style.cssText = `
                border: 2px solid #ddd; 
                padding: 15px; 
                border-radius: 8px; 
                cursor: pointer; 
                text-align: center;
                background: white;
                transition: all 0.3s ease;
            `;
            
            const name = prof.name || prof.professional_name || 'Nome não disponível';
            const hand = prof.hand || prof.hand_dominance || 'N/A';
            const camera = prof.camera || prof.camera_side || 'N/A';
            const video = prof.video || prof.filename || prof.video_path || 'N/A';
            const rating = prof.technique_rating || prof.rating || 'N/A';
            const exists = prof.video_exists !== undefined ? prof.video_exists : true;
            
            profCard.innerHTML = `
                <h4 style="margin: 0 0 10px 0; color: #4a7c59;">${name}</h4>
                <p style="margin: 5px 0;"><strong>Mão:</strong> ${hand}</p>
                <p style="margin: 5px 0;"><strong>Câmera:</strong> ${camera}</p>
                <p style="margin: 5px 0; font-size: 12px; color: #666;">${video}</p>
                <div style="margin-top: 10px; padding: 5px; background: #f8f9fa; border-radius: 3px; font-size: 11px;">
                    Rating: ${rating}% | Exists: ${exists ? '✅' : '❌'}
                </div>
            `;
            
            // Hover effects
            profCard.addEventListener('mouseenter', () => {
                profCard.style.borderColor = '#4a7c59';
                profCard.style.boxShadow = '0 4px 8px rgba(74, 124, 89, 0.2)';
            });
            
            profCard.addEventListener('mouseleave', () => {
                if (selectedProfessional !== prof) {
                    profCard.style.borderColor = '#ddd';
                    profCard.style.boxShadow = 'none';
                }
            });
            
            profCard.onclick = () => selectProfessional(prof, profCard);
            profList.appendChild(profCard);
        });
        
        debugLog("✅ " + professionals.length + " profissionais carregados com sucesso!");
        
    } catch (error) {
        debugLog("❌ ERRO na função loadProfessionals: " + error.message);
        debugLog("❌ Stack trace: " + error.stack);
        
        profList.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 20px; background: #ffebee; border: 1px solid #e57373; border-radius: 5px;">
                <h4 style="color: #c62828; margin: 0 0 10px 0;">❌ Erro ao carregar profissionais</h4>
                <p style="margin: 5px 0;"><strong>Erro:</strong> ${error.message}</p>
                <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">
                    Verifique se a API está funcionando e se o token é válido.
                </p>
                <button onclick="loadProfessionals()" style="margin-top: 10px; padding: 8px 16px; background: #4a7c59; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Tentar Novamente
                </button>
            </div>
        `;
    }
}'''

    # Encontrar e substituir a função loadProfessionals
    import re
    
    # Padrão para encontrar a função loadProfessionals
    pattern = r'async function loadProfessionals\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, debug_load_professionals.strip(), content, flags=re.DOTALL)
        print("✅ Função loadProfessionals substituída com debug")
    else:
        print("❌ Função loadProfessionals não encontrada")
        return False
    
    # Salvar arquivo atualizado
    with open('web_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Debug adicionado à função loadProfessionals!")
    print(f"💾 Backup disponível: {backup_filename}")
    
    return True

if __name__ == "__main__":
    print("🔍 ADICIONANDO DEBUG DETALHADO PARA loadProfessionals")
    print("="*60)
    
    if debug_professionals_loading():
        print("\n🧪 TESTE AGORA:")
        print("1. Recarregue página (Ctrl+F5)")
        print("2. Faça upload do arquivo")
        print("3. Configure movimento")
        print("4. Veja logs detalhados no console")
        print("5. Identifique onde loadProfessionals falha")
        
        print("\n🔍 LOGS ESPERADOS:")
        print("- Token OK/erro")
        print("- URL da requisição")
        print("- Status da resposta da API")
        print("- Dados recebidos")
        print("- Profissionais encontrados")
    else:
        print("\n❌ FALHA NO DEBUG")
        print("🔧 Verificar estrutura do arquivo")
