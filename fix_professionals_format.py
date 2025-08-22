"""
Correção para o formato de resposta da API /professionals
A API retorna {professionals: [...]} mas JavaScript espera [...]
"""

import re
from datetime import datetime

def fix_professionals_response_format():
    """Corrige o JavaScript para lidar com formato correto da API"""
    
    print("Corrigindo formato de resposta de profissionais...")
    
    # Backup
    backup_filename = f"web_interface_format_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open('web_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup criado: {backup_filename}")
    
    # Função loadProfessionals corrigida
    new_load_professionals = '''
async function loadProfessionals() {
    console.log('Carregando profissionais...');
    const profList = document.getElementById('professionals-list');
    profList.innerHTML = '<p>Carregando profissionais...</p>';
    
    try {
        // Obter configuração do usuário para filtrar profissionais
        const userConfig = getUserConfiguration();
        const movementFilter = `${userConfig.ladoRaquete.toLowerCase()}_${userConfig.tipoMovimento === 'D' ? 'drive' : 'push'}`;
        
        console.log('Filtro de movimento:', movementFilter);
        
        // Fazer requisição com filtros
        const url = `${API_BASE}/professionals?movement_type=${movementFilter}`;
        console.log('URL da requisição:', url);
        
        const response = await fetch(url, {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        
        console.log('Status da resposta:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Dados recebidos:', data);
        
        // CORREÇÃO PRINCIPAL: Acessar array dentro do objeto
        let professionals = [];
        
        if (data.success && Array.isArray(data.professionals)) {
            professionals = data.professionals;
            console.log(`✅ ${professionals.length} profissionais encontrados`);
        } else if (Array.isArray(data)) {
            // Fallback se API retornar array direto
            professionals = data;
            console.log(`✅ ${professionals.length} profissionais (array direto)`);
        } else {
            console.error('Formato inesperado de resposta:', data);
            throw new Error('Formato de resposta inválido');
        }
        
        // Limpar lista
        profList.innerHTML = '';
        
        if (professionals.length === 0) {
            profList.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 20px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
                    <p><strong>Nenhum profissional encontrado</strong></p>
                    <p>Movimento configurado: ${movementFilter}</p>
                    <p>Tente alterar as configurações de movimento.</p>
                </div>
            `;
            return;
        }
        
        // Criar cards dos profissionais
        professionals.forEach(prof => {
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
            
            profCard.innerHTML = `
                <h4 style="margin: 0 0 10px 0; color: #4a7c59;">${prof.name || 'Nome não disponível'}</h4>
                <p style="margin: 5px 0;"><strong>Mão:</strong> ${prof.hand || 'N/A'}</p>
                <p style="margin: 5px 0;"><strong>Câmera:</strong> ${prof.camera || 'N/A'}</p>
                <p style="margin: 5px 0; font-size: 12px; color: #666;">
                    ${prof.video || 'Vídeo não especificado'}
                </p>
                <div style="margin-top: 10px; padding: 5px; background: #f8f9fa; border-radius: 3px; font-size: 11px;">
                    Rating: ${prof.technique_rating || 'N/A'}% | Exists: ${prof.video_exists ? '✅' : '❌'}
                </div>
            `;
            
            // Adicionar hover effect
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
        
        console.log(`✅ ${professionals.length} profissionais carregados com sucesso`);
        
    } catch (error) {
        console.error('Erro ao carregar profissionais:', error);
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

    # Substituir função loadProfessionals
    pattern = r'async function loadProfessionals\(\)[^}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*}'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_load_professionals.strip(), content, flags=re.DOTALL)
        print("✅ Função loadProfessionals substituída")
    else:
        print("⚠️ Função loadProfessionals não encontrada para substituição")
    
    # Salvar arquivo corrigido
    with open('web_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Correção aplicada!")
    print(f"Backup disponível em: {backup_filename}")
    print("\nAgora:")
    print("1. Recarregue a página (F5)")
    print("2. Faça login (demo/demo123)")
    print("3. Upload teste_neutro.mp4") 
    print("4. Configure: Forehand + Push")
    print("5. Clique 'CARREGAR PROFISSIONAIS'")
    print("6. Deve funcionar perfeitamente!")

if __name__ == "__main__":
    fix_professionals_response_format()
