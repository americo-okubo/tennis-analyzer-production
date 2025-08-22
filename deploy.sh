#!/bin/bash
# Script de deploy para Google Cloud Platform
# Tennis Analyzer - Deploy automatizado

echo "🎾 Tennis Analyzer - Deploy para GCP"
echo "===================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para logs coloridos
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Verificar se gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    log_error "Google Cloud SDK não encontrado!"
    log_info "Instale em: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verificar autenticação
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    log_error "Não autenticado com Google Cloud!"
    log_info "Execute: gcloud auth login"
    exit 1
fi

# Obter projeto atual
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    log_error "Nenhum projeto configurado!"
    log_info "Execute: gcloud config set project SEU_PROJECT_ID"
    exit 1
fi

log_info "Projeto: $PROJECT_ID"

# Habilitar APIs necessárias
log_info "Habilitando APIs do Google Cloud..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build e deploy
log_info "Iniciando build e deploy..."
gcloud builds submit --config cloudbuild.yaml .

if [ $? -eq 0 ]; then
    log_info "✅ Deploy realizado com sucesso!"
    
    # Obter URL do serviço
    SERVICE_URL=$(gcloud run services describe tennis-analyzer --region=us-central1 --format="value(status.url)")
    
    echo ""
    log_info "🌐 Tennis Analyzer disponível em:"
    echo "   $SERVICE_URL"
    echo ""
    log_info "📊 Informações do deploy:"
    echo "   - Projeto: $PROJECT_ID"
    echo "   - Serviço: tennis-analyzer"
    echo "   - Região: us-central1"
    echo "   - Memória: 4GB"
    echo "   - CPU: 2 vCPUs"
    echo ""
    log_info "🔧 Comandos úteis:"
    echo "   - Logs: gcloud run services logs read tennis-analyzer --region=us-central1 --follow"
    echo "   - Atualizar: ./deploy.sh"
    echo "   - Deletar: gcloud run services delete tennis-analyzer --region=us-central1"
else
    log_error "❌ Falha no deploy!"
    log_info "Verifique os logs de build em: https://console.cloud.google.com/cloud-build"
    exit 1
fi