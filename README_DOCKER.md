# Tennis Analyzer - Deploy com Docker no GCP

Este guia mostra como fazer deploy do Tennis Analyzer no Google Cloud Platform usando Docker com **100% de funcionalidade**.

## 🚀 Quick Start

### 1. Pré-requisitos
```bash
# Instalar Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Autenticar
gcloud auth login

# Configurar projeto
gcloud config set project SEU_PROJECT_ID
```

### 2. Deploy Automático
```bash
# Windows
deploy.bat

# Linux/Mac
chmod +x deploy.sh
./deploy.sh
```

## 🐳 Como Funciona o Docker

### Dockerfile Otimizado
```dockerfile
# Imagem base Python 3.11
FROM python:3.11-slim

# Instalar dependências do sistema para MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Configurar ambiente headless
ENV DISPLAY=:99
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV MPLBACKEND=Agg

# Instalar dependências Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiar código
COPY . /app
WORKDIR /app

# Executar
CMD ["python", "api/main.py"]
```

## 🎯 Vantagens desta Solução

### ✅ **Funcionalidade Completa**
- **MediaPipe 100% funcional** no GCP
- **Análise biomecânica completa**
- **Detecção de poses em tempo real**
- **Comparação com profissionais**

### ✅ **Escalabilidade**
- **Auto-scaling** baseado na demanda
- **Load balancing** automático
- **Alta disponibilidade**

### ✅ **Facilidade de Deploy**
- **Um comando** para deploy completo
- **Updates automáticos** via Git
- **Rollback** fácil se necessário

## 🔧 Comandos Úteis

### Build Local
```bash
# Testar Docker localmente
docker build -t tennis-analyzer .
docker run -p 8080:8080 tennis-analyzer
```

### Deploy Manual
```bash
# Build e push para GCP
gcloud builds submit --tag gcr.io/SEU_PROJECT_ID/tennis-analyzer

# Deploy para Cloud Run
gcloud run deploy tennis-analyzer \
  --image gcr.io/SEU_PROJECT_ID/tennis-analyzer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Monitoramento
```bash
# Ver logs em tempo real
gcloud run services logs tail tennis-analyzer --region=us-central1

# Verificar status
gcloud run services describe tennis-analyzer --region=us-central1
```

## 📊 Configurações de Produção

### Recursos Alocados
- **Memória**: 4GB RAM
- **CPU**: 2 vCPUs
- **Timeout**: 15 minutos
- **Concorrência**: 100 requests simultâneas

### Variáveis de Ambiente
```bash
ENVIRONMENT=production
MEDIAPIPE_ENABLED=true
PORT=8080
```

## 🏗️ Arquitetura

```
Internet → Load Balancer → Cloud Run → Container (Docker)
                                    ├── MediaPipe
                                    ├── OpenCV
                                    ├── FastAPI
                                    └── Tennis Analyzer
```

## 💰 Custos Estimados

**Cloud Run** (pay-per-use):
- **CPU**: $0.000024 por vCPU/segundo
- **Memória**: $0.0000025 por GB/segundo
- **Requests**: $0.0000004 por request

**Exemplo mensal** (1000 análises):
- ~$10-30 dependendo do uso

## 🛠️ Troubleshooting

### Problema: Build falha
```bash
# Verificar logs
gcloud builds log BUILD_ID

# Verificar quota
gcloud compute project-info describe --project=SEU_PROJECT_ID
```

### Problema: Deploy falha
```bash
# Verificar serviço
gcloud run services list

# Verificar região
gcloud run regions list
```

### Problema: MediaPipe não funciona
- ✅ **Verificar**: Dockerfile tem todas as dependências
- ✅ **Verificar**: Variáveis de ambiente configuradas
- ✅ **Verificar**: Região do GCP suporta GPU

## 🔄 Atualizações

### Deploy Nova Versão
```bash
# Commit mudanças
git add .
git commit -m "Nova funcionalidade"

# Deploy automático
./deploy.sh
```

### Rollback
```bash
# Voltar para versão anterior
gcloud run services update tennis-analyzer \
  --image gcr.io/SEU_PROJECT_ID/tennis-analyzer:COMMIT_SHA_ANTERIOR \
  --region us-central1
```

## 📱 Próximos Passos

1. **✅ Deploy básico funcionando**
2. **📱 Adicionar gravação por câmera**
3. **📊 Adicionar analytics**
4. **🔐 Adicionar autenticação**
5. **💾 Adicionar banco de dados**

---

**Este setup garante que seu Tennis Analyzer funcione 100% no GCP com Docker! 🚀**