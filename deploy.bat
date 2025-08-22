@echo off
REM Script de deploy para Google Cloud Platform (Windows)
REM Tennis Analyzer - Deploy automatizado

echo 🎾 Tennis Analyzer - Deploy para GCP
echo ====================================

REM Verificar se gcloud está instalado
gcloud version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Google Cloud SDK não encontrado!
    echo 📥 Instale em: https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Verificar autenticação
for /f %%i in ('gcloud auth list --filter=status:ACTIVE --format="value(account)"') do set ACCOUNT=%%i
if "%ACCOUNT%"=="" (
    echo ❌ Não autenticado com Google Cloud!
    echo 🔑 Execute: gcloud auth login
    exit /b 1
)

REM Obter projeto atual
for /f %%i in ('gcloud config get-value project') do set PROJECT_ID=%%i
if "%PROJECT_ID%"=="" (
    echo ❌ Nenhum projeto configurado!
    echo 🔧 Execute: gcloud config set project SEU_PROJECT_ID
    exit /b 1
)

echo ✅ Projeto: %PROJECT_ID%

REM Habilitar APIs necessárias
echo 📋 Habilitando APIs do Google Cloud...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

REM Build e deploy
echo 🚀 Iniciando build e deploy...
gcloud builds submit --config cloudbuild.yaml .

if %errorlevel% equ 0 (
    echo.
    echo ✅ Deploy realizado com sucesso!
    
    REM Obter URL do serviço
    for /f %%i in ('gcloud run services describe tennis-analyzer --region=us-central1 --format="value(status.url)"') do set SERVICE_URL=%%i
    
    echo.
    echo 🌐 Tennis Analyzer disponível em:
    echo    %SERVICE_URL%
    echo.
    echo 📊 Informações do deploy:
    echo    - Projeto: %PROJECT_ID%
    echo    - Serviço: tennis-analyzer
    echo    - Região: us-central1
    echo    - Memória: 4GB
    echo    - CPU: 2 vCPUs
    echo.
    echo 🔧 Comandos úteis:
    echo    - Logs: gcloud run services logs read tennis-analyzer --region=us-central1 --follow
    echo    - Atualizar: deploy.bat
    echo    - Deletar: gcloud run services delete tennis-analyzer --region=us-central1
) else (
    echo ❌ Falha no deploy!
    echo 🔍 Verifique os logs em: https://console.cloud.google.com/cloud-build
    exit /b 1
)

pause