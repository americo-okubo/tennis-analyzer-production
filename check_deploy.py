#!/usr/bin/env python3
"""
Script de verificação antes do deploy
Verifica se todos os arquivos necessários estão presentes e corretos
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Verifica se um arquivo existe"""
    if os.path.exists(file_path):
        print(f"OK {description}: {file_path}")
        return True
    else:
        print(f"ERRO {description}: {file_path} - FALTANDO")
        return False

def check_requirements():
    """Verifica o arquivo requirements.txt"""
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"❌ {req_file} não encontrado")
        return False
    
    with open(req_file, 'r') as f:
        content = f.read()
        essential_packages = ['fastapi', 'uvicorn', 'opencv-python', 'mediapipe', 'numpy']
        missing = []
        
        for package in essential_packages:
            if package not in content:
                missing.append(package)
        
        if missing:
            print(f"ERRO Pacotes essenciais faltando no requirements.txt: {missing}")
            return False
        else:
            print("OK requirements.txt contém todos os pacotes essenciais")
            return True

def check_procfile():
    """Verifica o Procfile"""
    if not os.path.exists("Procfile"):
        print("ERRO Procfile não encontrado")
        return False
    
    with open("Procfile", 'r') as f:
        content = f.read().strip()
        if "uvicorn api.main:app" in content and "--host 0.0.0.0" in content and "--port $PORT" in content:
            print("OK Procfile está correto")
            return True
        else:
            print(f"ERRO Procfile incorreto: {content}")
            return False

def check_main_api():
    """Verifica se o arquivo principal da API existe"""
    api_main = "api/main.py"
    if not os.path.exists(api_main):
        print(f"ERRO {api_main} não encontrado")
        return False
    
    # Verifica se não há caminhos absolutos problemáticos
    with open(api_main, 'r', encoding='utf-8') as f:
        content = f.read()
        if "C:\\Users" in content:
            print("AVISO: Possíveis caminhos absolutos encontrados em api/main.py")
            return False
        else:
            print("OK api/main.py sem caminhos absolutos problemáticos")
            return True

def main():
    """Função principal de verificação"""
    print("Verificando arquivos para deploy no Railway...\n")
    
    checks = []
    
    # Verificar arquivos essenciais
    checks.append(check_file_exists("requirements.txt", "Requirements"))
    checks.append(check_file_exists("Procfile", "Procfile"))
    checks.append(check_file_exists("railway.json", "Railway config"))
    checks.append(check_file_exists(".env.example", "Environment example"))
    checks.append(check_file_exists("api/main.py", "Main API file"))
    checks.append(check_file_exists("web_interface.html", "Web interface"))
    checks.append(check_file_exists("professionals_biomech_data.json", "Professionals database"))
    
    # Verificações específicas
    checks.append(check_requirements())
    checks.append(check_procfile())
    checks.append(check_main_api())
    
    print(f"\nResultado: {sum(checks)}/{len(checks)} verificações passaram")
    
    if all(checks):
        print("\nProjeto pronto para deploy no Railway!")
        print("\nPróximos passos:")
        print("1. git add .")
        print("2. git commit -m 'Preparação para deploy Railway'")
        print("3. git push origin master")
        print("4. Configurar projeto no Railway")
        return True
    else:
        print("\nProjeto não está pronto para deploy. Corrija os problemas acima.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)