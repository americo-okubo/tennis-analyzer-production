#!/usr/bin/env python3
"""
Script para obter URL do ngrok
"""
import subprocess
import re
import time

def get_ngrok_url():
    try:
        # Executar comando para obter informações do ngrok
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
        
        # Verificar se port 4040 está ativo (interface web do ngrok)
        if ':4040' in result.stdout:
            print("Ngrok web interface detectada na porta 4040")
            print("\nPara obter a URL publica:")
            print("1. Abra http://localhost:4040 no navegador")
            print("2. Copie a URL que aparece (ex: https://abc123.ngrok.io)")
            print("3. Acesse: [URL]/web_interface.html")
            return True
        else:
            print("Ngrok web interface nao encontrada")
            return False
            
    except Exception as e:
        print(f"Erro: {e}")
        return False

if __name__ == "__main__":
    print("=== TENNIS ANALYZER - NGROK URL ===")
    print()
    
    # Verificar se ngrok está rodando
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        if 'ngrok.exe' in result.stdout:
            print("✓ Ngrok esta rodando")
            get_ngrok_url()
        else:
            print("✗ Ngrok nao esta rodando")
            print("Execute: run_ngrok.bat")
    except:
        print("Erro ao verificar processos")