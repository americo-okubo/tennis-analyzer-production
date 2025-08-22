#!/usr/bin/env python3
"""
Script para iniciar ngrok e obter URL pública
"""
import subprocess
import time
import requests
import json

def start_ngrok():
    print("Iniciando ngrok...")
    
    # Iniciar ngrok em background
    process = subprocess.Popen([
        "./ngrok.exe", "http", "8000"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("Aguardando ngrok inicializar...")
    time.sleep(8)
    
    try:
        # Obter informações do tunnel
        response = requests.get("http://localhost:4040/api/tunnels")
        data = response.json()
        
        if data.get("tunnels"):
            tunnel = data["tunnels"][0]
            public_url = tunnel["public_url"]
            print(f"Ngrok rodando!")
            print(f"URL Publica: {public_url}")
            print(f"Acesse no mobile: {public_url}/web_interface.html")
            print(f"API Docs: {public_url}/docs")
            return public_url
        else:
            print("Nenhum tunnel encontrado")
            return None
            
    except Exception as e:
        print(f"Erro ao obter URL do ngrok: {e}")
        return None

if __name__ == "__main__":
    start_ngrok()
    print("\nIMPORTANTE: Mantenha este terminal aberto para manter o tunel ativo!")
    print("Para parar: Ctrl+C")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nParando ngrok...")