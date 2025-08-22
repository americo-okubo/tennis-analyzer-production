#!/usr/bin/env python3
"""
Alternativa para acesso público sem ngrok
"""
import subprocess
import sys

def show_options():
    print("=== TENNIS ANALYZER - ACESSO PUBLICO ===")
    print()
    print("OPCOES DISPONIVEIS:")
    print()
    print("1. NGROK (Recomendado)")
    print("   - Precisa criar conta gratuita")
    print("   - Mais estavel e confiavel")
    print("   - Site: https://dashboard.ngrok.com/signup")
    print()
    print("2. ALTERNATIVAS SEM CADASTRO:")
    print("   - Localtunnel: npm install -g localtunnel")
    print("   - Cloudflared: Cloudflare tunnel")
    print()
    print("3. OPCAO TEMPORARIA:")
    print("   - Use hotspot do celular no PC")
    print("   - Conecte ambos no mesmo hotspot")
    print("   - Acesse pelo IP do hotspot")
    print()
    
    print("RECOMENDACAO:")
    print("1. Crie conta gratuita no ngrok")
    print("2. Configure o authtoken")
    print("3. Execute: ./ngrok.exe http 8000")
    print()
    print("OU use o hotspot como solucao imediata!")

if __name__ == "__main__":
    show_options()