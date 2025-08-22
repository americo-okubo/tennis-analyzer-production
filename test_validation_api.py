#!/usr/bin/env python3
"""
Test script for validation API
"""

import requests
import json

API_BASE = 'http://localhost:8006'

def test_login():
    """Test login functionality"""
    print("=== TESTING LOGIN ===")
    
    login_data = {
        "username": "demo",
        "password": "demo123"
    }
    
    response = requests.post(f"{API_BASE}/auth/login", json=login_data)
    print(f"Login status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        token = data.get('access_token')
        print(f"Token obtained: {token[:50]}...")
        return token
    else:
        print(f"Login failed: {response.text}")
        return None

def test_validation_with_forehand_config(token):
    """Test validation with Forehand configuration"""
    print("\n=== TESTING VALIDATION - FOREHAND CONFIG ===")
    
    # Simular configuração de Forehand
    metadata = {
        "maoDominante": "Destro",  # Will be normalized to "D"
        "ladoCamera": "Direita",   # Will be normalized to "D" 
        "ladoRaquete": "Forehand", # Will be normalized to "F"
        "tipoMovimento": "Drive (Ataque)" # Will be normalized to "D"
    }
    
    print(f"Sending metadata: {metadata}")
    
    # Create a dummy file for testing
    files = {'file': ('test.mp4', b'dummy content', 'video/mp4')}
    data = {'metadata': json.dumps(metadata)}
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.post(
            f"{API_BASE}/validate-and-get-professionals", 
            files=files,
            data=data,
            headers=headers
        )
        
        print(f"Validation status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            professionals = result.get('professionals', [])
            print(f"Found {len(professionals)} professionals")
            return True
        else:
            print(f"Validation failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_validation_with_backhand_config(token):
    """Test validation with Backhand configuration (should fail if video is Forehand)"""
    print("\n=== TESTING VALIDATION - BACKHAND CONFIG ===")
    
    # Simular configuração de Backhand
    metadata = {
        "maoDominante": "Destro",     # Will be normalized to "D"
        "ladoCamera": "Direita",      # Will be normalized to "D" 
        "ladoRaquete": "Backhand",    # Will be normalized to "B"
        "tipoMovimento": "Drive (Ataque)" # Will be normalized to "D"
    }
    
    print(f"Sending metadata: {metadata}")
    
    # Create a dummy file for testing
    files = {'file': ('test.mp4', b'dummy content', 'video/mp4')}
    data = {'metadata': json.dumps(metadata)}
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.post(
            f"{API_BASE}/validate-and-get-professionals", 
            files=files,
            data=data,
            headers=headers
        )
        
        print(f"Validation status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 422:
            print("✅ VALIDATION CORRECTLY REJECTED MISMATCHED CONFIGURATION")
            return True
        elif response.status_code == 200:
            print("⚠️ VALIDATION INCORRECTLY ACCEPTED MISMATCHED CONFIGURATION")
            return False
        else:
            print(f"Unexpected response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    token = test_login()
    if token:
        print("\n" + "="*60)
        test_validation_with_forehand_config(token)
        
        print("\n" + "="*60)
        test_validation_with_backhand_config(token)
    else:
        print("Cannot proceed without valid token")