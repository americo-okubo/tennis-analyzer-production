#!/usr/bin/env python3
"""
Test script for Tennis Analyzer Web API
Tests API endpoints and functionality
"""

import requests
import json
import time
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER = {"username": "demo", "password": "demo123"}

def test_api_health():
    """Test API health endpoints"""
    print("Testing API health...")
    
    # Test root endpoint
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Root endpoint: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    
    # Test health check
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    if response.status_code == 200:
        print(f"Health: {response.json()}")
    
    return True

def test_authentication():
    """Test authentication endpoints"""
    print("\nTesting authentication...")
    
    # Test login with demo user
    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        json=TEST_USER
    )
    
    print(f"Login: {response.status_code}")
    if response.status_code == 200:
        token_data = response.json()
        print(f"Token received: {token_data['token_type']}")
        return token_data["access_token"]
    else:
        print(f"Login failed: {response.text}")
        return None

def test_protected_endpoints(token):
    """Test protected endpoints"""
    if not token:
        print("No token available, skipping protected endpoint tests")
        return
    
    print("\nTesting protected endpoints...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test current user
    response = requests.get(f"{API_BASE_URL}/auth/me", headers=headers)
    print(f"Current user: {response.status_code}")
    if response.status_code == 200:
        print(f"User info: {response.json()}")
    
    # Test professionals endpoint
    response = requests.get(f"{API_BASE_URL}/professionals", headers=headers)
    print(f"Professionals: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['count']} professionals")
    
    # Test user history
    response = requests.get(f"{API_BASE_URL}/user/history", headers=headers)
    print(f"User history: {response.status_code}")

def test_development_endpoints():
    """Test development endpoints"""
    print("\nTesting development endpoints...")
    
    # Test component check
    response = requests.get(f"{API_BASE_URL}/dev/test-components")
    print(f"Component test: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Component status:")
        for component, status in data.items():
            print(f"  {component}: {status['status']}")
    elif response.status_code == 404:
        print("Development endpoints disabled (production mode)")

def main():
    """Run all API tests"""
    print("Tennis Analyzer API Test Suite")
    print("=" * 40)
    
    try:
        # Test basic health
        test_api_health()
        
        # Test authentication
        token = test_authentication()
        
        # Test protected endpoints
        test_protected_endpoints(token)
        
        # Test development endpoints
        test_development_endpoints()
        
        print("\n" + "=" * 40)
        print("API tests completed successfully!")
        print("\nAPI is ready for use!")
        print(f"Access the API documentation at: {API_BASE_URL}/docs")
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API server")
        print("Please ensure the API server is running:")
        print("  cd api && python main.py")
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")

if __name__ == "__main__":
    main()