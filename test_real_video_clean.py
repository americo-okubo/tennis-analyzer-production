#!/usr/bin/env python3
"""
Final test using real video files - no emojis
"""

import requests
import json
import os

API_BASE = 'http://localhost:8006'

def test_login():
    """Test login functionality"""
    print("[LOGIN] Testing login...")
    
    login_data = {
        "username": "demo",
        "password": "demo123"
    }
    
    response = requests.post(f"{API_BASE}/auth/login", json=login_data)
    
    if response.status_code == 200:
        data = response.json()
        token = data.get('access_token')
        print(f"[LOGIN] SUCCESS - Token: {token[:50]}...")
        return token
    else:
        print(f"[LOGIN] FAILED: {response.text}")
        return None

def test_forehand_video_with_correct_config(token):
    """Test Forehand video with CORRECT Forehand configuration - should PASS"""
    print("\n" + "="*80)
    print("[TEST1] Forehand video + Forehand config (should PASS)")
    print("="*80)
    
    video_path = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False
    
    # Configure for Forehand (matches the video content)
    metadata = {
        "maoDominante": "Destro",        # D - matches video
        "ladoCamera": "Direita",         # D - matches video  
        "ladoRaquete": "Forehand",       # F - matches video
        "tipoMovimento": "Drive (Ataque)" # D - matches video
    }
    
    print(f"[VIDEO] Zhang_Jike_FD_D_D.mp4 (Forehand Drive)")
    print(f"[CONFIG] {metadata}")
    print("[EXPECTED] APPROVED (video=Forehand, config=Forehand)")
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'file': ('Zhang_Jike_FD_D_D.mp4', video_file, 'video/mp4')}
            data = {'metadata': json.dumps(metadata)}
            headers = {'Authorization': f'Bearer {token}'}
            
            response = requests.post(
                f"{API_BASE}/validate-and-get-professionals", 
                files=files,
                data=data,
                headers=headers
            )
            
            print(f"[API] Response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                professionals = result.get('professionals', [])
                print(f"[SUCCESS] VALIDATION PASSED - Found {len(professionals)} professionals")
                print("[RESULT] CORRECT - Matching configuration was APPROVED")
                return True
            else:
                print(f"[FAILED] VALIDATION FAILED: {response.text}")
                print("[RESULT] ERROR - Matching configuration was incorrectly REJECTED")
                return False
                
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False

def test_forehand_video_with_wrong_config(token):
    """Test Forehand video with WRONG Backhand configuration - should FAIL"""
    print("\n" + "="*80)
    print("[TEST2] Forehand video + Backhand config (should FAIL)")
    print("="*80)
    
    video_path = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False
    
    # Configure for Backhand (WRONG - doesn't match video)
    metadata = {
        "maoDominante": "Destro",        # D - matches video
        "ladoCamera": "Direita",         # D - matches video
        "ladoRaquete": "Backhand",       # B - WRONG! Video is Forehand
        "tipoMovimento": "Drive (Ataque)" # D - matches video
    }
    
    print(f"[VIDEO] Zhang_Jike_FD_D_D.mp4 (Forehand Drive)")
    print(f"[CONFIG] {metadata}")
    print("[EXPECTED] REJECTED (video=Forehand, config=Backhand)")
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'file': ('Zhang_Jike_FD_D_D.mp4', video_file, 'video/mp4')}
            data = {'metadata': json.dumps(metadata)}
            headers = {'Authorization': f'Bearer {token}'}
            
            response = requests.post(
                f"{API_BASE}/validate-and-get-professionals", 
                files=files,
                data=data,
                headers=headers
            )
            
            print(f"[API] Response: {response.status_code}")
            
            if response.status_code in [400, 422]:
                print(f"[SUCCESS] VALIDATION CORRECTLY REJECTED: {response.text}")
                print("[RESULT] CORRECT - Mismatched configuration was REJECTED")
                return True
            elif response.status_code == 200:
                print("[FAILED] VALIDATION INCORRECTLY APPROVED mismatched configuration")
                print("[RESULT] ERROR - System should have REJECTED this")
                return False
            else:
                print(f"[UNKNOWN] Unexpected response: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False

def main():
    print("[START] REAL VIDEO VALIDATION TESTS")
    print("=" * 80)
    
    token = test_login()
    if not token:
        print("[ABORT] Cannot proceed without valid token")
        return False
    
    # Test both scenarios
    test1_passed = test_forehand_video_with_correct_config(token)
    test2_passed = test_forehand_video_with_wrong_config(token)
    
    # Final summary
    print("\n" + "="*80)
    print("[SUMMARY] FINAL TEST RESULTS")
    print("="*80)
    print(f"Test 1 (Correct Config): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Wrong Config):   {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n[SUCCESS] ALL TESTS PASSED! Video validation system is working correctly!")
        print("[OK] System correctly approves matching configurations")
        print("[OK] System correctly rejects mismatched configurations")
    else:
        print("\n[WARNING] Some tests failed - validation system needs review")
        
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)