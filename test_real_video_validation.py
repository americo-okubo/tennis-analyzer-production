#!/usr/bin/env python3
"""
Final test using real video files
"""

import requests
import json
import os

API_BASE = 'http://localhost:8006'

def test_login():
    """Test login functionality"""
    print("[LOGIN] TESTING LOGIN")
    
    login_data = {
        "username": "demo",
        "password": "demo123"
    }
    
    response = requests.post(f"{API_BASE}/auth/login", json=login_data)
    
    if response.status_code == 200:
        data = response.json()
        token = data.get('access_token')
        print(f"[OK] Login successful - Token: {token[:50]}...")
        return token
    else:
        print(f"[ERROR] Login failed: {response.text}")
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
    
    print(f"📹 Video: Zhang_Jike_FD_D_D.mp4 (Forehand Drive)")
    print(f"⚙️  Config: {metadata}")
    print("📝 Expected: APPROVED (video=Forehand, config=Forehand)")
    
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
            
            print(f"🔍 API Response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                professionals = result.get('professionals', [])
                print(f"✅ VALIDATION PASSED - Found {len(professionals)} professionals")
                print("🎯 RESULT: CORRECT - Matching configuration was APPROVED")
                return True
            else:
                print(f"❌ VALIDATION FAILED: {response.text}")
                print("🚨 RESULT: ERROR - Matching configuration was incorrectly REJECTED")
                return False
                
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def test_forehand_video_with_wrong_config(token):
    """Test Forehand video with WRONG Backhand configuration - should FAIL"""
    print("\n" + "="*80)
    print("🎾 TEST 2: Forehand video + Backhand config (should FAIL)")
    print("="*80)
    
    video_path = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Configure for Backhand (WRONG - doesn't match video)
    metadata = {
        "maoDominante": "Destro",        # D - matches video
        "ladoCamera": "Direita",         # D - matches video
        "ladoRaquete": "Backhand",       # B - WRONG! Video is Forehand
        "tipoMovimento": "Drive (Ataque)" # D - matches video
    }
    
    print(f"📹 Video: Zhang_Jike_FD_D_D.mp4 (Forehand Drive)")
    print(f"⚙️  Config: {metadata}")
    print("📝 Expected: REJECTED (video=Forehand, config=Backhand)")
    
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
            
            print(f"🔍 API Response: {response.status_code}")
            
            if response.status_code in [400, 422]:
                print(f"✅ VALIDATION CORRECTLY REJECTED: {response.text}")
                print("🎯 RESULT: CORRECT - Mismatched configuration was REJECTED")
                return True
            elif response.status_code == 200:
                print("❌ VALIDATION INCORRECTLY APPROVED mismatched configuration")
                print("🚨 RESULT: ERROR - System should have REJECTED this")
                return False
            else:
                print(f"❓ Unexpected response: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def main():
    print("🚀 STARTING REAL VIDEO VALIDATION TESTS")
    print("=" * 80)
    
    token = test_login()
    if not token:
        print("💥 Cannot proceed without valid token")
        return
    
    # Test both scenarios
    test1_passed = test_forehand_video_with_correct_config(token)
    test2_passed = test_forehand_video_with_wrong_config(token)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL TEST SUMMARY")
    print("="*80)
    print(f"Test 1 (Correct Config): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Wrong Config):   {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Video validation system is working correctly!")
        print("✅ System correctly approves matching configurations")
        print("✅ System correctly rejects mismatched configurations")
    else:
        print("\n⚠️  Some tests failed - validation system needs review")
        
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)