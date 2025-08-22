#!/usr/bin/env python3
"""
Single Cycle Analysis - Implements user's requested approach:
Analyze only the second cycle from each video to avoid noise from first/last cycles
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tennis_comparison_backend import TableTennisAnalyzer

class SingleCycleAnalyzer:
    def __init__(self):
        self.analyzer = TableTennisAnalyzer()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_single_cycle(self, video_path, metadata, cycle_index=1):
        """
        Analyze only a specific cycle (default: second cycle, index=1)
        Returns biomechanical parameters extracted from that single cycle
        """
        try:
            self.logger.info(f"[SINGLE_CYCLE] Analyzing cycle {cycle_index} from {os.path.basename(video_path)}")
            
            # Use a dummy professional video just to get the user analysis
            dummy_pro = video_path  # Will use same video but we only want user_analysis
            
            # Get full analysis 
            full_result = self.analyzer.compare_techniques(video_path, dummy_pro, metadata, metadata)
            
            if not full_result['success']:
                return full_result
            
            # Extract user analysis data
            user_analysis = full_result.get('user_analysis', {})
            cycles_details = user_analysis.get('cycles_details', [])
            
            if len(cycles_details) <= cycle_index:
                return {
                    'success': False,
                    'error': f'Insufficient cycles. Found {len(cycles_details)}, need at least {cycle_index + 1}',
                    'cycles_found': len(cycles_details)
                }
            
            # Extract the specific cycle
            target_cycle = cycles_details[cycle_index]
            self.logger.info(f"[SINGLE_CYCLE] Selected cycle {cycle_index}: duration={target_cycle.get('duration'):.3f}s, amplitude={target_cycle.get('amplitude'):.1f}")
            
            # Calculate single-cycle biomechanical parameters
            single_cycle_params = self._extract_single_cycle_parameters(target_cycle, user_analysis)
            
            return {
                'success': True,
                'cycle_index': cycle_index,
                'cycle_data': target_cycle,
                'single_cycle_params': single_cycle_params,
                'total_cycles_found': len(cycles_details),
                'analysis_type': 'single_cycle'
            }
            
        except Exception as e:
            self.logger.error(f"[SINGLE_CYCLE] Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_single_cycle_parameters(self, cycle, full_result):
        """
        Extract biomechanical parameters that can be calculated from a single cycle
        """
        params = {}
        
        # Direct single-cycle measurements
        params['duration'] = cycle.get('duration', 0.0)
        params['amplitude'] = cycle.get('amplitude', 0.0)
        params['quality'] = cycle.get('quality', 0.0)
        
        # Single-cycle derived parameters
        if cycle.get('duration', 0) > 0:
            params['frequency'] = 1.0 / cycle.get('duration', 1.0)
            params['amplitude_per_second'] = cycle.get('amplitude', 0.0) / cycle.get('duration', 1.0)
        else:
            params['frequency'] = 0.0
            params['amplitude_per_second'] = 0.0
        
        # Smoothness approximation from single cycle (using quality as proxy)
        params['movement_smoothness'] = cycle.get('quality', 0.0)
        
        # Efficiency (amplitude vs duration ratio, normalized)
        duration = cycle.get('duration', 0.0)
        amplitude = cycle.get('amplitude', 0.0)
        if duration > 0:
            params['movement_efficiency'] = min(1.0, amplitude / (duration * 10000))  # Normalized
        else:
            params['movement_efficiency'] = 0.0
        
        # Note: Parameters requiring multi-cycle analysis are not included:
        # - rhythm_variability (needs multiple cycles)
        # - consistency_score (needs multiple cycles)
        # - acceleration_smoothness (can be estimated but not precise with single cycle)
        
        self.logger.info(f"[SINGLE_CYCLE] Extracted parameters: {list(params.keys())}")
        
        return params
    
    def compare_single_cycles(self, user_video, pro_video, user_metadata, prof_metadata, cycle_index=1):
        """
        Compare single cycles from user and professional videos
        """
        try:
            self.logger.info(f"[COMPARISON] Starting single-cycle comparison (cycle {cycle_index})")
            
            # Analyze both videos
            user_result = self.analyze_single_cycle(user_video, user_metadata, cycle_index)
            pro_result = self.analyze_single_cycle(pro_video, prof_metadata, cycle_index)
            
            if not user_result['success']:
                return {
                    'success': False,
                    'error': f"User video analysis failed: {user_result.get('error')}",
                    'user_cycles_found': user_result.get('cycles_found', 0)
                }
            
            if not pro_result['success']:
                return {
                    'success': False,
                    'error': f"Professional video analysis failed: {pro_result.get('error')}",
                    'pro_cycles_found': pro_result.get('cycles_found', 0)
                }
            
            # Compare parameters
            user_params = user_result['single_cycle_params']
            pro_params = pro_result['single_cycle_params']
            
            comparison = self._compare_parameters(user_params, pro_params)
            
            # Calculate overall similarity score
            similarity_score = np.mean(list(comparison['similarities'].values()))
            
            result = {
                'success': True,
                'analysis_type': 'single_cycle_comparison',
                'cycle_index': cycle_index,
                'user_analysis': {
                    'cycle_data': user_result['cycle_data'],
                    'parameters': user_params,
                    'total_cycles': user_result['total_cycles_found']
                },
                'professional_analysis': {
                    'cycle_data': pro_result['cycle_data'], 
                    'parameters': pro_params,
                    'total_cycles': pro_result['total_cycles_found']
                },
                'comparison': {
                    'similarity_score': similarity_score,
                    'detailed_similarities': comparison['similarities'],
                    'differences': comparison['differences']
                },
                'final_score': similarity_score * 100  # Convert to percentage
            }
            
            self.logger.info(f"[COMPARISON] Single-cycle similarity: {similarity_score:.3f} ({similarity_score*100:.1f}%)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[COMPARISON] Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _compare_parameters(self, user_params, pro_params):
        """
        Compare single-cycle parameters between user and professional
        """
        similarities = {}
        differences = {}
        
        # Compare each parameter
        for param in user_params:
            if param in pro_params:
                user_val = user_params[param]
                pro_val = pro_params[param]
                
                if pro_val != 0:
                    # Calculate similarity (1 - relative difference)
                    rel_diff = abs(user_val - pro_val) / max(abs(pro_val), 0.001)
                    similarity = max(0.0, 1.0 - rel_diff)
                else:
                    similarity = 1.0 if user_val == 0 else 0.0
                
                similarities[param] = similarity
                differences[param] = {
                    'user': user_val,
                    'professional': pro_val,
                    'difference': user_val - pro_val,
                    'relative_difference': rel_diff if pro_val != 0 else 0.0
                }
        
        return {
            'similarities': similarities,
            'differences': differences
        }

def test_single_cycle_analysis():
    """Test the single cycle analysis with different videos"""
    analyzer = SingleCycleAnalyzer()
    
    user_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Zhang_Jike_FD_D_D.mp4"
    pro_video = r"C:\Users\aokub\OneDrive\tennis-analyzer-production\videos\Fan_Zhendong_FD_D_E.mp4"
    
    user_metadata = {
        'maoDominante': 'Destro',
        'ladoCamera': 'Direita',
        'ladoRaquete': 'F',
        'tipoMovimento': 'D'
    }
    
    prof_metadata = {
        'maoDominante': 'D',
        'ladoCamera': 'D',
        'ladoRaquete': 'F',
        'tipoMovimento': 'D'
    }
    
    print("=== TESTING SINGLE CYCLE ANALYSIS ===")
    print(f"User video: {os.path.basename(user_video)}")
    print(f"Professional video: {os.path.basename(pro_video)}")
    print()
    
    # Test with second cycle (index=1)
    result = analyzer.compare_single_cycles(
        user_video, pro_video, user_metadata, prof_metadata, cycle_index=1
    )
    
    if result['success']:
        print(f"[SUCCESS] Analysis successful!")
        print(f"Analyzing cycle index: {result['cycle_index']}")
        print(f"Final similarity score: {result['final_score']:.1f}%")
        print()
        
        print("USER CYCLE:")
        user_cycle = result['user_analysis']['cycle_data']
        user_params = result['user_analysis']['parameters']
        print(f"  Duration: {user_cycle['duration']:.3f}s")
        print(f"  Amplitude: {user_cycle['amplitude']:.1f}")
        print(f"  Quality: {user_cycle['quality']:.3f}")
        print(f"  Movement efficiency: {user_params['movement_efficiency']:.3f}")
        print(f"  Total cycles found: {result['user_analysis']['total_cycles']}")
        print()
        
        print("PROFESSIONAL CYCLE:")
        pro_cycle = result['professional_analysis']['cycle_data']
        pro_params = result['professional_analysis']['parameters']
        print(f"  Duration: {pro_cycle['duration']:.3f}s")
        print(f"  Amplitude: {pro_cycle['amplitude']:.1f}")
        print(f"  Quality: {pro_cycle['quality']:.3f}")
        print(f"  Movement efficiency: {pro_params['movement_efficiency']:.3f}")
        print(f"  Total cycles found: {result['professional_analysis']['total_cycles']}")
        print()
        
        print("SIMILARITIES:")
        for param, similarity in result['comparison']['detailed_similarities'].items():
            print(f"  {param}: {similarity:.3f} ({similarity*100:.1f}%)")
        
    else:
        print(f"[ERROR] Analysis failed: {result.get('error')}")

if __name__ == "__main__":
    test_single_cycle_analysis()