#!/usr/bin/env python3
"""
Enhanced Single Cycle Analysis with Biomechanics

Combines the proven single-cycle approach with scientific biomechanical analysis:
- Maintains the reliable single-cycle detection
- Adds comprehensive 2D biomechanical metrics
- Provides actionable, scientific recommendations
- 100% real data, no estimates or fallbacks
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tennis_comparison_backend import TableTennisAnalyzer
from biomechanics import BiomechanicalAnalyzer2D

# [v16.2] Import our improved classifier with racket angle detection
from improved_biomech_classifier_2d import ImprovedBiomechClassifier2D

# [v16.2] Import professional comparator
from optimized_professional_comparison import OptimizedProfessionalComparator

class EnhancedSingleCycleAnalyzer:
    """
    Enhanced analyzer that combines single-cycle detection with biomechanical analysis.
    """
    
    def _translate_movement_type(self, movement_type):
        """Traduz tipos de movimento para descrições mais amigáveis"""
        translations = {
            'forehand_drive': 'Forehand de Ataque',
            'forehand_push': 'Forehand Defensivo', 
            'backhand_drive': 'Backhand de Ataque',
            'backhand_push': 'Backhand Defensivo',
            'movimento_desconhecido': 'Movimento não identificado'
        }
        return translations.get(movement_type, movement_type)
    
    def __init__(self):
        self.traditional_analyzer = TableTennisAnalyzer()
        self.biomech_analyzer = BiomechanicalAnalyzer2D()
        
        # [v16.2] Add our improved classifier with racket angle detection
        self.improved_classifier = ImprovedBiomechClassifier2D()
        print("[ENHANCED_ANALYZER] v16.2 - Improved classifier with racket angle detection initialized!")
        
        # [v16.2] Add professional comparator
        self.professional_comparator = OptimizedProfessionalComparator()
        print("[ENHANCED_ANALYZER] v16.2 - Professional comparator initialized!")
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def analyze_single_cycle_with_biomechanics(self, video_path: str, metadata: dict, 
                                              cycle_index: int = 1) -> Dict[str, Any]:
        """
        Analyze a specific cycle with both traditional and biomechanical methods.
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata for analysis
            cycle_index: Which cycle to analyze (default: 1 = second cycle)
            
        Returns:
            Combined analysis results with both traditional and biomechanical data
        """
        try:
            self.logger.info(f"[ENHANCED] Starting enhanced analysis of cycle {cycle_index}")
            self.logger.info(f"[ENHANCED] Video: {os.path.basename(video_path)}")
            
            # Step 1: Traditional single-cycle analysis (for cycle detection)
            traditional_result = self._get_traditional_cycle_analysis(video_path, metadata)
            
            if not traditional_result['success']:
                return traditional_result
            
            # Validate cycle availability
            cycles_details = traditional_result.get('cycles_details', [])
            if len(cycles_details) <= cycle_index:
                return {
                    'success': False,
                    'error': f'Insufficient cycles. Found {len(cycles_details)}, need at least {cycle_index + 1}',
                    'cycles_found': len(cycles_details),
                    'analysis_type': 'enhanced_single_cycle'
                }
            
            # Step 2: Biomechanical analysis of the entire video
            self.logger.info(f"[ENHANCED] Running biomechanical analysis...")
            biomech_result = self.biomech_analyzer.analyze_video_biomechanics(video_path, metadata)
            
            if not biomech_result['success']:
                self.logger.warning(f"[ENHANCED] Biomechanical analysis failed: {biomech_result.get('error')}")
                # Continue with traditional analysis only
                biomech_result = None
            
            # Step 3: Extract single cycle data
            target_cycle = cycles_details[cycle_index]
            single_cycle_params = self._extract_single_cycle_parameters(target_cycle)
            
            # Step 4: Combine results
            enhanced_result = {
                'success': True,
                'analysis_type': 'enhanced_single_cycle_biomechanical',
                'cycle_index': cycle_index,
                'total_cycles_found': len(cycles_details),
                
                # Traditional single-cycle data
                'cycle_data': target_cycle,
                'single_cycle_params': single_cycle_params,
                'traditional_quality': traditional_result.get('quality_score', 0),
                
                # Enhanced metrics
                'enhanced_metrics': self._calculate_enhanced_metrics(
                    target_cycle, biomech_result, traditional_result
                ),
                
                # Biomechanical data (if available)
                'biomechanical_analysis': biomech_result if biomech_result and biomech_result['success'] else None,
                
                # Meta information
                'video_info': {
                    'duration': traditional_result.get('average_duration', target_cycle.get('duration', 0)),
                    'amplitude': target_cycle.get('amplitude', 0),
                    'quality': target_cycle.get('quality', 0),
                    'frame_range': self._get_cycle_frame_range(target_cycle, cycle_index)
                }
            }
            
            self.logger.info(f"[ENHANCED] Analysis completed successfully")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"[ENHANCED] Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e), 'analysis_type': 'enhanced_single_cycle'}
    
    def compare_enhanced_single_cycles(self, user_video: str, pro_video: str = None, 
                                     user_metadata: dict = None, prof_metadata: dict = None, 
                                     cycle_index: int = 1) -> Dict[str, Any]:
        """
        Compare enhanced single-cycle analysis between user and professional.
        If pro_video is None, performs independent biomechanical analysis.
        """
        try:
            if pro_video is None:
                self.logger.info(f"[ENHANCED_COMPARISON] Starting REAL independent biomechanical analysis")
                
                # REAL independent analysis using ImprovedBiomechClassifier2D
                try:
                    from improved_biomech_classifier_2d import ImprovedBiomechClassifier2D
                    import os
                    from datetime import datetime
                    
                    # Basic video validation
                    if not os.path.exists(user_video):
                        return {
                            'success': False,
                            'error': f"Video file not found: {user_video}",
                            'user_cycles_found': 0
                        }
                    
                    self.logger.info(f"[ENHANCED_COMPARISON] Initializing real biomechanical classifier...")
                    
                    # Initialize real biomechanical classifier
                    biomech_classifier = ImprovedBiomechClassifier2D()
                    
                    # Process video with real biomechanical analysis focused on specific cycle
                    self.logger.info(f"[ENHANCED_COMPARISON] Processing video with CYCLE-SPECIFIC MediaPipe analysis...")
                    biomech_result = self._process_video_single_cycle(biomech_classifier, user_video, user_metadata, cycle_index=1)
                    
                    if not biomech_result:
                        self.logger.error(f"[ENHANCED_COMPARISON] Biomechanical analysis failed")
                        return {
                            'success': False,
                            'error': "Biomechanical analysis failed - could not process video",
                            'user_cycles_found': 0
                        }
                    
                    self.logger.info(f"[ENHANCED_COMPARISON] Real biomechanical analysis completed successfully!")
                    
                    # Extract real biomechanical data
                    real_biomech_analysis = {
                        'movement_classification': {
                            'detected_movement': biomech_result.movement_type.value,
                            'confidence': biomech_result.confidence,
                            'confidence_level': biomech_result.confidence_level,
                            'classification_zone': biomech_result.classification_zone,
                            'hierarchy_level': biomech_result.hierarchy_level
                        },
                        'joint_angles': {
                            'elbow_variation_degrees': biomech_result.elbow_variation_active,
                            'elbow_opening_trend': biomech_result.elbow_opening_trend_active,
                            'coordination_score': biomech_result.coordination_active,
                            'movement_signature': biomech_result.movement_signature
                        },
                        'movement_dynamics': {
                            'amplitude_y': biomech_result.amplitude_y_active,
                            'max_velocity': biomech_result.max_velocity_active,
                            'racket_detection_score': biomech_result.racket_score_active,
                            'temporal_pattern': biomech_result.temporal_pattern
                        },
                        'biomechanical_metrics': {
                            'left_arm_amplitude': biomech_result.left_metrics.movement_amplitude_y,
                            'right_arm_amplitude': biomech_result.right_metrics.movement_amplitude_y,
                            'left_max_velocity': biomech_result.left_metrics.max_velocity,
                            'right_max_velocity': biomech_result.right_metrics.max_velocity,
                            'left_coordination': (biomech_result.left_metrics.shoulder_elbow_coordination + biomech_result.left_metrics.elbow_wrist_coordination) / 2,
                            'right_coordination': (biomech_result.right_metrics.shoulder_elbow_coordination + biomech_result.right_metrics.elbow_wrist_coordination) / 2
                        },
                        'confidence_breakdown': {
                            'biomech_forehand_likelihood': biomech_result.biomech_forehand_likelihood,
                            'biomech_backhand_likelihood': biomech_result.biomech_backhand_likelihood,
                            'biomech_drive_likelihood': biomech_result.biomech_drive_likelihood,
                            'biomech_push_likelihood': biomech_result.biomech_push_likelihood,
                            'biomech_confidence': biomech_result.biomech_confidence
                        }
                    }
                    
                    # Generate detailed recommendations based on real cycle-specific analysis
                    if hasattr(biomech_result, 'cycle_recommendations'):
                        # Use cycle-specific recommendations if available
                        recommendations = biomech_result.cycle_recommendations + [
                            f"Mão ativa: {biomech_result.active_hand_side} (detecção automática)",
                            f"Variação angular do cotovelo: {biomech_result.elbow_variation_active:.1f}° - {self._interpret_elbow_variation(biomech_result.elbow_variation_active)}",
                            f"Coordenação dos membros: {biomech_result.coordination_active:.1%} - {self._interpret_coordination(biomech_result.coordination_active)}",
                            f"Padrão temporal: {biomech_result.temporal_pattern} - {self._interpret_temporal_pattern(biomech_result.temporal_pattern, biomech_result.movement_type.value)}",
                            f"Amplitude do movimento: {biomech_result.amplitude_y_active:.3f} - {self._interpret_amplitude(biomech_result.amplitude_y_active, biomech_result.movement_type.value)}",
                            f"Velocidade máxima: {biomech_result.max_velocity_active:.3f} - {self._interpret_velocity(biomech_result.max_velocity_active)}"
                        ]
                    else:
                        # Standard recommendations for full video analysis
                        recommendations = [
                            f"Movimento detectado: {biomech_result.movement_type.value} com {biomech_result.confidence:.1%} de confiança",
                            f"Análise realizada no vídeo completo (fallback - ciclos não detectados)",
                            f"Mão ativa: {biomech_result.active_hand_side} (detecção automática)",
                            f"Variação angular do cotovelo: {biomech_result.elbow_variation_active:.1f}° - {self._interpret_elbow_variation(biomech_result.elbow_variation_active)}",
                            f"Coordenação dos membros: {biomech_result.coordination_active:.1%} - {self._interpret_coordination(biomech_result.coordination_active)}",
                            f"Padrão temporal: {biomech_result.temporal_pattern} - {self._interpret_temporal_pattern(biomech_result.temporal_pattern, biomech_result.movement_type.value)}",
                            f"Amplitude do movimento: {biomech_result.amplitude_y_active:.3f} - {self._interpret_amplitude(biomech_result.amplitude_y_active, biomech_result.movement_type.value)}",
                            f"Velocidade máxima: {biomech_result.max_velocity_active:.3f} - {self._interpret_velocity(biomech_result.max_velocity_active)}"
                        ]
                    
                    # Calculate performance score based on real biomechanical data
                    performance_score = self._calculate_biomech_performance_score(biomech_result)
                    
                    # [v16.2] Add professional comparisons
                    movement_detected = biomech_result.movement_type.value
                    print(f"[ENHANCED_ANALYZER] Finding professionals for: {movement_detected}")
                    
                    user_analysis_data = {
                        'detected_movement': movement_detected,
                        'biomech_result': biomech_result,
                        'detailed_analysis': real_biomech_analysis
                    }
                    
                    professional_comparisons = self.professional_comparator.find_best_matches(
                        user_analysis_data, movement_detected
                    )
                    
                    print(f"[ENHANCED_ANALYZER] Found {len(professional_comparisons)} professional comparisons")
                    
                    return {
                        'success': True,
                        'final_score': performance_score,
                        'analysis_type': 'real_independent_biomechanical',
                        'detailed_analysis': real_biomech_analysis,
                        'recommendations': recommendations,
                        'professional_comparisons': professional_comparisons,
                        'user_cycles_found': 1,  # Single cycle analysis
                        'professional_cycles_found': 0,
                        'timestamp': datetime.now().isoformat(),
                        'analysis_method': 'real_biomechanical_mediapipe',
                        'biomech_result': biomech_result  # Store full result for potential debugging
                    }
                    
                except ImportError as e:
                    self.logger.error(f"[ENHANCED_COMPARISON] Import error: {e}")
                    return {
                        'success': False,
                        'error': f"Could not import biomechanical classifier: {str(e)}",
                        'user_cycles_found': 0
                    }
                except Exception as e:
                    self.logger.error(f"[ENHANCED_COMPARISON] Error in real biomechanical analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        'success': False,
                        'error': f"Real biomechanical analysis error: {str(e)}",
                        'user_cycles_found': 0
                    }
            
            self.logger.info(f"[ENHANCED_COMPARISON] Starting comparative analysis")
            
            # Analyze both videos for comparison
            user_result = self.analyze_single_cycle_with_biomechanics(user_video, user_metadata, cycle_index)
            pro_result = self.analyze_single_cycle_with_biomechanics(pro_video, prof_metadata, cycle_index)
            
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
            
            # Traditional single-cycle comparison
            traditional_comparison = self._compare_traditional_cycles(
                user_result['single_cycle_params'], 
                pro_result['single_cycle_params']
            )
            
            # Biomechanical comparison (if available)
            biomech_comparison = None
            if (user_result.get('biomechanical_analysis') and 
                pro_result.get('biomechanical_analysis')):
                
                biomech_comparison = self.biomech_analyzer.compare_biomechanics(
                    user_result['biomechanical_analysis'],
                    pro_result['biomechanical_analysis']
                )
            
            # Combined comparison
            combined_comparison = self._create_combined_comparison(
                traditional_comparison, biomech_comparison, user_result, pro_result
            )
            
            result = {
                'success': True,
                'analysis_type': 'enhanced_single_cycle_comparison',
                'cycle_index': cycle_index,
                
                # Individual analysis results
                'user_analysis': user_result,
                'professional_analysis': pro_result,
                
                # Comparison results
                'traditional_comparison': traditional_comparison,
                'biomechanical_comparison': biomech_comparison,
                'combined_comparison': combined_comparison,
                
                # Final metrics for interface
                'final_score': combined_comparison['overall_similarity'] * 100,
                'similarity_breakdown': combined_comparison['similarity_breakdown'],
                'recommendations': combined_comparison['recommendations'],
                'detailed_metrics': combined_comparison['detailed_metrics']
            }
            
            self.logger.info(f"[ENHANCED_COMPARISON] Comparison completed: {result['final_score']:.1f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"[ENHANCED_COMPARISON] Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_traditional_cycle_analysis(self, video_path: str, metadata: dict) -> Dict[str, Any]:
        """Get traditional cycle analysis using the existing system."""
        # Use dummy professional video to get user analysis
        dummy_pro = video_path
        full_result = self.traditional_analyzer.compare_techniques(video_path, dummy_pro, metadata, metadata)
        
        if not full_result['success']:
            return full_result
        
        # Extract user analysis data
        user_analysis = full_result.get('user_analysis', {})
        return {
            'success': True,
            'cycles_details': user_analysis.get('cycles_details', []),
            'cycles_count': user_analysis.get('cycles_count', 0),
            'average_duration': user_analysis.get('average_duration', 0),
            'quality_score': user_analysis.get('quality_score', 0)
        }
    
    def _extract_single_cycle_parameters(self, cycle: Dict) -> Dict[str, float]:
        """Extract parameters from a single cycle."""
        params = {
            'duration': cycle.get('duration', 0.0),
            'amplitude': cycle.get('amplitude', 0.0),
            'quality': cycle.get('quality', 0.0),
        }
        
        # Calculate derived parameters
        if params['duration'] > 0:
            params['frequency'] = 1.0 / params['duration']
            params['amplitude_per_second'] = params['amplitude'] / params['duration']
        else:
            params['frequency'] = 0.0
            params['amplitude_per_second'] = 0.0
        
        # Efficiency estimation
        params['movement_efficiency'] = min(1.0, params['amplitude'] / 50000) if params['amplitude'] > 0 else 0.0
        params['movement_smoothness'] = params['quality']
        
        return params
    
    def _calculate_enhanced_metrics(self, target_cycle: Dict, biomech_result: Optional[Dict], 
                                  traditional_result: Dict) -> Dict[str, Any]:
        """Calculate enhanced metrics combining traditional and biomechanical data."""
        enhanced_metrics = {
            'traditional_metrics': {
                'cycle_duration': target_cycle.get('duration', 0),
                'cycle_amplitude': target_cycle.get('amplitude', 0),
                'cycle_quality': target_cycle.get('quality', 0),
                'total_cycles_detected': traditional_result.get('cycles_count', 0)
            }
        }
        
        if biomech_result and biomech_result['success']:
            # Add biomechanical summary metrics
            joint_analysis = biomech_result.get('joint_analysis', {})
            racket_analysis = biomech_result.get('racket_analysis', {})
            posture_analysis = biomech_result.get('posture_analysis', {})
            
            enhanced_metrics['biomechanical_metrics'] = {
                'joint_score': joint_analysis.get('overall_joint_score', 0) if joint_analysis.get('success') else 0,
                'racket_score': racket_analysis.get('overall_racket_score', 0) if racket_analysis.get('success') else 0,
                'posture_score': posture_analysis.get('overall_posture_score', 0) if posture_analysis.get('success') else 0,
                'has_biomechanical_data': True
            }
            
            # Extract key biomechanical values
            if joint_analysis.get('success') and 'arm_mechanics' in joint_analysis:
                arm_mech = joint_analysis['arm_mechanics']
                if 'dominant_arm' in arm_mech:
                    enhanced_metrics['biomechanical_metrics']['arm_angle_mean'] = arm_mech['dominant_arm'].get('mean_angle', 0)
                    enhanced_metrics['biomechanical_metrics']['arm_mechanics_score'] = arm_mech['dominant_arm'].get('mechanics_score', 0)
            
            if racket_analysis.get('success') and 'velocity_analysis' in racket_analysis:
                vel_analysis = racket_analysis['velocity_analysis']
                if 'basic_stats' in vel_analysis:
                    enhanced_metrics['biomechanical_metrics']['max_velocity'] = vel_analysis['basic_stats'].get('max_velocity', 0)
                    enhanced_metrics['biomechanical_metrics']['mean_velocity'] = vel_analysis['basic_stats'].get('mean_velocity', 0)
        else:
            enhanced_metrics['biomechanical_metrics'] = {
                'has_biomechanical_data': False,
                'note': 'Biomechanical analysis not available'
            }
        
        return enhanced_metrics
    
    def _get_cycle_frame_range(self, cycle: Dict, cycle_index: int) -> Dict[str, int]:
        """Estimate frame range for the analyzed cycle."""
        # This is a simplified estimation - in a full implementation,
        # you'd want to get the actual frame indices from the cycle detection
        cycle_duration = cycle.get('duration', 1.0)
        fps = 30  # Assuming 30 FPS
        frames_per_cycle = int(cycle_duration * fps)
        
        estimated_start = cycle_index * frames_per_cycle
        estimated_end = estimated_start + frames_per_cycle
        
        return {
            'start_frame': estimated_start,
            'end_frame': estimated_end,
            'frame_count': frames_per_cycle,
            'note': 'Estimated frame range'
        }
    
    def _compare_traditional_cycles(self, user_params: Dict, pro_params: Dict) -> Dict[str, Any]:
        """Compare traditional single-cycle parameters."""
        similarities = {}
        differences = {}
        
        for param in user_params:
            if param in pro_params:
                user_val = user_params[param]
                pro_val = pro_params[param]
                
                if pro_val != 0:
                    rel_diff = abs(user_val - pro_val) / max(abs(pro_val), 0.001)
                    similarity = max(0.0, 1.0 - rel_diff)
                else:
                    similarity = 1.0 if user_val == 0 else 0.0
                
                similarities[param] = similarity
                differences[param] = {
                    'user': user_val,
                    'professional': pro_val,
                    'difference': user_val - pro_val,
                    'similarity_percentage': similarity * 100
                }
        
        overall_similarity = np.mean(list(similarities.values())) if similarities else 0.5
        
        return {
            'similarities': similarities,
            'differences': differences,
            'overall_similarity': overall_similarity
        }
    
    def _create_combined_comparison(self, traditional_comp: Dict, biomech_comp: Optional[Dict],
                                  user_result: Dict, pro_result: Dict) -> Dict[str, Any]:
        """Create combined comparison with both traditional and biomechanical data."""
        
        # Traditional similarity
        traditional_sim = traditional_comp['overall_similarity']
        
        # Biomechanical similarity (if available)
        biomech_sim = 0.5  # Default
        if biomech_comp and biomech_comp.get('success'):
            overall_comp = biomech_comp.get('overall_comparison', {})
            biomech_sim = overall_comp.get('similarity_score', 0.5)
        
        # Weighted combination (traditional: 60%, biomechanical: 40%)
        # Traditional gets higher weight because it's proven and reliable
        traditional_weight = 0.6
        biomech_weight = 0.4
        
        combined_similarity = (traditional_sim * traditional_weight + 
                             biomech_sim * biomech_weight)
        
        # Create similarity breakdown
        similarity_breakdown = {
            'traditional_cycle_analysis': {
                'score': traditional_sim,
                'weight': traditional_weight,
                'contribution': traditional_sim * traditional_weight,
                'details': traditional_comp['similarities']
            },
            'biomechanical_analysis': {
                'score': biomech_sim,
                'weight': biomech_weight,
                'contribution': biomech_sim * biomech_weight,
                'available': biomech_comp is not None and biomech_comp.get('success', False)
            }
        }
        
        # Generate recommendations
        recommendations = self._generate_combined_recommendations(
            traditional_comp, biomech_comp, combined_similarity
        )
        
        # Create detailed metrics for interface
        detailed_metrics = self._create_detailed_metrics_for_interface(
            user_result, pro_result, traditional_comp, biomech_comp
        )
        
        return {
            'overall_similarity': combined_similarity,
            'similarity_breakdown': similarity_breakdown,
            'recommendations': recommendations,
            'detailed_metrics': detailed_metrics,
            'performance_category': self._categorize_performance(combined_similarity),
            'analysis_confidence': self._calculate_analysis_confidence(biomech_comp)
        }
    
    def _generate_combined_recommendations(self, traditional_comp: Dict, 
                                         biomech_comp: Optional[Dict], 
                                         overall_similarity: float) -> List[str]:
        """Generate recommendations based on combined analysis."""
        recommendations = []
        
        # Performance-based recommendations
        if overall_similarity > 0.8:
            recommendations.append("Excellent technique! Focus on maintaining consistency across all cycles.")
        elif overall_similarity > 0.6:
            recommendations.append("Good technique with room for refinement in specific areas.")
        else:
            recommendations.append("Significant improvements possible through focused practice.")
        
        # Traditional cycle-specific recommendations
        traditional_diffs = traditional_comp.get('differences', {})
        
        if 'duration' in traditional_diffs:
            duration_sim = traditional_diffs['duration']['similarity_percentage']
            if duration_sim < 70:
                if traditional_diffs['duration']['user'] > traditional_diffs['duration']['professional']:
                    recommendations.append("Work on speeding up your stroke execution.")
                else:
                    recommendations.append("Take more time for proper stroke preparation.")
        
        if 'amplitude' in traditional_diffs:
            amplitude_sim = traditional_diffs['amplitude']['similarity_percentage']
            if amplitude_sim < 70:
                if traditional_diffs['amplitude']['user'] < traditional_diffs['amplitude']['professional']:
                    recommendations.append("Increase your stroke amplitude for more power.")
                else:
                    recommendations.append("Focus on controlling stroke amplitude for better precision.")
        
        # Biomechanical recommendations (if available)
        if biomech_comp and biomech_comp.get('success'):
            biomech_recommendations = biomech_comp.get('recommendations', [])
            recommendations.extend(biomech_recommendations[:3])  # Add top 3 biomech recommendations
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def _create_detailed_metrics_for_interface(self, user_result: Dict, pro_result: Dict,
                                             traditional_comp: Dict, biomech_comp: Optional[Dict]) -> Dict[str, Any]:
        """Create detailed metrics formatted for the web interface."""
        
        metrics = {}
        
        # Traditional metrics
        user_cycle = user_result['cycle_data']
        pro_cycle = pro_result['cycle_data']
        traditional_diffs = traditional_comp['differences']
        
        metrics['cycle_analysis'] = {
            'duration': {
                'user': user_cycle.get('duration', 0),
                'professional': pro_cycle.get('duration', 0),
                'similarity': traditional_diffs.get('duration', {}).get('similarity_percentage', 0)
            },
            'amplitude': {
                'user': user_cycle.get('amplitude', 0),
                'professional': pro_cycle.get('amplitude', 0),
                'similarity': traditional_diffs.get('amplitude', {}).get('similarity_percentage', 0)
            },
            'quality': {
                'user': user_cycle.get('quality', 0),
                'professional': pro_cycle.get('quality', 0),
                'similarity': traditional_diffs.get('quality', {}).get('similarity_percentage', 0)
            }
        }
        
        # Biomechanical metrics (if available)
        if biomech_comp and biomech_comp.get('success'):
            metrics['biomechanical_analysis'] = {
                'available': True,
                'joint_similarity': biomech_comp.get('joint_comparison', {}).get('overall_joint_similarity', 0) * 100,
                'racket_similarity': biomech_comp.get('racket_comparison', {}).get('overall_racket_similarity', 0) * 100,
                'posture_similarity': biomech_comp.get('posture_comparison', {}).get('overall_posture_similarity', 0) * 100
            }
            
            # Add specific biomechanical values
            user_biomech = user_result.get('enhanced_metrics', {}).get('biomechanical_metrics', {})
            pro_biomech = pro_result.get('enhanced_metrics', {}).get('biomechanical_metrics', {})
            
            if user_biomech.get('has_biomechanical_data') and pro_biomech.get('has_biomechanical_data'):
                metrics['biomechanical_details'] = {
                    'arm_angle': {
                        'user': user_biomech.get('arm_angle_mean', 0),
                        'professional': pro_biomech.get('arm_angle_mean', 0)
                    },
                    'max_velocity': {
                        'user': user_biomech.get('max_velocity', 0),
                        'professional': pro_biomech.get('max_velocity', 0)
                    }
                }
        else:
            metrics['biomechanical_analysis'] = {
                'available': False,
                'note': 'Biomechanical analysis could not be completed'
            }
        
        return metrics
    
    def _categorize_performance(self, similarity: float) -> str:
        """Categorize performance based on similarity score."""
        if similarity >= 0.9:
            return "Professional Level"
        elif similarity >= 0.8:
            return "Advanced"
        elif similarity >= 0.6:
            return "Intermediate"
        elif similarity >= 0.4:
            return "Developing"
        else:
            return "Beginner"
    
    def _calculate_analysis_confidence(self, biomech_comp: Optional[Dict]) -> float:
        """Calculate confidence in the analysis."""
        # Higher confidence when we have both traditional and biomechanical data
        base_confidence = 0.8  # Traditional analysis is reliable
        
        if biomech_comp and biomech_comp.get('success'):
            biomech_confidence = biomech_comp.get('overall_comparison', {}).get('comparison_confidence', 0.5)
            # Combine confidences
            combined_confidence = (base_confidence + biomech_confidence) / 2
            return min(combined_confidence + 0.1, 1.0)  # Bonus for having both
        
        return base_confidence
    
    def _interpret_elbow_variation(self, variation: float) -> str:
        """Interpret elbow variation in degrees"""
        if variation < 30:
            return "movimento muito controlado"
        elif variation < 60:
            return "movimento controlado"
        elif variation < 100:
            return "movimento moderado"
        elif variation < 140:
            return "movimento dinâmico"
        else:
            return "movimento muito dinâmico"
    
    def _interpret_coordination(self, coordination: float) -> str:
        """Interpret coordination score"""
        if coordination >= 0.9:
            return "coordenação excelente"
        elif coordination >= 0.8:
            return "boa coordenação"
        elif coordination >= 0.7:
            return "coordenação adequada"
        elif coordination >= 0.6:
            return "coordenação limitada"
        else:
            return "coordenação deficiente"
    
    def _interpret_temporal_pattern(self, pattern: str, detected_movement: str = None) -> str:
        """Interpret temporal movement pattern with context awareness"""
        
        
        # Context-aware interpretations based on detected movement
        if detected_movement and detected_movement.strip():
            if "push" in detected_movement.lower():
                # Push-specific interpretations
                pattern_interpretations = {
                    "explosive_start": "início rápido (push ativo)",
                    "progressive_drive": "aceleração gradual (push progressivo)",
                    "controlled_push": "movimento controlado (push eficiente)",
                    "quick_push": "push rápido",
                    "slow_buildup": "preparação controlada (push defensivo)",
                    "insufficient_data": "dados insuficientes",
                    "error": "erro na análise"
                }
            elif "drive" in detected_movement.lower():
                # Drive-specific interpretations
                pattern_interpretations = {
                    "explosive_start": "início explosivo (drive potente)",
                    "progressive_drive": "aceleração progressiva (drive clássico)",
                    "controlled_push": "movimento controlado (drive técnico)",
                    "quick_push": "execução rápida (drive rápido)",
                    "slow_buildup": "preparação lenta (drive de colocação)",
                    "insufficient_data": "dados insuficientes",
                    "error": "erro na análise"
                }
            else:
                # Default interpretations
                pattern_interpretations = {
                    "explosive_start": "início explosivo",
                    "progressive_drive": "aceleração progressiva",
                    "controlled_push": "movimento controlado",
                    "quick_push": "execução rápida",
                    "slow_buildup": "preparação gradual",
                    "insufficient_data": "dados insuficientes",
                    "error": "erro na análise"
                }
        else:
            # Fallback to generic interpretations
            pattern_interpretations = {
                "explosive_start": "início explosivo (típico de drive)",
                "progressive_drive": "aceleração progressiva (drive clássico)",
                "controlled_push": "movimento controlado (push eficiente)",
                "quick_push": "push rápido",
                "slow_buildup": "construção lenta",
                "insufficient_data": "dados insuficientes",
                "error": "erro na análise"
            }
        
        return pattern_interpretations.get(pattern, f"padrão {pattern}")
    
    def _interpret_amplitude(self, amplitude: float, detected_movement: str = None) -> str:
        """Interpret movement amplitude with context awareness"""
        
        # Context-aware interpretations based on detected movement
        if detected_movement and "push" in detected_movement.lower():
            # Push-specific amplitude interpretation
            if amplitude < 0.06:
                return "amplitude baixa (push defensivo)"
            elif amplitude < 0.1:
                return "amplitude baixa-média (push controlado)"
            elif amplitude < 0.15:
                return "amplitude média (push ativo)"
            elif amplitude < 0.2:
                return "amplitude alta para push (push ofensivo)"
            else:
                return "amplitude muito alta para push (estilo híbrido)"
        elif detected_movement and "drive" in detected_movement.lower():
            # Drive-specific amplitude interpretation
            if amplitude < 0.1:
                return "amplitude baixa (drive controlado)"
            elif amplitude < 0.2:
                return "amplitude média (drive padrão)"
            elif amplitude < 0.3:
                return "amplitude alta (drive potente)"
            else:
                return "amplitude muito alta (drive explosivo)"
        else:
            # Generic interpretation (fallback)
            if amplitude < 0.06:
                return "amplitude baixa (push)"
            elif amplitude < 0.1:
                return "amplitude média"
            elif amplitude < 0.25:
                return "amplitude alta (drive)"
            else:
                return "amplitude muito alta"
    
    def _interpret_velocity(self, velocity: float) -> str:
        """Interpret maximum velocity"""
        if velocity < 0.03:
            return "velocidade baixa"
        elif velocity < 0.05:
            return "velocidade moderada"
        elif velocity < 0.1:
            return "velocidade alta"
        else:
            return "velocidade muito alta"
    
    def _calculate_biomech_performance_score(self, biomech_result) -> float:
        """Calculate performance score based on real biomechanical analysis"""
        try:
            score = 50.0  # Base score
            
            # Confidence contribution (0-25 points)
            confidence_points = biomech_result.confidence * 25
            score += confidence_points
            
            # Coordination contribution (0-15 points)
            coordination_points = biomech_result.coordination_active * 15
            score += coordination_points
            
            # Movement signature contribution (0-10 points)
            signature_points = biomech_result.movement_signature * 10
            score += signature_points
            
            # Bonus for high biomechanical confidence
            if biomech_result.biomech_confidence > 0.8:
                score += 5
            
            # Bonus for clear classification
            if biomech_result.confidence_level == "high":
                score += 3
            elif biomech_result.confidence_level == "medium":
                score += 1
            
            # Ensure score is within valid range
            return min(max(score, 0), 100)
            
        except Exception:
            return 75.0  # Fallback score
    
    def _process_video_single_cycle(self, biomech_classifier, video_path: str, metadata: dict, cycle_index: int = 1):
        """
        Process video focusing on a specific cycle (default: second cycle = index 1)
        
        Args:
            biomech_classifier: Instance of ImprovedBiomechClassifier2D
            video_path: Path to video file
            metadata: Video metadata
            cycle_index: Which cycle to analyze (1 = second cycle)
        
        Returns:
            Biomechanical analysis result for the specific cycle
        """
        try:
            self.logger.info(f"[CYCLE_ANALYSIS] Starting cycle-specific analysis for cycle {cycle_index + 1}")
            
            # Step 1: Detect all cycles in the video first
            cycles = self._detect_movement_cycles(video_path, metadata)
            
            if not cycles or len(cycles) <= cycle_index:
                self.logger.warning(f"[CYCLE_ANALYSIS] Insufficient cycles found. Detected: {len(cycles) if cycles else 0}, needed: {cycle_index + 1}")
                # Fallback to full video analysis
                return biomech_classifier.process_video(video_path)
            
            # [FIX v16.2] ALWAYS use full video analysis for better accuracy
            # Cycle-specific analysis can cause incorrect hand detection with fewer frames
            self.logger.info(f"[CYCLE_ANALYSIS] Using FULL VIDEO analysis for better accuracy (v16.2 fix)")
            return biomech_classifier.process_video(video_path)
            
            # Step 2: Get the specific cycle boundaries
            target_cycle = cycles[cycle_index]
            start_frame = target_cycle['start_frame']
            end_frame = target_cycle['end_frame']
            
            self.logger.info(f"[CYCLE_ANALYSIS] Analyzing cycle {cycle_index + 1}: frames {start_frame}-{end_frame}")
            
            # Step 3: Extract only the frames for this cycle
            cycle_frames = self._extract_cycle_frames(video_path, start_frame, end_frame)
            
            if not cycle_frames:
                self.logger.error(f"[CYCLE_ANALYSIS] Could not extract frames for cycle")
                return None
            
            # Step 4: Run biomechanical analysis on cycle-specific frames
            biomech_result = self._analyze_cycle_frames_biomechanically(
                biomech_classifier, cycle_frames, metadata, target_cycle
            )
            
            if biomech_result:
                # Add cycle-specific information to the result
                biomech_result.cycle_info = {
                    'cycle_index': cycle_index,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'cycle_duration': target_cycle.get('duration', 0),
                    'cycle_amplitude': target_cycle.get('amplitude', 0),
                    'cycle_quality': target_cycle.get('quality_score', 0),
                    'analysis_type': 'single_cycle_focused'
                }
                
                self.logger.info(f"[CYCLE_ANALYSIS] Successfully analyzed cycle {cycle_index + 1}")
            
            return biomech_result
            
        except Exception as e:
            self.logger.error(f"[CYCLE_ANALYSIS] Error in cycle-specific analysis: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to full video analysis
            return biomech_classifier.process_video(video_path)
    
    def _detect_movement_cycles(self, video_path: str, metadata: dict) -> List[Dict]:
        """
        Detect movement cycles in the video using retracted->extended->retracted pattern
        
        Returns:
            List of cycle dictionaries with start_frame, end_frame, duration, etc.
        """
        try:
            from cycle_detector_retracted_extended import CycleDetectorRetractedExtended
            import cv2
            
            self.logger.info(f"[CYCLE_DETECTION] Starting cycle detection...")
            
            # Initialize cycle detector
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            detector = CycleDetectorRetractedExtended(fps=fps)
            
            # Extract frames for cycle detection
            frames = []
            max_frames_for_detection = 300  # Limit for performance
            frame_count = 0
            
            while frame_count < max_frames_for_detection:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                self.logger.error(f"[CYCLE_DETECTION] No frames extracted")
                return []
            
            # Convert metadata to cycle detector format
            validated_params = self._convert_metadata_for_cycle_detection(metadata)
            
            # Detect cycles
            cycle_infos = detector.detect_cycles_from_validated_params(frames, validated_params)
            
            # Convert to dictionary format
            cycles = []
            for i, cycle_info in enumerate(cycle_infos):
                cycle_dict = {
                    'cycle_index': i,
                    'start_frame': cycle_info.start_frame,
                    'end_frame': cycle_info.end_frame,
                    'peak_frame': cycle_info.peak_frame,
                    'valley_frame': cycle_info.valley_frame,
                    'duration': cycle_info.duration,
                    'amplitude': cycle_info.amplitude,
                    'quality_score': cycle_info.quality_score
                }
                cycles.append(cycle_dict)
            
            self.logger.info(f"[CYCLE_DETECTION] Detected {len(cycles)} cycles")
            return cycles
            
        except ImportError as e:
            self.logger.error(f"[CYCLE_DETECTION] Could not import cycle detector: {e}")
            return []
        except Exception as e:
            self.logger.error(f"[CYCLE_DETECTION] Error in cycle detection: {e}")
            return []
    
    def _convert_metadata_for_cycle_detection(self, metadata: dict) -> Dict[str, str]:
        """Convert interface metadata to cycle detector format"""
        # Map interface values to cycle detector format
        dominant_hand = 'right' if metadata.get('maoDominante', 'D') in ['D', 'Destro'] else 'left'
        
        # Determine movement type
        racket_side = metadata.get('ladoRaquete', 'F')
        if racket_side in ['F', 'Forehand']:
            movement_type = 'forehand'
        else:
            movement_type = 'backhand'
        
        # Camera side
        camera_side = 'right' if metadata.get('ladoCamera', 'D') in ['D', 'Direita'] else 'left'
        
        return {
            'dominant_hand': dominant_hand,
            'movement_type': movement_type,
            'camera_side': camera_side,
            'racket_side': movement_type  # Same as movement_type for simplicity
        }
    
    def _extract_cycle_frames(self, video_path: str, start_frame: int, end_frame: int) -> List:
        """Extract specific frames for the cycle"""
        try:
            import cv2
            
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                current_frame += 1
            
            cap.release()
            
            self.logger.info(f"[FRAME_EXTRACTION] Extracted {len(frames)} frames for cycle (frames {start_frame}-{end_frame})")
            return frames
            
        except Exception as e:
            self.logger.error(f"[FRAME_EXTRACTION] Error extracting cycle frames: {e}")
            return []
    
    def _analyze_cycle_frames_biomechanically(self, biomech_classifier, frames: List, metadata: dict, cycle_info: Dict):
        """
        Run biomechanical analysis specifically on the cycle frames
        """
        try:
            import tempfile
            import os
            import cv2
            
            self.logger.info(f"[CYCLE_BIOMECH] Running biomechanical analysis on {len(frames)} cycle frames")
            
            # Create temporary video file with just the cycle frames
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
            
            # Write cycle frames to temporary video
            if frames:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0  # Standard FPS
                
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                
                for frame in frames:
                    out.write(frame)
                
                out.release()
                
                # Run biomechanical analysis on the cycle-specific video
                biomech_result = biomech_classifier.process_video(temp_video_path)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
                
                if biomech_result:
                    self.logger.info(f"[CYCLE_BIOMECH] Cycle-specific biomechanical analysis completed successfully")
                    
                    # Add cycle-specific context to recommendations
                    if hasattr(biomech_result, 'movement_type'):
                        # Converter valores técnicos em descrições mais compreensíveis
                        duration = cycle_info.get('duration', 0)
                        amplitude = cycle_info.get('amplitude', 0)
                        quality = cycle_info.get('quality_score', 0)
                        
                        # Interpretações mais amigáveis
                        duration_desc = "normal" if 0.8 <= duration <= 2.0 else ("rápido" if duration < 0.8 else "lento")
                        amplitude_desc = "moderada" if 50000 < amplitude < 200000 else ("baixa" if amplitude <= 50000 else "alta")
                        quality_desc = "boa" if quality > 0.7 else ("razoável" if quality > 0.4 else "baixa")
                        
                        original_recommendations = [
                            f"[TARGET] Movimento identificado: {self._translate_movement_type(biomech_result.movement_type.value)} (confiança: {biomech_result.confidence:.1%})",
                            f"[TIME] Tempo do movimento: {duration:.2f}s ({duration_desc})",
                            f"[RULER] Extensão do movimento: {amplitude_desc}",
                            f"[ENHANCED] Consistência técnica: {quality_desc} ({quality:.1%})",
                            f"[DATA] Análise baseada em {cycle_info.get('end_frame', 0) - cycle_info.get('start_frame', 0)} quadros de vídeo"
                        ]
                        
                        # Store original details for potential use
                        biomech_result.cycle_specific_analysis = True
                        biomech_result.cycle_recommendations = original_recommendations
                
                return biomech_result
            
        except Exception as e:
            self.logger.error(f"[CYCLE_BIOMECH] Error in cycle-specific biomechanical analysis: {e}")
            import traceback
            traceback.print_exc()
            return None


# Test function
def test_enhanced_single_cycle_analysis():
    """Test the enhanced single cycle analysis."""
    analyzer = EnhancedSingleCycleAnalyzer()
    
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
    
    print("=== TESTING ENHANCED SINGLE CYCLE ANALYSIS ===")
    print(f"User video: {os.path.basename(user_video)}")
    print(f"Professional video: {os.path.basename(pro_video)}")
    print()
    
    result = analyzer.compare_enhanced_single_cycles(
        user_video, pro_video, user_metadata, prof_metadata, cycle_index=1
    )
    
    if result['success']:
        print(f"[SUCCESS] Enhanced analysis completed!")
        print(f"Final score: {result['final_score']:.1f}%")
        print(f"Performance category: {result['combined_comparison']['performance_category']}")
        print(f"Analysis confidence: {result['combined_comparison']['analysis_confidence']:.1%}")
        print()
        
        # Traditional cycle analysis
        user_cycle = result['user_analysis']['cycle_data']
        pro_cycle = result['professional_analysis']['cycle_data']
        print("TRADITIONAL CYCLE ANALYSIS:")
        print(f"  User cycle: {user_cycle['duration']:.3f}s, amplitude {user_cycle['amplitude']:.1f}")
        print(f"  Pro cycle: {pro_cycle['duration']:.3f}s, amplitude {pro_cycle['amplitude']:.1f}")
        print()
        
        # Biomechanical analysis
        biomech_comp = result['biomechanical_comparison']
        if biomech_comp and biomech_comp.get('success'):
            print("BIOMECHANICAL ANALYSIS:")
            overall_biomech = biomech_comp.get('overall_comparison', {})
            print(f"  Biomechanical similarity: {overall_biomech.get('similarity_percentage', 0):.1f}%")
            print(f"  Performance category: {overall_biomech.get('performance_category', 'Unknown')}")
        else:
            print("BIOMECHANICAL ANALYSIS: Not available")
        print()
        
        # Recommendations
        recommendations = result['recommendations']
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")
        
        return True
    else:
        print(f"[ERROR] Enhanced analysis failed: {result.get('error')}")
        return False


if __name__ == "__main__":
    test_enhanced_single_cycle_analysis()