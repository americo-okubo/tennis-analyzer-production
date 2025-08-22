"""
Biomechanical Comparison Engine

Compares biomechanical data between user and professional movements using
scientific metrics instead of subjective estimates.

Provides detailed analysis with actionable recommendations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class BiomechanicalComparison:
    """
    Advanced comparison engine for biomechanical analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Weights for different aspects of comparison
        self.comparison_weights = {
            'joint_angles': 0.25,
            'racket_trajectory': 0.30,
            'body_posture': 0.25,
            'kinematics': 0.20
        }
    
    def compare(self, user_analysis: Dict, professional_analysis: Dict) -> Dict:
        """
        Comprehensive biomechanical comparison between user and professional.
        """
        try:
            comparison_result = {
                'success': True,
                'comparison_type': 'biomechanical_2d',
                'analysis_timestamp': self._get_timestamp()
            }
            
            # Compare different aspects
            if self._has_joint_analysis(user_analysis, professional_analysis):
                comparison_result['joint_comparison'] = self._compare_joint_angles(
                    user_analysis['joint_analysis'], 
                    professional_analysis['joint_analysis']
                )
            
            if self._has_racket_analysis(user_analysis, professional_analysis):
                comparison_result['racket_comparison'] = self._compare_racket_trajectory(
                    user_analysis['racket_analysis'], 
                    professional_analysis['racket_analysis']
                )
            
            if self._has_posture_analysis(user_analysis, professional_analysis):
                comparison_result['posture_comparison'] = self._compare_body_posture(
                    user_analysis['posture_analysis'], 
                    professional_analysis['posture_analysis']
                )
            
            if self._has_kinematics(user_analysis, professional_analysis):
                comparison_result['kinematics_comparison'] = self._compare_kinematics(
                    user_analysis['kinematics'], 
                    professional_analysis['kinematics']
                )
            
            # Calculate overall similarity and recommendations
            comparison_result['overall_comparison'] = self._calculate_overall_comparison(comparison_result)
            comparison_result['recommendations'] = self._generate_recommendations(comparison_result)
            comparison_result['detailed_metrics'] = self._generate_detailed_metrics(comparison_result)
            
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"[BIOMECH_COMPARISON] Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _compare_joint_angles(self, user_joints: Dict, pro_joints: Dict) -> Dict:
        """
        Compare joint angle analysis between user and professional.
        """
        joint_comparison = {}
        
        # Compare arm mechanics
        if ('arm_mechanics' in user_joints and 'arm_mechanics' in pro_joints):
            joint_comparison['arm_mechanics'] = self._compare_arm_mechanics(
                user_joints['arm_mechanics'], 
                pro_joints['arm_mechanics']
            )
        
        # Compare posture analysis
        if ('posture_analysis' in user_joints and 'posture_analysis' in pro_joints):
            joint_comparison['posture_comparison'] = self._compare_joint_posture(
                user_joints['posture_analysis'], 
                pro_joints['posture_analysis']
            )
        
        # Compare range of motion
        if ('range_of_motion' in user_joints and 'range_of_motion' in pro_joints):
            joint_comparison['rom_comparison'] = self._compare_range_of_motion(
                user_joints['range_of_motion'], 
                pro_joints['range_of_motion']
            )
        
        # Overall joint similarity
        joint_comparison['overall_joint_similarity'] = self._calculate_joint_similarity(joint_comparison)
        
        return joint_comparison
    
    def _compare_racket_trajectory(self, user_racket: Dict, pro_racket: Dict) -> Dict:
        """
        Compare racket trajectory analysis.
        """
        racket_comparison = {}
        
        # Compare velocity analysis
        if ('velocity_analysis' in user_racket and 'velocity_analysis' in pro_racket):
            racket_comparison['velocity_comparison'] = self._compare_velocity_patterns(
                user_racket['velocity_analysis'], 
                pro_racket['velocity_analysis']
            )
        
        # Compare trajectory analysis
        if ('trajectory_analysis' in user_racket and 'trajectory_analysis' in pro_racket):
            racket_comparison['trajectory_comparison'] = self._compare_trajectory_patterns(
                user_racket['trajectory_analysis'], 
                pro_racket['trajectory_analysis']
            )
        
        # Compare acceleration patterns
        if ('acceleration_analysis' in user_racket and 'acceleration_analysis' in pro_racket):
            racket_comparison['acceleration_comparison'] = self._compare_acceleration_patterns(
                user_racket['acceleration_analysis'], 
                pro_racket['acceleration_analysis']
            )
        
        # Overall racket similarity
        racket_comparison['overall_racket_similarity'] = self._calculate_racket_similarity(racket_comparison)
        
        return racket_comparison
    
    def _compare_body_posture(self, user_posture: Dict, pro_posture: Dict) -> Dict:
        """
        Compare body posture analysis.
        """
        posture_comparison = {}
        
        # Compare stance analysis
        if ('stance_analysis' in user_posture and 'stance_analysis' in pro_posture):
            posture_comparison['stance_comparison'] = self._compare_stance_analysis(
                user_posture['stance_analysis'], 
                pro_posture['stance_analysis']
            )
        
        # Compare balance analysis
        if ('balance_analysis' in user_posture and 'balance_analysis' in pro_posture):
            posture_comparison['balance_comparison'] = self._compare_balance_analysis(
                user_posture['balance_analysis'], 
                pro_posture['balance_analysis']
            )
        
        # Compare alignment
        if ('alignment_analysis' in user_posture and 'alignment_analysis' in pro_posture):
            posture_comparison['alignment_comparison'] = self._compare_alignment_analysis(
                user_posture['alignment_analysis'], 
                pro_posture['alignment_analysis']
            )
        
        # Overall posture similarity
        posture_comparison['overall_posture_similarity'] = self._calculate_posture_similarity(posture_comparison)
        
        return posture_comparison
    
    def _compare_kinematics(self, user_kinematics: Dict, pro_kinematics: Dict) -> Dict:
        """
        Compare kinematic data (velocities, accelerations, trajectories).
        """
        kinematics_comparison = {}
        
        # Compare velocity patterns
        if ('velocities' in user_kinematics and 'velocities' in pro_kinematics):
            kinematics_comparison['velocity_patterns'] = self._compare_velocity_kinematics(
                user_kinematics['velocities'], 
                pro_kinematics['velocities']
            )
        
        # Compare joint angles
        if ('joint_angles' in user_kinematics and 'joint_angles' in pro_kinematics):
            kinematics_comparison['angle_patterns'] = self._compare_angle_kinematics(
                user_kinematics['joint_angles'], 
                pro_kinematics['joint_angles']
            )
        
        # Overall kinematics similarity
        kinematics_comparison['overall_kinematics_similarity'] = self._calculate_kinematics_similarity(kinematics_comparison)
        
        return kinematics_comparison
    
    def _compare_arm_mechanics(self, user_arm: Dict, pro_arm: Dict) -> Dict:
        """Compare arm mechanics between user and professional."""
        comparison = {}
        
        if ('dominant_arm' in user_arm and 'dominant_arm' in pro_arm):
            user_dom = user_arm['dominant_arm']
            pro_dom = pro_arm['dominant_arm']
            
            # Compare key metrics
            metrics_comparison = {}
            
            for metric in ['mean_angle', 'angle_range', 'consistency', 'mechanics_score']:
                if metric in user_dom and metric in pro_dom:
                    similarity = self._calculate_similarity(user_dom[metric], pro_dom[metric])
                    metrics_comparison[metric] = {
                        'user_value': user_dom[metric],
                        'professional_value': pro_dom[metric],
                        'similarity': similarity,
                        'difference': abs(user_dom[metric] - pro_dom[metric])
                    }
            
            comparison['dominant_arm_comparison'] = metrics_comparison
            
            # Calculate overall arm similarity
            similarities = [comp['similarity'] for comp in metrics_comparison.values()]
            comparison['arm_similarity_score'] = np.mean(similarities) if similarities else 0.5
        
        return comparison
    
    def _compare_velocity_patterns(self, user_velocity: Dict, pro_velocity: Dict) -> Dict:
        """Compare velocity patterns."""
        comparison = {}
        
        if ('basic_stats' in user_velocity and 'basic_stats' in pro_velocity):
            user_stats = user_velocity['basic_stats']
            pro_stats = pro_velocity['basic_stats']
            
            stats_comparison = {}
            for stat in ['max_velocity', 'mean_velocity']:
                if stat in user_stats and stat in pro_stats:
                    similarity = self._calculate_similarity(user_stats[stat], pro_stats[stat])
                    stats_comparison[stat] = {
                        'user_value': user_stats[stat],
                        'professional_value': pro_stats[stat],
                        'similarity': similarity,
                        'difference': abs(user_stats[stat] - pro_stats[stat])
                    }
            
            comparison['velocity_stats'] = stats_comparison
        
        if ('performance' in user_velocity and 'performance' in pro_velocity):
            user_perf = user_velocity['performance']
            pro_perf = pro_velocity['performance']
            
            perf_comparison = {}
            for metric in ['velocity_efficiency', 'optimal_velocity_score']:
                if metric in user_perf and metric in pro_perf:
                    similarity = self._calculate_similarity(user_perf[metric], pro_perf[metric])
                    perf_comparison[metric] = {
                        'user_value': user_perf[metric],
                        'professional_value': pro_perf[metric],
                        'similarity': similarity
                    }
            
            comparison['velocity_performance'] = perf_comparison
        
        return comparison
    
    def _calculate_similarity(self, user_value: float, pro_value: float, 
                            scale_factor: float = 1.0) -> float:
        """
        Calculate similarity between two values (0-1 scale).
        """
        if pro_value == 0:
            return 1.0 if user_value == 0 else 0.0
        
        # Calculate relative difference
        relative_diff = abs(user_value - pro_value) / (abs(pro_value) * scale_factor)
        
        # Convert to similarity (1 = identical, 0 = completely different)
        similarity = 1.0 / (1.0 + relative_diff)
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_overall_comparison(self, comparison_result: Dict) -> Dict:
        """
        Calculate overall comparison metrics.
        """
        overall = {}
        
        # Collect all similarity scores
        similarities = []
        weights = []
        
        if 'joint_comparison' in comparison_result:
            joint_sim = comparison_result['joint_comparison'].get('overall_joint_similarity', 0.5)
            similarities.append(joint_sim)
            weights.append(self.comparison_weights['joint_angles'])
        
        if 'racket_comparison' in comparison_result:
            racket_sim = comparison_result['racket_comparison'].get('overall_racket_similarity', 0.5)
            similarities.append(racket_sim)
            weights.append(self.comparison_weights['racket_trajectory'])
        
        if 'posture_comparison' in comparison_result:
            posture_sim = comparison_result['posture_comparison'].get('overall_posture_similarity', 0.5)
            similarities.append(posture_sim)
            weights.append(self.comparison_weights['body_posture'])
        
        if 'kinematics_comparison' in comparison_result:
            kinematics_sim = comparison_result['kinematics_comparison'].get('overall_kinematics_similarity', 0.5)
            similarities.append(kinematics_sim)
            weights.append(self.comparison_weights['kinematics'])
        
        # Calculate weighted average
        if similarities and weights:
            overall_similarity = np.average(similarities, weights=weights)
        else:
            overall_similarity = 0.5
        
        overall['similarity_score'] = overall_similarity
        overall['similarity_percentage'] = overall_similarity * 100
        overall['comparison_confidence'] = self._calculate_confidence(similarities)
        
        # Categorize performance
        overall['performance_category'] = self._categorize_performance(overall_similarity)
        
        return overall
    
    def _generate_recommendations(self, comparison_result: Dict) -> List[str]:
        """
        Generate actionable recommendations based on comparison.
        """
        recommendations = []
        
        # Analyze each component and generate specific recommendations
        if 'joint_comparison' in comparison_result:
            joint_recs = self._generate_joint_recommendations(comparison_result['joint_comparison'])
            recommendations.extend(joint_recs)
        
        if 'racket_comparison' in comparison_result:
            racket_recs = self._generate_racket_recommendations(comparison_result['racket_comparison'])
            recommendations.extend(racket_recs)
        
        if 'posture_comparison' in comparison_result:
            posture_recs = self._generate_posture_recommendations(comparison_result['posture_comparison'])
            recommendations.extend(posture_recs)
        
        # Overall recommendations
        overall_sim = comparison_result.get('overall_comparison', {}).get('similarity_score', 0.5)
        
        if overall_sim > 0.8:
            recommendations.append("Excellent technique! Focus on maintaining consistency.")
        elif overall_sim > 0.6:
            recommendations.append("Good technique with room for refinement in specific areas.")
        else:
            recommendations.append("Significant improvements possible through focused practice.")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_detailed_metrics(self, comparison_result: Dict) -> Dict:
        """
        Generate detailed metrics for interface display.
        """
        metrics = {}
        
        # Extract key metrics for display
        if 'joint_comparison' in comparison_result:
            metrics['joint_metrics'] = self._extract_joint_metrics(comparison_result['joint_comparison'])
        
        if 'racket_comparison' in comparison_result:
            metrics['racket_metrics'] = self._extract_racket_metrics(comparison_result['racket_comparison'])
        
        if 'posture_comparison' in comparison_result:
            metrics['posture_metrics'] = self._extract_posture_metrics(comparison_result['posture_comparison'])
        
        return metrics
    
    # Helper methods for specific comparisons
    def _compare_trajectory_patterns(self, user_traj: Dict, pro_traj: Dict) -> Dict:
        """Compare trajectory patterns."""
        comparison = {}
        
        if ('basic_metrics' in user_traj and 'basic_metrics' in pro_traj):
            user_metrics = user_traj['basic_metrics']
            pro_metrics = pro_traj['basic_metrics']
            
            for metric in ['trajectory_efficiency', 'total_path_length']:
                if metric in user_metrics and metric in pro_metrics:
                    similarity = self._calculate_similarity(user_metrics[metric], pro_metrics[metric])
                    comparison[metric] = {
                        'user_value': user_metrics[metric],
                        'professional_value': pro_metrics[metric],
                        'similarity': similarity
                    }
        
        return comparison
    
    def _calculate_joint_similarity(self, joint_comparison: Dict) -> float:
        """Calculate overall joint similarity."""
        similarities = []
        
        for comp_type in joint_comparison.values():
            if isinstance(comp_type, dict) and 'arm_similarity_score' in comp_type:
                similarities.append(comp_type['arm_similarity_score'])
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_racket_similarity(self, racket_comparison: Dict) -> float:
        """Calculate overall racket similarity."""
        similarities = []
        
        for comp_type in racket_comparison.values():
            if isinstance(comp_type, dict):
                for metric_comp in comp_type.values():
                    if isinstance(metric_comp, dict) and 'similarity' in metric_comp:
                        similarities.append(metric_comp['similarity'])
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_posture_similarity(self, posture_comparison: Dict) -> float:
        """Calculate overall posture similarity."""
        return 0.5  # Placeholder
    
    def _calculate_kinematics_similarity(self, kinematics_comparison: Dict) -> float:
        """Calculate overall kinematics similarity."""
        return 0.5  # Placeholder
    
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
    
    def _calculate_confidence(self, similarities: List[float]) -> float:
        """Calculate confidence in the comparison."""
        if not similarities:
            return 0.5
        
        # Higher confidence when similarities are consistent
        consistency = 1.0 - np.std(similarities)
        # Higher confidence when we have more data points
        completeness = min(len(similarities) / 4.0, 1.0)
        
        return (consistency + completeness) / 2
    
    # Placeholder methods for detailed implementations
    def _has_joint_analysis(self, user: Dict, pro: Dict) -> bool:
        return 'joint_analysis' in user and 'joint_analysis' in pro
    
    def _has_racket_analysis(self, user: Dict, pro: Dict) -> bool:
        return 'racket_analysis' in user and 'racket_analysis' in pro
    
    def _has_posture_analysis(self, user: Dict, pro: Dict) -> bool:
        return 'posture_analysis' in user and 'posture_analysis' in pro
    
    def _has_kinematics(self, user: Dict, pro: Dict) -> bool:
        return 'kinematics' in user and 'kinematics' in pro
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Placeholder methods for specific recommendation generators
    def _generate_joint_recommendations(self, joint_comparison: Dict) -> List[str]:
        return ["Focus on elbow extension consistency"]
    
    def _generate_racket_recommendations(self, racket_comparison: Dict) -> List[str]:
        return ["Improve racket velocity timing"]
    
    def _generate_posture_recommendations(self, posture_comparison: Dict) -> List[str]:
        return ["Maintain better stance width"]
    
    # Placeholder methods for metric extraction
    def _extract_joint_metrics(self, joint_comparison: Dict) -> Dict:
        return {'elbow_angle_similarity': 0.8}
    
    def _extract_racket_metrics(self, racket_comparison: Dict) -> Dict:
        return {'velocity_similarity': 0.7}
    
    def _extract_posture_metrics(self, posture_comparison: Dict) -> Dict:
        return {'stance_similarity': 0.6}
    
    # Additional placeholder methods
    def _compare_joint_posture(self, user: Dict, pro: Dict) -> Dict:
        return {}
    
    def _compare_range_of_motion(self, user: Dict, pro: Dict) -> Dict:
        return {}
    
    def _compare_acceleration_patterns(self, user: Dict, pro: Dict) -> Dict:
        return {}
    
    def _compare_stance_analysis(self, user: Dict, pro: Dict) -> Dict:
        return {}
    
    def _compare_balance_analysis(self, user: Dict, pro: Dict) -> Dict:
        return {}
    
    def _compare_alignment_analysis(self, user: Dict, pro: Dict) -> Dict:
        return {}
    
    def _compare_velocity_kinematics(self, user: Dict, pro: Dict) -> Dict:
        return {}
    
    def _compare_angle_kinematics(self, user: Dict, pro: Dict) -> Dict:
        return {}