#!/usr/bin/env python3
"""
Guide d'évaluation RÉELLE de votre système de surveillance.
Ce script vous explique comment obtenir de vraies métriques.
"""

import os
import json
from datetime import datetime

class RealEvaluationGuide:
    """Guide pour évaluer réellement votre système."""
    
    def __init__(self):
        self.evaluation_methods = {
            'video_analysis': self._video_analysis_method,
            'synthetic_data': self._synthetic_data_method, 
            'benchmark_datasets': self._benchmark_datasets_method,
            'live_testing': self._live_testing_method
        }
    
    def _video_analysis_method(self):
        """Méthode 1: Évaluer avec vos propres vidéos"""
        return {
            'description': 'Analyser des vidéos de surveillance réelles',
            'steps': [
                '1. Collecter des vidéos de test (normales + suspectes)',
                '2. Créer un ground truth manuel (annoter les comportements)',
                '3. Faire tourner votre système sur ces vidéos',
                '4. Comparer les résultats avec vos annotations',
                '5. Calculer précision, recall, F1-score'
            ],
            'code_example': '''
# Utiliser votre script d'évaluation:
python evaluate_system.py --video_dir /path/to/your/videos --ground_truth annotations.json

# Ou directement en Python:
from evaluate_system import SystemEvaluator
evaluator = SystemEvaluator()
results = evaluator.evaluate_video("video_test.mp4", ground_truth_data)
print(f"Précision: {results['precision']:.2%}")
print(f"Recall: {results['recall']:.2%}")
            ''',
            'metrics_you_get': [
                'Temps de traitement par frame',
                'FPS (images par seconde)',
                'Précision de détection des comportements suspects',
                'Taux de faux positifs/négatifs',
                'Latence du système global'
            ]
        }
    
    def _synthetic_data_method(self):
        """Méthode 2: Données synthétiques contrôlées"""
        return {
            'description': 'Créer des scénarios de test contrôlés',
            'steps': [
                '1. Définir des scénarios types (vol, shopping normal, etc.)',
                '2. Créer/simuler des données pour ces scénarios',
                '3. Tester votre système sur ces données',
                '4. Mesurer les performances sur chaque scénario',
                '5. Analyser les points faibles'
            ],
            'example_scenarios': [
                'Personne met objets dans panier (NORMAL)',
                'Personne met objets dans son sac (SUSPECT)',
                'Personne regarde autour nerveusement (SUSPECT)',
                'Personne fait ses courses calmement (NORMAL)'
            ],
            'metrics_you_get': [
                'Précision par type de comportement',
                'Temps de réponse par scénario',
                'Consistency score (même comportement → même résultat)'
            ]
        }
    
    def _benchmark_datasets_method(self):
        """Méthode 3: Datasets publics de surveillance"""
        return {
            'description': 'Utiliser des datasets de recherche existants',
            'datasets': [
                'UCF-Crime Dataset (comportements anormaux)',
                'CCTV-Fights Dataset (détection d\'incidents)',
                'ShanghaiTech Campus Dataset (détection d\'anomalies)',
                'Avenue Dataset (comportements suspects)'
            ],
            'advantages': [
                'Ground truth déjà disponible',
                'Comparaison avec autres systèmes',
                'Métriques standardisées',
                'Crédibilité académique'
            ],
            'metrics_you_get': [
                'AUC (Area Under Curve)',
                'EER (Equal Error Rate)', 
                'Précision/Recall par classe',
                'Comparaison avec état de l\'art'
            ]
        }
    
    def _live_testing_method(self):
        """Méthode 4: Test en conditions réelles"""
        return {
            'description': 'Déployer et tester en environnement réel',
            'setup': [
                '1. Installer dans un environnement contrôlé',
                '2. Monitorer pendant une période définie',
                '3. Collecter feedback des utilisateurs/security',
                '4. Analyser les incidents détectés vs ratés',
                '5. Mesurer impact opérationnel'
            ],
            'metrics_you_get': [
                'Taux de détection en conditions réelles',
                'Temps de réponse opérationnel',
                'Satisfaction utilisateur',
                'Réduction des incidents non détectés',
                'ROI (retour sur investissement)'
            ]
        }
    
    def generate_evaluation_plan(self, method='video_analysis'):
        """Génère un plan d'évaluation détaillé."""
        
        if method not in self.evaluation_methods:
            return "Méthode non disponible. Choisissez: " + str(list(self.evaluation_methods.keys()))
        
        plan = self.evaluation_methods[method]()
        
        # Génération du plan complet
        evaluation_plan = {
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'plan': plan,
            'next_steps': self._get_next_steps(method),
            'tools_needed': self._get_tools_needed(method)
        }
        
        return evaluation_plan
    
    def _get_next_steps(self, method):
        """Prochaines étapes selon la méthode choisie."""
        steps = {
            'video_analysis': [
                '1. Enregistrer ou trouver 5-10 vidéos de test',
                '2. Annoter manuellement les comportements (ground truth)',
                '3. Modifier evaluate_system.py pour accept video directory',
                '4. Lancer l\'évaluation: python evaluate_system.py --videos ./test_videos',
                '5. Analyser les résultats et calculer métriques finales'
            ],
            'synthetic_data': [
                '1. Définir 10 scénarios types avec résultats attendus',
                '2. Créer dataset synthétique avec ces scénarios',
                '3. Tester votre système sur chaque scénario',
                '4. Calculer accuracy par scénario',
                '5. Identifier scenarios problématiques'
            ]
        }
        return steps.get(method, ['Voir documentation de la méthode'])
    
    def _get_tools_needed(self, method):
        """Outils nécessaires selon la méthode."""
        tools = {
            'video_analysis': [
                'Caméra/videos de test',
                'Outil d\'annotation (LabelImg, CVAT)',
                'evaluate_system.py (déjà créé)',
                'Calculs statistiques (pandas, sklearn.metrics)'
            ],
            'synthetic_data': [
                'Générateur de données synthétiques',
                'Scripts de scénarios prédéfinis',
                'Système de validation automatisée'
            ]
        }
        return tools.get(method, ['Voir documentation'])
    
    def save_plan(self, plan, filename=None):
        """Sauvegarde le plan d'évaluation."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_plan_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Plan d'évaluation sauvé: {filename}")

def main():
    """Fonction principale - génère votre plan d'évaluation."""
    guide = RealEvaluationGuide()
    
    print("🎯 GUIDE D'ÉVALUATION RÉELLE DE VOTRE SYSTÈME")
    print("=" * 50)
    print()
    print("Méthodes disponibles:")
    print("1. video_analysis - Analyser vos propres vidéos (RECOMMANDÉ)")
    print("2. synthetic_data - Données synthétiques contrôlées") 
    print("3. benchmark_datasets - Datasets publics de recherche")
    print("4. live_testing - Test en conditions réelles")
    print()
    
    # Générer plan pour analyse vidéo (méthode recommandée)
    method = 'video_analysis'
    plan = guide.generate_evaluation_plan(method)
    
    print(f"📋 Plan généré pour: {method}")
    print("-" * 30)
    print(f"Description: {plan['plan']['description']}")
    print()
    print("Étapes à suivre:")
    for step in plan['plan']['steps']:
        print(f"  {step}")
    print()
    print("Métriques que vous obtiendrez:")
    for metric in plan['plan']['metrics_you_get']:
        print(f"  • {metric}")
    print()
    print("Prochaines actions:")
    for action in plan['next_steps']:
        print(f"  ▶ {action}")
    
    # Sauvegarder le plan
    guide.save_plan(plan)
    
    print(f"\n🚀 VOTRE SYSTÈME EST PRÊT À ÊTRE ÉVALUÉ!")
    print("Commencez par la méthode 'video_analysis' pour des résultats rapides.")

if __name__ == "__main__":
    main()