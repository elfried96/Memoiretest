#!/usr/bin/env python3
"""Script principal pour exécuter tous les tests des outils avancés."""

import sys
import os
import subprocess
import time
import json
from datetime import datetime

def run_test_script(script_name):
    """Execute a test script and capture results."""
    print(f"\n{'='*80}")
    print(f"🧪 EXÉCUTION: {script_name}")
    print('='*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        execution_time = time.time() - start_time
        
        print(result.stdout)
        
        if result.stderr:
            print("\n🚨 STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        
        # Try to extract success information from output
        if "✅ TOUS LES TESTS" in result.stdout:
            test_success = True
        elif "❌ CERTAINS TESTS" in result.stdout:
            test_success = False
        else:
            test_success = success
        
        return {
            'script': script_name,
            'success': test_success,
            'return_code': result.returncode,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Erreur lors de l'exécution: {e}"
        print(f"\n❌ {error_msg}")
        
        return {
            'script': script_name,
            'success': False,
            'return_code': -1,
            'execution_time': execution_time,
            'stdout': '',
            'stderr': error_msg
        }

def generate_test_report(test_results):
    """Generate a comprehensive test report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(test_results),
        'successful_tests': sum(1 for r in test_results if r['success']),
        'failed_tests': sum(1 for r in test_results if not r['success']),
        'total_execution_time': sum(r['execution_time'] for r in test_results),
        'results': test_results
    }
    
    return report

def print_summary(test_results):
    """Print test execution summary."""
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ GLOBAL DES TESTS")
    print('='*80)
    
    total_tests = len(test_results)
    successful = sum(1 for r in test_results if r['success'])
    failed = total_tests - successful
    total_time = sum(r['execution_time'] for r in test_results)
    
    print(f"\n🎯 Statistiques globales:")
    print(f"   - Tests exécutés: {total_tests}")
    print(f"   - Succès: {successful} ✅")
    print(f"   - Échecs: {failed} ❌")
    print(f"   - Taux de réussite: {(successful/total_tests)*100:.1f}%")
    print(f"   - Temps total: {total_time:.2f}s")
    print(f"   - Temps moyen par test: {total_time/total_tests:.2f}s")
    
    print(f"\n📋 Détail par outil:")
    for result in test_results:
        tool_name = result['script'].replace('test_', '').replace('.py', '').replace('_', ' ').title()
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {tool_name:<25} ({result['execution_time']:.2f}s)")
    
    if failed > 0:
        print(f"\n🚨 Outils avec problèmes:")
        for result in test_results:
            if not result['success']:
                tool_name = result['script'].replace('test_', '').replace('.py', '').replace('_', ' ').title()
                print(f"   ❌ {tool_name}")
                if result['stderr']:
                    # Show first line of error
                    first_error_line = result['stderr'].split('\n')[0][:100]
                    print(f"      Erreur: {first_error_line}...")
    
    print(f"\n🎉 Tests terminés à {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Main function to run all tests."""
    print("🚀 SUITE DE TESTS - OUTILS AVANCÉS DE SURVEILLANCE")
    print(f"📅 Démarré à: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of test scripts in logical order
    test_scripts = [
        'test_sam2_segmentator.py',
        'test_dino_features.py', 
        'test_pose_estimation.py',
        'test_trajectory_analyzer.py',
        'test_multimodal_fusion.py',
        'test_temporal_transformer.py',
        'test_adversarial_detector.py',
        'test_domain_adapter.py'
    ]
    
    # Check if all test files exist
    missing_files = []
    for script in test_scripts:
        if not os.path.exists(script):
            missing_files.append(script)
    
    if missing_files:
        print(f"\n❌ Fichiers de test manquants:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\nVeuillez créer ces fichiers avant d'exécuter la suite de tests.")
        return False
    
    print(f"\n📂 Répertoire de travail: {os.getcwd()}")
    print(f"🧪 {len(test_scripts)} scripts de test trouvés")
    
    # Ask for confirmation
    response = input(f"\n▶️  Exécuter tous les tests? [o/N]: ").strip().lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("Tests annulés.")
        return False
    
    print(f"\n🏁 Démarrage des tests...")
    
    # Execute all tests
    test_results = []
    
    for i, script in enumerate(test_scripts, 1):
        print(f"\n📍 Test {i}/{len(test_scripts)}: {script}")
        result = run_test_script(script)
        test_results.append(result)
        
        # Brief status update
        status = "✅ SUCCÈS" if result['success'] else "❌ ÉCHEC"
        print(f"\n{status} - {script} ({result['execution_time']:.2f}s)")
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Generate and save report
    report = generate_test_report(test_results)
    
    try:
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Rapport sauvé dans: test_results.json")
    except Exception as e:
        print(f"\n⚠️  Impossible de sauver le rapport: {e}")
    
    # Print summary
    print_summary(test_results)
    
    # Return overall success
    return all(r['success'] for r in test_results)

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)