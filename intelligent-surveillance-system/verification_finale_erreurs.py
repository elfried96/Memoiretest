#!/usr/bin/env python3
"""
🔍 VÉRIFICATION FINALE DES ERREURS - Dashboard Intégré
=====================================================

Script de vérification exhaustive pour détecter toutes les erreurs
potentielles avant le lancement du dashboard.
"""

import ast
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

def check_python_syntax(file_path: Path) -> Tuple[bool, str]:
    """Vérifie la syntaxe Python d'un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, "✅ Syntaxe valide"
    except SyntaxError as e:
        return False, f"❌ Erreur syntaxe ligne {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"❌ Erreur: {e}"

def check_imports(file_path: Path) -> List[Dict]:
    """Vérifie les imports et détecte les problèmes potentiels."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        
        # Import dashboard. sans try/except
        if re.match(r'^from dashboard\\.', line) and 'try:' not in lines[max(0, i-3):i]:
            issues.append({
                'line': i,
                'type': 'import_risk',
                'message': f"Import dashboard absolu sans gestion d'erreur: {line}",
                'severity': 'medium'
            })
        
        # Imports manquants critiques
        critical_imports = ['streamlit', 'plotly', 'pandas', 'numpy']
        for imp in critical_imports:
            if f'import {imp}' in line or f'from {imp}' in line:
                if 'try:' not in lines[max(0, i-3):i]:
                    issues.append({
                        'line': i,
                        'type': 'missing_try_catch',
                        'message': f"Import critique sans gestion d'erreur: {imp}",
                        'severity': 'low'
                    })
    
    return issues

def check_function_definitions(file_path: Path) -> List[Dict]:
    """Vérifie les définitions de fonction."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fonctions critiques qui doivent être définies
    critical_functions = [
        'generate_auto_description',
        'trigger_integrated_alert', 
        'render_detection_timeline',
        'render_auto_descriptions',
        'render_alert_controls'
    ]
    
    for func in critical_functions:
        if f'def {func}(' not in content:
            issues.append({
                'type': 'missing_function',
                'message': f"Fonction critique manquante: {func}",
                'severity': 'high'
            })
        else:
            # Vérifie si la fonction est appelée
            if f'{func}()' not in content:
                issues.append({
                    'type': 'unused_function',
                    'message': f"Fonction définie mais non utilisée: {func}",
                    'severity': 'medium'
                })
    
    return issues

def check_session_state_usage(file_path: Path) -> List[Dict]:
    """Vérifie l'utilisation des variables session_state."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Variables session_state critiques
    critical_vars = [
        'alert_thresholds',
        'auto_descriptions', 
        'real_detections',
        'real_alerts'
    ]
    
    # Variables utilisées
    used_vars = re.findall(r'st\.session_state\.([a-zA-Z_]+)', content)
    used_vars.extend(re.findall(r"st\.session_state\[['\"]([a-zA-Z_]+)['\"]\]", content))
    
    # Variables initialisées
    init_patterns = [
        r"if ['\"]([a-zA-Z_]+)['\"] not in st\.session_state:",
        r"st\.session_state\.([a-zA-Z_]+)\s*=",
        r"st\.session_state\[['\"]([a-zA-Z_]+)['\"]\]\s*="
    ]
    
    initialized_vars = []
    for pattern in init_patterns:
        initialized_vars.extend(re.findall(pattern, content))
    
    # Vérifie les variables critiques
    for var in critical_vars:
        if var not in initialized_vars:
            issues.append({
                'type': 'uninitialized_session_var',
                'message': f"Variable session_state critique non initialisée: {var}",
                'severity': 'high'
            })
    
    # Variables utilisées mais non initialisées
    for var in set(used_vars):
        if var not in initialized_vars and var not in ['pipeline_initialized', 'cameras']:
            issues.append({
                'type': 'uninitialized_var',
                'message': f"Variable utilisée mais non initialisée: {var}",
                'severity': 'medium'
            })
    
    return issues

def check_audio_integration(file_path: Path) -> List[Dict]:
    """Vérifie l'intégration du système audio."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifications audio
    audio_checks = [
        ('AUDIO_AVAILABLE', "Variable AUDIO_AVAILABLE non définie"),
        ('AudioAlertSystem', "AudioAlertSystem non importé"),
        ('play_alert', "Fonction play_alert non importée"),
        ('trigger_integrated_alert', "Fonction trigger_integrated_alert non définie")
    ]
    
    for check, message in audio_checks:
        if check not in content:
            issues.append({
                'type': 'missing_audio_integration',
                'message': message,
                'severity': 'high' if 'AUDIO_AVAILABLE' in check else 'medium'
            })
    
    return issues

def check_tab_structure(file_path: Path) -> List[Dict]:
    """Vérifie la structure des onglets."""
    issues = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifie la présence du nouvel onglet
    if 'Timeline & Descriptions' not in content:
        issues.append({
            'type': 'missing_tab',
            'message': "Nouvel onglet 'Timeline & Descriptions' manquant",
            'severity': 'high'
        })
    
    # Vérifie la structure des onglets
    tab_pattern = r'tab\d+, tab\d+, tab\d+, tab\d+, tab\d+, tab\d+ = st\.tabs\(\['
    if not re.search(tab_pattern, content):
        issues.append({
            'type': 'incorrect_tab_structure',
            'message': "Structure des onglets incorrecte (doit avoir 6 onglets)",
            'severity': 'medium'
        })
    
    return issues

def run_comprehensive_check():
    """Lance une vérification complète."""
    print("🔍 VÉRIFICATION FINALE COMPLÈTE")
    print("=" * 50)
    
    dashboard_file = Path("dashboard/production_dashboard.py")
    audio_file = Path("dashboard/utils/audio_alerts.py")
    
    if not dashboard_file.exists():
        print("❌ ERREUR: Fichier dashboard principal non trouvé")
        return False
    
    if not audio_file.exists():
        print("⚠️ WARNING: Fichier audio_alerts.py non trouvé")
    
    all_issues = []
    
    # Tests de syntaxe
    print("\n📝 Vérification syntaxe Python...")
    syntax_ok, syntax_msg = check_python_syntax(dashboard_file)
    print(f"   {syntax_msg}")
    if not syntax_ok:
        return False
    
    if audio_file.exists():
        audio_syntax_ok, audio_syntax_msg = check_python_syntax(audio_file)
        print(f"   Audio: {audio_syntax_msg}")
    
    # Tests des imports
    print("\n📦 Vérification imports...")
    import_issues = check_imports(dashboard_file)
    all_issues.extend(import_issues)
    print(f"   {len(import_issues)} problèmes d'import détectés")
    
    # Tests des fonctions
    print("\n⚙️ Vérification fonctions...")
    function_issues = check_function_definitions(dashboard_file)
    all_issues.extend(function_issues)
    print(f"   {len(function_issues)} problèmes de fonction détectés")
    
    # Tests session_state
    print("\n💾 Vérification session_state...")
    session_issues = check_session_state_usage(dashboard_file)
    all_issues.extend(session_issues)
    print(f"   {len(session_issues)} problèmes session_state détectés")
    
    # Tests intégration audio
    print("\n🔊 Vérification intégration audio...")
    audio_issues = check_audio_integration(dashboard_file)
    all_issues.extend(audio_issues)
    print(f"   {len(audio_issues)} problèmes audio détectés")
    
    # Tests structure onglets
    print("\n📊 Vérification structure onglets...")
    tab_issues = check_tab_structure(dashboard_file)
    all_issues.extend(tab_issues)
    print(f"   {len(tab_issues)} problèmes onglets détectés")
    
    # Rapport détaillé
    print("\n" + "=" * 50)
    print("📋 RAPPORT DÉTAILLÉ")
    print("=" * 50)
    
    if not all_issues:
        print("🎉 AUCUNE ERREUR DÉTECTÉE !")
        print("✅ Le dashboard est prêt pour le lancement")
        return True
    
    # Grouper par sévérité
    high_issues = [i for i in all_issues if i.get('severity') == 'high']
    medium_issues = [i for i in all_issues if i.get('severity') == 'medium']
    low_issues = [i for i in all_issues if i.get('severity') == 'low']
    
    print(f"🚨 ERREURS CRITIQUES: {len(high_issues)}")
    for issue in high_issues:
        print(f"   ❌ {issue['message']}")
        if 'line' in issue:
            print(f"      Ligne {issue['line']}")
    
    print(f"\n⚠️ AVERTISSEMENTS: {len(medium_issues)}")
    for issue in medium_issues:
        print(f"   🟡 {issue['message']}")
        if 'line' in issue:
            print(f"      Ligne {issue['line']}")
    
    print(f"\n💡 SUGGESTIONS: {len(low_issues)}")
    for issue in low_issues:
        print(f"   🔵 {issue['message']}")
        if 'line' in issue:
            print(f"      Ligne {issue['line']}")
    
    # Verdict final
    if high_issues:
        print(f"\n❌ VERDICT: {len(high_issues)} erreurs critiques à corriger avant lancement")
        return False
    elif medium_issues:
        print(f"\n⚠️ VERDICT: {len(medium_issues)} avertissements - Lancement possible mais recommandations à suivre")
        return True
    else:
        print("\n✅ VERDICT: Dashboard prêt avec quelques suggestions mineures")
        return True

if __name__ == "__main__":
    success = run_comprehensive_check()
    
    if success:
        print(f"\n🚀 DASHBOARD PRÊT POUR LE LANCEMENT !")
        print("Commande: cd dashboard && streamlit run production_dashboard.py")
    else:
        print(f"\n🔧 CORRECTIONS NÉCESSAIRES")
    
    exit(0 if success else 1)