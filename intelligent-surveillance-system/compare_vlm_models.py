#!/usr/bin/env python3
"""Script de comparaison entre Kimi-VL et Qwen2-VL."""

import os
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime


class VLMComparison:
    """Outil de comparaison des modèles VLM."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "kimi_vl": None,
            "qwen2_vl": None,
            "comparison": {}
        }
    
    def run_benchmark(self, script_name: str, model_name: str):
        """Exécuter un benchmark et capturer les résultats."""
        
        print(f"\n🚀 Lancement benchmark {model_name}")
        print("=" * 50)
        
        try:
            # Exécuter le script de test
            result = subprocess.run(
                ["python", script_name], 
                capture_output=True, 
                text=True,
                timeout=1800  # 30 minutes max
            )
            
            if result.returncode == 0:
                print(f"✅ {model_name} - Test réussi")
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            else:
                print(f"❌ {model_name} - Test échoué (code: {result.returncode})")
                return {
                    "success": False,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} - Timeout (30 min dépassé)")
            return {
                "success": False,
                "error": "timeout",
                "timeout": True
            }
            
        except Exception as e:
            print(f"❌ {model_name} - Erreur: {e}")
            return {
                "success": False,
                "error": str(e),
                "exception": True
            }
    
    def parse_results(self, output: str) -> dict:
        """Parser les résultats d'un benchmark."""
        
        parsed = {
            "tests_realises": 0,
            "taux_succes": 0.0,
            "temps_moyen": 0.0,
            "temps_total": 0.0,
            "details_tests": [],
            "raw_output": output
        }
        
        lines = output.split('\n')
        
        for line in lines:
            # Extraction des métriques principales
            if "Tests réalisés:" in line:
                try:
                    parsed["tests_realises"] = int(line.split(':')[1].strip())
                except:
                    pass
            
            elif "Taux de succès:" in line:
                try:
                    parsed["taux_succes"] = float(line.split(':')[1].strip().rstrip('%'))
                except:
                    pass
            
            elif "Temps moyen:" in line:
                try:
                    parsed["temps_moyen"] = float(line.split(':')[1].strip().rstrip('s'))
                except:
                    pass
            
            elif "Temps total:" in line:
                try:
                    parsed["temps_total"] = float(line.split(':')[1].strip().rstrip('s'))
                except:
                    pass
            
            # Détails des tests individuels
            elif line.strip().startswith("✅") or line.strip().startswith("❌"):
                parsed["details_tests"].append(line.strip())
        
        return parsed
    
    def generate_comparison(self):
        """Générer la comparaison détaillée."""
        
        if not self.results["kimi_vl"] or not self.results["qwen2_vl"]:
            print("❌ Impossible de comparer - résultats manquants")
            return
        
        kimi_success = self.results["kimi_vl"]["success"]
        qwen_success = self.results["qwen2_vl"]["success"]
        
        comparison = {
            "both_working": kimi_success and qwen_success,
            "kimi_only": kimi_success and not qwen_success,
            "qwen_only": not kimi_success and qwen_success,
            "both_failed": not kimi_success and not qwen_success
        }
        
        if comparison["both_working"]:
            # Comparaison détaillée des performances
            kimi_parsed = self.parse_results(self.results["kimi_vl"]["stdout"])
            qwen_parsed = self.parse_results(self.results["qwen2_vl"]["stdout"])
            
            comparison["performance"] = {
                "kimi_vl": kimi_parsed,
                "qwen2_vl": qwen_parsed,
                "winner": self.determine_winner(kimi_parsed, qwen_parsed)
            }
        
        self.results["comparison"] = comparison
        return comparison
    
    def determine_winner(self, kimi_stats: dict, qwen_stats: dict) -> dict:
        """Déterminer le modèle gagnant selon différents critères."""
        
        winner = {
            "overall": None,
            "speed": None,
            "reliability": None,
            "efficiency": None,
            "scores": {
                "kimi_vl": 0,
                "qwen2_vl": 0
            },
            "reasoning": []
        }
        
        # Critère 1: Fiabilité (taux de succès)
        if kimi_stats["taux_succes"] > qwen_stats["taux_succes"]:
            winner["reliability"] = "kimi_vl"
            winner["scores"]["kimi_vl"] += 3
            winner["reasoning"].append(f"Kimi-VL plus fiable ({kimi_stats['taux_succes']:.1f}% vs {qwen_stats['taux_succes']:.1f}%)")
        elif qwen_stats["taux_succes"] > kimi_stats["taux_succes"]:
            winner["reliability"] = "qwen2_vl"
            winner["scores"]["qwen2_vl"] += 3
            winner["reasoning"].append(f"Qwen2-VL plus fiable ({qwen_stats['taux_succes']:.1f}% vs {kimi_stats['taux_succes']:.1f}%)")
        else:
            winner["reasoning"].append("Fiabilité égale")
        
        # Critère 2: Vitesse (temps moyen)
        if kimi_stats["temps_moyen"] < qwen_stats["temps_moyen"]:
            winner["speed"] = "kimi_vl"
            winner["scores"]["kimi_vl"] += 2
            winner["reasoning"].append(f"Kimi-VL plus rapide ({kimi_stats['temps_moyen']:.2f}s vs {qwen_stats['temps_moyen']:.2f}s)")
        elif qwen_stats["temps_moyen"] < kimi_stats["temps_moyen"]:
            winner["speed"] = "qwen2_vl"
            winner["scores"]["qwen2_vl"] += 2
            winner["reasoning"].append(f"Qwen2-VL plus rapide ({qwen_stats['temps_moyen']:.2f}s vs {kimi_stats['temps_moyen']:.2f}s)")
        else:
            winner["reasoning"].append("Vitesse similaire")
        
        # Critère 3: Efficacité (tests réussis par seconde)
        kimi_efficiency = (kimi_stats["taux_succes"] / 100) * kimi_stats["tests_realises"] / max(kimi_stats["temps_total"], 1)
        qwen_efficiency = (qwen_stats["taux_succes"] / 100) * qwen_stats["tests_realises"] / max(qwen_stats["temps_total"], 1)
        
        if kimi_efficiency > qwen_efficiency:
            winner["efficiency"] = "kimi_vl"
            winner["scores"]["kimi_vl"] += 1
            winner["reasoning"].append(f"Kimi-VL plus efficace ({kimi_efficiency:.2f} vs {qwen_efficiency:.2f} tests/s)")
        elif qwen_efficiency > kimi_efficiency:
            winner["efficiency"] = "qwen2_vl"
            winner["scores"]["qwen2_vl"] += 1
            winner["reasoning"].append(f"Qwen2-VL plus efficace ({qwen_efficiency:.2f} vs {kimi_efficiency:.2f} tests/s)")
        
        # Gagnant global
        if winner["scores"]["kimi_vl"] > winner["scores"]["qwen2_vl"]:
            winner["overall"] = "kimi_vl"
        elif winner["scores"]["qwen2_vl"] > winner["scores"]["kimi_vl"]:
            winner["overall"] = "qwen2_vl"
        else:
            winner["overall"] = "tie"
        
        return winner
    
    def print_comparison_report(self):
        """Afficher le rapport de comparaison."""
        
        print("\n" + "="*60)
        print("📊 RAPPORT DE COMPARAISON VLM")
        print("="*60)
        
        comparison = self.results["comparison"]
        
        # Status global
        print(f"\n🔍 Status des modèles:")
        print(f"  • Kimi-VL: {'✅ Fonctionnel' if self.results['kimi_vl']['success'] else '❌ Échec'}")
        print(f"  • Qwen2-VL: {'✅ Fonctionnel' if self.results['qwen2_vl']['success'] else '❌ Échec'}")
        
        # Scénarios
        if comparison["both_working"]:
            print(f"\n🎯 Les deux modèles fonctionnent - Comparaison possible!")
            
            perf = comparison["performance"]
            winner = perf["winner"]
            
            print(f"\n📈 Performances:")
            print(f"  Kimi-VL:   {perf['kimi_vl']['taux_succes']:.1f}% réussite, {perf['kimi_vl']['temps_moyen']:.2f}s moyen")
            print(f"  Qwen2-VL:  {perf['qwen2_vl']['taux_succes']:.1f}% réussite, {perf['qwen2_vl']['temps_moyen']:.2f}s moyen")
            
            print(f"\n🏆 Gagnant: {winner['overall'].upper() if winner['overall'] != 'tie' else 'ÉGALITÉ'}")
            print(f"  Scores: Kimi-VL {winner['scores']['kimi_vl']} - Qwen2-VL {winner['scores']['qwen2_vl']}")
            
            print(f"\n💡 Analyse détaillée:")
            for reason in winner["reasoning"]:
                print(f"  • {reason}")
            
            # Recommandation
            print(f"\n🎯 Recommandation:")
            if winner['overall'] == 'kimi_vl':
                print("  👑 Utilisez Kimi-VL pour ce projet")
                print("  🔧 Configuration recommandée: kimi-vl-a3b-thinking")
            elif winner['overall'] == 'qwen2_vl':
                print("  👑 Utilisez Qwen2-VL pour ce projet")
                print("  🔧 Configuration recommandée: qwen2-vl-7b-instruct")
            else:
                print("  ⚖️  Performance équivalente - choisir selon les préférences:")
                print("     • Kimi-VL: Plus récent, spécialisé surveillance")
                print("     • Qwen2-VL: Plus mature, OCR intégré")
        
        elif comparison["kimi_only"]:
            print(f"\n✅ Seul Kimi-VL fonctionne")
            print(f"  🎯 Recommandation: Utiliser Kimi-VL exclusivement")
        
        elif comparison["qwen_only"]:
            print(f"\n✅ Seul Qwen2-VL fonctionne") 
            print(f"  🎯 Recommandation: Utiliser Qwen2-VL exclusivement")
        
        else:
            print(f"\n❌ Aucun modèle ne fonctionne")
            print(f"  🔧 Action requise: Vérifier la configuration système")
    
    def save_results(self, filename: str = "vlm_comparison_results.json"):
        """Sauvegarder les résultats."""
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvés dans {filename}")
    
    def run_full_comparison(self):
        """Exécuter la comparaison complète."""
        
        print("🚀 COMPARAISON KIMI-VL vs QWEN2-VL")
        print("=" * 60)
        print("⚠️  Cette comparaison peut prendre 30-60 minutes")
        print("🔄 Les modèles seront testés séquentiellement pour éviter les conflits")
        
        input("\nPress Enter pour continuer ou Ctrl+C pour annuler...")
        
        # Test Kimi-VL
        self.results["kimi_vl"] = self.run_benchmark("test_kimi_vl_only.py", "Kimi-VL")
        
        # Test Qwen2-VL  
        self.results["qwen2_vl"] = self.run_benchmark("test_qwen2_vl_only.py", "Qwen2-VL")
        
        # Analyse comparative
        self.generate_comparison()
        
        # Rapport final
        self.print_comparison_report()
        
        # Sauvegarde
        self.save_results()


if __name__ == "__main__":
    comparator = VLMComparison()
    
    try:
        comparator.run_full_comparison()
    except KeyboardInterrupt:
        print("\n⏹️  Comparaison interrompue")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()