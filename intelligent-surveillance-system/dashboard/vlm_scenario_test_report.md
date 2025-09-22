# 📋 RAPPORT DE TESTS SCÉNARIOS VLM
==================================================
Date: 2025-09-18 06:53:54
Nombre de scénarios testés: 3

## Scénario 1: Personne qui fait du jogging le matin
**ID:** faux_positif_jogging
**Contexte:** {'time': '06:30', 'location': 'Zone résidentielle', 'weather': 'Ensoleillé', 'day_type': 'Jour de semaine'}

### Raisonnement VLM:
- Analyse temporelle: Heure diurne normale
- Analyse spatiale: Zone résidentielle autorisée
- Analyse comportementale: Mouvement sportif normal
- Objets détectés: person
- Conclusion: Activité normale

### Résultat:
- **Alerte:** Non
- **Classification:** Activité normale
- **Confiance:** high

### Évaluation:
**Résultat:** ✅ RÉUSSI

## Scénario 2: Intrusion nocturne dans zone commerciale
**ID:** vrai_positif_intrusion
**Contexte:** {'time': '02:15', 'location': 'Zone commerciale', 'weather': 'Nuageux', 'day_type': 'Nuit en semaine'}

### Raisonnement VLM:
- Analyse temporelle: Heure nocturne suspecte
- Analyse spatiale: Zone commerciale fermée
- Analyse comportementale: Évitement caméras suspect
- Objets détectés: person, bag, tool
- Conclusion: Intrusion probable

### Résultat:
- **Alerte:** Oui
- **Classification:** Intrusion probable
- **Confiance:** high

### Évaluation:
**Résultat:** ✅ RÉUSSI

## Scénario 3: Livreur tardif dans zone résidentielle
**ID:** cas_ambigu_livraison
**Contexte:** {'time': '21:45', 'location': 'Zone résidentielle', 'weather': 'Pluvieux', 'day_type': 'Vendredi soir'}

### Raisonnement VLM:
- Analyse temporelle: Heure diurne normale
- Analyse spatiale: Zone résidentielle autorisée
- Analyse comportementale: Vérification adresses normale
- Objets détectés: person, package, vehicle
- Conclusion: Activité à surveiller

### Résultat:
- **Alerte:** Oui
- **Classification:** Activité à surveiller
- **Confiance:** medium

### Évaluation:
**Résultat:** ✅ RÉUSSI
