# 🧪 Guide des Tests - Outils Avancés de Surveillance

## 📋 Vue d'ensemble

Ce guide vous explique comment tester individuellement chacun des 8 outils avancés du système de surveillance intelligente, ainsi que comment exécuter la suite complète de tests.

## 📂 Structure des Fichiers de Test

```
intelligent-surveillance-system/
├── src/
│   └── advanced_tools/           # Code source des outils
│       ├── __init__.py
│       ├── sam2_segmentation.py
│       ├── dino_features.py
│       ├── pose_estimation.py
│       ├── trajectory_analyzer.py
│       ├── multimodal_fusion.py
│       ├── temporal_transformer.py
│       ├── adversarial_detector.py
│       └── domain_adapter.py
├── test_sam2_segmentator.py      # Tests SAM2 segmentation
├── test_dino_features.py         # Tests extraction features DINO v2
├── test_pose_estimation.py       # Tests estimation de poses
├── test_trajectory_analyzer.py   # Tests analyse trajectoires
├── test_multimodal_fusion.py     # Tests fusion multimodale
├── test_temporal_transformer.py  # Tests analyse temporelle
├── test_adversarial_detector.py  # Tests détection adversariale
├── test_domain_adapter.py        # Tests adaptation domaines
├── run_all_tests.py              # Script principal pour tous les tests
├── RAPPORT_TESTS_OUTILS_AVANCES.md  # Rapport consolidé
└── README_TESTS.md               # Ce guide
```

## 🚀 Installation des Dépendances

### Dépendances Minimales (pour fallbacks)
```bash
pip install numpy opencv-python scikit-learn scipy
```

### Dépendances Complètes (pour tous les modèles)
```bash
pip install numpy opencv-python torch torchvision scikit-learn scipy transformers mediapipe pillow tensorflow
```

### Installation via requirements.txt (recommandé)
```bash
pip install -r requirements.txt
```

## 🎯 Exécution des Tests

### Option 1: Tests Individuels

Testez un outil spécifique :

```bash
# Test SAM2 Segmentation
python test_sam2_segmentator.py

# Test DINO v2 Features
python test_dino_features.py

# Test OpenPose Estimation
python test_pose_estimation.py

# Test Trajectory Analysis
python test_trajectory_analyzer.py

# Test Multimodal Fusion
python test_multimodal_fusion.py

# Test Temporal Transformer
python test_temporal_transformer.py

# Test Adversarial Detection
python test_adversarial_detector.py

# Test Domain Adaptation
python test_domain_adapter.py
```

### Option 2: Suite Complète

Exécutez tous les tests automatiquement :

```bash
python run_all_tests.py
```

## 📊 Interprétation des Résultats

### Symboles de Status
- ✅ **Succès** - Fonctionnalité opérationnelle
- ⚠️ **Attention** - Fonctionne avec limitations
- ❌ **Échec** - Problème détecté
- 🎯 **Information** - Détail technique

### Types de Tests

Chaque outil est testé sur :

1. **Fonctionnalités de base** - API principale
2. **Cas normaux** - Données typiques
3. **Cas limites** - Données extrêmes
4. **Gestion d'erreurs** - Resilience aux pannes
5. **Performance** - Temps d'exécution
6. **Fallbacks** - Alternatives en cas d'échec

## 🔧 Résolution des Problèmes Courants

### Problème: Module non trouvé
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution:** Installez les dépendances manquantes
```bash
pip install numpy opencv-python scikit-learn scipy
```

### Problème: Modèles non disponibles
```
WARNING: Could not load DINO v2: ..., using fallback
```
**Solution:** Normal ! Les fallbacks sont prévus. Pour les modèles complets :
```bash
pip install torch transformers mediapipe tensorflow
```

### Problème: Erreur GPU/CUDA
```
Expected all tensors to be on the same device
```
**Solution:** Les tests utilisent automatiquement les fallbacks CPU

### Problème: Permissions ou chemins
```
FileNotFoundError: ...
```
**Solution:** Exécutez depuis le répertoire racine du projet
```bash
cd intelligent-surveillance-system/
python test_sam2_segmentator.py
```

## 📈 Comprendre les Métriques

### SAM2Segmentator
- **Temps de traitement** - Performance segmentation
- **Nombre de masques** - Objets détectés
- **Propriétés masques** - Aire, périmètre, compacité, solidité

### DinoV2FeatureExtractor  
- **Similarité cosinus** - Comparaison features (0-1)
- **Clustering inertie** - Qualité regroupement (plus bas = mieux)
- **Temps extraction** - Performance modèle

### OpenPoseEstimator
- **Score comportement** - Suspicion détectée (0-1)
- **Nombre indicateurs** - Comportements suspects
- **Keypoints détectés** - Points corporels trouvés

### TrajectoryAnalyzer
- **Score anomalie** - Mouvement suspect (0-1)
- **Classification pattern** - Type de mouvement
- **Changements direction** - Erraticité du mouvement

### MultiModalFusion
- **Prédiction finale** - Score de confiance (0-1)
- **Poids attention** - Importance de chaque modalité
- **Shape features** - Dimension vecteur fusionné

### TemporalTransformer
- **Consistance séquence** - Stabilité temporelle (0-1)
- **Patterns temporels** - Types détectés
- **Score anomalie** - Changements suspects (0-1)

### AdversarialDetector
- **Est adversarial** - Attaque détectée (Oui/Non)
- **Type attaque** - FGSM, PGD, C&W, DeepFool
- **Score robustesse** - Résistance aux attaques (0-1)

### DomainAdapter
- **Score alignement** - Adaptation réussie (0-1)
- **Amélioration confiance** - Gain performance
- **Domaine détecté** - Environment identifié

## 🎯 Cas d'Usage des Tests

### Pour les Développeurs
- Validation après modifications du code
- Tests de régression automatiques
- Profiling de performance

### Pour les DevOps
- Tests d'intégration continue
- Validation déploiement
- Monitoring santé système

### Pour les Data Scientists
- Validation modèles ML
- Analyse performances algorithmes
- Tuning hyperparamètres

## 📝 Personnaliser les Tests

### Ajouter des Données de Test
```python
# Dans test_sam2_segmentator.py
def create_custom_test_image():
    # Votre image personnalisée
    return custom_image

# Modifier la fonction de test
test_image = create_custom_test_image()
```

### Modifier les Seuils
```python
# Ajuster les critères de validation
confidence_threshold = 0.8  # Plus strict
performance_threshold = 2.0  # Secondes max
```

### Ajouter des Métriques
```python
# Nouvelles métriques dans vos tests
custom_metrics = {
    "memory_usage": get_memory_usage(),
    "accuracy": calculate_accuracy(results),
    "precision": calculate_precision(results)
}
```

## 📊 Génération de Rapports

### Rapport JSON Automatique
```bash
python run_all_tests.py
# Génère: test_results.json
```

### Rapport Markdown
Le fichier `RAPPORT_TESTS_OUTILS_AVANCES.md` contient l'analyse détaillée complète.

### Logs Détaillés
Chaque test produit des logs détaillés avec :
- Timestamp d'exécution
- Métriques de performance
- Messages d'erreur si applicable
- Statistiques de validation

## 🎉 Prochaines Étapes

Après validation des tests individuels :

1. **Tests d'intégration** - Outils ensemble
2. **Tests de charge** - Performance sous stress  
3. **Tests end-to-end** - Pipeline complet
4. **Déploiement production** - Mise en service

---

## 📞 Support

Pour questions ou problèmes :
- 📧 Email : support-surveillance@company.com
- 📱 GitHub Issues : [Créer une issue](https://github.com/company/surveillance/issues)
- 📚 Documentation : [Wiki complet](https://wiki.company.com/surveillance)

---

*Guide maintenu par l'équipe Surveillance IA*  
*Dernière mise à jour : Août 2025*