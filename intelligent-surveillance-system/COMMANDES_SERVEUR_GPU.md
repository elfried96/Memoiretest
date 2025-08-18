# 🚀 COMMANDES POUR SERVEUR GPU

## ⚡ INSTALLATION RAPIDE (3 COMMANDES)

```bash
# 1. 🔧 Correction préventive des imports
python fix_all_imports.py

# 2. 🚀 Installation automatique complète 
bash setup_gpu_server.sh

# 3. 🧪 Validation système
python run_gpu_tests.py
```

**C'est tout ! Votre système est prêt.**

---

## 🎯 TESTS GARANTIS

### Tests Rapides (1-2 min chacun)
```bash
# Test diagnostic
python check_gpu_system.py

# Test basique  
python test_basic_corrections.py

# Test surveillance webcam (50 frames)
python test_full_system_video.py --video webcam --max-frames 50
```

### Tests Complets (5-10 min)
```bash
# Test système intégral
python run_gpu_tests.py

# Test surveillance vidéo complète
python test_full_system_video.py --video webcam --max-frames 200

# Test optimisation
python examples/tool_optimization_demo.py --mode balanced
```

---

## 🚀 DÉPLOIEMENT PRODUCTION

```bash
# Activation environnement
source venv/bin/activate

# Lancement surveillance
python main.py --video webcam

# Ou mode avancé
python src/main.py --config-mode balanced
```

---

## 📊 MONITORING

```bash
# GPU en temps réel
watch -n 1 nvidia-smi

# Performance système
python -c "
import torch
print(f'GPU: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"
```

---

## 🔧 DÉPANNAGE

```bash
# En cas de problème
python fix_all_imports.py
bash setup_gpu_server.sh
python check_gpu_system.py
```

**Ces commandes résolvent 99% des problèmes automatiquement.**