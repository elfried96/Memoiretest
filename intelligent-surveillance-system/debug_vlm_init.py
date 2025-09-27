#!/usr/bin/env python3
"""
🔍 Script de Debug VLM - Mode Console
=====================================

Ce script teste l'initialisation VLM en mode console
pour voir tous les logs de debug et identifier le problème.
"""

import sys
import logging
import asyncio
from datetime import datetime

# Configuration logging détaillé
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Ajout des paths
sys.path.append('dashboard')
sys.path.append('src')

def test_imports():
    """Test des imports de base."""
    try:
        logger.info("🧪 Test import dashboard.real_pipeline_integration...")
        from dashboard.real_pipeline_integration import (
            RealVLMPipeline,
            initialize_real_pipeline,
            is_real_pipeline_available,
            CORE_AVAILABLE
        )
        logger.info("✅ Import dashboard réussi")
        logger.info(f"📊 CORE_AVAILABLE: {CORE_AVAILABLE}")
        return True, CORE_AVAILABLE
    except Exception as e:
        logger.error(f"❌ Erreur import: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, False

def test_pipeline_creation():
    """Test création instance pipeline."""
    try:
        logger.info("🏗️ Test création RealVLMPipeline...")
        from dashboard.real_pipeline_integration import RealVLMPipeline
        
        pipeline = RealVLMPipeline(
            vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            enable_optimization=False  # Désactiver pour debug
        )
        logger.info("✅ Instance pipeline créée")
        logger.info(f"📊 Pipeline model: {pipeline.vlm_model_name}")
        logger.info(f"📊 Pipeline initialized: {pipeline.initialized}")
        return True, pipeline
    except Exception as e:
        logger.error(f"❌ Erreur création pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None

async def test_pipeline_init(pipeline):
    """Test initialisation async de la pipeline."""
    try:
        logger.info("🚀 Test initialisation async pipeline...")
        success = await pipeline.initialize()
        logger.info(f"📈 Résultat initialisation: {success}")
        
        if success:
            logger.info("✅ Pipeline initialisée avec succès!")
            logger.info(f"📊 State initialized: {pipeline.initialized}")
            
            # Test des composants
            if hasattr(pipeline, 'orchestrator'):
                logger.info(f"🎯 Orchestrator: {pipeline.orchestrator is not None}")
            if hasattr(pipeline, 'vlm_model'):
                logger.info(f"🤖 VLM Model: {pipeline.vlm_model is not None}")
            if hasattr(pipeline, 'tools_manager'):
                logger.info(f"🛠️ Tools Manager: {pipeline.tools_manager is not None}")
                
        return success
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_sync_init():
    """Test initialisation synchrone."""
    try:
        logger.info("🔄 Test initialize_real_pipeline (sync)...")
        from dashboard.real_pipeline_integration import initialize_real_pipeline
        
        success = initialize_real_pipeline(
            vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            enable_optimization=False
        )
        logger.info(f"📈 Résultat sync init: {success}")
        return success
    except Exception as e:
        logger.error(f"❌ Erreur sync init: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Fonction principale de debug."""
    logger.info("=" * 60)
    logger.info("🔍 DÉMARRAGE DEBUG VLM PIPELINE")
    logger.info("=" * 60)
    
    print(f"🐍 Python: {sys.version}")
    print(f"⏰ Time: {datetime.now()}")
    
    # Étape 1: Test imports
    logger.info("\n📦 ÉTAPE 1: Test des imports")
    import_success, core_available = test_imports()
    if not import_success:
        logger.error("❌ Imports échoués - Arrêt")
        return 1
    
    # Étape 2: Test création pipeline
    logger.info("\n🏗️ ÉTAPE 2: Test création pipeline")
    creation_success, pipeline = test_pipeline_creation()
    if not creation_success:
        logger.error("❌ Création pipeline échouée - Arrêt")
        return 1
    
    # Étape 3: Test initialisation async
    logger.info("\n🚀 ÉTAPE 3: Test initialisation async")
    async_success = asyncio.run(test_pipeline_init(pipeline))
    
    # Étape 4: Test initialisation sync
    logger.info("\n🔄 ÉTAPE 4: Test initialisation sync")
    sync_success = test_sync_init()
    
    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    logger.info(f"✅ Imports: {import_success}")
    logger.info(f"✅ Core disponible: {core_available}")
    logger.info(f"✅ Création pipeline: {creation_success}")
    logger.info(f"✅ Init async: {async_success}")
    logger.info(f"✅ Init sync: {sync_success}")
    
    if async_success or sync_success:
        logger.info("🎉 SUCCÈS: Pipeline VLM fonctionnelle!")
        return 0
    else:
        logger.error("❌ ÉCHEC: Pipeline VLM non fonctionnelle")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n🏁 Exit code: {exit_code}")
    sys.exit(exit_code)