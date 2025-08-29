"""
Pipeline Tester pour les tests d'intégration.
"""

from typing import Dict, Any, List
import asyncio
import time
from dataclasses import dataclass


@dataclass
class TestConfiguration:
    """Configuration pour les tests."""
    name: str
    description: str = ""
    timeout: float = 600.0  # 10 minutes
    retry_count: int = 3
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class PipelineTester:
    """Testeur simple pour les pipelines."""
    
    def __init__(self):
        self.test_results = []
    
    async def run_test(self, test_name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
        """Exécute un test et retourne le résultat."""
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func(*args, **kwargs)
            else:
                result = test_func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            test_result = {
                "name": test_name,
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            test_result = {
                "name": test_name,
                "success": False,
                "result": None,
                "execution_time": execution_time,
                "error": str(e)
            }
        
        self.test_results.append(test_result)
        return test_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des tests."""
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r["success"]])
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / max(total_tests, 1),
            "total_time": sum(r["execution_time"] for r in self.test_results)
        }