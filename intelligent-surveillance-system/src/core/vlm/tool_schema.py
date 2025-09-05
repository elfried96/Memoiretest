"""Schémas officiels des outils pour Tool Calling selon standards OpenAI/Kimi 2024."""

from typing import Dict, List, Any


class ToolSchemaBuilder:
    """Constructeur de schémas d'outils pour Tool Calling officiel."""
    
    @staticmethod
    def get_surveillance_tools_schema() -> List[Dict[str, Any]]:
        """Schémas officiels des outils de surveillance selon format OpenAI 2024."""
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "sam2_segmentator",
                    "description": "Segmentation SAM2 pour masques précis des objets détectés",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "detection_boxes": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4
                                },
                                "description": "Boîtes englobantes [x1, y1, x2, y2] des objets à segmenter"
                            },
                            "segment_everything": {
                                "type": "boolean",
                                "description": "Segmenter tous les objets si aucune boîte fournie",
                                "default": False
                            }
                        },
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "dino_features",
                    "description": "Extraction de features DINO v2 pour représentations visuelles robustes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "detection_regions": {
                                "type": "array",
                                "items": {
                                    "type": "array", 
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4
                                },
                                "description": "Régions spécifiques [x1, y1, x2, y2] pour extraction"
                            },
                            "global_features": {
                                "type": "boolean",
                                "description": "Extraire features globales de l'image", 
                                "default": True
                            }
                        },
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "pose_estimator", 
                    "description": "Estimation poses OpenPose pour analyse postures et gestes suspects",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "person_boxes": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"}, 
                                    "minItems": 4,
                                    "maxItems": 4
                                },
                                "description": "Boîtes des personnes [x1, y1, x2, y2] à analyser"
                            },
                            "analyze_behavior": {
                                "type": "boolean",
                                "description": "Activer l'analyse comportementale des poses",
                                "default": True
                            }
                        },
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "trajectory_analyzer",
                    "description": "Analyse trajectoires avancée pour patterns de mouvement suspects",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "person_id": {
                                "type": "string",
                                "description": "ID de tracking de la personne à analyser"
                            },
                            "time_window": {
                                "type": "number",
                                "description": "Fenêtre temporelle d'analyse en secondes",
                                "minimum": 1,
                                "maximum": 300,
                                "default": 30
                            }
                        },
                        "required": ["person_id"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "multimodal_fusion",
                    "description": "Fusion multimodale - agrégation intelligente des données visuelles",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "visual_features": {
                                "type": "boolean",
                                "description": "Inclure les features visuelles DINO",
                                "default": True
                            },
                            "pose_features": {
                                "type": "boolean", 
                                "description": "Inclure les features de pose",
                                "default": True
                            },
                            "motion_features": {
                                "type": "boolean",
                                "description": "Inclure les features de mouvement", 
                                "default": True
                            }
                        },
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "adversarial_detector",
                    "description": "Détection d'attaques adversariales - protection contre manipulations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sensitivity": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": "Niveau de sensibilité de détection",
                                "default": "medium"
                            }
                        },
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]
    
    @staticmethod
    def get_tool_by_name(tool_name: str) -> Dict[str, Any]:
        """Récupère le schéma d'un outil spécifique."""
        tools = ToolSchemaBuilder.get_surveillance_tools_schema()
        for tool in tools:
            if tool["function"]["name"] == tool_name:
                return tool
        return None
    
    @staticmethod  
    def get_tool_names() -> List[str]:
        """Liste des noms d'outils disponibles."""
        tools = ToolSchemaBuilder.get_surveillance_tools_schema()
        return [tool["function"]["name"] for tool in tools]