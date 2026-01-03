"""Utilitaires partagés pour le projet VLM-Bot."""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging
from dotenv import load_dotenv
import os


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis config.yaml.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Configuration sous forme de dictionnaire
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(level: str = "INFO") -> None:
    """
    Configure le système de logging.
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_environment() -> None:
    """Charge les variables d'environnement depuis .env."""
    load_dotenv()
    
    # Vérifier les variables requises
    required_vars = ['HF_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Variables d'environnement manquantes: {', '.join(missing_vars)}\n"
            f"Copiez .env.example vers .env et remplissez les valeurs."
        )


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Crée les répertoires nécessaires s'ils n'existent pas.
    
    Args:
        config: Configuration du projet
    """
    dirs_to_create = [
        'data/raw',
        'data/processed',
        'models',
        'logs'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def format_prompt(
    context: str,
    rag_context: str,
    prompt_type: str = "direct"
) -> str:
    """
    Construit le prompt pour le VLM.

    Args:
        context: Contexte textuel optionnel (ex: mesures, notes cliniques)
        rag_context: Contexte médical depuis RAG
        prompt_type: Type de prompt ('with_context' ou 'direct')

    Returns:
        Prompt formaté
    """
    if prompt_type == "with_context":
        return f"""
{context}

================================================================================
RELEVANT MEDICAL LITERATURE:
================================================================================
{rag_context}

================================================================================
EVIDENCE-BASED ANALYSIS INSTRUCTIONS:
================================================================================

Using the provided clinical context and the retrieved medical literature, provide:

1. DIFFERENTIAL DIAGNOSIS (ranked)
2. Key concerning features with evidence
3. Comparison to literature patterns
4. Clinical recommendations and urgency
5. Patient communication guidance

Include citations [Source X] for factual claims.
"""
    else:
        return f"""
COMPREHENSIVE DERMATOLOGICAL LESION ANALYSIS:

Analyze this skin lesion image or clinical description using the retrieved literature.

================================================================================
RELEVANT MEDICAL LITERATURE:
================================================================================
{rag_context}

Provide a clear, evidence-based clinical interpretation with differential diagnosis
and recommended next steps. Cite sources where appropriate.
"""
