"""
Pages Dash de l'application hydrologique
Vues individuelles pour chaque module
"""

from .home import create_home_page
from .eto_page import create_etp_page
from .bias_page import create_bias_page
from .modeling_page import create_modeling_page
from .prediction_page import create_prediction_page
from .help_page import create_help_page
from .about_page import create_about_page

# Dictionnaire des pages
PAGES = {
    "home": {
        "name": "Accueil",
        "function": create_home_page,
        "description": "Tableau de bord principal",
        "icon": "fas fa-home"
    },
    "eto": {
        "name": "Calcul ETP",
        "function": create_etp_page,
        "description": "Calcul d'évapotranspiration potentielle",
        "icon": "fas fa-sun"
    },
    "bias": {
        "name": "Correction de Biais",
        "function": create_bias_page,
        "description": "Correction statistique des modèles climatiques",
        "icon": "fas fa-adjust"
    },
    "modeling": {
        "name": "Modélisation",
        "function": create_modeling_page,
        "description": "Modèles hydrologiques ModHyPMA et LSTM",
        "icon": "fas fa-project-diagram"
    },
    "prediction": {
        "name": "Prédiction",
        "function": create_prediction_page,
        "description": "Prédictions LSTM avec modèles climatiques",
        "icon": "fas fa-chart-line"
    },
    "help": {
        "name": "Aide",
        "function": create_help_page,
        "description": "Documentation et support",
        "icon": "fas fa-question-circle"
    },
    "about": {  # Nouvelle page
        "name": "À propos",
        "function": create_about_page,
        "description": "Le développeur et la technologie",
        "icon": "fas fa-info-circle"
    }
}

def get_page_function(page_name):
    """Retourne la fonction de création d'une page"""
    page_info = PAGES.get(page_name)
    if page_info:
        return page_info["function"]
    return create_home_page  # Page par défaut

def get_page_info(page_name):
    """Retourne les informations d'une page"""
    return PAGES.get(page_name, {})

def get_all_pages():
    """Retourne la liste de toutes les pages"""
    return [
        {
            "id": page_id,
            "name": info["name"],
            "description": info["description"],
            "icon": info["icon"]
        }
        for page_id, info in PAGES.items()
    ]

__all__ = [
    'create_home_page',
    'create_etp_page',
    'create_bias_page',
    'create_modeling_page',
    'create_prediction_page',
    'create_help_page',
    'PAGES',
    'get_page_function',
    'get_page_info',
    'get_all_pages'
]

print(f"Pages Dash chargées - {len(PAGES)} pages disponibles")