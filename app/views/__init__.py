"""
Interface Dash pour l'application hydrologique
Vues et composants d'interface utilisateur
"""


from .layout import  init_dash_callbacks



# Import des composants
from .components import (
    create_header,
    create_sidebar,
    create_alert,
    #create_loading_overlay,
    #create_error_card
)

# Import des pages
from .pages import (
    create_home_page,
    create_etp_page,
    create_bias_page,
    create_modeling_page,
    create_prediction_page,
    #create_help_page
)

# Configuration des vues
VIEWS_CONFIG = {
    "home": {
        "name": "Accueil",
        "icon": "fas fa-home",
        "path": "/dash/home",
        "component": "create_home_page"
    },
    "eto": {
        "name": "Calcul ETP",
        "icon": "fas fa-sun",
        "path": "/dash/eto",
        "component": "create_etp_page"
    },
    "bias": {
        "name": "Correction Biais",
        "icon": "fas fa-adjust",
        "path": "/dash/bias",
        "component": "create_bias_page"
    },
    "modeling": {
        "name": "Modélisation",
        "icon": "fas fa-project-diagram",
        "path": "/dash/modeling",
        "component": "create_modeling_page"
    },
    "prediction": {
        "name": "Prédiction",
        "icon": "fas fa-chart-line",
        "path": "/dash/prediction",
        "component": "create_prediction_page"
    }#,
    #"help": {
        #"name": "Aide",
        #"icon": "fas fa-question-circle",
        #"path": "/dash/help",
       # "component": "create_help_page"
    #}
}

# Navigation principale
MAIN_NAVIGATION = [
    {"name": "Accueil", "path": "/dash/home", "icon": "fas fa-home"},
    {"name": "ETP", "path": "/dash/eto", "icon": "fas fa-sun"},
    {"name": "Correction Biais", "path": "/dash/bias", "icon": "fas fa-adjust"},
    {"name": "Modélisation", "path": "/dash/modeling", "icon": "fas fa-project-diagram"},
    {"name": "Prédiction", "path": "/dash/prediction", "icon": "fas fa-chart-line"},
    #{"name": "Aide", "path": "/dash/help", "icon": "fas fa-question-circle"}
]

# Navigation secondaire
SECONDARY_NAVIGATION = [
    {"name": "Paramètres", "path": "#", "icon": "fas fa-cog"},
    {"name": "Documentation", "path": "#", "icon": "fas fa-book"},
    {"name": "À propos", "path": "#", "icon": "fas fa-info-circle"}
]

# Thème par défaut
THEME_CONFIG = {
    "primary_color": "#3498db",
    "secondary_color": "#2ecc71",
    "accent_color": "#e74c3c",
    "background_color": "#f8f9fa",
    "text_color": "#495057",
    "font_family": "'Inter', sans-serif",
    "font_size_base": "13px",
    "font_size_small": "11px",
    "border_radius": "6px"
}

def get_navigation_items():
    """Retourne les éléments de navigation"""
    return {
        "main": MAIN_NAVIGATION,
        "secondary": SECONDARY_NAVIGATION
    }

def get_page_config(page_name):
    """Retourne la configuration d'une page"""
    return VIEWS_CONFIG.get(page_name, {})

__all__ = [
    # Application Dash
    'create_dash_app',
    
    # Composants
    'create_header',
    'create_sidebar',
    'create_alert',
    #'create_loading_overlay',
    #'create_error_card',
    
    # Pages
    'create_home_page',
    'create_etp_page',
    'create_bias_page',
    'create_modeling_page',
    'create_prediction_page',
    #'create_help_page',
    
    # Configuration
    'VIEWS_CONFIG',
    'MAIN_NAVIGATION',
    'SECONDARY_NAVIGATION',
    'THEME_CONFIG',
    'get_navigation_items',
    'get_page_config'
]

print(f"Interface Dash chargée - {len(VIEWS_CONFIG)} pages disponibles")