"""
Composants Dash réutilisables
Éléments d'interface utilisateur communs
"""

from .header import create_header
from .sidebar import create_sidebar
from .alerts import (
    create_alert,
    create_loading_overlay,
    create_error_card
)

# Types d'alertes disponibles
ALERT_TYPES = {
    "success": {
        "color": "#27ae60",
        "icon": "check-circle",
        "bg_color": "#d1f7c4"
    },
    "danger": {
        "color": "#e74c3c",
        "icon": "exclamation-circle",
        "bg_color": "#ffe6e6"
    },
    "warning": {
        "color": "#f39c12",
        "icon": "exclamation-triangle",
        "bg_color": "#fff3cd"
    },
    "info": {
        "color": "#3498db",
        "icon": "info-circle",
        "bg_color": "#e8f4fd"
    }
}

# Types de boutons
BUTTON_VARIANTS = {
    "primary": {
        "color": "#3498db",
        "hover_color": "#2980b9",
        "text_color": "#ffffff"
    },
    "secondary": {
        "color": "#6c757d",
        "hover_color": "#5a6268",
        "text_color": "#ffffff"
    },
    "success": {
        "color": "#2ecc71",
        "hover_color": "#27ae60",
        "text_color": "#ffffff"
    },
    "danger": {
        "color": "#e74c3c",
        "hover_color": "#c0392b",
        "text_color": "#ffffff"
    },
    "warning": {
        "color": "#f39c12",
        "hover_color": "#d68910",
        "text_color": "#212529"
    },
    "info": {
        "color": "#17a2b8",
        "hover_color": "#138496",
        "text_color": "#ffffff"
    },
    "light": {
        "color": "#f8f9fa",
        "hover_color": "#e2e6ea",
        "text_color": "#212529"
    },
    "dark": {
        "color": "#343a40",
        "hover_color": "#23272b",
        "text_color": "#ffffff"
    }
}

# Classes CSS utilitaires
UTILITY_CLASSES = {
    "text": {
        "small": "text-tiny",
        "muted": "text-muted",
        "primary": "text-primary",
        "success": "text-success",
        "danger": "text-danger",
        "warning": "text-warning",
        "info": "text-info"
    },
    "background": {
        "light": "bg-light",
        "primary": "bg-primary",
        "success": "bg-success",
        "danger": "bg-danger",
        "warning": "bg-warning",
        "info": "bg-info"
    },
    "border": {
        "light": "border-light",
        "primary": "border-primary",
        "success": "border-success",
        "danger": "border-danger",
        "warning": "border-warning",
        "info": "border-info"
    },
    "spacing": {
        "small": {
            "margin": "m-1",
            "padding": "p-1"
        },
        "medium": {
            "margin": "m-2",
            "padding": "p-2"
        },
        "large": {
            "margin": "m-3",
            "padding": "p-3"
        }
    }
}

def get_alert_config(alert_type):
    """Retourne la configuration d'une alerte"""
    return ALERT_TYPES.get(alert_type, ALERT_TYPES["info"])

def get_button_config(variant):
    """Retourne la configuration d'un bouton"""
    return BUTTON_VARIANTS.get(variant, BUTTON_VARIANTS["primary"])

def create_custom_button(text, variant="primary", size="md", **kwargs):
    """Crée un bouton personnalisé"""
    from dash import html
    import dash_bootstrap_components as dbc
    
    config = get_button_config(variant)
    
    size_classes = {
        "sm": "btn-sm",
        "md": "",
        "lg": "btn-lg"
    }
    
    return dbc.Button(
        text,
        color=variant,
        className=size_classes.get(size, ""),
        style={
            "backgroundColor": config["color"],
            "borderColor": config["color"],
            "color": config["text_color"],
            "fontSize": "0.8125rem",
            "fontWeight": "500"
        },
        **kwargs
    )

__all__ = [
    # Composants principaux
    'create_header',
    'create_sidebar',
    
    # Alertes
    'create_alert',
    'create_loading_overlay',
    'create_error_card',
    
    # Configuration
    'ALERT_TYPES',
    'BUTTON_VARIANTS',
    'UTILITY_CLASSES',
    
    # Fonctions utilitaires
    'get_alert_config',
    'get_button_config',
    'create_custom_button'
]

print("Composants Dash chargés - Interface utilisateur prête")