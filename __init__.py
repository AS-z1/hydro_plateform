"""
Plateforme Hydrologique Professionnelle
Application Dash + FastAPI pour l'analyse hydrologique
"""

__version__ = "1.0.0"
__author__ = "Équipe Hydrologique"
__email__ = "contact@hydrological-platform.com"

# Export des composants principaux
from main import app
from config import settings

# Modules principaux
__all__ = [
    'app',
    'settings',
    '__version__',
    '__author__',
    '__email__'
]

print(f"Initialisation de {__name__} version {__version__}")