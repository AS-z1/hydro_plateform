"""
Fonctions utilitaires
"""

import uuid
from datetime import datetime
import numpy as np

def generate_request_id():
    """Génère un ID unique pour une requête"""
    return str(uuid.uuid4())

def safe_divide(a, b, default=0):
    """Division sécurisée"""
    try:
        return a / b if b != 0 else default
    except:
        return default

def format_bytes(size):
    """Formate une taille en octets"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"