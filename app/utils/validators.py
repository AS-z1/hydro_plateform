"""
Validateurs pour les fichiers et données
"""

from fastapi import HTTPException
import os
from typing import List

def validate_file_extension(filename: str, allowed_extensions: List[str]):
    """Valide l'extension d'un fichier"""
    if not filename:
        raise HTTPException(400, "Nom de fichier manquant")
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Extension {ext} non autorisée. Utilisez: {', '.join(allowed_extensions)}")

def validate_file_size(size: int, max_size_mb: int = 100):
    """Valide la taille d'un fichier"""
    max_size = max_size_mb * 1024 * 1024
    if size > max_size:
        raise HTTPException(400, f"Fichier trop volumineux ({size/1024/1024:.1f} MB > {max_size_mb} MB)")