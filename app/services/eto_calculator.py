"""
Module de calcul d'évapotranspiration potentielle (ETo)
Classes: EToData, EToCalculator, EToDataManager
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyet
import warnings
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Union
warnings.filterwarnings('ignore')

# ======================================================
# 1. CLASSES DE DONNÉES
# ======================================================
@dataclass
class EToData:
    """Structure de données pour le calcul de l'ETo"""
    tmean: Optional[pd.Series] = None       # Température moyenne (°C)
    tmin: Optional[pd.Series] = None        # Température minimale (°C)
    tmax: Optional[pd.Series] = None        # Température maximale (°C)
    wind: Optional[pd.Series] = None        # Vitesse du vent (m/s)
    rs: Optional[pd.Series] = None          # Rayonnement solaire (MJ/m²/jour)
    rn: Optional[pd.Series] = None          # Rayonnement net (MJ/m²/jour)
    rh: Optional[pd.Series] = None          # Humidité relative (%)
    elevation: Optional[float] = None        # Altitude (m)
    lat: Optional[float] = None              # Latitude (radians)
    dates: Optional[pd.DatetimeIndex] = None # Dates
    
    def validate(self, method: str) -> bool:
        """Valider les données pour une méthode spécifique"""
        required_params = {
            "Penman": ["tmean", "wind", "rn", "rh", "elevation", "lat"],
            "Penman-Monteith": ["tmean", "wind", "rn", "rh", "elevation", "lat"],
            "Hamon": ["tmean", "lat"],
            "Turc": ["tmean", "rs", "rh"],
            "Hargreaves": ["tmean", "tmin", "tmax", "lat"],
            "Oudin": ["tmean", "lat"],
            "FAO-56": ["tmean", "wind", "rn", "rh", "elevation", "lat"]
        }
        
        if method not in required_params:
            return False
        
        for param in required_params[method]:
            value = getattr(self, param)
            if value is None:
                return False
            if isinstance(value, pd.Series) and len(value) == 0:
                return False
        
        return True
    
    def get_params(self, method: str) -> Dict:
        """Obtenir les paramètres pour une méthode spécifique"""
        params = {}
        method_params = {
            "Penman": ["tmean", "wind", "rn", "rh", "elevation", "lat"],
            "Penman-Monteith": ["tmean", "wind", "rn", "rh", "elevation", "lat"],
            "Hamon": ["tmean", "lat"],
            "Turc": ["tmean", "rs", "rh"],
            "Hargreaves": ["tmean", "tmin", "tmax", "lat"],
            "Oudin": ["tmean", "lat"],
            "FAO-56": ["tmean", "wind", "rn", "rh", "elevation", "lat"]
        }
        
        for param in method_params.get(method, []):
            value = getattr(self, param)
            if value is not None:
                params[param] = value
        
        return params


# ======================================================
# 2. CALCULATEUR D'ETo
# ======================================================
class EToCalculator:
    """
    Classe implémentant les méthodes de calcul de l'évapotranspiration potentielle (ETo).
    """
    
    def __init__(self, data: EToData):
        self._data = data
        self._results = {}
        self._methods_info = {
            "Penman": {
                "description": "Équation de Penman originale (1948)",
                "complexity": "Haute",
                "precision": "Élevée",
                "requirements": "tmean, wind, rn, rh, elevation, lat"
            },
            "Penman-Monteith": {
                "description": "Équation standard FAO Penman-Monteith",
                "complexity": "Haute",
                "precision": "Très élevée",
                "requirements": "tmean, wind, rn, rh, elevation, lat"
            },
            "Hamon": {
                "description": "Méthode simplifiée basée sur la température",
                "complexity": "Basse",
                "precision": "Moyenne",
                "requirements": "tmean, lat"
            },
            "Turc": {
                "description": "Méthode adaptée aux climats humides",
                "complexity": "Moyenne",
                "precision": "Bonne",
                "requirements": "tmean, rs, rh"
            },
            "Hargreaves": {
                "description": "Méthode pour régions avec données limitées",
                "complexity": "Basse",
                "precision": "Acceptable",
                "requirements": "tmean, tmin, tmax, lat"
            },
            "Oudin": {
                "description": "Méthode simplifiée pour modélisation hydrologique",
                "complexity": "Très basse",
                "precision": "Basique",
                "requirements": "tmean, lat"
            },
            "FAO-56": {
                "description": "Méthode de référence FAO-56 Penman-Monteith",
                "complexity": "Haute",
                "precision": "Référence",
                "requirements": "tmean, wind, rn, rh, elevation, lat"
            }
        }
    
    def calculate(self, method: str) -> pd.Series:
        """
        Calculer l'ETo avec la méthode spécifiée
        """
        # Vérifier que la méthode est disponible
        if method not in self._methods_info:
            available_methods = ", ".join(self._methods_info.keys())
            raise ValueError(f"Méthode '{method}' non disponible. "
                           f"Méthodes disponibles: {available_methods}")
        
        # Valider les données pour cette méthode
        if not self._data.validate(method):
            requirements = self._methods_info[method]["requirements"]
            raise ValueError(f"Données insuffisantes pour la méthode '{method}'. "
                           f"Données requises: {requirements}")
        
        # Obtenir les paramètres
        params = self._data.get_params(method)
        
        try:
            # Vérifier et préparer les paramètres
            for key, value in params.items():
                if isinstance(value, pd.Series):
                    # Assurer que la série a un index temporel
                    if value.index is None or len(value.index) == 0:
                        raise ValueError(f"La série '{key}' n'a pas d'index temporel valide")
            
            # Calcul selon la méthode choisie
            if method == "Penman":
                eto_calculated = pyet.penman(**params)
            elif method == "Penman-Monteith":
                eto_calculated = pyet.pm(**params)
            elif method == "Hamon":
                eto_calculated = pyet.hamon(**params)
            elif method == "Turc":
                eto_calculated = pyet.turc(**params)
            elif method == "Hargreaves":
                eto_calculated = pyet.hargreaves(**params)
            elif method == "Oudin":
                eto_calculated = pyet.oudin(**params)
            elif method == "FAO-56":
                eto_calculated = pyet.pm_fao56(**params)
            else:
                raise ValueError(f"Méthode '{method}' non implémentée")
            
            # Convertir en numpy array pour les calculs statistiques
            eto_values = eto_calculated.values if isinstance(eto_calculated, pd.Series) else eto_calculated
            
            # Stocker les résultats
            self._results[method] = {
                'values': eto_values,
                'series': eto_calculated,
                'mean': float(np.nanmean(eto_values)),
                'std': float(np.nanstd(eto_values)),
                'min': float(np.nanmin(eto_values)),
                'max': float(np.nanmax(eto_values)),
                'sum': float(np.nansum(eto_values))
            }
            
            return eto_calculated
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du calcul avec la méthode '{method}': {str(e)}")
    
    def get_method_info(self, method: str = None) -> Union[Dict, Dict[str, Dict]]:
        """Obtenir des informations sur les méthodes de calcul"""
        if method:
            if method in self._methods_info:
                return self._methods_info[method]
            else:
                raise ValueError(f"Méthode '{method}' non disponible")
        else:
            return self._methods_info
    
    def get_results(self, method: str = None) -> Union[Dict, Dict[str, Dict]]:
        """Obtenir les résultats des calculs"""
        if method:
            if method in self._results:
                return self._results[method]
            else:
                raise ValueError(f"Aucun résultat pour la méthode '{method}'")
        else:
            return self._results


# ======================================================
# 3. GESTIONNAIRE DE DONNÉES
# ======================================================
class EToDataManager:
    """Gestionnaire de données pour le calcul d'ETo"""
    
    def __init__(self):
        self.data = None
        self.filepath = None
    
    def load_data(self, filepath: str, date_col: str = "date"):
        """Charger les données depuis un fichier"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        self.filepath = filepath
        
        # Charger selon l'extension
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, parse_dates=[date_col])
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath, parse_dates=[date_col])
        else:
            raise ValueError("Format non supporté. Utilisez .csv ou .xlsx")
        
        # Vérifier la colonne de date
        if date_col not in df.columns:
            raise ValueError(f"Colonne de date '{date_col}' non trouvée")
        
        # Définir l'index
        df = df.set_index(date_col).sort_index()
        
        return df
    
    def prepare_etodata(self, df: pd.DataFrame, lat: float, elevation: float = None) -> EToData:
        """Préparer les données pour le calcul d'ETo"""
        eto_data = EToData()
        eto_data.dates = df.index
        eto_data.lat = np.deg2rad(lat)  # Convertir en radians
        eto_data.elevation = elevation
        
        # Mapping des colonnes possibles
        column_mapping = {
            'tmean': ['tmean', 'tavg', 'temp_mean', 'temperature_mean', 'temp', 'temperature'],
            'tmin': ['tmin', 'temp_min', 'temperature_min'],
            'tmax': ['tmax', 'temp_max', 'temperature_max'],
            'wind': ['wind', 'wind_speed', 'ws', 'u2', 'windspeed'],
            'rs': ['rs', 'solar_radiation', 'rad_sol', 'srad', 'radiation'],
            'rn': ['rn', 'net_radiation', 'rad_net'],
            'rh': ['rh', 'humidity', 'relative_humidity', 'hum', 'rel_hum']
        }
        
        # Chercher et assigner les données
        for param, possible_names in column_mapping.items():
            for name in possible_names:
                if name in df.columns:
                    setattr(eto_data, param, df[name])
                    break
        
        # Calculer tmean si non fourni mais tmin et tmax disponibles
        if eto_data.tmean is None and eto_data.tmin is not None and eto_data.tmax is not None:
            eto_data.tmean = (eto_data.tmin + eto_data.tmax) / 2
        
        return eto_data
    
    def validate_data_completeness(self, eto_data: EToData) -> Dict[str, list]:
        """Valider quelles méthodes peuvent être calculées avec les données disponibles"""
        available_methods = [
            "Penman", "Penman-Monteith", "Hamon", "Turc", 
            "Hargreaves", "Oudin", "FAO-56"
        ]
        
        results = {
            'available': [],
            'unavailable': []
        }
        
        for method in available_methods:
            if eto_data.validate(method):
                results['available'].append(method)
            else:
                results['unavailable'].append(method)
        
        return results