"""
Module de correction de biais climatique
Classes: ClimateBiasCorrection, DataManager, CorrectionResults
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from ibicus.debias import ISIMIP, QuantileDeltaMapping, LinearScaling, DeltaChange, ScaledDistributionMapping
    IBICUS_AVAILABLE = True
except ImportError:
    print("Attention: ibicus n'est pas installé. Installez-le avec: pip install ibicus")
    IBICUS_AVAILABLE = False
    # Création de classes factices pour éviter les erreurs
    class ISIMIP: pass
    class QuantileDeltaMapping: pass
    class LinearScaling: pass
    class DeltaChange: pass
    class ScaledDistributionMapping: pass


class ClimateDataManager:
    """Gestionnaire de données pour la correction de biais climatique."""
    
    def __init__(self):
        self.data_loaded = False
        self.df_hist = None
        self.df_fut = None
        self.col_names = {}
        self.data_summary = {}
    
    def load_dataframes(self, df_hist: pd.DataFrame, df_fut: pd.DataFrame):
        """Charge les DataFrames directement."""
        self.df_hist = df_hist
        self.df_fut = df_fut
        self.data_loaded = True
        
        # Analyse automatique des colonnes
        self._analyze_columns()
        
        return self._get_summary()
    
    def _analyze_columns(self):
        """Analyse automatique des colonnes disponibles."""
        hist_cols = self.df_hist.columns.tolist()
        fut_cols = self.df_fut.columns.tolist()
        
        # Détection des colonnes de date
        date_candidates = ['date', 'Date', 'DATE', 'time', 'Time', 'TIME', 
                          'datetime', 'Datetime', 'DATETIME']
        
        self.col_names['date_hist'] = next((col for col in date_candidates if col in hist_cols), None)
        self.col_names['date_fut'] = next((col for col in date_candidates if col in fut_cols), None)
        
        # Si pas de colonne date trouvée, en créer une
        if not self.col_names['date_hist']:
            self.df_hist['date'] = pd.date_range(start='2000-01-01', periods=len(self.df_hist), freq='D')
            self.col_names['date_hist'] = 'date'
        
        if not self.col_names['date_fut']:
            self.df_fut['date'] = pd.date_range(start='2050-01-01', periods=len(self.df_fut), freq='D')
            self.col_names['date_fut'] = 'date'
        
        # Détection des variables potentielles
        numeric_cols_hist = self.df_hist.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_fut = self.df_fut.select_dtypes(include=[np.number]).columns.tolist()
        
        self.data_summary = {
            'hist_columns': hist_cols,
            'fut_columns': fut_cols,
            'numeric_hist': numeric_cols_hist,
            'numeric_fut': numeric_cols_fut,
            'hist_shape': self.df_hist.shape,
            'fut_shape': self.df_fut.shape,
            'hist_period': f"{self.df_hist[self.col_names['date_hist']].min()} to {self.df_hist[self.col_names['date_hist']].max()}",
            'fut_period': f"{self.df_fut[self.col_names['date_fut']].min()} to {self.df_fut[self.col_names['date_fut']].max()}"
        }
    
    def get_variable_candidates(self):
        """Retourne les variables candidates pour la correction."""
        hist_num = self.data_summary.get('numeric_hist', [])
        fut_num = self.data_summary.get('numeric_fut', [])
        
        # Variables communes
        common_vars = list(set(hist_num) & set(fut_num))
        
        # Variables uniques
        unique_hist = list(set(hist_num) - set(fut_num))
        unique_fut = list(set(fut_num) - set(hist_num))
        
        return {
            'common': common_vars,
            'unique_hist': unique_hist,
            'unique_fut': unique_fut
        }
    
    def prepare_data(self, obs_col: str, hist_col: str, fut_col: str):
        """Prépare les données pour la correction."""
        if not self.data_loaded:
            raise ValueError("Données non chargées")
        
        # Vérifier l'existence des colonnes
        for col, name in [(obs_col, 'obs'), (hist_col, 'hist'), (fut_col, 'fut')]:
            if col not in self.df_hist.columns and name != 'fut':
                raise ValueError(f"Colonne {col} non trouvée dans données historiques")
            if col not in self.df_fut.columns and name == 'fut':
                raise ValueError(f"Colonne {col} non trouvée dans données futures")
        
        # Extraire les séries
        obs = self.df_hist[obs_col].values
        hist = self.df_hist[hist_col].values
        fut = self.df_fut[fut_col].values
        
        # Dates
        dates_obs = pd.to_datetime(self.df_hist[self.col_names['date_hist']])
        dates_hist = pd.to_datetime(self.df_hist[self.col_names['date_hist']])
        dates_fut = pd.to_datetime(self.df_fut[self.col_names['date_fut']])
        
        # Stocker les noms
        self.col_names.update({
            'obs': obs_col,
            'hist': hist_col,
            'fut': fut_col
        })
        
        return {
            'obs': obs,
            'hist': hist,
            'fut': fut,
            'dates_obs': dates_obs,
            'dates_hist': dates_hist,
            'dates_fut': dates_fut
        }
    
    def _get_summary(self):
        """Retourne un résumé des données."""
        return self.data_summary


class ClimateBiasCorrection:
    """Classe principale pour la correction de biais climatique."""
    
    METHOD_MAP = {
        "ISIMIP": ISIMIP if IBICUS_AVAILABLE else None,
        "QuantileDeltaMapping": QuantileDeltaMapping if IBICUS_AVAILABLE else None,
        "LinearScaling": LinearScaling if IBICUS_AVAILABLE else None,
        "DeltaChange": DeltaChange if IBICUS_AVAILABLE else None,
        "ScaledDistributionMapping": ScaledDistributionMapping if IBICUS_AVAILABLE else None
    }
    
    METHOD_INFO = {
        "ISIMIP": {
            "description": "Méthode ISIMIP (Inter-Sectoral Impact Model Intercomparison Project)",
            "requirements": "Observations, historique, futur",
            "precision": "Très élevée",
            "complexity": "Complexe",
            "best_for": "Toutes les variables, résultats robustes"
        },
        "QuantileDeltaMapping": {
            "description": "Mapping quantile-delta (QDM)",
            "requirements": "Observations, historique, futur",
            "precision": "Élevée",
            "complexity": "Moyenne",
            "best_for": "Précipitation, variables non-linéaires"
        },
        "LinearScaling": {
            "description": "Mise à l'échelle linéaire",
            "requirements": "Observations, historique, futur",
            "precision": "Moyenne",
            "complexity": "Simple",
            "best_for": "Température, variables linéaires"
        },
        "DeltaChange": {
            "description": "Méthode delta-change",
            "requirements": "Historique, futur",
            "precision": "Bonne",
            "complexity": "Simple",
            "best_for": "Changements moyens, température"
        },
        "ScaledDistributionMapping": {
            "description": "Mapping de distribution ajustée",
            "requirements": "Observations, historique, futur",
            "precision": "Élevée",
            "complexity": "Moyenne",
            "best_for": "Distribution complète, extrêmes"
        }
    }
    
    def __init__(self):
        """Initialise le correcteur de biais."""
        self.data_manager = ClimateDataManager()
        self.results = {}
        self.current_correction = None
    
    def apply_correction(self, data: Dict, method: str, variable_type: str = "tas") -> Optional[Dict]:
        """
        Applique la correction de biais.
        
        Args:
            data: Dictionnaire avec obs, hist, fut, dates
            method: Nom de la méthode
            variable_type: 'tas' (température) ou 'pr' (précipitation)
            
        Returns:
            Dictionnaire avec résultats ou None
        """
        if not IBICUS_AVAILABLE:
            raise ImportError("La bibliothèque ibicus n'est pas installée.")
        
        if method not in self.METHOD_MAP or self.METHOD_MAP[method] is None:
            raise ValueError(f"Méthode {method} non disponible")
        
        try:
            # Créer le debiaser
            debiaser_class = self.METHOD_MAP[method]
            debiaser = debiaser_class.from_variable(variable_type)
            
            # Appliquer la correction
            corrected_data = debiaser.apply_location(
                data['obs'],
                data['hist'],
                data['fut'],
                data['dates_obs'],
                data['dates_hist'],
                data['dates_fut']
            )
            
            # Calculer les statistiques
            stats = self._calculate_statistics(data['fut'], corrected_data)
            
            # Préparer les résultats
            result_id = f"{method}_{variable_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.results[result_id] = {
                'method': method,
                'variable_type': variable_type,
                'original_data': data['fut'],
                'corrected_data': corrected_data,
                'dates': data['dates_fut'],
                'stats': stats,
                'timestamp': datetime.now()
            }
            
            self.current_correction = result_id
            
            return self.results[result_id]
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la correction: {str(e)}")
    
    def _calculate_statistics(self, original: np.ndarray, corrected: np.ndarray) -> Dict:
        """Calcule les statistiques comparatives."""
        return {
            'original_mean': float(np.nanmean(original)),
            'corrected_mean': float(np.nanmean(corrected)),
            'original_std': float(np.nanstd(original)),
            'corrected_std': float(np.nanstd(corrected)),
            'original_min': float(np.nanmin(original)),
            'corrected_min': float(np.nanmin(corrected)),
            'original_max': float(np.nanmax(original)),
            'corrected_max': float(np.nanmax(corrected)),
            'mean_change': float(np.nanmean(corrected) - np.nanmean(original)),
            'mean_change_percent': float(((np.nanmean(corrected) - np.nanmean(original)) / np.nanmean(original) * 100) 
                                         if np.nanmean(original) != 0 else 0)
        }
    
    def get_method_info(self, method: str = None) -> Dict:
        """Retourne les informations sur les méthodes."""
        if method:
            return self.METHOD_INFO.get(method, {})
        return self.METHOD_INFO
    
    def get_results(self, result_id: str = None) -> Dict:
        """Retourne les résultats."""
        if result_id:
            return self.results.get(result_id)
        return self.results
    
    def create_results_dataframe(self, result_id: str) -> pd.DataFrame:
        """Crée un DataFrame avec les résultats."""
        result = self.results.get(result_id)
        if not result:
            raise ValueError(f"Résultat {result_id} non trouvé")
        
        df = pd.DataFrame({
            'date': result['dates'],
            'original': result['original_data'],
            'corrected': result['corrected_data'],
            'difference': result['corrected_data'] - result['original_data']
        })
        
        return df


class CorrectionResults:
    """Classe pour gérer et visualiser les résultats."""
    
    @staticmethod
    def get_statistics_table(stats: Dict) -> pd.DataFrame:
        """Crée un tableau des statistiques."""
        stats_df = pd.DataFrame([
            {"Statistique": "Moyenne", 
             "Original": f"{stats['original_mean']:.4f}",
             "Corrigé": f"{stats['corrected_mean']:.4f}",
             "Différence": f"{stats['mean_change']:+.4f} ({stats['mean_change_percent']:+.2f}%)"},
            
            {"Statistique": "Écart-type",
             "Original": f"{stats['original_std']:.4f}",
             "Corrigé": f"{stats['corrected_std']:.4f}",
             "Différence": f"{stats['corrected_std'] - stats['original_std']:+.4f}"},
            
            {"Statistique": "Minimum",
             "Original": f"{stats['original_min']:.4f}",
             "Corrigé": f"{stats['corrected_min']:.4f}",
             "Différence": f"{stats['corrected_min'] - stats['original_min']:+.4f}"},
            
            {"Statistique": "Maximum",
             "Original": f"{stats['original_max']:.4f}",
             "Corrigé": f"{stats['corrected_max']:.4f}",
             "Différence": f"{stats['corrected_max'] - stats['original_max']:+.4f}"}
        ])
        
        return stats_df