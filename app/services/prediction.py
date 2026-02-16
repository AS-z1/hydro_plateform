"""
Service de prédiction hydrologique avec LSTM
Version professionnelle pour Dash avec modèles depuis app/models
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow non installé. Installation: pip install tensorflow")


class LSTMHydrologyPredictor:
    """
    Prédicteur hydrologique LSTM
    Utilise les modèles pré-entraînés depuis app/models
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialise le prédicteur avec le répertoire des modèles
        
        Args:
            models_dir: Chemin vers le dossier des modèles (par défaut: app/models)
        """
        # Déterminer le chemin des modèles
        if models_dir is None:
            # Chercher le dossier app/models
            current_path = Path(__file__).resolve()
            if 'app' in str(current_path):
                # On est dans app/services/
                self.models_dir = current_path.parent.parent / 'models'
            else:
                # Fallback
                self.models_dir = Path('app/models')
        else:
            self.models_dir = Path(models_dir)
        
        # S'assurer que le dossier existe
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Modèles chargés
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.seq_length = 19  # Longueur de séquence par défaut
        
        # Statistiques du modèle
        self.model_info = {
            'max': {
                'name': 'Qmax/Q90',
                'description': 'Prédiction des débits extrêmes',
                'file_model': 'lstm_qmax_q90.h5',
                'file_scaler_X': 'scaler_X_max.pkl',
                'file_scaler_y': 'scaler_y_max.pkl',
                'loaded': False
            },
            'mean': {
                'name': 'Débits moyens',
                'description': 'Prédiction des débits moyens journaliers',
                'file_model': 'lstm_mean.h5',
                'file_scaler_X': 'scaler_X_mean.pkl',
                'file_scaler_y': 'scaler_y_mean.pkl',
                'loaded': False
            }
        }
        
        # Tentative de chargement automatique
        self._load_available_models()
    
    def _load_available_models(self):
        """Charge tous les modèles disponibles"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow non disponible - modèles non chargés")
            return
        
        for model_type, info in self.model_info.items():
            model_path = self.models_dir / info['file_model']
            scaler_x_path = self.models_dir / info['file_scaler_X']
            scaler_y_path = self.models_dir / info['file_scaler_y']
            
            if all(p.exists() for p in [model_path, scaler_x_path, scaler_y_path]):
                try:
                    self.models[model_type] = load_model(str(model_path), compile=False)
                    self.scalers_X[model_type] = joblib.load(str(scaler_x_path))
                    self.scalers_y[model_type] = joblib.load(str(scaler_y_path))
                    self.model_info[model_type]['loaded'] = True
                    print(f"✓ Modèle {info['name']} chargé avec succès")
                except Exception as e:
                    print(f"✗ Erreur chargement {info['name']}: {e}")
    
    def is_model_available(self, model_type: str) -> bool:
        """
        Vérifie si un modèle est disponible
        
        Args:
            model_type: 'max' ou 'mean'
            
        Returns:
            bool: True si le modèle est chargé
        """
        return model_type in self.models and self.model_info[model_type]['loaded']
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des modèles disponibles
        
        Returns:
            List[Dict]: Informations sur les modèles chargés
        """
        available = []
        for model_type, info in self.model_info.items():
            if info['loaded']:
                available.append({
                    'type': model_type,
                    'name': info['name'],
                    'description': info['description']
                })
        return available
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les features pour la prédiction LSTM
        
        Args:
            df: DataFrame avec colonnes 'Pluie' et 'ETP'
            
        Returns:
            pd.DataFrame: Features calculées
        """
        df = df.copy()
        
        # Fenêtres pour cumuls
        windows_common = [5, 7, 14, 30]
        windows_pluie = [45, 47, 54, 60, 65]
        
        # Cumuls communs
        for w in windows_common:
            df[f'pluie_cum{w}'] = df['Pluie'].rolling(w, min_periods=1).sum()
            df[f'ETP_cum{w}'] = df['ETP'].rolling(w, min_periods=1).sum()
        
        # Cumuls spécifiques pluie
        for w in windows_pluie:
            df[f'pluie_cum{w}'] = df['Pluie'].rolling(w, min_periods=1).sum()
        
        return df
    
    def predict(self, 
                model_type: str,
                dates: pd.DatetimeIndex,
                pluie: np.ndarray,
                etp: np.ndarray) -> pd.DataFrame:
        """
        Prédit le débit à partir des données de pluie et ETP
        
        Args:
            model_type: 'max' ou 'mean'
            dates: Dates des observations
            pluie: Valeurs de précipitation (mm/jour)
            etp: Valeurs d'évapotranspiration (mm/jour)
            
        Returns:
            pd.DataFrame: DataFrame avec dates et prédictions
        """
        # Vérifications
        if not self.is_model_available(model_type):
            raise ValueError(f"Modèle {model_type} non disponible")
        
        if len(pluie) != len(etp) or len(pluie) != len(dates):
            raise ValueError("Dimensions incohérentes entre les entrées")
        
        # Créer le DataFrame de base
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'Pluie': pluie,
            'ETP': etp
        })
        
        # Préparer les features
        df = self.prepare_features(df)
        df = df.dropna().reset_index(drop=True)
        
        # Récupérer le scaler et les features
        scaler_X = self.scalers_X[model_type]
        
        # Vérifier que toutes les features nécessaires sont présentes
        FEATURES = list(scaler_X.feature_names_in_)
        missing_features = [f for f in FEATURES if f not in df.columns]
        
        if missing_features:
            # Ajouter les features manquantes avec des zéros
            for f in missing_features:
                df[f] = 0
        
        # Normalisation
        X_scaled = scaler_X.transform(df[FEATURES])
        
        # Création des séquences LSTM
        if len(X_scaled) <= self.seq_length:
            raise ValueError(f"Pas assez de données: {len(X_scaled)} < {self.seq_length}")
        
        X_seq = np.array([
            X_scaled[i:i + self.seq_length]
            for i in range(len(X_scaled) - self.seq_length)
        ])
        
        # Prédiction
        model = self.models[model_type]
        y_scaled = model.predict(X_seq, verbose=0)
        
        # Dénormalisation
        scaler_y = self.scalers_y[model_type]
        y_pred = scaler_y.inverse_transform(y_scaled).flatten()
        
        # Éviter les valeurs négatives
        y_pred = np.maximum(y_pred, 0)
        
        # Alignement des dates avec les prédictions
        result = df.iloc[self.seq_length:].copy()
        result['Prediction'] = y_pred
        
        return result[['date', 'Prediction']]
    
    def calculate_qmax_q90(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule les sommes annuelles Qmax et Q90
        
        Args:
            df: DataFrame avec colonnes 'date' et 'Prediction'
            
        Returns:
            Dict: Statistiques Qmax/Q90
        """
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['year'] = df_copy['date'].dt.year
        
        # Fonction pour Q90
        def q90_percentile(x):
            x_clean = x.dropna()
            return np.percentile(x_clean, 90) if len(x_clean) > 0 else 0
        
        # Statistiques annuelles
        annual_stats = df_copy.groupby('year').agg({
            'Prediction': ['max', q90_percentile]
        })
        
        annual_stats.columns = ['Qmax', 'Q90']
        
        # Sommes sur toutes les années
        qmax_sum = float(annual_stats['Qmax'].sum())
        q90_sum = float(annual_stats['Q90'].sum())
        
        return {
            'Qmax_sum': qmax_sum,
            'Q90_sum': q90_sum,
            'Qmax_mean': float(annual_stats['Qmax'].mean()),
            'Q90_mean': float(annual_stats['Q90'].mean()),
            'Qmax_max': float(annual_stats['Qmax'].max()),
            'Q90_max': float(annual_stats['Q90'].max()),
            'n_years': len(annual_stats)
        }
    
    def calculate_flow_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule les statistiques de débit
        
        Args:
            df: DataFrame avec colonnes 'date' et 'Prediction'
            
        Returns:
            Dict: Statistiques complètes
        """
        predictions = df['Prediction'].values
        
        return {
            'sum': float(np.sum(predictions)),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'max': float(np.max(predictions)),
            'min': float(np.min(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75)),
            'q90': float(np.percentile(predictions, 90)),
            'q10': float(np.percentile(predictions, 10)),
            'n_days': len(predictions)
        }
    
    def calculate_change_percentage(self, future_value: float, reference_value: float) -> float:
        """
        Calcule le pourcentage de changement
        
        Args:
            future_value: Valeur future
            reference_value: Valeur de référence
            
        Returns:
            float: Pourcentage de changement
        """
        if reference_value == 0:
            return 0.0
        return ((future_value - reference_value) / reference_value) * 100


# Instance globale pour réutilisation
_predictor_instance = None

def get_predictor() -> LSTMHydrologyPredictor:
    """
    Retourne une instance singleton du prédicteur
    
    Returns:
        LSTMHydrologyPredictor: Instance du prédicteur
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = LSTMHydrologyPredictor()
    return _predictor_instance