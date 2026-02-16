"""
Constantes de l'application
"""

from enum import Enum
from typing import Dict, List, Tuple

# Version de l'application
APP_VERSION = "1.0.0"
APP_NAME = "Hydrological Platform"

# Formats de fichiers supportés
SUPPORTED_FILE_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.json', '.txt']
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']

# Tailles maximales
MAX_FILE_SIZE_MB = 100  # 100 MB
MAX_UPLOAD_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

# Périodes par défaut
DEFAULT_PERIODS = {
    "2021_2040": ("2021-01-01", "2040-12-31"),
    "2041_2070": ("2041-01-01", "2070-12-31"),
    "2071_2100": ("2071-01-01", "2100-12-31"),
}

# Modèles climatiques par défaut
DEFAULT_CLIMATE_MODELS = [
    "EC_Earth585_cor",
    "INM_CM5585_cor", 
    "MPI585_cor"
]

# Méthodes ETP disponibles
ETO_METHODS = [
    "FAO-56",
    "Hargreaves", 
    "Oudin",
    "Turc",
    "Hamon",
    "Penman",
    "Penman-Monteith"
]

# Méthodes de correction de biais
BIAS_CORRECTION_METHODS = [
    "ISIMIP",
    "QuantileDeltaMapping",
    "LinearScaling",
    "DeltaChange",
    "ScaledDistributionMapping"
]

# Types d'analyse
ANALYSIS_TYPES = [
    "qmax_q90",
    "mean"
]

# Paramètres ModHyPMA par défaut
MODHY_PMA_DEFAULT_PARAMS = {
    'm': 1.1,
    'l': 50.0,
    'P2': 3.5,
    'TX': 0.1
}

# Plages de paramètres ModHyPMA
MODHY_PMA_PARAM_RANGES = {
    'm': (0.9, 1.45),
    'l': (26.0, 150.0),
    'P2': (2.2, 10.0),
    'TX': (0.00001, 0.8)
}

# Paramètres LSTM par défaut
LSTM_DEFAULT_PARAMS = {
    'seq_length': 10,
    'units': 50,
    'epochs': 20,
    'learning_rate': 0.001,
    'batch_size': 32,
    'dropout_rate': 0.25
}

# Variables climatiques
CLIMATE_VARIABLES = {
    'tas': 'Temperature',
    'pr': 'Precipitation',
    'hurs': 'Relative Humidity',
    'rsds': 'Surface Solar Radiation',
    'sfcWind': 'Wind Speed'
}

# Unités
UNITS = {
    'temperature': '°C',
    'precipitation': 'mm',
    'flow': 'm³/s',
    'eto': 'mm/jour',
    'wind': 'm/s',
    'radiation': 'MJ/m²/jour'
}

# Codes de couleur
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'success': '#27ae60',
    'danger': '#e74c3c',
    'light': '#ecf0f1',
    'dark': '#2c3e50'
}

# Seuils de performance
PERFORMANCE_THRESHOLDS = {
    'excellent': {'nse': 0.75, 'r2': 0.8, 'kge': 0.75},
    'good': {'nse': 0.65, 'r2': 0.7, 'kge': 0.65},
    'fair': {'nse': 0.5, 'r2': 0.6, 'kge': 0.5},
    'poor': {'nse': 0.0, 'r2': 0.0, 'kge': 0.0}
}

# Messages d'erreur
ERROR_MESSAGES = {
    'file_not_found': "Fichier non trouvé",
    'invalid_format': "Format de fichier invalide",
    'missing_columns': "Colonnes manquantes",
    'invalid_date': "Format de date invalide",
    'calculation_error': "Erreur lors du calcul",
    'model_error': "Erreur du modèle",
    'validation_error': "Erreur de validation",
    'server_error': "Erreur interne du serveur"
}

# Codes HTTP personnalisés
HTTP_CODES = {
    'success': 200,
    'created': 201,
    'bad_request': 400,
    'unauthorized': 401,
    'forbidden': 403,
    'not_found': 404,
    'method_not_allowed': 405,
    'conflict': 409,
    'internal_error': 500,
    'service_unavailable': 503
}

# Chemins des répertoires
DIRECTORIES = {
    'data': 'data',
    'models': 'models',
    'logs': 'logs',
    'temp': 'temp',
    'exports': 'exports'
}

# Formats de date
DATE_FORMATS = {
    'short': '%Y-%m-%d',
    'long': '%Y-%m-%d %H:%M:%S',
    'file': '%Y%m%d_%H%M%S',
    'display': '%d/%m/%Y'
}

class AnalysisType(Enum):
    """Types d'analyse"""
    QMAX_Q90 = "qmax_q90"
    MEAN = "mean"

class ModelType(Enum):
    """Types de modèles"""
    MODHY_PMA = "ModHyPMA"
    LSTM = "LSTM"

class VariableType(Enum):
    """Types de variables"""
    TEMPERATURE = "tas"
    PRECIPITATION = "pr"

# Configuration des graphiques
PLOT_CONFIG = {
    'template': 'plotly_white',
    'font_size': 10,
    'font_family': 'Inter, sans-serif',
    'width': 800,
    'height': 400,
    'margin': {'t': 40, 'r': 20, 'b': 40, 'l': 60}
}

# Limites
LIMITS = {
    'max_rows_display': 1000,
    'max_columns_display': 50,
    'max_points_plot': 10000,
    'max_file_size': MAX_UPLOAD_SIZE
}