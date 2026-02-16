"""
Services métier pour l'application hydrologique
Logique de calcul et traitement des données
"""

from .eto_calculator import (
    EToData,
    EToCalculator,
    EToDataManager
)

from .bias_correction import (
    ClimateBiasCorrection,
    ClimateDataManager,
    CorrectionResults
)

from .hydrological_modeling import (
    set_seed,
    Metrics,
    ModHyPMA_Model,
    DataLoader,
    FeatureEngineer,
    SequenceBuilder,
    LSTMModelBuilder,
    LSTMTrainer,
    #SimpleModelOptimizer,
    ModHyPMA_Evaluator,
    LSTM_Evaluator
)

from .prediction import (
    LSTMHydrologyPredictor,
    #PeriodManager,
    #ReferenceValueManager
)

# Configuration des services
SERVICES_CONFIG = {
    "eto": {
        "name": "Calcul ETP",
        "class": "EToCalculator",
        "methods": ["FAO-56", "Hargreaves", "Oudin", "Turc", "Hamon", "Penman", "Penman-Monteith"],
        "dependencies": ["pandas", "numpy", "pyet"]
    },
    "bias": {
        "name": "Correction de biais",
        "class": "ClimateBiasCorrection",
        "methods": ["ISIMIP", "QuantileDeltaMapping", "LinearScaling", "DeltaChange", "ScaledDistributionMapping"],
        "dependencies": ["pandas", "numpy", "ibicus"]
    },
    "modeling": {
        "name": "Modélisation hydrologique",
        "classes": ["ModHyPMA_Model", "LSTMTrainer"],
        "dependencies": ["pandas", "numpy", "tensorflow", "scikit-learn"]
    },
    "prediction": {
        "name": "Prédiction hydrologique",
        "class": "LSTMHydrologyPredictor",
        "dependencies": ["pandas", "numpy", "tensorflow", "joblib"]
    }
}

# Vérification des dépendances
def check_dependencies():
    """Vérifie les dépendances des services"""
    import importlib
    import sys
    
    missing_deps = {}
    
    for service_name, config in SERVICES_CONFIG.items():
        if "dependencies" in config:
            missing = []
            for dep in config["dependencies"]:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    missing.append(dep)
            
            if missing:
                missing_deps[service_name] = missing
    
    return missing_deps

# Initialisation
missing = check_dependencies()
if missing:
    print("⚠️ Dépendances manquantes détectées:")
    for service, deps in missing.items():
        print(f"  - {service}: {', '.join(deps)}")
else:
    print("✅ Toutes les dépendances sont installées")

__all__ = [
    # ETP
    'EToData',
    'EToCalculator',
    'EToDataManager',
    
    # Correction de biais
    'ClimateBiasCorrection',
    'ClimateDataManager',
    'CorrectionResults',
    
    # Modélisation
    'set_seed',
    'Metrics',
    'ModHyPMA_Model',
    'DataLoader',
    'FeatureEngineer',
    'SequenceBuilder',
    'LSTMModelBuilder',
    'LSTMTrainer',
    'SimpleModelOptimizer',
    #'ModHyPMA_Evaluator',
    'LSTM_Evaluator',
    
    # Prédiction
    'LSTMHydrologyPredictor',
    #'PeriodManager',
    #'ReferenceValueManager',
    
    # Configuration
    'SERVICES_CONFIG',
    'check_dependencies'
]

print(f"Services métier chargés - {len(SERVICES_CONFIG)} services disponibles")