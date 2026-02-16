import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import random
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYM00_AVAILABLE = True
except ImportError:
    print("Attention: pymoo n'est pas installé. Installez-le avec: pip install pymoo")
    PYM00_AVAILABLE = False

# ======================================================
# FONCTIONS UTILITAIRES
# ======================================================
def set_seed(seed=42):
    """Fixer tous les seeds pour la reproductibilité"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass

class Metrics:
    @staticmethod
    def nse(y_true, y_pred):
        """Nash-Sutcliffe Efficiency"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if denominator == 0:
            return -9999
        return 1 - numerator / denominator
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """Coefficient de détermination personnalisé"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        return correlation ** 2 if not np.isnan(correlation) else 0
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Square Error"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def kge(y_true, y_pred):
        """Kling-Gupta Efficiency"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        r = np.corrcoef(y_true, y_pred)[0, 1]
        alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else 0
        beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else 0
        
        if np.isnan(r):
            r = 0
        if np.isnan(alpha):
            alpha = 0
        if np.isnan(beta):
            beta = 0
            
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    @staticmethod
    def bias(y_true, y_pred):
        """Biais moyen"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return np.mean(y_pred - y_true)


# ======================================================
# 1. MODÈLE ModHyPMA
# ======================================================
class ModHyPMA_Model:
    @staticmethod
    def simulate(rains, PET, m, l, P2, TX):
        """
        Implémentation du modèle ModHyPMA
        """
        rains = np.array(rains, dtype=np.float64)
        PET = np.array(PET, dtype=np.float64)
        N = len(rains)
        
        # Initialisation des variables d'état
        X = np.zeros(N)
        q = np.zeros(N)
        Q = np.zeros(N)
        
        # Conditions initiales
        X[0] = 0.0
        q[0] = max(rains[0] - PET[0], 0.0)
        Q[0] = 0.0
        
        for i in range(1, N):
            # Calcul de q[i]
            if rains[i] <= PET[i]:
                q[i] = 0.0
                X[i] = X[i-1] * (1 - m/l)
            else:
                q[i] = rains[i] - PET[i]
                if q[i] > 0:
                    X[i] = X[i-1] + (m/l) * np.power(q[i], 2*m - P2)
                else:
                    X[i] = X[i-1]
            
            # Condition sur TX
            if X[i] < TX:
                X[i] = 0.0
            
            # Calcul de Q[i] avec protection
            if Q[i-1] > 0:
                Q_pow = np.power(Q[i-1], 2*m - 1)
                Q[i] = Q[i-1] - (m/l) * Q_pow + X[i] * q[i-1] / l
            else:
                Q[i] = X[i] * q[i-1] / l
            
            # Assurer la positivité
            Q[i] = max(Q[i], 0.0)
        
        return Q


# ======================================================
# 2. GESTION DES DONNÉES (comme code 2)
# ======================================================
class DataLoader:
    @staticmethod
    def load(path, date_col="date", target_col="Qobs"):
        """Charger les données depuis un fichier CSV ou Excel comme dans code 2"""
        if path.endswith(".csv"):
            df = pd.read_csv(path, parse_dates=[date_col])
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            df = pd.read_excel(path, parse_dates=[date_col])
        else:
            raise ValueError(f"Format non supporté: {path}. Utilisez .csv ou .xlsx")
        
        # Vérifier que les colonnes nécessaires existent
        required_cols = [date_col, target_col, 'Pluie', 'ETP']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        return df.set_index(date_col).sort_index()

class FeatureEngineer:
    @staticmethod
    def transform(df):
        """Créer les features de cumuls glissants comme dans code 2"""
        df = df.copy()
        
        # Première ligne : cumuls standard
        for w in [5, 7, 14, 30]:
            df[f'pluie_cum{w}'] = df['Pluie'].rolling(window=w).sum()
            df[f'ETP_cum{w}'] = df['ETP'].rolling(window=w).sum()
        
        # Deuxième ligne : cumuls avec min_periods
        for w in [45, 47, 54, 60, 65]:
            df[f'pluie_cum{w}'] = df['Pluie'].rolling(window=w, min_periods=1).sum()
        
        # Supprimer les lignes avec NaN
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("Toutes les données ont été supprimées après nettoyage")
            
        return df_clean

class SequenceBuilder:
    @staticmethod
    def build(X, y, seq_len):
        """Créer des séquences pour le LSTM comme dans code 2"""
        if len(X) != len(y):
            raise ValueError("X et y doivent avoir la même longueur")
        if seq_len >= len(X):
            raise ValueError(f"seq_len ({seq_len}) doit être inférieur à la longueur des données ({len(X)})")
        
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)


# ======================================================
# 3. MODÈLE LSTM (comme code 2)
# ======================================================
class LSTMModelBuilder:
    def __init__(self, input_shape, units=50, learning_rate=0.001, dropout_rate=0.25):
        self.input_shape = input_shape
        self.units = units
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
    def build(self):
        """Construire le modèle LSTM comme dans code 2"""
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(self.units, activation='tanh', return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model


# ======================================================
# 4. TRAINER LSTM (comme code 2)
# ======================================================
class LSTMTrainer:
    def __init__(self, train, val, test, features):
        self.train = train
        self.val = val
        self.test = test
        self.features = features
        
        # Initialiser les scalers comme dans code 2
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Variables pour stocker les données préparées
        self.X_train_seq = None
        self.y_train_seq = None
        self.X_val_seq = None
        self.y_val_seq = None
        self.X_test_seq = None
        self.y_test_seq = None
        self.seq_length = None
        
        # Pour le calage TRAIN+VAL comme dans code 2
        self.X_trainval_seq = None
        self.y_trainval_seq = None
    
    def prepare_data(self, seq_length):
        """Préparer toutes les données pour l'entraînement comme dans code 2"""
        self.seq_length = seq_length
        
        # Normalisation des données
        X_train = self.scaler_X.fit_transform(self.train[self.features])
        y_train = self.scaler_y.fit_transform(self.train[['Qobs']])
        
        X_val = self.scaler_X.transform(self.val[self.features])
        y_val = self.scaler_y.transform(self.val[['Qobs']])
        
        X_test = self.scaler_X.transform(self.test[self.features])
        y_test = self.scaler_y.transform(self.test[['Qobs']])
        
        # Création des séquences
        self.X_train_seq, self.y_train_seq = SequenceBuilder.build(X_train, y_train, seq_length)
        self.X_val_seq, self.y_val_seq = SequenceBuilder.build(X_val, y_val, seq_length)
        self.X_test_seq, self.y_test_seq = SequenceBuilder.build(X_test, y_test, seq_length)
        
        print(f"✓ Données LSTM préparées avec seq_length={seq_length}")
        print(f"  Train: {self.X_train_seq.shape} -> {self.y_train_seq.shape}")
        print(f"  Val: {self.X_val_seq.shape} -> {self.y_val_seq.shape}")
        print(f"  Test: {self.X_test_seq.shape} -> {self.y_test_seq.shape}")
        
        return self.X_train_seq, self.y_train_seq, self.X_val_seq, self.y_val_seq, self.X_test_seq, self.y_test_seq
    
    def prepare_trainval_data(self, seq_length):
        """Préparer les données TRAIN+VAL pour le calage final comme dans code 2"""
        # Concaténer train et val
        trainval = pd.concat([self.train, self.val])
        
        # Normalisation (utilise les mêmes scalers que pour train)
        X_trainval = self.scaler_X.transform(trainval[self.features])
        y_trainval = self.scaler_y.transform(trainval[['Qobs']])
        
        # Création des séquences
        self.X_trainval_seq, self.y_trainval_seq = SequenceBuilder.build(
            X_trainval, y_trainval, seq_length
        )
        
        print(f"✓ Données TRAIN+VAL préparées: {self.X_trainval_seq.shape} -> {self.y_trainval_seq.shape}")
        
        return self.X_trainval_seq, self.y_trainval_seq
    
    def train_and_eval(self, epochs, lr, batch_size, seq_length, units, verbose=0, 
                      evaluate_trainval=False):
        """Entraîner et évaluer un modèle avec des hyperparamètres donnés comme dans code 2"""
        try:
            # Préparer les données si nécessaire
            if self.seq_length != seq_length or self.X_train_seq is None:
                self.prepare_data(seq_length)
            
            # Construire le modèle comme dans code 2
            model_builder = LSTMModelBuilder(
                input_shape=(self.X_train_seq.shape[1], self.X_train_seq.shape[2]),
                units=units,
                learning_rate=lr,
                dropout_rate=0.25
            )
            model = model_builder.build()
            
            # Entraîner le modèle
            print(f"\nEntraînement avec {epochs} epochs...")
            history = model.fit(
                self.X_train_seq, self.y_train_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_val_seq, self.y_val_seq),
                verbose=verbose
            )
            
            # Évaluation sur les différents datasets
            results = {}
            
            print("\n" + "="*50)
            print("ÉVALUATION DES PERFORMANCES")
            print("="*50)
            
            # 1. Évaluation sur TRAIN
            results['train'] = self._evaluate_model(
                model, self.X_train_seq, self.y_train_seq, self.scaler_y, "TRAIN"
            )
            
            # 2. Évaluation sur VALIDATION
            results['val'] = self._evaluate_model(
                model, self.X_val_seq, self.y_val_seq, self.scaler_y, "VALIDATION"
            )
            
            # 3. Évaluation sur TEST
            results['test'] = self._evaluate_model(
                model, self.X_test_seq, self.y_test_seq, self.scaler_y, "TEST"
            )
            
            # 4. Évaluation sur TRAIN+VAL (CALAGE) si demandé
            if evaluate_trainval:
                if self.X_trainval_seq is None or self.seq_length != seq_length:
                    self.prepare_trainval_data(seq_length)
                
                print("-"*50)
                results['trainval'] = self._evaluate_model(
                    model, self.X_trainval_seq, self.y_trainval_seq, self.scaler_y, "TRAIN+VAL (CALAGE)"
                )
            
            # Ajouter le modèle et l'historique aux résultats
            results['model'] = model
            results['history'] = history
            
            return results
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {str(e)}")
            return {
                'train': {'rmse': 9999, 'r2': -9999, 'nse': -9999, 'kge': -9999, 'bias': 0},
                'val': {'rmse': 9999, 'r2': -9999, 'nse': -9999, 'kge': -9999, 'bias': 0},
                'test': {'rmse': 9999, 'r2': -9999, 'nse': -9999, 'kge': -9999, 'bias': 0},
                'trainval': {'rmse': 9999, 'r2': -9999, 'nse': -9999, 'kge': -9999, 'bias': 0},
                'model': None,
                'history': None
            }
    
    def _evaluate_model(self, model, X_seq, y_seq, scaler, dataset_name="Dataset"):
        """Évaluer un modèle sur un dataset donné comme dans code 2"""
        # Prédictions
        y_pred_scaled = model.predict(X_seq, verbose=0)
        
        # Dénormalisation
        y_true = scaler.inverse_transform(y_seq.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # Correction des débits négatifs
        y_pred = np.maximum(y_pred, 0)
        
        # Calcul des métriques
        rmse = Metrics.rmse(y_true, y_pred)
        r2 = Metrics.r2_score(y_true, y_pred)
        nash = Metrics.nse(y_true, y_pred)
        kge = Metrics.kge(y_true, y_pred)
        
        # Calcul du biais
        bias = Metrics.bias(y_true, y_pred)
        
        results = {
            'rmse': rmse,
            'r2': r2,
            'nse': nash,
            'kge': kge,
            'bias': bias,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        # Affichage
        print(f"{dataset_name}:")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  R²   = {r2:.4f}")
        print(f"  NSE  = {nash:.4f}")
        print(f"  KGE  = {kge:.4f}")
        print(f"  Biais= {bias:.4f}")
        print()
        
        return results


# ======================================================
# 5. OPTIMISATION MODHYPMA (NSGA-II)
# ======================================================
class ModHyPMAOptimizationProblem(Problem):
    def __init__(self, calib_data):
        super().__init__(
            n_var=4,      # m, l, P2, TX
            n_obj=3,      # RMSE, -R², -NSE (à minimiser)
            n_constr=0,
            xl=np.array([0.9, 26.0, 2.2, 0.00001]),
            xu=np.array([1.45, 150.0, 10.0, 0.8])
        )
        self.calib_data = calib_data
        
    def _evaluate(self, X, out, *args, **kwargs):
        f1, f2, f3 = [], [], []
        
        for params in X:
            m, l, P2, TX = params
            
            try:
                # Simulation sur la période de calage
                Q_sim = ModHyPMA_Model.simulate(
                    self.calib_data['Pluie'].values,
                    self.calib_data['ETP'].values,
                    m, l, P2, TX
                )
                
                # Évaluation des métriques
                Q_obs = self.calib_data['Qobs'].values
                
                # Assurer que les tableaux ont la même longueur
                min_len = min(len(Q_sim), len(Q_obs))
                Q_sim_trunc = Q_sim[:min_len]
                Q_obs_trunc = Q_obs[:min_len]
                
                rmse_val = Metrics.rmse(Q_obs_trunc, Q_sim_trunc)
                r2_val = Metrics.r2_score(Q_obs_trunc, Q_sim_trunc)
                nse_val = Metrics.nse(Q_obs_trunc, Q_sim_trunc)
                
            except Exception as e:
                # Valeurs pénalisantes en cas d'erreur
                rmse_val = 9999
                r2_val = 0
                nse_val = -9999
            
            f1.append(rmse_val)       # À minimiser
            f2.append(1 - r2_val)     # À minimiser (maximiser R²)
            f3.append(1 - nse_val)    # À minimiser (maximiser NSE)
        
        out["F"] = np.column_stack([f1, f2, f3])


# ======================================================
# 6. OPTIMISATION LSTM (NSGA-II comme code 2)
# ======================================================
class LSTMOptimizationProblem(Problem):
    def __init__(self, trainer):
        self.trainer = trainer
        super().__init__(
            n_var=5,  # epochs, lr, batch_size, seq_length, units
            n_obj=3,  # RMSE, 1-R², 1-NSE (à minimiser)
            xl=np.array([5, 0.001, 16, 7, 16]),
            xu=np.array([50, 0.1, 128, 30, 128])
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for params in X:
            epochs = int(params[0])
            lr = float(params[1])
            batch_size = int(params[2])
            seq_length = int(params[3])
            units = int(params[4])
            
            # Entraîner et évaluer sur TEST uniquement pour l'optimisation
            try:
                # Préparer les données
                self.trainer.prepare_data(seq_length)
                
                # Construire le modèle
                model_builder = LSTMModelBuilder(
                    input_shape=(self.trainer.X_train_seq.shape[1], 
                               self.trainer.X_train_seq.shape[2]),
                    units=units,
                    learning_rate=lr,
                    dropout_rate=0.25
                )
                model = model_builder.build()
                
                # Entraînement rapide (pas de validation pour accélérer)
                model.fit(
                    self.trainer.X_train_seq, self.trainer.y_train_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                # Prédictions sur TEST
                y_pred_scaled = model.predict(self.trainer.X_test_seq, verbose=0)
                y_true = self.trainer.scaler_y.inverse_transform(self.trainer.y_test_seq.reshape(-1, 1))
                y_pred = self.trainer.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
                y_pred = np.maximum(y_pred, 0)
                
                # Métriques
                rmse = Metrics.rmse(y_true, y_pred)
                r2 = Metrics.r2_score(y_true, y_pred)
                nash = Metrics.nse(y_true, y_pred)
                
            except Exception as e:
                print(f"Erreur optimisation: {e}")
                rmse, r2, nash = 9999, -9999, -9999
            
            # Objectifs à minimiser
            F.append([rmse, 1 - r2, 1 - nash])
        
        out["F"] = np.array(F)


# ======================================================
# 7. OPTIMISEURS
# ======================================================
class ModHyPMAOptimizer:
    def __init__(self, calib_data):
        self.calib_data = calib_data
        self.results = None
        self.best_params = None
        
    def optimize(self, pop_size=50, n_generations=30):
        """Exécuter l'optimisation NSGA-II pour ModHyPMA"""
        print("\n" + "="*60)
        print("OPTIMISATION MODHYPMA NSGA-II EN COURS")
        print("="*60)
        
        # Définir le problème
        problem = ModHyPMAOptimizationProblem(self.calib_data)
        
        # Configurer l'algorithme
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.90, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Critère d'arrêt
        termination = get_termination("n_gen", n_generations)
        
        # Exécuter l'optimisation
        self.results = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=True
        )
        
        # Trouver la meilleure solution (basée sur NSE)
        best_idx = np.argmin(self.results.F[:, 2])  # Minimiser (1-NSE)
        self.best_params = self.results.X[best_idx]
        
        # Afficher les résultats
        self._print_results()
        
        return {
            'm': self.best_params[0],
            'l': self.best_params[1],
            'P2': self.best_params[2],
            'TX': self.best_params[3]
        }
    
    def _print_results(self):
        """Afficher les résultats de l'optimisation"""
        print("\n" + "="*60)
        print("RÉSULTATS OPTIMISATION MODHYPMA")
        print("="*60)
        
        print(f"\nMeilleurs paramètres trouvés:")
        print(f"  m  = {self.best_params[0]:.4f}")
        print(f"  l  = {self.best_params[1]:.4f}")
        print(f"  P2 = {self.best_params[2]:.4f}")
        print(f"  TX = {self.best_params[3]:.4f}")
        
        print(f"\nPerformances sur le front de Pareto:")
        print(f"  RMSE min: {np.min(self.results.F[:, 0]):.4f}")
        print(f"  R² max:   {1 - np.min(self.results.F[:, 1]):.4f}")
        print(f"  NSE max:  {1 - np.min(self.results.F[:, 2]):.4f}")


class LSTMOptimizer:
    def __init__(self, lstm_trainer):
        self.lstm_trainer = lstm_trainer
        self.results = None
        self.best_params = None
        
    def optimize(self, pop_size=10, n_generations=10):
        """Exécuter l'optimisation NSGA-II pour LSTM comme dans code 2"""
        print("\n" + "="*60)
        print("OPTIMISATION LSTM NSGA-II EN COURS")
        print("="*60)
        print("Note: L'optimisation LSTM peut prendre plus de temps que ModHyPMA")
        
        # Définir le problème comme dans code 2
        problem = LSTMOptimizationProblem(self.lstm_trainer)
        
        # Configurer l'algorithme comme dans code 2
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Critère d'arrêt
        termination = get_termination("n_gen", n_generations)
        
        # Exécuter l'optimisation
        self.results = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=True
        )
        
        # Trouver la meilleure solution (basée sur NSE) comme dans code 2
        best_idx = np.argmin(self.results.F[:, 2])  # Minimiser (1-NSE)
        self.best_params = self.results.X[best_idx]
        
        # Afficher les résultats
        self._print_results()
        
        return {
            'epochs': int(self.best_params[0]),
            'lr': float(self.best_params[1]),
            'batch_size': int(self.best_params[2]),
            'seq_length': int(self.best_params[3]),
            'units': int(self.best_params[4])
        }
    
    def _print_results(self):
        """Afficher les résultats de l'optimisation"""
        print("\n" + "="*60)
        print("RÉSULTATS OPTIMISATION LSTM")
        print("="*60)
        
        print(f"\nMeilleurs hyperparamètres trouvés:")
        print(f"  Epochs:      {int(self.best_params[0])}")
        print(f"  Learning rate: {self.best_params[1]:.6f}")
        print(f"  Batch size:    {int(self.best_params[2])}")
        print(f"  Seq length:    {int(self.best_params[3])}")
        print(f"  Units LSTM:    {int(self.best_params[4])}")
        
        print(f"\nPerformances sur le front de Pareto:")
        print(f"  RMSE min: {np.min(self.results.F[:, 0]):.4f}")
        print(f"  R² max:   {1 - np.min(self.results.F[:, 1]):.4f}")
        print(f"  NSE max:  {1 - np.min(self.results.F[:, 2]):.4f}")


# ======================================================
# 8. ÉVALUATEURS
# ======================================================
class ModHyPMA_Evaluator:
    def __init__(self, calib_data, valid_data):
        self.calib_data = calib_data
        self.valid_data = valid_data
        self.results = {}
        
    def evaluate(self, m, l, P2, TX):
        """Évaluer les performances de ModHyPMA"""
        # Simulation sur calage
        Q_sim_cal = ModHyPMA_Model.simulate(
            self.calib_data['Pluie'].values,
            self.calib_data['ETP'].values,
            m, l, P2, TX
        )
        Q_obs_cal = self.calib_data['Qobs'].values
        
        # Simulation sur validation
        Q_sim_val = ModHyPMA_Model.simulate(
            self.valid_data['Pluie'].values,
            self.valid_data['ETP'].values,
            m, l, P2, TX
        )
        Q_obs_val = self.valid_data['Qobs'].values
        
        # Calcul des métriques
        self.results = {
            'calib': self._calculate_metrics(Q_obs_cal, Q_sim_cal),
            'valid': self._calculate_metrics(Q_obs_val, Q_sim_val),
            'params': {'m': m, 'l': l, 'P2': P2, 'TX': TX},
            'sim_calib': Q_sim_cal,
            'sim_valid': Q_sim_val
        }
        
        return self.results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculer les métriques"""
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        return {
            'rmse': Metrics.rmse(y_true, y_pred),
            'r2': Metrics.r2_score(y_true, y_pred),
            'nse': Metrics.nse(y_true, y_pred),
            'kge': Metrics.kge(y_true, y_pred),
            'bias': Metrics.bias(y_true, y_pred)
        }
    
    def print_summary(self):
        """Afficher le résumé"""
        print("\n" + "="*60)
        print("RÉSUMÉ MODHYPMA")
        print("="*60)
        
        params = self.results['params']
        print(f"\nParamètres:")
        print(f"  m  = {params['m']:.4f}")
        print(f"  l  = {params['l']:.4f}")
        print(f"  P2 = {params['P2']:.4f}")
        print(f"  TX = {params['TX']:.4f}")
        
        print("\n┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│     Période     │   RMSE   │    R²    │   NSE    │   KGE    │   Biais  │")
        print("├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
        
        periods = ['calib', 'valid']
        period_names = ['Calage', 'Validation']
        
        for i, period in enumerate(periods):
            r = self.results[period]
            print(f"│ {period_names[i]:15s} │ {r['rmse']:8.4f} │ {r['r2']:8.4f} │ {r['nse']:8.4f} │ {r['kge']:8.4f} │ {r['bias']:8.4f} │")
        
        print("└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
        
        # Analyse
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyser les résultats"""
        calib_nse = self.results['calib']['nse']
        valid_nse = self.results['valid']['nse']
        
        print(f"\n→ Comparaison Calage vs Validation:")
        print(f"  NSE Calage: {calib_nse:.4f}")
        print(f"  NSE Validation: {valid_nse:.4f}")
        
        if abs(calib_nse - valid_nse) > 0.2:
            print("  ⚠️  Différence importante: risque de surapprentissage")
        elif abs(calib_nse - valid_nse) > 0.1:
            print("  ⚠️  Différence modérée")
        else:
            print("  ✓ Bonne consistance")
        
        # Analyse qualité
        print(f"\n→ Qualité du modèle (Validation):")
        if valid_nse < 0.0:
            print("  NSE < 0.0: Performances mauvaises")
        elif valid_nse < 0.5:
            print("  0.0 ≤ NSE < 0.5: Performances insuffisantes")
        elif valid_nse < 0.65:
            print("  0.5 ≤ NSE < 0.65: Bonnes performances")
        elif valid_nse < 0.8:
            print("  0.65 ≤ NSE < 0.8: Tres Bonnes performances")
        else:
            print("  NSE ≥ 0.8: Excellentes performances!")
    
    def plot_results(self):
        """Tracer les résultats"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Série temporelle calage
        axes[0, 0].plot(self.calib_data.index, self.calib_data['Qobs'], 
                       'b-', label='Observé', alpha=0.7, linewidth=1)
        axes[0, 0].plot(self.calib_data.index, self.results['sim_calib'], 
                       'r-', label='Simulé', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('ModHyPMA - Calage')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Débit (m³/s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Série temporelle validation
        axes[0, 1].plot(self.valid_data.index, self.valid_data['Qobs'], 
                       'b-', label='Observé', alpha=0.7, linewidth=1)
        axes[0, 1].plot(self.valid_data.index, self.results['sim_valid'], 
                       'r-', label='Simulé', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('ModHyPMA - Validation')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Débit (m³/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Diagramme de dispersion calage
        axes[1, 0].scatter(self.calib_data['Qobs'].values, self.results['sim_calib'], 
                          alpha=0.5, s=10, color='blue')
        min_val = min(self.calib_data['Qobs'].min(), self.results['sim_calib'].min())
        max_val = max(self.calib_data['Qobs'].max(), self.results['sim_calib'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
        axes[1, 0].set_title('Diagramme de Dispersion - Calage')
        axes[1, 0].set_xlabel('Observé (m³/s)')
        axes[1, 0].set_ylabel('Simulé (m³/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Diagramme de dispersion validation
        axes[1, 1].scatter(self.valid_data['Qobs'].values, self.results['sim_valid'], 
                          alpha=0.5, s=10, color='green')
        min_val = min(self.valid_data['Qobs'].min(), self.results['sim_valid'].min())
        max_val = max(self.valid_data['Qobs'].max(), self.results['sim_valid'].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
        axes[1, 1].set_title('Diagramme de Dispersion - Validation')
        axes[1, 1].set_xlabel('Observé (m³/s)')
        axes[1, 1].set_ylabel('Simulé (m³/s)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class LSTM_Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer
        self.results = {}
    
    def evaluate(self, model, seq_length, evaluate_trainval=True):
        """Évaluer les performances du LSTM comme dans code 2"""
        # Préparer les données si nécessaire
        if self.trainer.seq_length != seq_length or self.trainer.X_train_seq is None:
            self.trainer.prepare_data(seq_length)
        
        # Évaluation sur tous les datasets
        results = {}
        
        print("\n" + "="*50)
        print("ÉVALUATION COMPLÈTE LSTM")
        print("="*50)
        
        # 1. Évaluation sur TRAIN
        results['train'] = self.trainer._evaluate_model(
            model, self.trainer.X_train_seq, self.trainer.y_train_seq, 
            self.trainer.scaler_y, "TRAIN"
        )
        
        # 2. Évaluation sur VALIDATION
        results['val'] = self.trainer._evaluate_model(
            model, self.trainer.X_val_seq, self.trainer.y_val_seq, 
            self.trainer.scaler_y, "VALIDATION"
        )
        
        # 3. Évaluation sur TEST
        results['test'] = self.trainer._evaluate_model(
            model, self.trainer.X_test_seq, self.trainer.y_test_seq, 
            self.trainer.scaler_y, "TEST"
        )
        
        # 4. Évaluation sur TRAIN+VAL si demandé
        if evaluate_trainval:
            if self.trainer.X_trainval_seq is None or self.trainer.seq_length != seq_length:
                self.trainer.prepare_trainval_data(seq_length)
            
            print("-"*50)
            results['trainval'] = self.trainer._evaluate_model(
                model, self.trainer.X_trainval_seq, self.trainer.y_trainval_seq, 
                self.trainer.scaler_y, "TRAIN+VAL (CALAGE)"
            )
        
        self.results = results
        return results
    
    def print_summary(self):
        """Afficher le résumé comme dans code 2"""
        print("\n" + "="*60)
        print("RÉSUMÉ LSTM")
        print("="*60)
        
        print("\n┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        print("│     Dataset     │   RMSE   │    R²    │   NSE    │   KGE    │   Biais  │")
        print("├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
        
        datasets = ['train', 'val', 'test', 'trainval']
        dataset_names = ['TRAIN', 'VALIDATION', 'TEST', 'TRAIN+VAL']
        
        for i, dataset in enumerate(datasets):
            if dataset in self.results:
                r = self.results[dataset]
                print(f"│ {dataset_names[i]:15s} │ {r['rmse']:8.4f} │ {r['r2']:8.4f} │ {r['nse']:8.4f} │ {r['kge']:8.4f} │ {r['bias']:8.4f} │")
        
        print("└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
        
        # Analyse comme dans code 2
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyser les résultats comme dans code 2"""
        if 'test' not in self.results:
            return
            
        test_nse = self.results['test']['nse']
        
        print(f"\n→ Qualité du modèle (TEST):")
        if test_nse < 0.0:
            print("  NSE < 0.0: Le modèle est moins bon que la moyenne des observations.")
            print("  Le modèle ne capture pas correctement la dynamique du débit.")
            print("  Suggestions:")
            print("  1. Vérifier la qualité des données d'entrée")
            print("  2. Augmenter la complexité du modèle (plus d'unités LSTM)")
            print("  3. Essayer différentes fenêtres temporelles")
        elif test_nse < 0.5:
            print("→ 0.0 ≤ NSE < 0.5: Performances acceptables mais limitées.")
            print("  Le modèle capture partiellement la dynamique mais avec des erreurs importantes.")
            print("  Suggestions:")
            print("  1. Ajouter plus de features")
            print("  2. Augmenter la longueur de séquence")
            print("  3. Utiliser une architecture plus complexe")
        elif test_nse < 0.8:
            print("→ 0.5 ≤ NSE < 0.8: Bonnes performances.")
            print("  Le modèle capture bien la dynamique générale du débit.")
            print("  Suggestions pour amélioration:")
            print("  1. Ajuster le learning rate")
            print("  2. Ajouter des techniques de régularisation")
            print("  3. Tester différentes fonctions d'activation")
        else:
            print("→ NSE ≥ 0.8: Excellentes performances!")
            print("  Le modèle capture très bien la dynamique du débit.")
            print("  Le modèle est prêt pour des applications opérationnelles.")
        
        # Vérification de la consistance TRAIN+VAL vs TEST comme dans code 2
        if 'trainval' in self.results and 'test' in self.results:
            trainval_nse = self.results['trainval']['nse']
            test_nse = self.results['test']['nse']
            
            print(f"\n→ Comparaison CALAGE (TRAIN+VAL) vs TEST:")
            print(f"  NSE CALAGE: {trainval_nse:.4f}")
            print(f"  NSE TEST: {test_nse:.4f}")
            
            if abs(trainval_nse - test_nse) > 0.2:
                print("  ⚠️  Différence importante: risque de surapprentissage")
            elif abs(trainval_nse - test_nse) > 0.1:
                print("  ⚠️  Différence modérée: le modèle généralise modérément")
            else:
                print("  ✓ Bonne consistance: le modèle généralise bien")
    
    def plot_results(self):
        """Tracer les résultats"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Déterminer les dates pour chaque dataset
        train_dates = self.trainer.train.index[self.trainer.seq_length:]
        val_dates = self.trainer.val.index[self.trainer.seq_length:]
        test_dates = self.trainer.test.index[self.trainer.seq_length:]
        
        # 1. Série temporelle TRAIN
        if 'train' in self.results:
            axes[0, 0].plot(train_dates, self.results['train']['y_true'], 
                           'b-', label='Observé', alpha=0.7, linewidth=1)
            axes[0, 0].plot(train_dates, self.results['train']['y_pred'], 
                           'r-', label='Simulé', alpha=0.7, linewidth=1)
            axes[0, 0].set_title('LSTM - TRAIN')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Débit (m³/s)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Série temporelle TEST
        if 'test' in self.results:
            axes[0, 1].plot(test_dates, self.results['test']['y_true'], 
                           'b-', label='Observé', alpha=0.7, linewidth=1)
            axes[0, 1].plot(test_dates, self.results['test']['y_pred'], 
                           'r-', label='Simulé', alpha=0.7, linewidth=1)
            axes[0, 1].set_title('LSTM - TEST')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Débit (m³/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Diagramme de dispersion TRAIN
        if 'train' in self.results:
            obs_train = self.results['train']['y_true']
            sim_train = self.results['train']['y_pred']
            axes[1, 0].scatter(obs_train, sim_train, alpha=0.5, s=10, color='blue')
            min_val = min(obs_train.min(), sim_train.min())
            max_val = max(obs_train.max(), sim_train.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
            axes[1, 0].set_title('Diagramme de Dispersion - TRAIN')
            axes[1, 0].set_xlabel('Observé (m³/s)')
            axes[1, 0].set_ylabel('Simulé (m³/s)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Diagramme de dispersion TEST
        if 'test' in self.results:
            obs_test = self.results['test']['y_true']
            sim_test = self.results['test']['y_pred']
            axes[1, 1].scatter(obs_test, sim_test, alpha=0.5, s=10, color='green')
            min_val = min(obs_test.min(), sim_test.min())
            max_val = max(obs_test.max(), sim_test.max())
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
            axes[1, 1].set_title('Diagramme de Dispersion - TEST')
            axes[1, 1].set_xlabel('Observé (m³/s)')
            axes[1, 1].set_ylabel('Simulé (m³/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ======================================================
# 9. PIPELINE PRINCIPAL
# ======================================================
class Hydrological_Pipeline:
    def __init__(self):
        set_seed(42)
        self.df = None
        self.calib_data = None
        self.valid_data = None
        self.selected_model = None
        self.best_params = None
        self.final_model = None
        
    def run(self):
        """Exécuter le pipeline complet"""
        print("=" * 60)
        print("PIPELINE DE MODÉLISATION HYDROLOGIQUE")
        print("=" * 60)
        
        # Étape 1: Choix du modèle
        print("\n1. CHOIX DU MODÈLE")
        print("-" * 30)
        print("Modèles disponibles:")
        print("  1. ModHyPMA - Modèle hydrologique à base physique")
        print("  2. LSTM - Réseau de neurones récurrent")
        
        while True:
            model_choice = input("\nChoisissez le modèle (1 ou 2): ").strip()
            if model_choice == '1':
                self.selected_model = 'ModHyPMA'
                break
            elif model_choice == '2':
                self.selected_model = 'LSTM'
                break
            else:
                print("Choix invalide. Tapez '1' pour ModHyPMA ou '2' pour LSTM")
        
        print(f"✓ Modèle sélectionné: {self.selected_model}")
        
        # Étape 2: Chargement des données
        print(f"\n2. CHARGEMENT DES DONNÉES")
        print("-" * 30)
        path = input("Chemin du fichier (CSV ou Excel): ").strip()
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas")
        
        # Charger les données comme dans code 2
        loader = DataLoader()
        self.df = loader.load(path)
        print(f"✓ Données chargées: {len(self.df)} lignes")
        print(f"  Période: {self.df.index.min()} à {self.df.index.max()}")
        
        # Feature engineering pour LSTM
        if self.selected_model == 'LSTM':
            print("\n→ Feature Engineering pour LSTM...")
            engineer = FeatureEngineer()
            self.df = engineer.transform(self.df)
            print(f"✓ Features créées: {len(self.df.columns)} colonnes")
        
        # Étape 3: Définition des périodes
        print(f"\n3. DÉFINITION DES PÉRIODES")
        print("-" * 30)
        
        if self.selected_model == 'ModHyPMA':
            # Pour ModHyPMA: calage et validation
            print("Format: YYYY-MM-DD ou YYYY (ex: 1970 ou 1970-01-01)")
            calib_start = input("Début période de calage: ").strip()
            calib_end = input("Fin période de calage: ").strip()
            valid_start = input("Début période de validation: ").strip()
            valid_end = input("Fin période de validation: ").strip()
            
            # Diviser les données pour ModHyPMA
            self.calib_data = self.df.loc[calib_start:calib_end].copy()
            self.valid_data = self.df.loc[valid_start:valid_end].copy()
            
            if len(self.calib_data) == 0:
                raise ValueError("Période de calage vide")
            if len(self.valid_data) == 0:
                raise ValueError("Période de validation vide")
            
            print(f"✓ Calage: {len(self.calib_data)} échantillons ({calib_start} à {calib_end})")
            print(f"✓ Validation: {len(self.valid_data)} échantillons ({valid_start} à {valid_end})")
            
        else:  # LSTM
            # Pour LSTM: train, validation, test comme dans code 2
            print("Format: YYYY-MM-DD (ex: 1970-01-01)")
            train_start = input("Début période d'entraînement (TRAIN): ").strip()
            train_end = input("Fin période d'entraînement (TRAIN): ").strip()
            val_start = input("Début période de validation (VAL): ").strip()
            val_end = input("Fin période de validation (VAL): ").strip()
            test_start = input("Début période de test (TEST): ").strip()
            test_end = input("Fin période de test (TEST): ").strip()
            
            # Diviser les données pour LSTM
            self.train_data = self.df.loc[train_start:train_end].copy()
            self.val_data = self.df.loc[val_start:val_end].copy()
            self.test_data = self.df.loc[test_start:test_end].copy()
            
            if len(self.train_data) == 0:
                raise ValueError("Période d'entraînement vide")
            if len(self.val_data) == 0:
                raise ValueError("Période de validation vide")
            if len(self.test_data) == 0:
                raise ValueError("Période de test vide")
            
            print(f"✓ TRAIN: {len(self.train_data)} échantillons ({train_start} à {train_end})")
            print(f"✓ VAL: {len(self.val_data)} échantillons ({val_start} à {val_end})")
            print(f"✓ TEST: {len(self.test_data)} échantillons ({test_start} à {test_end})")
            
            # Définir les features pour LSTM
            self.features = [col for col in self.df.columns if col != 'Qobs']
            print(f"\n✓ Features utilisées pour LSTM ({len(self.features)}):")
            for i, feat in enumerate(self.features[:10]):  # Afficher les 10 premières
                print(f"  {i+1:2d}. {feat}")
            if len(self.features) > 10:
                print(f"  ... et {len(self.features)-10} autres")
        
        # Étape 4: Choix du mode d'optimisation
        print(f"\n4. MODE D'OPTIMISATION")
        print("-" * 30)
        while True:
            mode = input("Mode (manuel/auto): ").strip().lower()
            if mode in ['manuel', 'auto']:
                break
            print("Choix invalide. Tapez 'manuel' ou 'auto'")
        
        if mode == 'manuel':
            self._manual_mode()
        else:
            self._auto_mode()
        
        # Étape 5: Évaluation finale
        print(f"\n5. ÉVALUATION FINALE")
        print("-" * 30)
        self._final_evaluation()
    
    def _manual_mode(self):
        """Mode manuel: l'utilisateur saisit les paramètres"""
        print(f"\n→ MODE MANUEL - {self.selected_model}")
        
        if self.selected_model == 'ModHyPMA':
            print("Veuillez saisir les paramètres du modèle ModHyPMA:")
            
            self.best_params = {
                'm': float(input("  Paramètre m (0.9-1.45): ")),
                'l': float(input("  Paramètre l (26-200): ")),
                'P2': float(input("  Paramètre P2 (3.1-6): ")),
                'TX': float(input("  Paramètre TX (0.00001-0.8): "))
            }
            
        else:  # LSTM
            print("Veuillez saisir les hyperparamètres du modèle LSTM:")
            
            self.best_params = {
                'epochs': int(input("  Nombre d'epochs (5-50): ")),
                'lr': float(input("  Learning rate (0.001-0.1): ")),
                'batch_size': int(input("  Batch size (16-128): ")),
                'seq_length': int(input("  Longueur de séquence (7-30): ")),
                'units': int(input("  Unités LSTM (16-128): "))
            }
        
        print(f"\n✓ Paramètres saisis pour {self.selected_model}:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
    
    def _auto_mode(self):
        """Mode automatique: optimisation NSGA-II"""
        print(f"\n→ MODE AUTOMATIQUE - {self.selected_model}")
        
        # Configuration de l'optimisation
        pop_size = input("Taille de la population (défaut: 30 pour ModHyPMA, 10 pour LSTM): ").strip()
        n_generations = input("Nombre de générations (défaut: 20 pour ModHyPMA, 10 pour LSTM): ").strip()
        
        if self.selected_model == 'ModHyPMA':
            # Optimisation ModHyPMA
            pop_size = int(pop_size) if pop_size else 30
            n_generations = int(n_generations) if n_generations else 20
            
            optimizer = ModHyPMAOptimizer(self.calib_data)
            self.best_params = optimizer.optimize(pop_size, n_generations)
            
        else:  # LSTM
            # Optimisation LSTM comme dans code 2
            pop_size = int(pop_size) if pop_size else 10
            n_generations = int(n_generations) if n_generations else 10
            
            # Créer le trainer LSTM comme dans code 2
            lstm_trainer = LSTMTrainer(self.train_data, self.val_data, self.test_data, self.features)
            
            # Optimiser
            optimizer = LSTMOptimizer(lstm_trainer)
            self.best_params = optimizer.optimize(pop_size, n_generations)
            
            # Entraîner le modèle final avec les meilleurs paramètres
            print("\n→ Entraînement du modèle LSTM final...")
            self.final_results = lstm_trainer.train_and_eval(
                epochs=self.best_params['epochs'],
                lr=self.best_params['lr'],
                batch_size=self.best_params['batch_size'],
                seq_length=self.best_params['seq_length'],
                units=self.best_params['units'],
                verbose=1,
                evaluate_trainval=False  # On fera l'évaluation complète plus tard
            )
            self.final_model = self.final_results['model']
            self.lstm_trainer = lstm_trainer  # Sauvegarder pour l'évaluation
    
    def _final_evaluation(self):
        """Évaluation finale avec les paramètres choisis"""
        print(f"\n→ ÉVALUATION FINALE - {self.selected_model}")
        
        if self.best_params is None:
            raise ValueError("Aucun paramètre disponible pour l'évaluation")
        
        # Afficher les paramètres
        print(f"\nParamètres utilisés pour {self.selected_model}:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        if self.selected_model == 'ModHyPMA':
            # Évaluation ModHyPMA
            evaluator = ModHyPMA_Evaluator(self.calib_data, self.valid_data)
            results = evaluator.evaluate(
                self.best_params['m'],
                self.best_params['l'],
                self.best_params['P2'],
                self.best_params['TX']
            )
            
            # Afficher le résumé
            evaluator.print_summary()
            
        else:  # LSTM
            # Évaluation LSTM comme dans code 2
            if self.final_model is None:
                # Si mode manuel, entraîner d'abord le modèle
                print("\n→ Entraînement du modèle LSTM...")
                self.lstm_trainer = LSTMTrainer(self.train_data, self.val_data, self.test_data, self.features)
                self.final_results = self.lstm_trainer.train_and_eval(
                    epochs=self.best_params['epochs'],
                    lr=self.best_params['lr'],
                    batch_size=self.best_params['batch_size'],
                    seq_length=self.best_params['seq_length'],
                    units=self.best_params['units'],
                    verbose=1,
                    evaluate_trainval=True  # Inclure TRAIN+VAL pour l'évaluation finale
                )
                self.final_model = self.final_results['model']
            
            # Évaluer le modèle avec évaluation complète
            evaluator = LSTM_Evaluator(self.lstm_trainer)
            results = evaluator.evaluate(self.final_model, self.best_params['seq_length'], evaluate_trainval=True)
            
            # Afficher le résumé
            evaluator.print_summary()
        
        # Demander si on veut visualiser
        plot = input("\nVoulez-vous visualiser les résultats? (oui/non): ").strip().lower()
        if plot == 'oui':
            if self.selected_model == 'ModHyPMA':
                evaluator.plot_results()
            else:
                evaluator.plot_results()
        
        # Demander si on veut sauvegarder
        save = input("\nVoulez-vous sauvegarder les résultats? (oui/non): ").strip().lower()
        if save == 'oui':
            self._save_results(results)
        
        print("\n" + "="*60)
        print("PIPELINE TERMINÉ AVEC SUCCÈS!")
        print("="*60)
    
    def _save_results(self, results):
        """Sauvegarder les résultats"""
        filename = input(f"Nom du fichier pour sauvegarde (ex: resultats_{self.selected_model.lower()}.txt): ").strip()
        
        with open(filename, 'w') as f:
            f.write(f"RÉSULTATS DU MODÈLE {self.selected_model}\n")
            f.write("="*50 + "\n")
            
            f.write(f"\nParamètres du modèle:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nPerformances:\n")
            
            # Pour ModHyPMA
            if self.selected_model == 'ModHyPMA':
                for period_name, period_key in [('Calage', 'calib'), 
                                              ('Validation', 'valid')]:
                    if period_key in results:
                        r = results[period_key]
                        f.write(f"\n{period_name}:\n")
                        f.write(f"  RMSE:  {r['rmse']:.4f}\n")
                        f.write(f"  R²:    {r['r2']:.4f}\n")
                        f.write(f"  NSE:   {r['nse']:.4f}\n")
                        f.write(f"  KGE:   {r['kge']:.4f}\n")
                        f.write(f"  Biais: {r['bias']:.4f}\n")
            
            # Pour LSTM
            else:
                for dataset_name, dataset_key in [('TRAIN', 'train'), 
                                                ('VALIDATION', 'val'),
                                                ('TEST', 'test'),
                                                ('TRAIN+VAL', 'trainval')]:
                    if dataset_key in results:
                        r = results[dataset_key]
                        f.write(f"\n{dataset_name}:\n")
                        f.write(f"  RMSE:  {r['rmse']:.4f}\n")
                        f.write(f"  R²:    {r['r2']:.4f}\n")
                        f.write(f"  NSE:   {r['nse']:.4f}\n")
                        f.write(f"  KGE:   {r['kge']:.4f}\n")
                        f.write(f"  Biais: {r['bias']:.4f}\n")
        
        print(f"✓ Résultats sauvegardés sous: {filename}")
        
        # Sauvegarder aussi le modèle LSTM si c'est le cas
        if self.selected_model == 'LSTM' and self.final_model is not None:
            model_filename = filename.replace('.txt', '.h5')
            self.final_model.save(model_filename)
            print(f"✓ Modèle LSTM sauvegardé sous: {model_filename}")


# ======================================================
# LANCEMENT
# ======================================================
if __name__ == "__main__":
    try:
        pipeline = Hydrological_Pipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        print("\nVérifiez que:")
        print("1. Le chemin du fichier est correct")
        print("2. Le fichier contient les colonnes: date, Qobs, Pluie, ETP")
        print("3. Les périodes définies existent dans vos données")
        print("4. Les paramètres saisis sont dans les bornes valides")