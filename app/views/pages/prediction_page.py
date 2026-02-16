"""
Page de prédiction hydrologique avec VRAI modèle LSTM
Utilise les modèles depuis app/models
Version complète et corrigée
"""

from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import du service de prédiction
from app.services.prediction import get_predictor, TENSORFLOW_AVAILABLE
from app.views.components.alerts import create_alert


def create_prediction_page():
    """Crée la page de prédiction hydrologique avec vrai modèle LSTM"""
    
    # Vérifier la disponibilité des modèles
    predictor = get_predictor()
    available_models = predictor.get_available_models()
    
    if not TENSORFLOW_AVAILABLE:
        return create_error_page(
            "TensorFlow non installé",
            "Pour utiliser les prédictions, installez tensorflow:",
            "pip install tensorflow"
        )
    
    if not available_models:
        return create_error_page(
            "Modèles LSTM non trouvés",
            "Aucun modèle trouvé dans app/models/",
            "Fichiers attendus: lstm_qmax_q90.h5, lstm_mean.h5, scaler_*.pkl"
        )
    
    return dbc.Container([
        # Header avec statut des modèles
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("🔮 Prédiction Hydrologique LSTM", 
                           className="mb-2",
                           style={"fontSize": "22px", "fontWeight": "600", "color": "#2c3e50"}),
                    html.P("Prédiction des débits futurs avec modèles LSTM pré-entraînés",
                          className="text-muted mb-0",
                          style={"fontSize": "14px"}),
                    # Badge statut modèle
                    html.Div([
                        html.Span("✓ Modèles LSTM chargés", 
                                 className="badge bg-success me-2",
                                 style={"fontSize": "12px"}),
                        html.Span(f"{len(available_models)} modèle(s) disponible(s)", 
                                 className="badge bg-info",
                                 style={"fontSize": "12px"})
                    ], className="mt-2")
                ], className="text-center")
            ])
        ], className="mb-4 pt-3",
           style={"borderBottom": "1px solid #eaeaea", "backgroundColor": "white"}),
        
        # Section principale: Import des données
        dbc.Row([
            # Colonne gauche - Précipitation
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-cloud-rain me-2"),
                            "Données de précipitation"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Upload précipitation
                        html.Div([
                            dbc.Label("Importation données P", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dcc.Upload(
                                id="precipitation-upload",
                                children=html.Div([
                                    html.Div([
                                        html.I(className="fas fa-file-upload me-2"),
                                        "Projections de précipitation"
                                    ], className="text-center"),
                                    html.Small("Format Excel recommandé (.xlsx)", 
                                             className="text-muted d-block mt-1")
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '100px',
                                    'lineHeight': '100px',
                                    'borderWidth': '2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '8px',
                                    'borderColor': '#4a6fa5',
                                    'textAlign': 'center',
                                    'backgroundColor': '#f8f9fa',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s'
                                }
                            ),
                        ], className="mb-4"),
                        
                        # Statut de l'upload
                        html.Div(id="p-upload-status", className="mb-4"),
                        
                        # Sélection du modèle climatique
                        html.Div([
                            dbc.Label("Modèle climatique", 
                                     className="small mb-1 fw-bold",
                                     style={"color": "#495057"}),
                            dcc.Dropdown(
                                id="p-model-selector",
                                placeholder="Sélectionnez un modèle...",
                                className="mb-3",
                                style={"fontSize": "13px", "borderRadius": "6px"}
                            ),
                        ]),
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=6, className="mb-3"),
            
            # Colonne droite - ETP
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-sun me-2"),
                            "Données d'évapotranspiration"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Upload ETP
                        html.Div([
                            dbc.Label("Importation données ETP", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dcc.Upload(
                                id="etp-upload",
                                children=html.Div([
                                    html.Div([
                                        html.I(className="fas fa-file-upload me-2"),
                                        "Projections d'évapotranspiration"
                                    ], className="text-center"),
                                    html.Small("Format Excel recommandé (.xlsx)", 
                                             className="text-muted d-block mt-1")
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '100px',
                                    'lineHeight': '100px',
                                    'borderWidth': '2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '8px',
                                    'borderColor': '#4a6fa5',
                                    'textAlign': 'center',
                                    'backgroundColor': '#f8f9fa',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s'
                                }
                            ),
                        ], className="mb-4"),
                        
                        # Statut de l'upload
                        html.Div(id="e-upload-status", className="mb-4"),
                        
                        # Sélection du modèle climatique pour ETP
                        html.Div([
                            dbc.Label("Modèle climatique", 
                                     className="small mb-1 fw-bold",
                                     style={"color": "#495057"}),
                            dcc.Dropdown(
                                id="e-model-selector",
                                placeholder="Sélectionnez un modèle...",
                                className="mb-3",
                                style={"fontSize": "13px", "borderRadius": "6px"}
                            ),
                        ]),
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=6, className="mb-3"),
        ], className="mb-4"),
        
        # Section configuration et lancement
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-sliders-h me-2"),
                            "Configuration de la prédiction"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Sélection du modèle LSTM
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Modèle LSTM", 
                                        className="form-label small fw-bold text-secondary mb-2"),
                                dcc.Dropdown(
                                    id="lstm-model-selector",
                                    options=[
                                        {"label": "📊 Qmax/Q90 - Débits extrêmes", "value": "max"},
                                        {"label": "📈 Moyen - Débits journaliers", "value": "mean"}
                                    ],
                                    value="max",
                                    clearable=False,
                                    style={"fontSize": "13px", "borderRadius": "6px"}
                                ),
                            ], md=6),
                            dbc.Col([
                                dbc.Label("Type d'analyse", 
                                        className="form-label small fw-bold text-secondary mb-2"),
                                dcc.Dropdown(
                                    id="analysis-type",
                                    options=[
                                        {"label": "📊 Qmax/Q90 - Analyse annuelle", "value": "qmax_q90"},
                                        {"label": "📈 Statistiques complètes", "value": "full"},
                                    ],
                                    value="qmax_q90",
                                    clearable=False,
                                    style={"fontSize": "13px", "borderRadius": "6px"}
                                ),
                            ], md=6),
                        ], className="mb-4"),
                        
                        # Valeurs de référence (optionnelles)
                        dbc.Collapse(
                            id="reference-values-collapse",
                            is_open=False,
                            children=[
                                dbc.Card([
                                    dbc.CardHeader([
                                        html.I(className="fas fa-chart-line me-2"),
                                        "Valeurs de référence (optionnel)"
                                    ], className="py-2", style={"backgroundColor": "#f8f9fa"}),
                                    dbc.CardBody([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Qmax référence", className="small"),
                                                dbc.Input(
                                                    id="qmax-ref",
                                                    type="number",
                                                    placeholder="ex: 12887.41",
                                                    step=0.01,
                                                    className="mb-2"
                                                ),
                                            ], md=6),
                                            dbc.Col([
                                                dbc.Label("Q90 référence", className="small"),
                                                dbc.Input(
                                                    id="q90-ref",
                                                    type="number",
                                                    placeholder="ex: 6130.45",
                                                    step=0.01,
                                                    className="mb-2"
                                                ),
                                            ], md=6),
                                        ])
                                    ], className="p-3")
                                ], className="mb-3")
                            ]
                        ),
                        
                        # Bouton pour afficher/cacher les références
                        dbc.Button(
                            [
                                html.I(className="fas fa-cog me-2"),
                                "Paramètres avancés"
                            ],
                            id="toggle-advanced-btn",
                            color="secondary",
                            size="sm",
                            outline=True,
                            className="mb-3",
                            style={"fontSize": "12px"}
                        ),
                        
                        # Zone de validation
                        html.Div(id="prediction-warning", className="mb-4"),
                        
                        # Bouton principal
                        dbc.Button(
                            [
                                html.I(className="fas fa-magic me-2"),
                                "Exécuter la prédiction LSTM"
                            ],
                            id="run-prediction-btn",
                            color="primary",
                            size="lg",
                            className="w-100 py-3",
                            disabled=True,
                            style={
                                "backgroundColor": "#4a6fa5", 
                                "border": "none", 
                                "borderRadius": "8px", 
                                "fontSize": "16px",
                                "fontWeight": "500"
                            }
                        ),
                        
                        # Informations sur le modèle
                        html.Div([
                            html.Hr(className="my-4"),
                            html.Div(id="model-info-display", className="mt-2")
                        ]),
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], className="mb-3")
        ]),
        
        # Section visualisation
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-line me-2"),
                            "Visualisation des prédictions LSTM"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-prediction",
                            type="circle",
                            color="#4a6fa5",
                            children=[
                                dcc.Graph(
                                    id="prediction-hydrograph",
                                    config={
                                        'displayModeBar': True,
                                        'displaylogo': False,
                                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                                    },
                                    style={'height': '450px'}
                                )
                            ]
                        ),
                        html.Div(id="prediction-stats", className="mt-4")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], className="mb-4")
        ]),
        
        # Section résultats détaillés
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-table me-2"),
                            "Résultats détaillés"
                        ], className="d-flex align-items-center"),
                        html.Div([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-file-csv me-2"),
                                        "CSV"
                                    ],
                                    id="download-csv-btn",
                                    color="success",
                                    size="sm",
                                    className="py-1 px-3",
                                    style={"borderRadius": "4px", "fontSize": "13px"}
                                ),
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-file-excel me-2"),
                                        "Excel"
                                    ],
                                    id="download-excel-btn",
                                    color="warning",
                                    size="sm",
                                    className="py-1 px-3",
                                    style={"borderRadius": "4px", "fontSize": "13px"}
                                ),
                            ], className="ms-auto")
                        ], className="d-flex align-items-center")
                    ], className="py-2 d-flex justify-content-between align-items-center", 
                       style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div(id="prediction-results-table", className="table-responsive")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], className="mb-4")
        ]),
        
        # Section analyse statistique
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-bar me-2"),
                            "Analyse statistique détaillée"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div(id="prediction-detailed-stats")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], className="mb-4")
        ]),
        
        # Stores
        dcc.Store(id="precipitation-store"),
        dcc.Store(id="etp-store"),
        dcc.Store(id="prediction-results-store"),
        dcc.Store(id="prediction-metadata-store"),
        
        # Downloads
        dcc.Download(id="download-csv-pred"),
        dcc.Download(id="download-excel-pred"),
        
    ], fluid=False, className="py-3", style={'backgroundColor': '#f8f9fa', "marginLeft": "200px"})


def create_error_page(title, message, detail=""):
    """Crée une page d'erreur élégante"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle fa-4x mb-4", 
                                  style={"color": "#e74c3c"}),
                            html.H4(title, className="mb-3", style={"color": "#2c3e50"}),
                            html.P(message, className="text-muted mb-3"),
                            html.Code(detail, 
                                     className="d-block p-3 bg-dark text-white rounded",
                                     style={"fontSize": "13px"}),
                            html.Hr(className="my-4"),
                            dbc.Button(
                                [html.I(className="fas fa-home me-2"), "Retour accueil"],
                                href="/dash/home",
                                color="primary",
                                className="px-5",
                                style={"backgroundColor": "#4a6fa5", "border": "none"}
                            )
                        ], className="text-center p-5")
                    ])
                ], className="shadow border-0 mx-auto",
                   style={"maxWidth": "600px", "borderRadius": "12px", "marginTop": "50px"})
            ])
        ])
    ], fluid=False, className="py-3", style={'backgroundColor': '#f8f9fa'})


# ====================================================
# CALLBACKS
# ====================================================

@callback(
    Output("precipitation-store", "data"),
    Output("p-upload-status", "children"),
    Output("p-model-selector", "options"),
    Output("p-model-selector", "value"),
    Input("precipitation-upload", "contents"),
    State("precipitation-upload", "filename"),
    prevent_initial_call=True
)
def load_precipitation(contents, filename):
    """Charge les données de précipitation"""
    if not contents:
        return None, None, [], None
    
    try:
        # Lecture du fichier
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(decoded))
        else:
            return None, create_alert("danger", "Format non supporté. Utilisez .xlsx ou .csv"), [], None
        
        # Vérifier colonne date
        if 'date' not in df.columns:
            return None, create_alert("danger", "Colonne 'date' non trouvée"), [], None
        
        # Préparer les données
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Nettoyer
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        
        # Obtenir les modèles climatiques
        model_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not model_cols:
            return None, create_alert("danger", "Aucune colonne numérique trouvée"), [], None
        
        # Options pour dropdown
        model_options = [{"label": col, "value": col} for col in model_cols]
        
        # Message de succès
        alert = create_alert("success", 
            html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle me-2"),
                    f"✓ {len(df)} jours de précipitation chargés"
                ], className="d-flex align-items-center fw-bold"),
                html.Div([
                    html.Span("Période: ", className="fw-bold"),
                    f"{df.index[0].strftime('%Y-%m-%d')} au {df.index[-1].strftime('%Y-%m-%d')}"
                ], className="mt-2 small"),
                html.Div([
                    html.Span("Modèles: ", className="fw-bold"),
                    f"{len(model_cols)} modèles climatiques"
                ], className="mt-1 small")
            ])
        )
        
        return df.to_dict('records'), alert, model_options, model_cols[0]
        
    except Exception as e:
        return None, create_alert("danger", f"Erreur: {str(e)[:100]}"), [], None


@callback(
    Output("etp-store", "data"),
    Output("e-upload-status", "children"),
    Output("e-model-selector", "options"),
    Output("e-model-selector", "value"),
    Input("etp-upload", "contents"),
    State("etp-upload", "filename"),
    prevent_initial_call=True
)
def load_etp(contents, filename):
    """Charge les données d'évapotranspiration"""
    if not contents:
        return None, None, [], None
    
    try:
        # Lecture du fichier
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(decoded))
        else:
            return None, create_alert("danger", "Format non supporté. Utilisez .xlsx ou .csv"), [], None
        
        # Vérifier colonne date
        if 'date' not in df.columns:
            return None, create_alert("danger", "Colonne 'date' non trouvée"), [], None
        
        # Préparer les données
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Nettoyer
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        
        # Obtenir les modèles climatiques
        model_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not model_cols:
            return None, create_alert("danger", "Aucune colonne numérique trouvée"), [], None
        
        # Options pour dropdown
        model_options = [{"label": col, "value": col} for col in model_cols]
        
        # Message de succès
        alert = create_alert("success", 
            html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle me-2"),
                    f"✓ {len(df)} jours d'ETP chargés"
                ], className="d-flex align-items-center fw-bold"),
                html.Div([
                    html.Span("Période: ", className="fw-bold"),
                    f"{df.index[0].strftime('%Y-%m-%d')} au {df.index[-1].strftime('%Y-%m-%d')}"
                ], className="mt-2 small"),
                html.Div([
                    html.Span("Modèles: ", className="fw-bold"),
                    f"{len(model_cols)} modèles climatiques"
                ], className="mt-1 small")
            ])
        )
        
        return df.to_dict('records'), alert, model_options, model_cols[0]
        
    except Exception as e:
        return None, create_alert("danger", f"Erreur: {str(e)[:100]}"), [], None


@callback(
    Output("run-prediction-btn", "disabled"),
    Output("prediction-warning", "children"),
    Output("model-info-display", "children"),
    Input("precipitation-store", "data"),
    Input("etp-store", "data"),
    Input("p-model-selector", "value"),
    Input("e-model-selector", "value"),
    Input("lstm-model-selector", "value"),
    prevent_initial_call=True
)
def validate_prediction(p_data, e_data, p_model, e_model, lstm_model):
    """Valide les données et affiche les informations du modèle"""
    
    # Vérifier disponibilité du modèle LSTM
    predictor = get_predictor()
    model_available = predictor.is_model_available(lstm_model)
    
    # Informations sur le modèle
    if model_available:
        model_info = html.Div([
            html.H6("Modèle LSTM actif", className="mb-2 fw-bold"),
            dbc.Badge("✓ Prêt", color="success", className="me-2"),
            html.Span(f"{predictor.model_info[lstm_model]['name']}", className="text-muted"),
            html.Div([
                html.Small(f"Description: {predictor.model_info[lstm_model]['description']}", 
                          className="text-muted d-block mt-2")
            ])
        ], className="p-3 bg-light rounded")
    else:
        model_info = html.Div([
            html.H6("Modèle LSTM non disponible", className="mb-2 fw-bold"),
            dbc.Badge("✗ Non chargé", color="danger", className="me-2"),
            html.Div([
                html.Small(f"Fichiers attendus dans app/models/:", className="d-block mt-2"),
                html.Code(f"  - {predictor.model_info[lstm_model]['file_model']}", className="d-block"),
                html.Code(f"  - {predictor.model_info[lstm_model]['file_scaler_X']}", className="d-block"),
                html.Code(f"  - {predictor.model_info[lstm_model]['file_scaler_y']}", className="d-block"),
            ], className="mt-2 text-muted")
        ], className="p-3 bg-light rounded")
    
    # Validation des données
    if not p_data or not e_data:
        return True, create_alert("warning", 
            html.Div([
                html.I(className="fas fa-info-circle me-2"),
                "Importez les données de précipitation ET d'évapotranspiration"
            ])
        ), model_info
    
    if not p_model or not e_model:
        return True, create_alert("warning", 
            html.Div([
                html.I(className="fas fa-info-circle me-2"),
                "Sélectionnez un modèle climatique pour P et ETP"
            ])
        ), model_info
    
    if not model_available:
        return True, create_alert("danger", 
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Modèle LSTM '{lstm_model}' non disponible"
            ])
        ), model_info
    
    try:
        # Vérifier cohérence des données
        df_p = pd.DataFrame(p_data)
        df_p.index = pd.to_datetime(df_p.index)
        df_e = pd.DataFrame(e_data)
        df_e.index = pd.to_datetime(df_e.index)
        
        # Vérifier les dates
        common_dates = df_p.index.intersection(df_e.index)
        
        if len(common_dates) < 100:
            return True, create_alert("warning", 
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Seulement {len(common_dates)} dates communes. Minimum requis: 100 jours"
                ])
            ), model_info
        
        # Vérifier les modèles sélectionnés
        if p_model not in df_p.columns:
            return True, create_alert("danger", f"Modèle '{p_model}' non trouvé dans précipitation"), model_info
        
        if e_model not in df_e.columns:
            return True, create_alert("danger", f"Modèle '{e_model}' non trouvé dans ETP"), model_info
        
        # Succès
        return False, create_alert("success", 
            html.Div([
                html.I(className="fas fa-check-circle me-2"),
                f"✅ Prêt pour prédiction avec modèle {predictor.model_info[lstm_model]['name']}"
            ])
        ), model_info
        
    except Exception as e:
        return True, create_alert("danger", f"Erreur de validation: {str(e)[:100]}"), model_info


@callback(
    Output("reference-values-collapse", "is_open"),
    Input("toggle-advanced-btn", "n_clicks"),
    State("reference-values-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_advanced(n_clicks, is_open):
    """Affiche/masque les paramètres avancés"""
    return not is_open


@callback(
    Output("prediction-hydrograph", "figure"),
    Output("prediction-stats", "children"),
    Output("prediction-results-table", "children"),
    Output("prediction-detailed-stats", "children"),
    Output("prediction-results-store", "data"),
    Output("prediction-metadata-store", "data"),
    Input("run-prediction-btn", "n_clicks"),
    State("precipitation-store", "data"),
    State("etp-store", "data"),
    State("p-model-selector", "value"),
    State("e-model-selector", "value"),
    State("lstm-model-selector", "value"),
    State("analysis-type", "value"),
    State("qmax-ref", "value"),
    State("q90-ref", "value"),
    prevent_initial_call=True
)
def run_prediction(n_clicks, p_data, e_data, p_model, e_model, lstm_model, analysis_type, qmax_ref, q90_ref):
    """
    Exécute la prédiction avec le VRAI modèle LSTM
    """
    
    # Figure vide par défaut
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Cliquez sur 'Exécuter la prédiction LSTM'",
        xaxis_title="Date",
        yaxis_title="Débit (m³/s)",
        template='plotly_white',
        height=400
    )
    
    if not n_clicks:
        return empty_fig, html.Div(), html.Div(), html.Div(), None, None
    
    try:
        # 1. Charger les données
        df_p = pd.DataFrame(p_data)
        df_p.index = pd.to_datetime(df_p.index)
        df_e = pd.DataFrame(e_data)
        df_e.index = pd.to_datetime(df_e.index)
        
        # Aligner sur dates communes
        common_dates = df_p.index.intersection(df_e.index)
        df_p = df_p.loc[common_dates].sort_index()
        df_e = df_e.loc[common_dates].sort_index()
        
        # Extraire les séries
        precipitation = df_p[p_model].values
        etp = df_e[e_model].values
        dates = df_p.index
        
        # 2. Initialiser le prédicteur LSTM
        predictor = get_predictor()
        
        if not predictor.is_model_available(lstm_model):
            raise ValueError(f"Modèle LSTM '{lstm_model}' non disponible")
        
        # 3. Exécuter la prédiction avec le VRAI modèle
        df_pred = predictor.predict(
            model_type=lstm_model,
            dates=dates,
            pluie=precipitation,
            etp=etp
        )
        
        # 4. Calculer les statistiques selon le type d'analyse
        if analysis_type == 'qmax_q90':
            stats = predictor.calculate_qmax_q90(df_pred)
            
            # Valeurs de référence
            ref_qmax = qmax_ref if qmax_ref is not None else stats['Qmax_sum']
            ref_q90 = q90_ref if q90_ref is not None else stats['Q90_sum']
            
            # Calcul des changements
            qmax_change = predictor.calculate_change_percentage(stats['Qmax_sum'], ref_qmax)
            q90_change = predictor.calculate_change_percentage(stats['Q90_sum'], ref_q90)
            
            stats_display = create_qmax_q90_stats_display(stats, qmax_change, q90_change, ref_qmax, ref_q90)
            
        else:  # full statistics
            stats = predictor.calculate_flow_statistics(df_pred)
            stats_display = create_full_stats_display(stats)
        
        # 5. Créer la figure
        fig = create_prediction_figure(
            df_pred, 
            dates, 
            precipitation, 
            p_model, 
            lstm_model,
            analysis_type
        )
        
        # 6. Créer le tableau des résultats - VERSION CORRIGÉE
        results_table = create_results_table(df_pred, precipitation, etp, dates)
        
        # 7. Statistiques détaillées
        detailed_stats = create_detailed_stats(df_pred, precipitation, etp, stats, analysis_type)
        
        # 8. Préparer les données pour téléchargement
        download_data = prepare_download_data(df_pred, precipitation, etp, dates, p_model)
        
        # 9. Métadonnées
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_lstm': lstm_model,
            'model_climate': p_model,
            'analysis_type': analysis_type,
            'n_days': len(df_pred),
            'period_start': df_pred['date'].iloc[0].strftime('%Y-%m-%d') if len(df_pred) > 0 else '',
            'period_end': df_pred['date'].iloc[-1].strftime('%Y-%m-%d') if len(df_pred) > 0 else ''
        }
        
        return fig, stats_display, results_table, detailed_stats, download_data, metadata
        
    except Exception as e:
        import traceback
        print(f"Erreur prédiction: {str(e)}")
        print(traceback.format_exc())
        
        error_fig = go.Figure()
        error_fig.update_layout(
            title=f"Erreur de prédiction: {str(e)[:100]}",
            xaxis_title="Date",
            yaxis_title="Débit (m³/s)",
            template='plotly_white',
            height=400,
            annotations=[{
                'text': "Vérifiez vos données et modèles",
                'xref': "paper",
                'yref': "paper",
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 14, 'color': 'red'}
            }]
        )
        
        error_alert = create_alert("danger", f"Erreur: {str(e)[:200]}")
        return error_fig, error_alert, html.Div(), html.Div(), None, None


# ====================================================
# FONCTIONS UTILITAIRES POUR L'AFFICHAGE
# ====================================================

def create_prediction_figure(df_pred, dates, precipitation, climate_model, lstm_model, analysis_type):
    """Crée la figure de visualisation"""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Limiter à 365 jours pour lisibilité
    max_points = min(365, len(df_pred))
    
    # Précipitations (barres)
    fig.add_trace(
        go.Bar(
            x=df_pred['date'].iloc[:max_points],
            y=precipitation[:max_points],
            name="Précipitations",
            marker_color='#3498db',
            opacity=0.5,
            hovertemplate='Date: %{x}<br>Précipitation: %{y:.1f} mm<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Débit prédit LSTM
    fig.add_trace(
        go.Scatter(
            x=df_pred['date'].iloc[:max_points],
            y=df_pred['Prediction'].iloc[:max_points],
            mode='lines',
            name=f'Débit prédit LSTM',
            line=dict(color='#e74c3c', width=2.5),
            hovertemplate='Date: %{x}<br>Débit: %{y:.1f} m³/s<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Moyenne mobile 7 jours
    rolling_mean = df_pred['Prediction'].rolling(7, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=df_pred['date'].iloc[:max_points],
            y=rolling_mean.iloc[:max_points],
            mode='lines',
            name='Moyenne mobile (7j)',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Moyenne: %{y:.1f} m³/s<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Mise en page
    model_name = "Qmax/Q90" if lstm_model == 'max' else "Moyen"
    
    fig.update_layout(
        title={
            'text': f'Prédiction LSTM {model_name} - Modèle climatique: {climate_model}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        template='plotly_white',
        height=450,
        margin=dict(t=60, b=60, l=80, r=80),
        font=dict(size=12),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Débit (m³/s)", secondary_y=False)
    fig.update_yaxes(title="Précipitation (mm)", secondary_y=True)
    
    return fig


def create_qmax_q90_stats_display(stats, qmax_change, q90_change, ref_qmax, ref_q90):
    """Affiche les statistiques Qmax/Q90"""
    
    # Déterminer la couleur du changement
    qmax_color = "#e74c3c" if qmax_change > 0 else "#2ecc71"
    q90_color = "#e74c3c" if q90_change > 0 else "#2ecc71"
    
    return html.Div([
        html.H5("Analyse Qmax/Q90 - Sommes annuelles", 
               className="mb-3",
               style={"color": "#2c3e50", "fontSize": "16px"}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Qmax (somme)", className="card-title text-center"),
                        html.H3(f"{stats['Qmax_sum']:.1f}", 
                               className="card-text text-center",
                               style={"color": "#3498db"}),
                        html.P("m³/s", className="text-center text-muted small"),
                        html.Div([
                            html.Span(f"{qmax_change:+.1f}%", 
                                    className=f"badge bg-{'danger' if qmax_change > 0 else 'success'}"),
                            html.Small(f" réf: {ref_qmax:.0f}", className="ms-2 text-muted")
                        ], className="text-center mt-2")
                    ], className="p-3")
                ], className="shadow-sm h-100")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Q90 (somme)", className="card-title text-center"),
                        html.H3(f"{stats['Q90_sum']:.1f}", 
                               className="card-text text-center",
                               style={"color": "#f39c12"}),
                        html.P("m³/s", className="text-center text-muted small"),
                        html.Div([
                            html.Span(f"{q90_change:+.1f}%", 
                                    className=f"badge bg-{'danger' if q90_change > 0 else 'success'}"),
                            html.Small(f" réf: {ref_q90:.0f}", className="ms-2 text-muted")
                        ], className="text-center mt-2")
                    ], className="p-3")
                ], className="shadow-sm h-100")
            ], width=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("Années analysées: ", className="fw-bold"),
                            html.Span(f"{stats['n_years']} ans", className="text-muted ms-2")
                        ]),
                        html.Div([
                            html.Span("Qmax moyen annuel: ", className="fw-bold"),
                            html.Span(f"{stats['Qmax_mean']:.1f} m³/s", className="text-muted ms-2")
                        ], className="mt-2"),
                        html.Div([
                            html.Span("Q90 moyen annuel: ", className="fw-bold"),
                            html.Span(f"{stats['Q90_mean']:.1f} m³/s", className="text-muted ms-2")
                        ], className="mt-2")
                    ], className="p-3")
                ], className="shadow-sm")
            ], width=12)
        ])
    ])


def create_full_stats_display(stats):
    """Affiche les statistiques complètes"""
    
    return html.Div([
        html.H5("Statistiques des débits prédits", 
               className="mb-3",
               style={"color": "#2c3e50", "fontSize": "16px"}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Moyenne", className="card-title text-center"),
                        html.H3(f"{stats['mean']:.1f}", 
                               className="card-text text-center",
                               style={"color": "#3498db"}),
                        html.P("m³/s", className="text-center text-muted small")
                    ], className="p-3")
                ], className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Maximum", className="card-title text-center"),
                        html.H3(f"{stats['max']:.1f}", 
                               className="card-text text-center",
                               style={"color": "#e74c3c"}),
                        html.P("m³/s", className="text-center text-muted small")
                    ], className="p-3")
                ], className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Minimum", className="card-title text-center"),
                        html.H3(f"{stats['min']:.1f}", 
                               className="card-text text-center",
                               style={"color": "#2ecc71"}),
                        html.P("m³/s", className="text-center text-muted small")
                    ], className="p-3")
                ], className="shadow-sm")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Écart-type", className="card-title text-center"),
                        html.H3(f"{stats['std']:.1f}", 
                               className="card-text text-center",
                               style={"color": "#9b59b6"}),
                        html.P("m³/s", className="text-center text-muted small")
                    ], className="p-3")
                ], className="shadow-sm")
            ], width=3),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("Médiane: ", className="fw-bold"),
                            html.Span(f"{stats['median']:.1f} m³/s", className="text-muted ms-2")
                        ]),
                        html.Div([
                            html.Span("Q25 - Q75: ", className="fw-bold"),
                            html.Span(f"{stats['q25']:.1f} - {stats['q75']:.1f} m³/s", 
                                    className="text-muted ms-2")
                        ], className="mt-2"),
                        html.Div([
                            html.Span("Q90 (percentile 90): ", className="fw-bold"),
                            html.Span(f"{stats['q90']:.1f} m³/s", className="text-muted ms-2")
                        ], className="mt-2"),
                        html.Div([
                            html.Span("Jours analysés: ", className="fw-bold"),
                            html.Span(f"{stats['n_days']} jours", className="text-muted ms-2")
                        ], className="mt-2")
                    ], className="p-3")
                ], className="shadow-sm")
            ], width=12)
        ])
    ])


def create_results_table(df_pred, precipitation, etp, dates):
    """Crée le tableau des résultats - VERSION CORRIGÉE"""
    
    # Convertir en datetime
    dates_pred = pd.to_datetime(df_pred['date'])
    dates_original = pd.to_datetime(dates)
    
    # Créer des ensembles pour l'intersection
    dates_pred_set = set(dates_pred)
    dates_original_set = set(dates_original)
    
    # Trouver l'intersection
    common_dates = sorted(list(dates_pred_set.intersection(dates_original_set)))
    
    results_data = []
    for i, date in enumerate(common_dates[:50]):  # 50 premiers jours
        # Trouver les index
        idx_pred = dates_pred == date
        idx_orig = dates_original == date
        
        if idx_pred.any() and idx_orig.any():
            pred_value = df_pred.loc[idx_pred, 'Prediction'].iloc[0]
            precip_value = precipitation[idx_orig][0]
            etp_value = etp[idx_orig][0]
            
            results_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Précipitation (mm)': round(float(precip_value), 1),
                'ETP (mm)': round(float(etp_value), 1),
                'Débit prédit (m³/s)': round(float(pred_value), 1)
            })
    
    # Fallback si aucune date commune
    if not results_data:
        for i in range(min(50, len(df_pred))):
            results_data.append({
                'Date': pd.to_datetime(df_pred['date'].iloc[i]).strftime('%Y-%m-%d'),
                'Précipitation (mm)': round(float(precipitation[i]), 1) if i < len(precipitation) else 0,
                'ETP (mm)': round(float(etp[i]), 1) if i < len(etp) else 0,
                'Débit prédit (m³/s)': round(float(df_pred['Prediction'].iloc[i]), 1)
            })
    
    df_display = pd.DataFrame(results_data)
    
    return dash_table.DataTable(
        data=df_display.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_display.columns],
        page_size=10,
        style_table={
            'overflowX': 'auto',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontSize': '12px',
            'fontFamily': 'Segoe UI, Arial, sans-serif',
            'border': '1px solid #eaeaea'
        },
        style_header={
            'backgroundColor': '#4a6fa5',
            'color': 'white',
            'fontWeight': '600',
            'fontSize': '13px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            }
        ]
    )


def create_detailed_stats(df_pred, precipitation, etp, stats, analysis_type):
    """Crée l'affichage des statistiques détaillées selon le type d'analyse"""
    
    if analysis_type == 'qmax_q90':
        # Affichage pour Qmax/Q90 annuels
        return html.Div([
            html.H5("Analyse statistique détaillée - Qmax/Q90 annuels", 
                   className="mb-3",
                   style={"color": "#2c3e50", "fontSize": "16px"}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Statistiques Qmax", className="mb-0")),
                        dbc.CardBody([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Span("Somme sur toutes les années: ", className="fw-bold"),
                                    html.Span(f"{stats['Qmax_sum']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Moyenne annuelle: ", className="fw-bold"),
                                    html.Span(f"{stats['Qmax_mean']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Maximum annuel: ", className="fw-bold"),
                                    html.Span(f"{stats['Qmax_max']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                            ], flush=True)
                        ], className="p-3")
                    ], className="border-0 shadow-sm h-100")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Statistiques Q90", className="mb-0")),
                        dbc.CardBody([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Span("Somme sur toutes les années: ", className="fw-bold"),
                                    html.Span(f"{stats['Q90_sum']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Moyenne annuelle: ", className="fw-bold"),
                                    html.Span(f"{stats['Q90_mean']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Maximum annuel: ", className="fw-bold"),
                                    html.Span(f"{stats['Q90_max']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                            ], flush=True)
                        ], className="p-3")
                    ], className="border-0 shadow-sm h-100")
                ], md=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Données d'entrée", className="mb-0")),
                        dbc.CardBody([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Span("Précipitation totale: ", className="fw-bold"),
                                    html.Span(f"{np.sum(precipitation):.0f} mm", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Précipitation moyenne: ", className="fw-bold"),
                                    html.Span(f"{np.mean(precipitation):.1f} mm/j", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("ETP totale: ", className="fw-bold"),
                                    html.Span(f"{np.sum(etp):.0f} mm", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("ETP moyenne: ", className="fw-bold"),
                                    html.Span(f"{np.mean(etp):.1f} mm/j", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                            ], flush=True)
                        ], className="p-3")
                    ], className="border-0 shadow-sm h-100")
                ], md=12),
            ])
        ])
    
    else:  # full statistics
        predictions = df_pred['Prediction'].values
        return html.Div([
            html.H5("Analyse statistique détaillée", 
                   className="mb-3",
                   style={"color": "#2c3e50", "fontSize": "16px"}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Statistiques des débits", className="mb-0")),
                        dbc.CardBody([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Span("Somme totale: ", className="fw-bold"),
                                    html.Span(f"{stats['sum']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Moyenne: ", className="fw-bold"),
                                    html.Span(f"{stats['mean']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Écart-type: ", className="fw-bold"),
                                    html.Span(f"{stats['std']:.1f} m³/s", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Jours > Q90: ", className="fw-bold"),
                                    html.Span(f"{(predictions > stats['q90']).sum()} jours", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Jours < Q10: ", className="fw-bold"),
                                    html.Span(f"{(predictions < stats['q10']).sum()} jours", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                            ], flush=True)
                        ], className="p-3")
                    ], className="border-0 shadow-sm h-100")
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Données d'entrée", className="mb-0")),
                        dbc.CardBody([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.Span("Précipitation totale: ", className="fw-bold"),
                                    html.Span(f"{np.sum(precipitation):.0f} mm", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("Précipitation moyenne: ", className="fw-bold"),
                                    html.Span(f"{np.mean(precipitation):.1f} mm/j", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("ETP totale: ", className="fw-bold"),
                                    html.Span(f"{np.sum(etp):.0f} mm", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                                dbc.ListGroupItem([
                                    html.Span("ETP moyenne: ", className="fw-bold"),
                                    html.Span(f"{np.mean(etp):.1f} mm/j", className="text-muted ms-2")
                                ], className="border-0 py-2"),
                            ], flush=True)
                        ], className="p-3")
                    ], className="border-0 shadow-sm h-100")
                ], md=6),
            ])
        ])


def prepare_download_data(df_pred, precipitation, etp, dates, climate_model):
    """Prépare les données pour téléchargement"""
    
    # Convertir en datetime
    dates_pred = pd.to_datetime(df_pred['date'])
    dates_orig = pd.to_datetime(dates)
    
    # Créer des ensembles pour l'intersection
    dates_pred_set = set(dates_pred)
    dates_orig_set = set(dates_orig)
    
    # Trouver l'intersection
    common_dates = sorted(list(dates_pred_set.intersection(dates_orig_set)))
    
    data = []
    for date in common_dates:
        idx_pred = dates_pred == date
        idx_orig = dates_orig == date
        
        if idx_pred.any() and idx_orig.any():
            pred_value = df_pred.loc[idx_pred, 'Prediction'].iloc[0]
            precip_value = precipitation[idx_orig][0]
            etp_value = etp[idx_orig][0]
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Modèle_climatique': climate_model,
                'Précipitation_mm': round(float(precip_value), 2),
                'ETP_mm': round(float(etp_value), 2),
                'Debit_predit_m3s': round(float(pred_value), 2)
            })
    
    # Fallback
    if not data:
        for i in range(len(df_pred)):
            data.append({
                'Date': pd.to_datetime(df_pred['date'].iloc[i]).strftime('%Y-%m-%d'),
                'Modèle_climatique': climate_model,
                'Précipitation_mm': round(float(precipitation[i]), 2) if i < len(precipitation) else 0,
                'ETP_mm': round(float(etp[i]), 2) if i < len(etp) else 0,
                'Debit_predit_m3s': round(float(df_pred['Prediction'].iloc[i]), 2)
            })
    
    return data


# ====================================================
# CALLBACKS DE TÉLÉCHARGEMENT
# ====================================================

@callback(
    Output("download-csv-pred", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("prediction-results-store", "data"),
    State("p-model-selector", "value"),
    State("lstm-model-selector", "value"),
    State("analysis-type", "value"),
    prevent_initial_call=True
)
def download_csv(n_clicks, results_data, climate_model, lstm_model, analysis_type):
    """Télécharge les résultats au format CSV"""
    
    if n_clicks and results_data:
        try:
            df = pd.DataFrame(results_data)
            
            # Nom de fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = "qmax_q90" if lstm_model == 'max' else "mean"
            filename = f"lstm_prediction_{model_name}_{climate_model}_{timestamp}.csv"
            
            return dcc.send_data_frame(df.to_csv, filename, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"Erreur CSV: {e}")
    
    return no_update


@callback(
    Output("download-excel-pred", "data"),
    Input("download-excel-btn", "n_clicks"),
    State("prediction-results-store", "data"),
    State("prediction-metadata-store", "data"),
    State("p-model-selector", "value"),
    State("lstm-model-selector", "value"),
    State("analysis-type", "value"),
    prevent_initial_call=True
)
def download_excel(n_clicks, results_data, metadata, climate_model, lstm_model, analysis_type):
    """Télécharge les résultats au format Excel"""
    
    if n_clicks and results_data:
        try:
            df = pd.DataFrame(results_data)
            
            # Nom de fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = "qmax_q90" if lstm_model == 'max' else "mean"
            filename = f"lstm_prediction_{model_name}_{climate_model}_{timestamp}.xlsx"
            
            # Créer buffer Excel
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Feuille principale
                df.to_excel(writer, sheet_name='Prédictions', index=False)
                
                # Feuille métadonnées
                if metadata:
                    metadata_df = pd.DataFrame([metadata])
                    metadata_df.to_excel(writer, sheet_name='Métadonnées', index=False)
                
                # Ajuster largeur colonnes
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            output.seek(0)
            return dcc.send_bytes(output.getvalue(), filename=filename)
            
        except Exception as e:
            print(f"Erreur Excel: {e}")
            # Fallback CSV
            try:
                df = pd.DataFrame(results_data)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lstm_prediction_{model_name}_{climate_model}_{timestamp}.csv"
                return dcc.send_data_frame(df.to_csv, filename, index=False, encoding='utf-8-sig')
            except:
                pass
    
    return no_update