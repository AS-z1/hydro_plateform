"""
Page de modélisation hydrologique – Conforme au script original modelisation.txt
✅ Graphiques 2x2 (séries temporelles + scatter) identiques à matplotlib
✅ Tableau récapitulatif des métriques (console)
✅ Analyse textuelle (qualité, comparaison, etc.)
✅ Pas de fallback sur les périodes : toutes doivent être fournies
✅ Visualisation de l'évolution de l'optimisation (mode auto)
✅ Téléchargement des résultats (CSV)
"""

from dash import dcc, html, Input, Output, State, callback, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from app.services.hydrological_modeling import (
    ModHyPMA_Model,
    DataLoader,
    FeatureEngineer,
    LSTMTrainer,
    ModHyPMA_Evaluator,
    LSTM_Evaluator,
    ModHyPMAOptimizer,
    LSTMOptimizer,
    set_seed,
    Metrics,
    PYM00_AVAILABLE
)
from app.views.components.alerts import create_alert

set_seed(42)

def create_modeling_page():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(" Modélisation Hydrologique", 
                           className="mb-2",
                           style={"fontSize": "22px", "fontWeight": "600", "color": "#2c3e50"}),
                    html.P("Simulation et prédiction des débits avec ModHyPMA et LSTM",
                          className="text-muted mb-0",
                          style={"fontSize": "14px"})
                ], className="text-center")
            ])
        ], className="mb-4 pt-3",
           style={"borderBottom": "1px solid #eaeaea", "backgroundColor": "white"}),

        # Section principale
        dbc.Row([
            # Colonne gauche - Configuration
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-cogs me-2"),
                            "Configuration"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Upload de données
                        html.Div([
                            dbc.Label("Importation des données", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dcc.Upload(
                                id="upload-data-modeling",
                                children=html.Div([
                                    html.Div([
                                        html.I(className="fas fa-file-upload me-2"),
                                        "Données hydrologiques"
                                    ], className="text-center"),
                                    html.Small("CSV, Excel - Colonnes: Qobs, Pluie, ETP, date", 
                                             className="text-muted d-block mt-1")
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '80px',
                                    'lineHeight': '80px',
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
                        html.Div(id="upload-status-modeling", className="mb-4"),

                        # Sélection du modèle
                        html.Div([
                            dbc.Label("Modèle hydrologique", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dbc.RadioItems(
                                id="model-selector",
                                options=[
                                    {"label": "📊 ModHyPMA (Physique)", "value": "ModHyPMA"},
                                    {"label": "🧠 LSTM (Intelligence Artificielle)", "value": "LSTM"},
                                ],
                                value="ModHyPMA",
                                inline=False,
                                className="mb-4"
                            ),
                        ]),

                        # Mode d'optimisation
                        html.Div([
                            dbc.Label("Mode d'optimisation", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dbc.RadioItems(
                                id="optimization-mode",
                                options=[
                                    {"label": "⚡ Manuel", "value": "manuel"},
                                    {"label": "🎯 Automatique (NSGA-II)", "value": "auto"},
                                ],
                                value="manuel",
                                inline=False,
                                className="mb-4"
                            ),
                        ]),

                        # Message pymoo
                        html.Div(id="pymoo-warning", className="mb-2"),

                        # Paramètres NSGA-II (défauts dynamiques)
                        html.Div([
                            dbc.Label("Paramètres NSGA-II", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Input(
                                        id="pop-size",
                                        type="number",
                                        placeholder="Taille population",
                                        value=30,
                                        min=5,
                                        max=100,
                                        step=1,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("Taille population", className="text-muted d-block"),
                                ], md=6),
                                dbc.Col([
                                    dbc.Input(
                                        id="n-generations",
                                        type="number",
                                        placeholder="Générations",
                                        value=20,
                                        min=5,
                                        max=100,
                                        step=1,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("Nombre de générations", className="text-muted d-block"),
                                ], md=6),
                            ], className="g-2")
                        ], id="nsga2-params", className="mb-3", style={"display": "none"}),

                        # Paramètres manuels ModHyPMA
                        html.Div([
                            dbc.Label("Paramètres ModHyPMA", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Input(
                                        id="param-m",
                                        type="number",
                                        placeholder="m",
                                        value=1.1,
                                        min=0.9,
                                        max=1.45,
                                        step=0.00000001,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "8px"}
                                    ),
                                    html.Small("m (0.9-1.45)", className="text-muted d-block"),
                                ], md=3),
                                dbc.Col([
                                    dbc.Input(
                                        id="param-l",
                                        type="number",
                                        placeholder="l",
                                        value=50.0,
                                        min=26.0,
                                        max=150.0,
                                        step=1.00000001,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "8px"}
                                    ),
                                    html.Small("l (26-150)", className="text-muted d-block"),
                                ], md=3),
                                dbc.Col([
                                    dbc.Input(
                                        id="param-p2",
                                        type="number",
                                        placeholder="P2",
                                        value=3.5,
                                        min=2.2,
                                        max=10.0,
                                        step=0.00000001,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("P2 (2.2-10.0)", className="text-muted d-block"),
                                ], md=3),
                                dbc.Col([
                                    dbc.Input(
                                        id="param-tx",
                                        type="number",
                                        placeholder="TX",
                                        value=0.1,
                                        min=0.00001,
                                        max=0.8,
                                        step=0.0000001,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("TX (0.00001-0.8)", className="text-muted d-block"),
                                ], md=3),
                            ], className="g-2")
                        ], id="manual-params-modhypma", className="mb-3", style={"display": "block"}),

                        # Paramètres manuels LSTM (sans dropout)
                        html.Div([
                            dbc.Label("Hyperparamètres LSTM", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Input(
                                        id="param-epochs",
                                        type="number",
                                        placeholder="Epochs",
                                        value=20,
                                        min=5,
                                        max=50,
                                        step=1,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("Epochs (5-50)", className="text-muted d-block"),
                                ], md=2),
                                dbc.Col([
                                    dbc.Input(
                                        id="param-lr",
                                        type="number",
                                        placeholder="Learning rate",
                                        value=0.001,
                                        min=0.0001,
                                        max=0.1,
                                        step=0.00000001,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("LR (0.0001-0.1)", className="text-muted d-block"),
                                ], md=2),
                                dbc.Col([
                                    dbc.Input(
                                        id="param-batch",
                                        type="number",
                                        placeholder="Batch size",
                                        value=32,
                                        min=16,
                                        max=128,
                                        step=1,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("Batch (16-128)", className="text-muted d-block"),
                                ], md=2),
                                dbc.Col([
                                    dbc.Input(
                                        id="param-seq",
                                        type="number",
                                        placeholder="Seq length",
                                        value=10,
                                        min=7,
                                        max=30,
                                        step=1,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("Séquence (7-30)", className="text-muted d-block"),
                                ], md=2),
                                dbc.Col([
                                    dbc.Input(
                                        id="param-units",
                                        type="number",
                                        placeholder="Units",
                                        value=50,
                                        min=16,
                                        max=128,
                                        step=1,
                                        size="sm",
                                        className="mb-2",
                                        style={"fontSize": "12px"}
                                    ),
                                    html.Small("Units LSTM (16-128)", className="text-muted d-block"),
                                ], md=2),
                                # Pas de champ dropout (fixé à 0.25 dans le code)
                            ], className="g-2")
                        ], id="manual-params-lstm", className="mb-3", style={"display": "none"}),

                        # Périodes de simulation
                        html.Div([
                            dbc.Label("Périodes de simulation", 
                                     className="form-label small fw-bold text-secondary mb-3"),

                            # PÉRIODE 1 : CALAGE / ENTRAÎNEMENT
                            html.Div([
                                html.Small(id="period1-label", children="Période de calage", 
                                          className="text-muted d-block mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Input(
                                            id="train-start",
                                            placeholder="Début (YYYY-MM-DD)",
                                            size="sm",
                                            className="mb-2",
                                            style={"fontSize": "12px"}
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Input(
                                            id="train-end",
                                            placeholder="Fin (YYYY-MM-DD)",
                                            size="sm",
                                            className="mb-2",
                                            style={"fontSize": "12px"}
                                        )
                                    ], md=6),
                                ], className="g-2 mb-3"),
                            ]),

                            # PÉRIODE 2 : VALIDATION
                            html.Div([
                                html.Small(id="period2-label", children="Période de validation", 
                                          className="text-muted d-block mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Input(
                                            id="valid-start",
                                            placeholder="Début (YYYY-MM-DD)",
                                            size="sm",
                                            className="mb-2",
                                            style={"fontSize": "12px"}
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Input(
                                            id="valid-end",
                                            placeholder="Fin (YYYY-MM-DD)",
                                            size="sm",
                                            className="mb-2",
                                            style={"fontSize": "12px"}
                                        )
                                    ], md=6),
                                ], className="g-2"),
                            ], className="mb-3"),

                            # PÉRIODE 3 : TEST (LSTM SEULEMENT)
                            html.Div([
                                html.Small("Période de test", className="text-muted d-block mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Input(
                                            id="test-start",
                                            placeholder="Début (YYYY-MM-DD)",
                                            size="sm",
                                            className="mb-2",
                                            style={"fontSize": "12px"}
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Input(
                                            id="test-end",
                                            placeholder="Fin (YYYY-MM-DD)",
                                            size="sm",
                                            className="mb-2",
                                            style={"fontSize": "12px"}
                                        )
                                    ], md=6),
                                ], className="g-2"),
                            ], id="test-period-container", style={"display": "none"}),

                        ], className="mb-4"),

                        # Boutons d'exécution
                        html.Div([
                            dbc.Button(
                                [
                                    html.I(className="fas fa-play me-2"),
                                    "Lancer ModHyPMA"
                                ],
                                id="run-modhypma",
                                color="primary",
                                size="sm",
                                className="w-100 py-2 mb-2",
                                disabled=True,
                                style={"backgroundColor": "#4a6fa5", "border": "none", "borderRadius": "6px"}
                            ),

                            dbc.Button(
                                [
                                    html.I(className="fas fa-brain me-2"),
                                    "Lancer LSTM"
                                ],
                                id="run-lstm",
                                color="success",
                                size="sm",
                                className="w-100 py-2",
                                disabled=True,
                                style={"backgroundColor": "#2ecc71", "border": "none", "borderRadius": "6px"}
                            ),
                        ])
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=4, className="mb-3"),

            # Colonne droite - Visualisation
            dbc.Col([
                # Graphiques 2x2
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-line me-2"),
                            "Résultats graphiques"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        dcc.Graph(
                            id="modeling-subplots",
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '650px'}
                        )
                    ], className="p-3")
                ], className="shadow border-0 mb-3", style={"borderRadius": "10px"}),

                # Graphique d'évolution de l'optimisation (visible seulement en mode auto)
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-line me-2"),
                            "Évolution de l'optimisation (NSGA-II)"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        dcc.Graph(
                            id="optimization-history-plot",
                            config={'displayModeBar': False},
                            style={'height': '250px'}
                        )
                    ], className="p-3")
                ], id="optim-history-card", className="shadow border-0", style={"borderRadius": "10px", "display": "none"}),

                # Bouton de téléchargement des résultats
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [
                                html.I(className="fas fa-download me-2"),
                                "Télécharger les résultats (CSV)"
                            ],
                            id="btn-download-results",
                            color="success",
                            size="sm",
                            className="mt-3",
                            disabled=True
                        ),
                        dcc.Download(id="download-results-csv-mod")
                    ], width=11, className="text-end")
                ])
            ], md=8, className="mb-3"),
        ], className="mb-4"),

        # Section tableau récapitulatif (métriques)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-table me-2"),
                            "Tableau récapitulatif des performances"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div(id="modeling-summary-table")
                    ], className="p-4")
                ], className="shadow border-0", style={"borderRadius": "10px"})
            ], width=12, className="mb-4")
        ]),

        # Section analyse textuelle
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-bar me-2"),
                            "Analyse des performances"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div(id="modeling-analysis-text", className="p-3")
                    ], className="p-3")
                ], className="shadow border-0", style={"borderRadius": "10px"})
            ], width=12, className="mb-4")
        ]),

        # Section paramètres et informations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-cog me-2"),
                            "Paramètres et informations"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div(id="modeling-detailed-stats", className="p-3")
                    ], className="p-3")
                ], className="shadow border-0", style={"borderRadius": "10px"})
            ], width=12, className="mb-4")
        ]),

        # Stockage
        dcc.Store(id="data-store-modeling"),
        dcc.Store(id="model-type-store"),
        dcc.Store(id="model-params-store"),
        dcc.Store(id="lstm-trainer-store"),
        dcc.Store(id="opt-history-store"),        # pour l'historique de l'optimisation
        dcc.Store(id="results-mod-store"),            # pour les séries observées/simulées (export)
    ], fluid=False, className="py-3", style={'backgroundColor': '#f8f9fa', "marginLeft": "200px"})


# ======================================================
# CALLBACKS (configuration)
# ======================================================

@callback(
    Output("pymoo-warning", "children"),
    Input("optimization-mode", "value"),
    prevent_initial_call=False
)
def check_pymoo_availability(mode):
    if mode == "auto" and not PYM00_AVAILABLE:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "PyMOO n'est pas installé. L'optimisation NSGA-II n'est pas disponible. ",
            "Utilisez le mode manuel ou installez pymoo: pip install pymoo"
        ], color="warning", size="sm", className="mb-2")
    return None


@callback(
    Output("nsga2-params", "style"),
    Output("manual-params-modhypma", "style"),
    Output("manual-params-lstm", "style"),
    Input("optimization-mode", "value"),
    Input("model-selector", "value"),
    prevent_initial_call=False
)
def toggle_optimization_params(mode, model):
    if mode == "auto":
        nsga2_style = {"display": "block"}
        modhypma_style = {"display": "none"}
        lstm_style = {"display": "none"}
    else:
        nsga2_style = {"display": "none"}
        if model == "ModHyPMA":
            modhypma_style = {"display": "block"}
            lstm_style = {"display": "none"}
        else:
            modhypma_style = {"display": "none"}
            lstm_style = {"display": "block"}
    return nsga2_style, modhypma_style, lstm_style


@callback(
    Output("period1-label", "children"),
    Output("period2-label", "children"),
    Output("test-period-container", "style"),
    Input("model-selector", "value"),
    prevent_initial_call=False
)
def update_periods(model):
    if model == "ModHyPMA":
        return ("Période de calage", "Période de validation", {"display": "none"})
    else:
        return ("Période d'entraînement", "Période de validation", {"display": "block"})


@callback(
    Output("pop-size", "value"),
    Output("n-generations", "value"),
    Input("model-selector", "value"),
    prevent_initial_call=False
)
def update_nsga_defaults(model):
    if model == "ModHyPMA":
        return 30, 20
    else:
        return 10, 10


@callback(
    Output("data-store-modeling", "data"),
    Output("upload-status-modeling", "children"),
    Output("run-modhypma", "disabled"),
    Output("run-lstm", "disabled"),
    Input("upload-data-modeling", "contents"),
    State("upload-data-modeling", "filename"),
    prevent_initial_call=True
)
def load_modeling_data(contents, filename):
    if not contents:
        return None, None, True, True

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if filename.endswith(".csv"):
            df = pd.read_csv(
                io.StringIO(decoded.decode("utf-8")),
                sep=None,
                engine='python',
                decimal=',',
                dayfirst=True,
                na_values=['', 'NA', 'NaN']
            )
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, create_alert("danger", "Format non supporté. Utilisez .csv ou .xlsx"), True, True

        required = ['Qobs', 'Pluie', 'ETP', 'date']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return None, create_alert("danger", 
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Colonnes manquantes: {', '.join(missing)} (exactement 'date', 'Qobs', 'Pluie', 'ETP')"
                ])), True, True

        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['date'])
        df = df.set_index('date').sort_index()

        for col in ['Qobs', 'Pluie', 'ETP']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        df = df.dropna(subset=['Qobs', 'Pluie', 'ETP'])
        final_len = len(df)

        if final_len == 0:
            return None, create_alert("danger", 
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Aucune donnée valide après nettoyage. {initial_len} lignes initiales, 0 conservées."
                ])), True, True

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df_reset = df.reset_index()

        alert = create_alert("success", 
            html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle me-2"),
                    f"{final_len} lignes de données chargées et nettoyées (sur {initial_len})"
                ], className="d-flex align-items-center fw-bold"),
                html.Div([
                    html.Span("Période: ", className="fw-bold"),
                    f"{df_reset['date'].min().date()} au {df_reset['date'].max().date()}"
                ], className="mt-2")
            ])
        )

        return df_reset.to_dict('records'), alert, False, False

    except Exception as e:
        import traceback
        print(f"Erreur dans load_modeling_data: {str(e)}")
        print(traceback.format_exc())
        return None, create_alert("danger", f"Erreur: {str(e)[:100]}"), True, True


# ======================================================
# CALLBACK PRINCIPAL (exécution)
# ======================================================
@callback(
    Output("modeling-subplots", "figure"),
    Output("modeling-summary-table", "children"),
    Output("modeling-analysis-text", "children"),
    Output("modeling-detailed-stats", "children"),
    Output("model-params-store", "data"),
    Output("lstm-trainer-store", "data"),
    Output("opt-history-store", "data"),
    Output("optim-history-card", "style"),
    Output("optimization-history-plot", "figure"),
    Output("results-mod-store", "data"),
    Output("btn-download-results", "disabled"),
    Input("run-modhypma", "n_clicks"),
    Input("run-lstm", "n_clicks"),
    State("data-store-modeling", "data"),
    State("train-start", "value"),
    State("train-end", "value"),
    State("valid-start", "value"),
    State("valid-end", "value"),
    State("test-start", "value"),
    State("test-end", "value"),
    State("model-selector", "value"),
    State("optimization-mode", "value"),
    State("pop-size", "value"),
    State("n-generations", "value"),
    State("param-m", "value"),
    State("param-l", "value"),
    State("param-p2", "value"),
    State("param-tx", "value"),
    State("param-epochs", "value"),
    State("param-lr", "value"),
    State("param-batch", "value"),
    State("param-seq", "value"),
    State("param-units", "value"),
    prevent_initial_call=True
)
def run_modeling(modhypma_clicks, lstm_clicks, data, 
                train_start, train_end, valid_start, valid_end, test_start, test_end,
                model_type, opt_mode, pop_size, n_gen,
                param_m, param_l, param_p2, param_tx,
                param_epochs, param_lr, param_batch, param_seq, param_units):
    ctx = callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "run-modhypma":
        model_to_run = "ModHyPMA"
    elif button_id == "run-lstm":
        model_to_run = "LSTM"
    else:
        fig = make_subplots(rows=2, cols=2, subplot_titles=["", "", "", ""])
        fig.update_layout(title="Importez des données et lancez une simulation", height=650)
        return fig, html.Div(), html.Div(), html.Div(), None, None, None, {"display": "none"}, go.Figure(), None, True

    if not data:
        fig = make_subplots(rows=2, cols=2)
        fig.update_layout(title="Aucune donnée - importez d'abord")
        return fig, html.Div("Aucune donnée"), html.Div(), html.Div(), None, None, None, {"display": "none"}, go.Figure(), None, True

    try:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['date'])
        df = df.set_index('date').sort_index()

        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]

        if model_to_run == "ModHyPMA":
            return _run_modhypma(df, train_start, train_end, valid_start, valid_end,
                               opt_mode, pop_size, n_gen,
                               param_m, param_l, param_p2, param_tx)
        else:
            return _run_lstm(df, train_start, train_end, valid_start, valid_end, test_start, test_end,
                           opt_mode, pop_size, n_gen,
                           param_epochs, param_lr, param_batch, param_seq, param_units)

    except Exception as e:
        import traceback
        print(f"Erreur dans run_modeling: {str(e)}")
        print(traceback.format_exc())
        fig = make_subplots(rows=2, cols=2)
        fig.update_layout(title=f"Erreur: {str(e)[:50]}")
        alert = create_alert("danger", f"Erreur: {str(e)[:100]}")
        return fig, alert, html.Div(), html.Div(), None, None, None, {"display": "none"}, go.Figure(), None, True


# ======================================================
# FONCTION MODHYPMA (avec historique d'optimisation)
# ======================================================
def _run_modhypma(df, calib_start, calib_end, valid_start, valid_end,
                 opt_mode, pop_size, n_gen,
                 param_m, param_l, param_p2, param_tx):
    # Vérification des périodes
    if not calib_start or not calib_end:
        raise ValueError("Les dates de début et fin de calage sont obligatoires")
    if not valid_start or not valid_end:
        raise ValueError("Les dates de début et fin de validation sont obligatoires")

    try:
        start_ts = pd.Timestamp(calib_start)
        end_ts = pd.Timestamp(calib_end)
        df_calib = df.loc[start_ts:end_ts].copy()
        if len(df_calib) == 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            raise ValueError(
                f"Aucune donnée dans la plage de calage {start_ts.date()} - {end_ts.date()}. "
                f"Les données disponibles vont du {data_min} au {data_max}."
            )
    except Exception as e:
        raise ValueError(f"Erreur période de calage: {e}")

    try:
        start_ts = pd.Timestamp(valid_start)
        end_ts = pd.Timestamp(valid_end)
        df_valid = df.loc[start_ts:end_ts].copy()
        if len(df_valid) == 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            raise ValueError(
                f"Aucune donnée dans la plage de validation {start_ts.date()} - {end_ts.date()}. "
                f"Les données disponibles vont du {data_min} au {data_max}."
            )
    except Exception as e:
        raise ValueError(f"Erreur période de validation: {e}")

    params = {'m': 1.1, 'l': 50.0, 'P2': 3.5, 'TX': 0.1}
    history = []  # pour stocker l'évolution (si mode auto)
    opt_history_fig = go.Figure()
    show_history = False

    if opt_mode == "manuel":
        if param_m is not None:
            params['m'] = float(param_m)
        if param_l is not None:
            params['l'] = float(param_l)
        if param_p2 is not None:
            params['P2'] = float(param_p2)
        if param_tx is not None:
            params['TX'] = float(param_tx)

    elif opt_mode == "auto" and PYM00_AVAILABLE:
        pop_size = int(pop_size) if pop_size else 30
        n_gen = int(n_gen) if n_gen else 20
        optimizer = ModHyPMAOptimizer(df_calib)
        # Appel sans return_history (car non supporté)
        opt_params = optimizer.optimize(pop_size=pop_size, n_generations=n_gen)
        # Essayer de récupérer l'historique si disponible (attribut history)
        if hasattr(optimizer, 'history'):
            history = optimizer.history
        else:
            history = []
        params.update(opt_params)
        if history:
            show_history = True
            # Construction du graphique d'évolution
            generations = list(range(1, len(history)+1))
            opt_history_fig = go.Figure()
            opt_history_fig.add_trace(go.Scatter(x=generations, y=history, mode='lines+markers',
                                                  name='Meilleure fitness', line=dict(color='blue')))
            opt_history_fig.update_layout(title="Évolution de la fitness (NSE)", xaxis_title="Génération",
                                          yaxis_title="NSE", template='plotly_white', height=200)

    # Simulation calage
    Q_sim_cal = ModHyPMA_Model.simulate(
        df_calib['Pluie'].values,
        df_calib['ETP'].values,
        params['m'], params['l'], params['P2'], params['TX']
    )
    min_len_cal = min(len(df_calib), len(Q_sim_cal))
    df_calib = df_calib.iloc[:min_len_cal].copy()
    Q_sim_cal = Q_sim_cal[:min_len_cal]

    # Simulation validation
    Q_sim_val = ModHyPMA_Model.simulate(
        df_valid['Pluie'].values,
        df_valid['ETP'].values,
        params['m'], params['l'], params['P2'], params['TX']
    )
    min_len_val = min(len(df_valid), len(Q_sim_val))
    df_valid = df_valid.iloc[:min_len_val].copy()
    Q_sim_val = Q_sim_val[:min_len_val]

    # Calcul des métriques
    cal_metrics = {
        'rmse': Metrics.rmse(df_calib['Qobs'].values, Q_sim_cal),
        'r2': Metrics.r2_score(df_calib['Qobs'].values, Q_sim_cal),
        'nse': Metrics.nse(df_calib['Qobs'].values, Q_sim_cal),
        'kge': Metrics.kge(df_calib['Qobs'].values, Q_sim_cal),
        'bias': Metrics.bias(df_calib['Qobs'].values, Q_sim_cal)
    }
    val_metrics = {
        'rmse': Metrics.rmse(df_valid['Qobs'].values, Q_sim_val),
        'r2': Metrics.r2_score(df_valid['Qobs'].values, Q_sim_val),
        'nse': Metrics.nse(df_valid['Qobs'].values, Q_sim_val),
        'kge': Metrics.kge(df_valid['Qobs'].values, Q_sim_val),
        'bias': Metrics.bias(df_valid['Qobs'].values, Q_sim_val)
    }

    # Construction du graphique 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("ModHyPMA - Calage", "ModHyPMA - Validation",
                        "Diagramme de dispersion - Calage", "Diagramme de dispersion - Validation"),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # Time series calage
    fig.add_trace(go.Scatter(x=df_calib.index, y=df_calib['Qobs'].values,
                             mode='lines', name='Observé (Calage)',
                             line=dict(color='#3498db', width=1.5), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_calib.index, y=Q_sim_cal,
                             mode='lines', name='Simulé (Calage)',
                             line=dict(color='#e74c3c', width=1.5), showlegend=True), row=1, col=1)

    # Time series validation
    fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Qobs'].values,
                             mode='lines', name='Observé (Validation)',
                             line=dict(color='#2ecc71', width=1.5), showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_valid.index, y=Q_sim_val,
                             mode='lines', name='Simulé (Validation)',
                             line=dict(color='#f39c12', width=1.5), showlegend=True), row=1, col=2)

    # Scatter calage
    fig.add_trace(go.Scatter(x=df_calib['Qobs'].values, y=Q_sim_cal,
                             mode='markers', name='Calage',
                             marker=dict(color='blue', size=4, opacity=0.5),
                             showlegend=False), row=2, col=1)
    min_cal = min(df_calib['Qobs'].min(), Q_sim_cal.min())
    max_cal = max(df_calib['Qobs'].max(), Q_sim_cal.max())
    fig.add_trace(go.Scatter(x=[min_cal, max_cal], y=[min_cal, max_cal],
                             mode='lines', name='1:1',
                             line=dict(color='red', dash='dash'),
                             showlegend=False), row=2, col=1)

    # Scatter validation
    fig.add_trace(go.Scatter(x=df_valid['Qobs'].values, y=Q_sim_val,
                             mode='markers', name='Validation',
                             marker=dict(color='green', size=4, opacity=0.5),
                             showlegend=False), row=2, col=2)
    min_val = min(df_valid['Qobs'].min(), Q_sim_val.min())
    max_val = max(df_valid['Qobs'].max(), Q_sim_val.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                             mode='lines', name='1:1',
                             line=dict(color='red', dash='dash'),
                             showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Observé (m³/s)", row=2, col=1)
    fig.update_xaxes(title_text="Observé (m³/s)", row=2, col=2)
    fig.update_yaxes(title_text="Débit (m³/s)", row=1, col=1)
    fig.update_yaxes(title_text="Débit (m³/s)", row=1, col=2)
    fig.update_yaxes(title_text="Simulé (m³/s)", row=2, col=1)
    fig.update_yaxes(title_text="Simulé (m³/s)", row=2, col=2)

    fig.update_layout(height=650, template='plotly_white',
                      title_text=f"ModHyPMA - {'Mode manuel' if opt_mode=='manuel' else 'NSGA-II'}",
                      hovermode='x unified')

    # Tableau récapitulatif (métriques avec 3 décimales)
    summary_table = html.Div([
        html.H6("Récapitulatif des performances", className="mb-3"),
        html.Table(
            [html.Tr([html.Th("Période"), html.Th("RMSE"), html.Th("R²"), html.Th("NSE"), html.Th("KGE"), html.Th("Biais")])] +
            [html.Tr([html.Td("Calage"),
                      html.Td(f"{cal_metrics['rmse']:.3f}"),
                      html.Td(f"{cal_metrics['r2']:.3f}"),
                      html.Td(f"{cal_metrics['nse']:.3f}"),
                      html.Td(f"{cal_metrics['kge']:.3f}"),
                      html.Td(f"{cal_metrics['bias']:.3f}")]),
             html.Tr([html.Td("Validation"),
                      html.Td(f"{val_metrics['rmse']:.3f}"),
                      html.Td(f"{val_metrics['r2']:.3f}"),
                      html.Td(f"{val_metrics['nse']:.3f}"),
                      html.Td(f"{val_metrics['kge']:.3f}"),
                      html.Td(f"{val_metrics['bias']:.3f}")])],
            style={'width': '100%', 'textAlign': 'center', 'borderCollapse': 'collapse'},
            className="table table-bordered"
        )
    ])

    # Analyse textuelle (métriques avec 3 décimales)
    analysis = []
    analysis.append(html.H6("Analyse des performances", className="mb-3"))
    analysis.append(html.Div([
        html.P("→ Comparaison Calage vs Validation :"),
        html.P(f"  NSE Calage : {cal_metrics['nse']:.3f}"),
        html.P(f"  NSE Validation : {val_metrics['nse']:.3f}"),
        html.P("  " + ("⚠️ Différence importante: risque de surapprentissage" if abs(cal_metrics['nse'] - val_metrics['nse']) > 0.2 else
                       ("⚠️ Différence modérée" if abs(cal_metrics['nse'] - val_metrics['nse']) > 0.1 else
                        "✓ Bonne consistance"))),
        html.Br(),
        html.P("→ Qualité du modèle (Validation) :"),
        html.P("  " + ("NSE < 0.0: Performances mauvaises" if val_metrics['nse'] < 0 else
                       ("0.0 ≤ NSE < 0.5: Performances insuffisantes" if val_metrics['nse'] < 0.5 else
                        ("0.5 ≤ NSE < 0.65: Bonnes performances" if val_metrics['nse'] < 0.65 else
                         ("0.65 ≤ NSE < 0.8: Très Bonnes performances" if val_metrics['nse'] < 0.8 else
                          "NSE ≥ 0.8: Excellentes performances!")))),
    )], className="p-3", style={"backgroundColor": "#f8f9fa", "borderRadius": "6px"}))

    # Paramètres détaillés (paramètres avec 8 décimales)
    detailed = html.Div([
        html.H6("Paramètres du modèle", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("m"), html.H4(f"{params['m']:.8f}")])], className="border-0 shadow-sm"), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("l"), html.H4(f"{params['l']:.8f}")])], className="border-0 shadow-sm"), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("P2"), html.H4(f"{params['P2']:.8f}")])], className="border-0 shadow-sm"), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("TX"), html.H4(f"{params['TX']:.8f}")])], className="border-0 shadow-sm"), md=3),
        ], className="mb-3"),
        html.H6("Informations", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.Strong("Calage: "), f"{len(df_calib)} jours"]),
                    dbc.Col([html.Strong("Validation: "), f"{len(df_valid)} jours"]),
                    dbc.Col([html.Strong("Mode: "), "Manuel" if opt_mode == "manuel" else "NSGA-II"]),
                ])
            ])
        ], className="border-0 shadow-sm", style={"backgroundColor": "#f8f9fa"})
    ])

    # Préparation des données pour le téléchargement
    results_for_export = {
        'calibration': {
            'dates': df_calib.index.strftime('%Y-%m-%d').tolist(),
            'Qobs': df_calib['Qobs'].tolist(),
            'Qsim': Q_sim_cal.tolist()
        },
        'validation': {
            'dates': df_valid.index.strftime('%Y-%m-%d').tolist(),
            'Qobs': df_valid['Qobs'].tolist(),
            'Qsim': Q_sim_val.tolist()
        },
        'metrics': {
            'calibration': cal_metrics,
            'validation': val_metrics
        },
        'params': params
    }

    # Style de la carte d'historique (affichée seulement si mode auto et historique présent)
    history_card_style = {"display": "block"} if show_history else {"display": "none"}

    return (fig, summary_table, analysis, detailed, params, None,
            history, history_card_style, opt_history_fig, results_for_export, False)


# ======================================================
# FONCTION LSTM (avec historique d'optimisation)
# ======================================================
def _run_lstm(df, train_start, train_end, val_start, val_end, test_start, test_end,
             opt_mode, pop_size, n_gen,
             param_epochs, param_lr, param_batch, param_seq, param_units):
    # Vérification des périodes
    if not train_start or not train_end:
        raise ValueError("Les dates de début et fin d'entraînement sont obligatoires")
    if not val_start or not val_end:
        raise ValueError("Les dates de début et fin de validation sont obligatoires")
    if not test_start or not test_end:
        raise ValueError("Les dates de début et fin de test sont obligatoires")

    try:
        start_ts = pd.Timestamp(train_start)
        end_ts = pd.Timestamp(train_end)
        df_train = df.loc[start_ts:end_ts].copy()
        if len(df_train) == 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            raise ValueError(
                f"Aucune donnée dans la plage d'entraînement {start_ts.date()} - {end_ts.date()}. "
                f"Les données disponibles vont du {data_min} au {data_max}."
            )
    except Exception as e:
        raise ValueError(f"Erreur période entraînement: {e}")

    try:
        start_ts = pd.Timestamp(val_start)
        end_ts = pd.Timestamp(val_end)
        df_val = df.loc[start_ts:end_ts].copy()
        if len(df_val) == 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            raise ValueError(
                f"Aucune donnée dans la plage de validation {start_ts.date()} - {end_ts.date()}. "
                f"Les données disponibles vont du {data_min} au {data_max}."
            )
    except Exception as e:
        raise ValueError(f"Erreur période validation: {e}")

    try:
        start_ts = pd.Timestamp(test_start)
        end_ts = pd.Timestamp(test_end)
        df_test = df.loc[start_ts:end_ts].copy()
        if len(df_test) == 0:
            data_min = df.index.min().date()
            data_max = df.index.max().date()
            raise ValueError(
                f"Aucune donnée dans la plage de test {start_ts.date()} - {end_ts.date()}. "
                f"Les données disponibles vont du {data_min} au {data_max}."
            )
    except Exception as e:
        raise ValueError(f"Erreur période test: {e}")

    # Feature engineering
    df_train = FeatureEngineer.transform(df_train)
    df_val = FeatureEngineer.transform(df_val)
    df_test = FeatureEngineer.transform(df_test)

    features = [col for col in df_train.columns if col != 'Qobs']

    # Paramètres par défaut
    params = {
        'epochs': 20,
        'lr': 0.001,
        'batch_size': 32,
        'seq_length': 10,
        'units': 50,
    }
    history = []
    opt_history_fig = go.Figure()
    show_history = False

    if opt_mode == "manuel":
        if param_epochs is not None:
            params['epochs'] = int(float(param_epochs))
        if param_lr is not None:
            params['lr'] = float(param_lr)
        if param_batch is not None:
            params['batch_size'] = int(float(param_batch))
        if param_seq is not None:
            params['seq_length'] = int(float(param_seq))
        if param_units is not None:
            params['units'] = int(float(param_units))

    # Optimisation si auto
    elif opt_mode == "auto" and PYM00_AVAILABLE:
        pop_size = int(pop_size) if pop_size else 10
        n_gen = int(n_gen) if n_gen else 10
        lstm_trainer = LSTMTrainer(df_train, df_val, df_test, features)
        optimizer = LSTMOptimizer(lstm_trainer)
        # Appel sans return_history
        opt_params = optimizer.optimize(pop_size=pop_size, n_generations=n_gen)
        if hasattr(optimizer, 'history'):
            history = optimizer.history
        else:
            history = []
        params.update(opt_params)
        if history:
            show_history = True
            generations = list(range(1, len(history)+1))
            opt_history_fig = go.Figure()
            opt_history_fig.add_trace(go.Scatter(x=generations, y=history, mode='lines+markers',
                                                  name='Meilleure fitness', line=dict(color='blue')))
            opt_history_fig.update_layout(title="Évolution de la fitness (NSE sur validation)",
                                          xaxis_title="Génération", yaxis_title="NSE",
                                          template='plotly_white', height=200)
    else:
        # Création du trainer même en mode manuel (pour la suite)
        lstm_trainer = LSTMTrainer(df_train, df_val, df_test, features)

    # Entraînement (en mode manuel ou avec les paramètres optimisés)
    # On récupère le trainer (s'il n'a pas été créé dans la branche auto)
    if opt_mode == "manuel" or not PYM00_AVAILABLE:
        lstm_trainer = LSTMTrainer(df_train, df_val, df_test, features)

    results = lstm_trainer.train_and_eval(
        epochs=params['epochs'],
        lr=params['lr'],
        batch_size=params['batch_size'],
        seq_length=params['seq_length'],
        units=params['units'],
        verbose=0,
        evaluate_trainval=True
    )

    # Récupération des dates (après séquences)
    train_dates = df_train.index[params['seq_length']:]
    val_dates = df_val.index[params['seq_length']:]
    test_dates = df_test.index[params['seq_length']:]

    # Données pour les graphiques
    y_true_train = results['train']['y_true'].flatten()
    y_pred_train = results['train']['y_pred'].flatten()
    y_true_val = results['val']['y_true'].flatten()
    y_pred_val = results['val']['y_pred'].flatten()
    y_true_test = results['test']['y_true'].flatten()
    y_pred_test = results['test']['y_pred'].flatten()

    # Construction du graphique 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("LSTM - Entraînement", "LSTM - Test",
                        "Diagramme de dispersion - Entraînement", "Diagramme de dispersion - Test"),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # Time series train
    fig.add_trace(go.Scatter(x=train_dates, y=y_true_train,
                             mode='lines', name='Observé (Train)',
                             line=dict(color='#3498db', width=1.5), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_dates, y=y_pred_train,
                             mode='lines', name='Simulé (Train)',
                             line=dict(color='#e74c3c', width=1.5), showlegend=True), row=1, col=1)

    # Time series test
    fig.add_trace(go.Scatter(x=test_dates, y=y_true_test,
                             mode='lines', name='Observé (Test)',
                             line=dict(color='#2ecc71', width=1.5), showlegend=True), row=1, col=2)
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred_test,
                             mode='lines', name='Simulé (Test)',
                             line=dict(color='#f39c12', width=1.5), showlegend=True), row=1, col=2)

    # Scatter train
    fig.add_trace(go.Scatter(x=y_true_train, y=y_pred_train,
                             mode='markers', name='Train',
                             marker=dict(color='blue', size=4, opacity=0.5),
                             showlegend=False), row=2, col=1)
    min_train = min(y_true_train.min(), y_pred_train.min())
    max_train = max(y_true_train.max(), y_pred_train.max())
    fig.add_trace(go.Scatter(x=[min_train, max_train], y=[min_train, max_train],
                             mode='lines', name='1:1',
                             line=dict(color='red', dash='dash'),
                             showlegend=False), row=2, col=1)

    # Scatter test
    fig.add_trace(go.Scatter(x=y_true_test, y=y_pred_test,
                             mode='markers', name='Test',
                             marker=dict(color='green', size=4, opacity=0.5),
                             showlegend=False), row=2, col=2)
    min_test = min(y_true_test.min(), y_pred_test.min())
    max_test = max(y_true_test.max(), y_pred_test.max())
    fig.add_trace(go.Scatter(x=[min_test, max_test], y=[min_test, max_test],
                             mode='lines', name='1:1',
                             line=dict(color='red', dash='dash'),
                             showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Observé (m³/s)", row=2, col=1)
    fig.update_xaxes(title_text="Observé (m³/s)", row=2, col=2)
    fig.update_yaxes(title_text="Débit (m³/s)", row=1, col=1)
    fig.update_yaxes(title_text="Débit (m³/s)", row=1, col=2)
    fig.update_yaxes(title_text="Simulé (m³/s)", row=2, col=1)
    fig.update_yaxes(title_text="Simulé (m³/s)", row=2, col=2)

    fig.update_layout(height=650, template='plotly_white',
                      title_text=f"LSTM - {'Mode manuel' if opt_mode=='manuel' else 'NSGA-II'} (dropout=0.25)",
                      hovermode='x unified')

    # Tableau récapitulatif des métriques (avec 3 décimales)
    summary_table = html.Div([
        html.H6("Récapitulatif des performances", className="mb-3"),
        html.Table(
            [html.Tr([html.Th("Dataset"), html.Th("RMSE"), html.Th("R²"), html.Th("NSE"), html.Th("KGE"), html.Th("Biais")])] +
            [html.Tr([html.Td("TRAIN"),
                      html.Td(f"{results['train']['rmse']:.3f}"),
                      html.Td(f"{results['train']['r2']:.3f}"),
                      html.Td(f"{results['train']['nse']:.3f}"),
                      html.Td(f"{results['train']['kge']:.3f}"),
                      html.Td(f"{results['train']['bias']:.3f}")]),
             html.Tr([html.Td("VALIDATION"),
                      html.Td(f"{results['val']['rmse']:.3f}"),
                      html.Td(f"{results['val']['r2']:.3f}"),
                      html.Td(f"{results['val']['nse']:.3f}"),
                      html.Td(f"{results['val']['kge']:.3f}"),
                      html.Td(f"{results['val']['bias']:.3f}")]),
             html.Tr([html.Td("TEST"),
                      html.Td(f"{results['test']['rmse']:.3f}"),
                      html.Td(f"{results['test']['r2']:.3f}"),
                      html.Td(f"{results['test']['nse']:.3f}"),
                      html.Td(f"{results['test']['kge']:.3f}"),
                      html.Td(f"{results['test']['bias']:.3f}")]),
             html.Tr([html.Td("TRAIN+VAL"),
                      html.Td(f"{results['trainval']['rmse']:.3f}"),
                      html.Td(f"{results['trainval']['r2']:.3f}"),
                      html.Td(f"{results['trainval']['nse']:.3f}"),
                      html.Td(f"{results['trainval']['kge']:.3f}"),
                      html.Td(f"{results['trainval']['bias']:.3f}")])],
            style={'width': '100%', 'textAlign': 'center', 'borderCollapse': 'collapse'},
            className="table table-bordered"
        )
    ])

    # Analyse textuelle (métriques avec 3 décimales)
    test_nse = results['test']['nse']
    analysis = []
    analysis.append(html.H6("Analyse des performances", className="mb-3"))
    analysis.append(html.Div([
        html.P("→ Qualité du modèle (TEST) :"),
        html.P("  " + ("NSE < 0.0: Le modèle est moins bon que la moyenne des observations." if test_nse < 0 else
                       ("0.0 ≤ NSE < 0.5: Performances acceptables mais limitées." if test_nse < 0.5 else
                        ("0.5 ≤ NSE < 0.8: Bonnes performances." if test_nse < 0.8 else
                         "NSE ≥ 0.8: Excellentes performances!")))),
        html.Br(),
        html.P(f"→ Comparaison CALAGE (TRAIN+VAL) vs TEST :"),
        html.P(f"  NSE CALAGE : {results['trainval']['nse']:.3f}"),
        html.P(f"  NSE TEST : {test_nse:.3f}"),
        html.P("  " + ("⚠️ Différence importante: risque de surapprentissage" if abs(results['trainval']['nse'] - test_nse) > 0.2 else
                       ("⚠️ Différence modérée: le modèle généralise modérément" if abs(results['trainval']['nse'] - test_nse) > 0.1 else
                        "✓ Bonne consistance: le modèle généralise bien"))),
    ], className="p-3", style={"backgroundColor": "#f8f9fa", "borderRadius": "6px"}))

    # Paramètres détaillés (lr avec 8 décimales, autres entiers)
    detailed = html.Div([
        html.H6("Hyperparamètres LSTM", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Epochs"), html.H4(f"{params['epochs']}")])], className="border-0 shadow-sm"), md=2),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("LR"), html.H4(f"{params['lr']:.8f}")])], className="border-0 shadow-sm"), md=2),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Batch"), html.H4(f"{params['batch_size']}")])], className="border-0 shadow-sm"), md=2),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Seq"), html.H4(f"{params['seq_length']}")])], className="border-0 shadow-sm"), md=2),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Units"), html.H4(f"{params['units']}")])], className="border-0 shadow-sm"), md=2),
        ], className="mb-3"),
        html.H6("Informations", className="mb-3"),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.Strong("Entraînement: "), f"{len(df_train)} jours"]),
                    dbc.Col([html.Strong("Validation: "), f"{len(df_val)} jours"]),
                    dbc.Col([html.Strong("Test: "), f"{len(df_test)} jours"]),
                    dbc.Col([html.Strong("Mode: "), "Manuel" if opt_mode == "manuel" else "NSGA-II"]),
                ])
            ])
        ], className="border-0 shadow-sm", style={"backgroundColor": "#f8f9fa"})
    ])

    trainer_data = {
        'features': features,
        'seq_length': params['seq_length'],
        'units': params['units'],
        'lr': params['lr'],
        'batch_size': params['batch_size'],
        'epochs': params['epochs'],
    }

    # Préparation des données pour le téléchargement
    results_for_export = {
        'train': {
            'dates': train_dates.strftime('%Y-%m-%d').tolist(),
            'Qobs': y_true_train.tolist(),
            'Qsim': y_pred_train.tolist()
        },
        'validation': {
            'dates': val_dates.strftime('%Y-%m-%d').tolist(),
            'Qobs': y_true_val.tolist(),
            'Qsim': y_pred_val.tolist()
        },
        'test': {
            'dates': test_dates.strftime('%Y-%m-%d').tolist(),
            'Qobs': y_true_test.tolist(),
            'Qsim': y_pred_test.tolist()
        },
        'metrics': {
            'train': results['train'],
            'validation': results['val'],
            'test': results['test'],
            'trainval': results['trainval']
        },
        'params': params
    }

    # Style de la carte d'historique
    history_card_style = {"display": "block"} if show_history else {"display": "none"}

    return (fig, summary_table, analysis, detailed, params, trainer_data,
            history, history_card_style, opt_history_fig, results_for_export, False)


# ======================================================
# CALLBACK DE TÉLÉCHARGEMENT
# ======================================================
@callback(
    Output("download-results-csv-mod", "data"),
    Input("btn-download-results", "n_clicks"),
    State("results-mod-store", "data"),
    prevent_initial_call=True
)
def download_results(n_clicks, results_data):
    if not results_data:
        return no_update

    # Construction d'un DataFrame unique avec toutes les périodes
    all_rows = []
    for period in ['train', 'validation', 'test']:
        if period in results_data:
            p_data = results_data[period]
            if p_data and 'dates' in p_data and len(p_data['dates']) > 0:
                df_period = pd.DataFrame({
                    'date': p_data['dates'],
                    'Qobs': p_data['Qobs'],
                    'Qsim': p_data['Qsim'],
                    'period': period.upper()
                })
                all_rows.append(df_period)

    # Pour ModHyPMA, les clés sont 'calibration' et 'validation'
    for period in ['calibration', 'validation']:
        if period in results_data:
            p_data = results_data[period]
            if p_data and 'dates' in p_data and len(p_data['dates']) > 0:
                df_period = pd.DataFrame({
                    'date': p_data['dates'],
                    'Qobs': p_data['Qobs'],
                    'Qsim': p_data['Qsim'],
                    'period': period.upper()
                })
                all_rows.append(df_period)

    if not all_rows:
        return no_update

    df_export = pd.concat(all_rows, ignore_index=True)
    df_export = df_export.sort_values('date')

    # Ajout des métriques en fin de fichier (en commentaires) avec 8 décimales
    metrics_lines = ["# Métriques de performance :"]
    for period, metrics in results_data.get('metrics', {}).items():
        if isinstance(metrics, dict):
            line = f"# {period.upper()}: RMSE={metrics.get('rmse', 'N/A'):.8f}, R²={metrics.get('r2', 'N/A'):.8f}, NSE={metrics.get('nse', 'N/A'):.8f}, KGE={metrics.get('kge', 'N/A'):.8f}, Biais={metrics.get('bias', 'N/A'):.8f}"
            metrics_lines.append(line)

    # Création du contenu CSV avec les métriques en tête
    csv_string = df_export.to_csv(index=False, sep=',', decimal='.')
    csv_string = "\n".join(metrics_lines) + "\n" + csv_string

    return dict(content=csv_string, filename="resultats_modelisation.csv", type="text/csv")