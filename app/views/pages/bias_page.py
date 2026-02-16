"""
Page de correction de biais climatique
Version professionnelle et esthétique avec téléchargement des résultats
"""

from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from app.services.bias_correction import ClimateBiasCorrection, ClimateDataManager
from app.views.components.alerts import create_alert

def create_bias_page():
    """Crée la page de correction de biais avec une structure professionnelle"""
    
    return dbc.Container([
        # Header avec titre et description
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("📊 Correction de Biais Climatique", 
                           className="mb-2",
                           style={"fontSize": "22px", "fontWeight": "600", "color": "#2c3e50"}),
                    html.P("Correction des projections climatiques avec méthodes statistiques avancées",
                          className="text-muted mb-0",
                          style={"fontSize": "14px"})
                ], className="text-center")
            ])
        ], className="mb-4 pt-3",
           style={"borderBottom": "1px solid #eaeaea", "backgroundColor": "white"}),
        
        # Section principale: Import des données
        dbc.Row([
            # Colonne gauche - Données historiques
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-history me-2"),
                            "Données historiques"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Upload historique
                        html.Div([
                            dbc.Label("Importation données", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dcc.Upload(
                                id="historical-upload",
                                children=html.Div([
                                    html.Div([
                                        html.I(className="fas fa-file-upload me-2"),
                                        "Observations + Modèle"
                                    ], className="text-center"),
                                    html.Small("CSV, Excel", 
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
                        
                        # Statut de l'upload historique
                        html.Div(id="historical-upload-status", className="mb-3"),
                        
                        # Sélection des colonnes
                        html.Div([
                            dbc.Label("Observations", 
                                     className="small mb-1",
                                     style={"color": "#495057"}),
                            dcc.Dropdown(
                                id="obs-col",
                                placeholder="Sélectionnez colonne observations...",
                                className="mb-3",
                                style={"fontSize": "13px", "borderRadius": "6px"}
                            ),
                            
                            dbc.Label("Modèle historique", 
                                     className="small mb-1",
                                     style={"color": "#495057"}),
                            dcc.Dropdown(
                                id="hist-col",
                                placeholder="Sélectionnez colonne historique...",
                                style={"fontSize": "13px", "borderRadius": "6px"}
                            ),
                        ])
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=4, className="mb-3"),
            
            # Colonne centrale - Données futures
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-line me-2"),
                            "Données futures"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Upload futur
                        html.Div([
                            dbc.Label("Importation projections", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dcc.Upload(
                                id="future-upload",
                                children=html.Div([
                                    html.Div([
                                        html.I(className="fas fa-file-upload me-2"),
                                        "Projections futures"
                                    ], className="text-center"),
                                    html.Small("CSV, Excel", 
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
                        
                        # Statut de l'upload futur
                        html.Div(id="future-upload-status", className="mb-3"),
                        
                        # Sélection des colonnes
                        html.Div([
                            dbc.Label("Modèle futur", 
                                     className="small mb-1",
                                     style={"color": "#495057"}),
                            dcc.Dropdown(
                                id="fut-col",
                                placeholder="Sélectionnez colonne futur...",
                                className="mb-3",
                                style={"fontSize": "13px", "borderRadius": "6px"}
                            ),
                            
                            dbc.Label("Type de variable", 
                                     className="small mb-1",
                                     style={"color": "#495057"}),
                            dcc.Dropdown(
                                id="variable-type",
                                options=[
                                    {"label": "Température (tas)", "value": "tas"},
                                    {"label": "Précipitation (pr)", "value": "pr"}
                                ],
                                value="tas",
                                clearable=False,
                                style={"fontSize": "13px", "borderRadius": "6px"}
                            ),
                        ])
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=4, className="mb-3"),
            
            # Colonne droite - Configuration
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-sliders-h me-2"),
                            "Configuration"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Méthode de correction
                        html.Div([
                            dbc.Label("Méthode de correction", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dcc.Dropdown(
                                id="method-selector",
                                options=[
                                    {"label": "ISIMIP", "value": "ISIMIP"},
                                    {"label": "Quantile Delta Mapping", "value": "QuantileDeltaMapping"},
                                    {"label": "Linear Scaling", "value": "LinearScaling"},
                                    {"label": "Delta Change", "value": "DeltaChange"},
                                    {"label": "Scaled Distribution Mapping", "value": "ScaledDistributionMapping"},
                                ],
                                value="QuantileDeltaMapping",
                                clearable=False,
                                className="mb-4",
                                style={"fontSize": "13px", "borderRadius": "6px"}
                            ),
                        ]),
                        
                        # Informations méthode
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(id="method-info-display", 
                                        style={"fontSize": "12px"})
                            ], className="p-3")
                        ], className="mb-4", 
                           style={"backgroundColor": "#f8f9fa", "border": "1px solid #eaeaea"}),
                        
                        # Validation et bouton
                        html.Div(id="correction-warning", className="mb-3"),
                        
                        dbc.Button(
                            [
                                html.I(className="fas fa-magic me-2"),
                                "Appliquer la correction"
                            ],
                            id="apply-correction-btn",
                            color="primary",
                            size="sm",
                            className="w-100 py-2",
                            disabled=True,
                            style={"backgroundColor": "#4a6fa5", "border": "none", "borderRadius": "6px"}
                        ),
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=4, className="mb-3"),
        ], className="mb-4"),
        
        # Section visualisation
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-line me-2"),
                            "Visualisation"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div([
                            dcc.Graph(
                                id="bias-hydrograph",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                                },
                                style={'height': '400px'}
                            )
                        ], className="graph-container"),
                        html.Div(id="bias-stats", className="mt-4")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], className="mb-4")
        ]),
        
        # Section résultats détaillés avec boutons de téléchargement
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-table me-2"),
                            "Résultats détaillés"
                        ], className="d-flex align-items-center"),
                        # Boutons de téléchargement dans le header
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
                        html.Div(id="results-table", className="table-responsive")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], className="mb-4")
        ]),
        
        # Section statistiques
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-bar me-2"),
                            "Analyse statistique"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div(id="detailed-stats")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], className="mb-4")
        ]),
        
        # Stockage
        dcc.Store(id="historical-store"),
        dcc.Store(id="future-store"),
        dcc.Store(id="results-store"),
        dcc.Store(id="correction-metadata-store"),
        
        # Composants de téléchargement (invisibles)
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-excel"),
        
        # Styles CSS
        
    ], fluid=False, className="py-3", style={'backgroundColor': '#f8f9fa', "marginLeft": "240px"})


# Callbacks pour l'upload historique
@callback(
    [Output("historical-store", "data"),
     Output("obs-col", "options"),
     Output("hist-col", "options"),
     Output("historical-upload-status", "children")],
    Input("historical-upload", "contents"),
    State("historical-upload", "filename"),
    prevent_initial_call=True
)
def load_historical_data(contents, filename):
    """Charge les données historiques"""
    if not contents:
        return None, [], [], None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, [], [], create_alert("danger", "Format non supporté")
        
        # Options pour les dropdowns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        options = [{"label": col, "value": col} for col in numeric_cols]
        
        # Message de succès
        status = create_alert("success", 
            html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle me-2"),
                    f"{len(df)} lignes chargées"
                ], className="d-flex align-items-center fw-bold"),
                html.Div([
                    html.Span("Colonnes numériques: ", className="fw-bold"),
                    f"{len(numeric_cols)} détectées"
                ], className="mt-1 small")
            ])
        )
        
        return df.to_dict('records'), options, options, status
        
    except Exception as e:
        return None, [], [], create_alert("danger", f"Erreur: {str(e)[:100]}")


# Callback pour l'upload futur
@callback(
    [Output("future-store", "data"),
     Output("fut-col", "options"),
     Output("future-upload-status", "children")],
    Input("future-upload", "contents"),
    State("future-upload", "filename"),
    prevent_initial_call=True
)
def load_future_data(contents, filename):
    """Charge les données futures"""
    if not contents:
        return None, [], None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, [], create_alert("danger", "Format non supporté")
        
        # Options pour le dropdown
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        options = [{"label": col, "value": col} for col in numeric_cols]
        
        # Message de succès
        status = create_alert("success", 
            html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle me-2"),
                    f"{len(df)} lignes chargées"
                ], className="d-flex align-items-center fw-bold"),
                html.Div([
                    html.Span("Colonnes numériques: ", className="fw-bold"),
                    f"{len(numeric_cols)} détectées"
                ], className="mt-1 small")
            ])
        )
        
        return df.to_dict('records'), options, status
        
    except Exception as e:
        return None, [], create_alert("danger", f"Erreur: {str(e)[:100]}")


# Callback pour la validation
@callback(
    Output("apply-correction-btn", "disabled"),
    Output("correction-warning", "children"),
    Output("method-info-display", "children"),
    Input("historical-store", "data"),
    Input("future-store", "data"),
    Input("obs-col", "value"),
    Input("hist-col", "value"),
    Input("fut-col", "value"),
    Input("method-selector", "value"),
    Input("variable-type", "value"),
    prevent_initial_call=True
)
def validate_correction(hist_data, fut_data, obs_col, hist_col, fut_col, method, var_type):
    """Valide les paramètres de correction"""
    if not hist_data or not fut_data:
        return True, create_alert("warning", "Importez les données historiques et futures"), None
    
    missing_cols = []
    if not obs_col:
        missing_cols.append("Observations")
    if not hist_col:
        missing_cols.append("Modèle historique")
    if not fut_col:
        missing_cols.append("Modèle futur")
    
    if missing_cols:
        message = f"Sélectionnez: {', '.join(missing_cols)}"
        return True, create_alert("warning", message), None
    
    # Informations sur la méthode
    method_info = get_method_info(method, var_type)
    
    return False, create_alert("success", "✅ Prêt pour la correction"), method_info


def get_method_info(method, var_type):
    """Obtenir les informations sur la méthode"""
    method_descriptions = {
        "ISIMIP": {
            "description": "Méthode développée par l'Institut ISIMIP",
            "best_for": "Variables climatiques standard",
            "precision": "⭐️⭐️⭐️⭐️"
        },
        "QuantileDeltaMapping": {
            "description": "Mapping par quantiles avec ajustement delta",
            "best_for": "Correction distributionnelle complète",
            "precision": "⭐️⭐️⭐️⭐️⭐️"
        },
        "LinearScaling": {
            "description": "Ajustement linéaire simple",
            "best_for": "Corrections rapides et basiques",
            "precision": "⭐️⭐️⭐️"
        },
        "DeltaChange": {
            "description": "Application du changement moyen",
            "best_for": "Préservation des tendances",
            "precision": "⭐️⭐️⭐️"
        },
        "ScaledDistributionMapping": {
            "description": "Mapping distributionnel avec échelle",
            "best_for": "Précipitations, variables asymétriques",
            "precision": "⭐️⭐️⭐️⭐️"
        }
    }
    
    info = method_descriptions.get(method, {
        "description": "Méthode de correction climatique",
        "best_for": "Variables climatiques générales",
        "precision": "⭐️⭐️⭐️"
    })
    
    var_name = "Température" if var_type == "tas" else "Précipitation"
    
    return html.Div([
        html.Strong(f"{method}", className="d-block mb-2"),
        html.P(info["description"], className="small mb-2"),
        html.Div([
            html.Span("Précision: ", className="fw-bold"),
            html.Span(info["precision"])
        ], className="small mb-1"),
        html.Div([
            html.Span("Variable: ", className="fw-bold"),
            html.Span(var_name)
        ], className="small")
    ])


# Callback principal pour la correction
@callback(
    Output("bias-hydrograph", "figure"),
    Output("bias-stats", "children"),
    Output("results-table", "children"),
    Output("detailed-stats", "children"),
    Output("results-store", "data"),
    Input("apply-correction-btn", "n_clicks"),
    State("historical-store", "data"),
    State("future-store", "data"),
    State("obs-col", "value"),
    State("hist-col", "value"),
    State("fut-col", "value"),
    State("method-selector", "value"),
    State("variable-type", "value"),
    prevent_initial_call=True
)
def apply_correction(n_clicks, hist_data, fut_data, obs_col, hist_col, fut_col, method, var_type):
    """Applique la correction de biais"""
    if not n_clicks:
        fig_vide = go.Figure()
        fig_vide.update_layout(
            title="Appliquer la correction pour voir les résultats",
            xaxis_title="Index",
            yaxis_title="Valeur",
            template='plotly_white',
            height=380,
            plot_bgcolor='rgba(240, 240, 240, 0.1)'
        )
        return fig_vide, html.Div(), html.Div(), html.Div(), None
    
    try:
        df_hist = pd.DataFrame(hist_data)
        df_fut = pd.DataFrame(fut_data)
        
        # Préparer données
        data_manager = ClimateDataManager()
        data_manager.load_dataframes(df_hist, df_fut)
        data = data_manager.prepare_data(obs_col, hist_col, fut_col)
        
        # Appliquer correction
        corrector = ClimateBiasCorrection()
        result = corrector.apply_correction(data, method, var_type)
        
        if not result:
            alert = create_alert("danger", "Échec de la correction")
            return go.Figure(), alert, html.Div(), html.Div(), None
        
        # Créer hydrogramme amélioré
        fig = go.Figure()
        
        # Original
        fig.add_trace(go.Scatter(
            x=list(range(len(result['original_data'][:100]))),
            y=result['original_data'][:100],
            mode='lines',
            name='Original',
            line=dict(color='#3498db', width=2),
            opacity=0.7,
            hovertemplate='Index: %{x}<br>Valeur: %{y:.2f}<extra>Original</extra>'
        ))
        
        # Corrigé
        fig.add_trace(go.Scatter(
            x=list(range(len(result['corrected_data'][:100]))),
            y=result['corrected_data'][:100],
            mode='lines',
            name='Corrigé',
            line=dict(color='#2ecc71', width=2.5),
            hovertemplate='Index: %{x}<br>Valeur: %{y:.2f}<extra>Corrigé</extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f"Correction - {method}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': '#2c3e50'}
            },
            xaxis_title="Index",
            yaxis_title="Valeur",
            template='plotly_white',
            height=400,
            margin=dict(t=60, b=60, l=80, r=40),
            font=dict(size=12),
            hovermode="x unified",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Statistiques principales
        stats = result['stats']
        stats_display = html.Div([
            html.H5("Statistiques de correction", 
                   className="mb-3",
                   style={"color": "#2c3e50", "fontSize": "16px"}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Changement moyen", className="card-title text-center"),
                            html.H4(f"{stats['mean_change']:+.2f}", 
                                   className="card-text text-center",
                                   style={"color": "#e74c3c"})
                        ], className="p-2 text-center")
                    ], className="shadow-sm")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Moyenne originale", className="card-title text-center"),
                            html.H4(f"{stats['original_mean']:.2f}", 
                                   className="card-text text-center",
                                   style={"color": "#3498db"})
                        ], className="p-2 text-center")
                    ], className="shadow-sm")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Moyenne corrigée", className="card-title text-center"),
                            html.H4(f"{stats['corrected_mean']:.2f}", 
                                   className="card-text text-center",
                                   style={"color": "#2ecc71"})
                        ], className="p-2 text-center")
                    ], className="shadow-sm")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Réduction biais", className="card-title text-center"),
                            html.H4(f"{abs(stats['mean_change']/stats['original_mean']*100):.1f}%", 
                                   className="card-text text-center",
                                   style={"color": "#9b59b6"})
                        ], className="p-2 text-center")
                    ], className="shadow-sm")
                ], width=3),
            ], className="g-2")
        ])
        
        # Tableau des résultats
        results_df = pd.DataFrame({
            'Index': range(min(50, len(result['corrected_data']))),
            'Original': result['original_data'][:50],
            'Corrigé': result['corrected_data'][:50],
            'Différence': result['corrected_data'][:50] - result['original_data'][:50]
        })
        
        # Arrondir les valeurs pour l'affichage
        results_df['Original'] = results_df['Original'].round(2)
        results_df['Corrigé'] = results_df['Corrigé'].round(2)
        results_df['Différence'] = results_df['Différence'].round(2)
        
        results_table = dash_table.DataTable(
            data=results_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in results_df.columns],
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
                },
                {
                    'if': {'column_id': 'Différence'},
                    'fontWeight': '600',
                    'color': '#e74c3c'
                }
            ]
        )
        
        # Statistiques détaillées
        detailed_stats = html.Div([
            html.H5("Analyse statistique détaillée", 
                   className="mb-3",
                   style={"color": "#2c3e50", "fontSize": "16px"}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Variabilité", className="card-title text-center mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("Écart-type original: ", className="fw-bold"),
                                    html.Span(f"{stats.get('original_std', np.std(result['original_data'][:100])):.2f}")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Écart-type corrigé: ", className="fw-bold"),
                                    html.Span(f"{stats.get('corrected_std', np.std(result['corrected_data'][:100])):.2f}")
                                ])
                            ])
                        ], className="p-3")
                    ], className="border-0 shadow-sm")
                ], md=4, className="mb-3"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Distribution", className="card-title text-center mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("Médiane originale: ", className="fw-bold"),
                                    html.Span(f"{np.median(result['original_data'][:100]):.2f}")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Médiane corrigée: ", className="fw-bold"),
                                    html.Span(f"{np.median(result['corrected_data'][:100]):.2f}")
                                ])
                            ])
                        ], className="p-3")
                    ], className="border-0 shadow-sm")
                ], md=4, className="mb-3"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Extrêmes", className="card-title text-center mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("Max original: ", className="fw-bold"),
                                    html.Span(f"{np.max(result['original_data'][:100]):.2f}")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Max corrigé: ", className="fw-bold"),
                                    html.Span(f"{np.max(result['corrected_data'][:100]):.2f}")
                                ])
                            ])
                        ], className="p-3")
                    ], className="border-0 shadow-sm")
                ], md=4, className="mb-3"),
            ]),
            
            # Informations de calcul
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Small("Méthode", className="text-muted d-block"),
                                html.Strong(method)
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Small("Variable", className="text-muted d-block"),
                                html.Strong("Température" if var_type == "tas" else "Précipitation")
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Small("Échantillon", className="text-muted d-block"),
                                html.Strong(f"{len(result['corrected_data'])} valeurs")
                            ])
                        ], md=4),
                    ], className="align-items-center")
                ], className="p-3")
            ], className="border-0 shadow-sm",
               style={"backgroundColor": "#f8f9fa"})
        ])
        
        # Préparer les données pour le téléchargement
        download_df = results_df.copy()
        
        return fig, stats_display, results_table, detailed_stats, download_df.to_dict('records')
        
    except Exception as e:
        import traceback
        print(f"Erreur dans apply_correction: {str(e)}")
        print(traceback.format_exc())
        alert = create_alert("danger", f"Erreur: {str(e)[:100]}")
        return go.Figure(), alert, html.Div(), html.Div(), None


# Callback pour le téléchargement CSV
@callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("results-store", "data"),
    State("method-selector", "value"),
    State("variable-type", "value"),
    prevent_initial_call=True
)
def download_csv(n_clicks, results_data, method, var_type):
    """Télécharge les résultats au format CSV"""
    if n_clicks and results_data:
        try:
            # Convertir les données en DataFrame
            df = pd.DataFrame(results_data)
            
            # Créer un nom de fichier avec la date et la méthode
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            var_name = "temperature" if var_type == "tas" else "precipitation"
            filename = f"bias_correction_{method}_{var_name}_{date_str}.csv"
            
            # Convertir en CSV
            csv_string = df.to_csv(index=False, encoding='utf-8-sig')
            
            return dict(content=csv_string, filename=filename)
            
        except Exception as e:
            print(f"Erreur lors de la création du CSV: {e}")
            return None
    return None


# Callback pour le téléchargement Excel
@callback(
    Output("download-excel", "data"),
    Input("download-excel-btn", "n_clicks"),
    State("results-store", "data"),
    State("method-selector", "value"),
    State("variable-type", "value"),
    prevent_initial_call=True
)
def download_excel(n_clicks, results_data, method, var_type):
    """Télécharge les résultats au format Excel"""
    if n_clicks and results_data:
        try:
            # Convertir les données en DataFrame
            df = pd.DataFrame(results_data)
            
            # Créer un nom de fichier avec la date et la méthode
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            var_name = "temperature" if var_type == "tas" else "precipitation"
            filename = f"bias_correction_{method}_{var_name}_{date_str}.xlsx"
            
            # Créer un buffer pour le fichier Excel
            output = io.BytesIO()
            
            # Écrire directement le DataFrame dans Excel sans formatage complexe
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Correction_Biais')
            
            output.seek(0)
            
            return dcc.send_bytes(output.getvalue(), filename=filename)
            
        except Exception as e:
            print(f"Erreur lors de la création du fichier Excel: {e}")
            # Fallback: créer un CSV si Excel échoue
            try:
                from datetime import datetime
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"bias_correction_{method}_{var_name}_{date_str}.csv"
                csv_string = df.to_csv(index=False, encoding='utf-8-sig')
                print("Utilisation du format CSV en fallback")
                return dict(content=csv_string, filename=csv_filename)
            except:
                return None
    return None