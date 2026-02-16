from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from app.services.eto_calculator import EToData, EToCalculator, EToDataManager
from app.views.components.alerts import create_alert

def create_etp_page():
    """Crée la page ETP avec une structure professionnelle et esthétique"""
    
    return dbc.Container([
        # Header avec titre et description
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("☀️ Calcul d'Évapotranspiration (ETo)", 
                           className="mb-2",
                           style={"fontSize": "22px", "fontWeight": "600", "color": "#2c3e50"}),
                    html.P("Importez vos données météorologiques et calculez l'évapotranspiration de référence",
                          className="text-muted mb-0",
                          style={"fontSize": "14px"})
                ], className="text-center")
            ])
        ], className="mb-4 pt-3",
           style={"borderBottom": "1px solid #eaeaea", "backgroundColor": "white"}),
        
        # Section principale: Paramètres + Graphique
        dbc.Row([
            # Colonne gauche - Configuration
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-sliders-h me-2"),
                            "Configuration"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        # Upload de données
                        html.Div([
                            dbc.Label("Importation des données", 
                                     className="form-label small fw-bold text-secondary mb-2"),
                            dcc.Upload(
                                id="etp-data-upload",
                                children=html.Div([
                                    html.Div([
                                        html.I(className="fas fa-file-upload me-2"),
                                        "Cliquez ou glissez-déposez"
                                    ], className="text-center"),
                                    html.Small("Formats supportés: CSV, Excel, TXT", 
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
                                },
                                multiple=False
                            ),
                        ], className="mb-4"),
                        
                        # Statut de l'upload
                        html.Div(id="etp-upload-status", className="mb-4"),
                        
                        # Paramètres de calcul
                        html.Div([
                            dbc.Label("Paramètres de calcul", 
                                     className="form-label small fw-bold text-secondary mb-3"),
                            
                            # Latitude
                            html.Div([
                                dbc.Label("Latitude (°)", 
                                        className="small mb-1",
                                        style={"color": "#495057"}),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="latitude-input",
                                        type="number",
                                        placeholder="ex: 48.86",
                                        min=-90,
                                        max=90,
                                        step=0.01,
                                        value=48.86,
                                        size="sm",
                                        className="border-end-0"
                                    ),
                                    dbc.InputGroupText("°", className="bg-light")
                                ], size="sm")
                            ], className="mb-3"),
                            
                            # Méthode
                            html.Div([
                                dbc.Label("📚 Méthode de calcul", 
                                        className="small mb-1",
                                        style={"color": "#495057"}),
                                dcc.Dropdown(
                                    id="method-selector",
                                    options=[
                                        {"label": "FAO-56 ", "value": "FAO-56"},
                                        {"label": "Penman", "value": "Penman"},
                                        {"label": "Penman-Monteith", "value": "Penman-Monteith"},
                                        {"label": "Hargreaves", "value": "Hargreaves"},
                                        {"label": "Oudin", "value": "Oudin"},
                                        {"label": "Turc", "value": "Turc"},
                                        {"label": "Hamon", "value": "Hamon"},
                                    ],
                                    value="FAO-56",
                                    clearable=False,
                                    className="mb-4",
                                    style={"fontSize": "13px", "borderRadius": "6px"}
                                ),
                            ]),
                            
                            # Bouton calcul
                            dbc.Button(
                                [
                                    html.I(className="fas fa-calculator me-2"),
                                    "Calculer l'ETo"
                                ],
                                id="etp-calculate-btn",
                                color="primary",
                                size="sm",
                                className="w-100 py-2",
                                disabled=True,
                                style={"backgroundColor": "#4a6fa5", "border": "none", "borderRadius": "6px"}
                            ),
                        ])
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=5, className="mb-1"),
            
            # Colonne droite - Visualisation
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-line me-2"),
                            "Visualisation  👁️"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div([
                            dcc.Graph(
                                id="eto-hydrograph",
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                                },
                                style={'height': '400px'}
                            )
                        ], className="graph-container"),
                        html.Div(id="eto-stats", className="mt-4")
                    ], className="p-4")
                ], className="shadow border-0 h-100",
                   style={"borderRadius": "10px"})
            ], md=7, className="mb-3"),
        ], className="mb-4"),
        
        # Section résultats - Tableau avec bouton de téléchargement
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-table me-2"),
                            "Résultats détaillés"
                        ], className="d-flex align-items-center"),
                        # Bouton de téléchargement dans le header
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
                        html.Div(id="eto-results-table", className="table-responsive")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], width=12, className="mb-4")
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
                        html.Div(id="eto-detailed-stats")
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ], width=12, className="mb-4")
        ]),
        
        # Stockage
        dcc.Store(id="etp-data-store"),
        dcc.Store(id="eto-results-store"),
        
        # Composants de téléchargement (invisibles)
        dcc.Download(id="download-csv-etp"),
        dcc.Download(id="download-excel-etp"),
        
        # Styles CSS supplémentaires
        
    ], fluid=False, className="py-3", style={'backgroundColor': '#f8f9fa', "marginLeft": "200px" })


# Callbacks (inchangés - seul le layout a été modifié)
@callback(
    Output("etp-data-store", "data"),
    Output("etp-upload-status", "children"),
    Output("etp-calculate-btn", "disabled"),
    Input("etp-data-upload", "contents"),
    State("etp-data-upload", "filename"),
    prevent_initial_call=True
)
def charger_donnees_etp(contenu, nom_fichier):
    """Charge les données ETP depuis le fichier uploadé"""
    if not contenu:
        return None, None, True
    
    try:
        content_type, content_string = contenu.split(',')
        decoded = base64.b64decode(content_string)
        
        # Vérifier le format du fichier
        if nom_fichier and nom_fichier.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif nom_fichier and nom_fichier.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(decoded))
        elif nom_fichier and nom_fichier.endswith(".txt"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter='\t')
        else:
            return None, create_alert("danger", "Format non supporté. Utilisez .csv, .xlsx ou .txt"), True
        
        # Détecter la colonne date
        colonnes_date = [col for col in df.columns if 'date' in col.lower() or 'jour' in col.lower() or 'time' in col.lower()]
        if not colonnes_date:
            return None, create_alert("danger", "Colonne date non trouvée. Colonnes disponibles: " + ", ".join(df.columns)), True
        
        colonne_date = colonnes_date[0]
        df[colonne_date] = pd.to_datetime(df[colonne_date], errors='coerce')
        df = df.dropna(subset=[colonne_date])
        df = df.set_index(colonne_date).sort_index()
        
        # Vérifier que le DataFrame n'est pas vide
        if df.empty:
            return None, create_alert("danger", "Aucune donnée valide après traitement"), True
        
        # Vérifier les colonnes nécessaires pour les différentes méthodes
        colonnes_presentes = df.columns.tolist()
        message_colonnes = f"Colonnes détectées: {', '.join(colonnes_presentes[:5])}" + ("..." if len(colonnes_presentes) > 5 else "")
            
        message = create_alert("success", 
            html.Div([
                html.Div([
                    html.I(className="fas fa-check-circle me-2"),
                    f"{len(df)} lignes chargées avec succès"
                ], className="d-flex align-items-center fw-bold"),
                html.Div([
                    html.Span("Période: ", className="fw-bold"),
                    f"{df.index[0].date()} au {df.index[-1].date()}"
                ], className="mt-2"),
                html.Div([
                    html.Span("Données: ", className="fw-bold"),
                    message_colonnes
                ], className="mt-1")
            ])
        )
        
        # Préparer les données pour le stockage JSON
        df_reinitialise = df.reset_index()
        df_reinitialise[colonne_date] = df_reinitialise[colonne_date].astype(str)
        
        return df_reinitialise.to_dict('records'), message, False
        
    except Exception as e:
        import traceback
        print(f"Erreur dans charger_donnees_etp: {str(e)}")
        print(traceback.format_exc())
        return None, create_alert("danger", f"Erreur: {str(e)[:100]}"), True


@callback(
    Output("eto-hydrograph", "figure"),
    Output("eto-stats", "children"),
    Output("eto-results-table", "children"),
    Output("eto-detailed-stats", "children"),
    Output("eto-results-store", "data"),
    Input("etp-calculate-btn", "n_clicks"),
    State("etp-data-store", "data"),
    State("latitude-input", "value"),
    State("method-selector", "value"),
    prevent_initial_call=True
)
def calculer_etp(n_clics, donnees, latitude, methode):
    """Calcule l'ETo et affiche les résultats"""
    if not n_clics or not donnees:
        # Retourner des figures et div vides
        fig_vide = go.Figure()
        fig_vide.update_layout(
            title="Cliquez sur 'Calculer l'ETo' après avoir chargé des données",
            xaxis_title="Date",
            yaxis_title="ETo (mm/jour)",
            template='plotly_white',
            height=380,
            plot_bgcolor='rgba(240, 240, 240, 0.1)'
        )
        fig_vide.add_annotation(
            text="Graphique ETo",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig_vide, html.Div(), html.Div(), html.Div(), None
    
    try:
        # Convertir les données en DataFrame
        df = pd.DataFrame(donnees)
        
        # Trouver la colonne date
        colonnes_date = [col for col in df.columns if 'date' in col.lower() or 'jour' in col.lower()]
        if not colonnes_date:
            return go.Figure(), create_alert("danger", "Colonne date non trouvée"), html.Div(), html.Div(), None
        
        colonne_date = colonnes_date[0]
        df[colonne_date] = pd.to_datetime(df[colonne_date])
        df = df.set_index(colonne_date).sort_index()
        
        # Préparer les données
        gestionnaire_donnees = EToDataManager()
        donnees_eto = gestionnaire_donnees.prepare_etodata(df, latitude)
        
        if not donnees_eto.validate(methode):
            alerte = create_alert("warning", 
                f"Méthode {methode} nécessite des données spécifiques. Vérifiez votre fichier.")
            return go.Figure(), alerte, html.Div(), html.Div(), None
        
        # Calculer
        calculateur = EToCalculator(donnees_eto)
        serie_eto = calculateur.calculate(methode)
        resultats = calculateur.get_results(methode)
        
        # Créer l'hydrogramme
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=serie_eto.values,
            mode='lines+markers',
            name=f'ETo {methode}',
            line=dict(color='#4a6fa5', width=2.5),
            marker=dict(size=5, color='#4a6fa5', opacity=0.7),
            fill='tozeroy',
            fillcolor='rgba(74, 111, 165, 0.1)',
            hovertemplate='<b>Date:</b> %{x}<br><b>ETo:</b> %{y:.2f} mm/jour<extra></extra>'
        ))
        
        # Ajouter une ligne de moyenne
        moyenne = serie_eto.mean()
        fig.add_hline(
            y=moyenne,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Moyenne: {moyenne:.2f} mm/jour",
            annotation_position="bottom right",
            annotation_font=dict(color="#e74c3c", size=12)
        )
        
        fig.update_layout(
            title={
                'text': f'Évapotranspiration (ETo) - Méthode {methode}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial'}
            },
            xaxis_title="Date",
            yaxis_title="ETo (mm/jour)",
            template='plotly_white',
            height=400,
            margin=dict(t=60, b=60, l=80, r=40),
            font=dict(family="Arial", size=12),
            hovermode="x unified",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.5)',
                showline=True,
                linecolor='rgba(200, 200, 200, 0.5)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.5)',
                showline=True,
                linecolor='rgba(200, 200, 200, 0.5)'
            )
        )
        
        # Statistiques simplifiées avec meilleur design
        stats = html.Div([
            html.H5("Résumé statistique", 
                   className="mb-4",
                   style={"color": "#2c3e50", "fontSize": "16px"}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-calculator fa-lg", 
                                      style={"color": "#3498db"}),
                            ], className="text-center mb-3"),
                            html.H4(f"{resultats['mean']:.2f}", 
                                   className="card-text text-center fw-bold",
                                   style={"color": "#2c3e50"}),
                            html.P("Moyenne (mm/j)", 
                                  className="small text-center text-muted mb-0")
                        ], className="p-3 text-center")
                    ], className="border-0 shadow-sm stat-card")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-arrow-up fa-lg", 
                                      style={"color": "#e74c3c"}),
                            ], className="text-center mb-3"),
                            html.H4(f"{resultats['max']:.2f}", 
                                   className="card-text text-center fw-bold",
                                   style={"color": "#2c3e50"}),
                            html.P("Maximum (mm/j)", 
                                  className="small text-center text-muted mb-0")
                        ], className="p-3 text-center")
                    ], className="border-0 shadow-sm stat-card")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-arrow-down fa-lg", 
                                      style={"color": "#2ecc71"}),
                            ], className="text-center mb-3"),
                            html.H4(f"{resultats['min']:.2f}", 
                                   className="card-text text-center fw-bold",
                                   style={"color": "#2c3e50"}),
                            html.P("Minimum (mm/j)", 
                                  className="small text-center text-muted mb-0")
                        ], className="p-3 text-center")
                    ], className="border-0 shadow-sm stat-card")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-chart-bar fa-lg", 
                                      style={"color": "#9b59b6"}),
                            ], className="text-center mb-3"),
                            html.H4(f"{resultats['sum']:.0f}", 
                                   className="card-text text-center fw-bold",
                                   style={"color": "#2c3e50"}),
                            html.P("Total (mm)", 
                                  className="small text-center text-muted mb-0")
                        ], className="p-3 text-center")
                    ], className="border-0 shadow-sm stat-card")
                ], width=3),
            ], className="g-4")
        ])
        
        # Tableau des résultats amélioré
        df_resultats = pd.DataFrame({
            'Date': df.index.strftime('%Y-%m-%d'),
            f'ETo {methode} (mm/jour)': serie_eto.values.round(2),
            'Cumul (mm)': serie_eto.cumsum().values.round(1)
        })
        
        tableau = dash_table.DataTable(
            data=df_resultats.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_resultats.columns],
            page_size=15,
            style_table={
                'overflowX': 'auto',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
            style_cell={
                'textAlign': 'center',
                'padding': '12px',
                'fontSize': '13px',
                'fontFamily': 'Segoe UI, Arial, sans-serif',
                'border': '1px solid #eaeaea'
            },
            style_header={
                'backgroundColor': '#4a6fa5',
                'color': 'white',
                'fontWeight': '600',
                'fontSize': '14px',
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                },
                {
                    'if': {'column_id': f'ETo {methode} (mm/jour)'},
                    'fontWeight': '600',
                    'color': '#2c3e50'
                },
                {
                    'if': {'column_id': 'Cumul (mm)'},
                    'color': '#4a6fa5',
                    'fontWeight': '500'
                }
            ],
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            page_current=0,
            page_action='native'
        )
        
        # Statistiques détaillées avec meilleur design
        detailed_stats = html.Div([
            html.H5("Analyse statistique détaillée", 
                   className="mb-4",
                   style={"color": "#2c3e50", "fontSize": "16px"}),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Variabilité", className="card-title text-center mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("Écart-type: ", className="fw-bold"),
                                    html.Span(f"{serie_eto.std():.2f} mm/j")
                                ]),
                                html.Div([
                                    html.Span("Coefficient variation: ", className="fw-bold"),
                                    html.Span(f"{(serie_eto.std()/serie_eto.mean()*100):.1f}%")
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
                                    html.Span("Médiane: ", className="fw-bold"),
                                    html.Span(f"{serie_eto.median():.2f} mm/j")
                                ]),
                                html.Div([
                                    html.Span("Q1-Q3: ", className="fw-bold"),
                                    html.Span(f"{np.percentile(serie_eto, 25):.2f} - {np.percentile(serie_eto, 75):.2f} mm/j")
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
                                    html.Span("Jours > 5mm: ", className="fw-bold"),
                                    html.Span(f"{(serie_eto > 5).sum()} jours")
                                ]),
                                html.Div([
                                    html.Span("Jours < 1mm: ", className="fw-bold"),
                                    html.Span(f"{(serie_eto < 1).sum()} jours")
                                ])
                            ])
                        ], className="p-3")
                    ], className="border-0 shadow-sm")
                ], md=4, className="mb-3"),
            ], className="mb-4"),
            
            # Informations de calcul
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Small("Période analysée", className="text-muted d-block"),
                                html.Strong(f"{len(df)} jours"),
                                html.Div([
                                    html.Small(f"{df.index[0].date()} au {df.index[-1].date()}")
                                ], className="text-muted")
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Small("Latitude", className="text-muted d-block"),
                                html.Strong(f"{latitude}°")
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Small("Méthode", className="text-muted d-block"),
                                html.Strong(methode)
                            ])
                        ], md=4),
                    ], className="align-items-center")
                ], className="p-3")
            ], className="border-0 shadow-sm",
               style={"backgroundColor": "#f8f9fa"})
        ])
        
        # Stocker les résultats pour le téléchargement
        results_data = df_resultats.to_dict('records')
        
        return fig, stats, tableau, detailed_stats, results_data
        
    except Exception as e:
        import traceback
        print(f"Erreur dans calculer_etp: {str(e)}")
        print(traceback.format_exc())
        alerte = create_alert("danger", 
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Erreur de calcul avec la méthode {methode}: {str(e)[:150]}"
            ])
        )
        return go.Figure(), alerte, html.Div(), html.Div(), None


# Callback pour le téléchargement CSV
@callback(
    Output("download-csv-etp", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("eto-results-store", "data"),
    State("method-selector", "value"),
    prevent_initial_call=True
)
def download_csv(n_clicks, results_data, methode):
    """Télécharge les résultats au format CSV"""
    if n_clicks and results_data:
        try:
            # Convertir les données en DataFrame
            df = pd.DataFrame(results_data)
            
            # Créer un nom de fichier avec la date et la méthode
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eto_results_{methode}_{date_str}.csv"
            
            # Convertir en CSV
            csv_string = df.to_csv(index=False, encoding='utf-8-sig')
            
            return dict(content=csv_string, filename=filename)
        except Exception as e:
            print(f"Erreur lors de la création du CSV: {e}")
            return None
    return None


# Callback pour le téléchargement Excel (version simplifiée)
@callback(
    Output("download-excel-etp", "data"),
    Input("download-excel-btn", "n_clicks"),
    State("eto-results-store", "data"),
    State("method-selector", "value"),
    prevent_initial_call=True
)
def download_excel(n_clicks, results_data, methode):
    """Télécharge les résultats au format Excel (version simplifiée)"""
    if n_clicks and results_data:
        try:
            # Convertir les données en DataFrame
            df = pd.DataFrame(results_data)
            
            # Créer un nom de fichier avec la date et la méthode
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eto_results_{methode}_{date_str}.xlsx"
            
            # Créer un buffer pour le fichier Excel
            output = io.BytesIO()
            
            # Écrire directement le DataFrame dans Excel sans formatage complexe
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='ETo_Results')
            
            output.seek(0)
            
            return dcc.send_bytes(output.getvalue(), filename=filename)
            
        except Exception as e:
            print(f"Erreur lors de la création du fichier Excel: {e}")
            # Fallback: créer un CSV si Excel échoue
            try:
                from datetime import datetime
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"eto_results_{methode}_{date_str}.csv"
                csv_string = df.to_csv(index=False, encoding='utf-8-sig')
                print("Utilisation du format CSV en fallback")
                return dict(content=csv_string, filename=csv_filename)
            except:
                return None
    return None