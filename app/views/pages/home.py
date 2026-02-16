import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import settings
from datetime import datetime

def create_home_page():
    """Crée la page d'accueil professionnelle"""
    
    # Graphique de bienvenue
    welcome_fig = go.Figure()
    welcome_fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=95,
        title={'text': "Performance système", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': settings.PRIMARY_COLOR},
            'steps': [
                {'range': [0, 50], 'color': "#e74c3c"},
                {'range': [50, 80], 'color': "#f39c12"},
                {'range': [80, 100], 'color': settings.SECONDARY_COLOR}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    welcome_fig.update_layout(height=250, margin=dict(t=30, b=30))
    
    return dbc.Container([
        # Header avec titre
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1(" Plateforme Hydrologique", 
                           className="mb-3",
                           style={"fontSize": "28px", "fontWeight": "700", "color": "#2c3e50"}),
                    html.P("Outils professionnels pour l'analyse et la modélisation hydrologique",
                          className="text-muted mb-4",
                          style={"fontSize": "16px"})
                ], className="text-center py-4")
            ])
        ], className="mb-4",
           style={"borderBottom": "2px solid #4a6fa5", "backgroundColor": "white", "borderRadius": "8px"}),
        
        # Cartes des fonctionnalités principales
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-calculator me-2"),
                            "Calcul ETP"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-temperature-high fa-2x mb-3", 
                                  style={"color": "#e74c3c"}),
                            html.H5("Évapotranspiration", className="card-title mb-2"),
                            html.P(
                                "Calcul d'évapotranspiration potentielle avec différentes méthodes (FAO-56, Hargreaves, Oudin, etc.)",
                                className="card-text small text-muted mb-3"
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-arrow-right me-2"),
                                    "Accéder au module"
                                ],
                                href="/dash/eto",
                                color="primary",
                                size="sm",
                                className="w-100 py-2",
                                style={"backgroundColor": "#4a6fa5", "border": "none", "borderRadius": "6px"}
                            )
                        ], className="text-center")
                    ], className="p-4")
                ], className="h-100 shadow border-0",
                   style={"borderRadius": "10px", "transition": "transform 0.2s"})
            ], md=4, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-adjust me-2"),
                            "Correction Biais"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-adjust fa-2x mb-3", 
                                  style={"color": "#4a6fa5"}),
                            html.H5("Correction Climatique", className="card-title mb-2"),
                            html.P(
                                "Correction statistique des modèles climatiques avec méthodes avancées (Quantile Mapping, ISIMIP, etc.)",
                                className="card-text small text-muted mb-3"
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-arrow-right me-2"),
                                    "Accéder au module"
                                ],
                                href="/dash/bias",
                                color="primary",
                                size="sm",
                                className="w-100 py-2",
                                style={"backgroundColor": "#4a6fa5", "border": "none", "borderRadius": "6px"}
                            )
                        ], className="text-center")
                    ], className="p-4")
                ], className="h-100 shadow border-0",
                   style={"borderRadius": "10px", "transition": "transform 0.2s"})
            ], md=4, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-project-diagram me-2"),
                            "Modélisation"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-project-diagram fa-2x mb-3", 
                                  style={"color": "#4a6fa5"}),
                            html.H5("Modèles Hydrologiques", className="card-title mb-2"),
                            html.P(
                                "Modèles hydrologiques ModHyPMA et LSTM pour la simulation et prédiction des débits",
                                className="card-text small text-muted mb-3"
                            ),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-arrow-right me-2"),
                                    "Accéder au module"
                                ],
                                href="/dash/modeling",
                                color="primary",
                                size="sm",
                                className="w-100 py-2",
                                style={"backgroundColor": "#4a6fa5", "border": "none", "borderRadius": "6px"}
                            )
                        ], className="text-center")
                    ], className="p-4")
                ], className="h-100 shadow border-0",
                   style={"borderRadius": "10px", "transition": "transform 0.2s"})
            ], md=4, className="mb-3"),
        ], className="mb-4"),
        
        # Section graphique et activité récente
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-tachometer-alt me-2"),
                            "Performances système"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        dcc.Graph(
                            figure=welcome_fig,
                            config={'displayModeBar': False}
                        )
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px", "height": "100%"})
            ], md=6, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-history me-2"),
                            "Activité récente"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(
                                        className="fas fa-check-circle me-2",
                                        style={"color": "#2ecc71"}
                                    ),
                                    html.Div([
                                        html.Span(
                                            "Session utilisateur démarrée",
                                            className="fw-medium"
                                        ),
                                        html.Small(
                                            "Il y a 2 minutes",
                                            className="text-muted d-block"
                                        )
                                    ])
                                ], className="d-flex align-items-center")
                            ], className="border-0 py-3"),
                            
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(
                                        className="fas fa-file-upload me-2",
                                        style={"color": "#3498db"}
                                    ),
                                    html.Div([
                                        html.Span(
                                            "Données météorologiques importées",
                                            className="fw-medium"
                                        ),
                                        html.Small(
                                            "Il y a 10 minutes",
                                            className="text-muted d-block"
                                        )
                                    ])
                                ], className="d-flex align-items-center")
                            ], className="border-0 py-3"),
                            
                            dbc.ListGroupItem([
                                html.Div([
                                    html.I(
                                        className="fas fa-chart-line me-2",
                                        style={"color": "#e74c3c"}
                                    ),
                                    html.Div([
                                        html.Span(
                                            "Calcul ETP terminé",
                                            className="fw-medium"
                                        ),
                                        html.Small(
                                            "Il y a 1 heure",
                                            className="text-muted d-block"
                                        )
                                    ])
                                ], className="d-flex align-items-center")
                            ], className="border-0 py-3"),
                        ], flush=True)
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px", "height": "100%"})
            ], md=6, className="mb-3"),
        ], className="mb-4"),
        
        # Documentation rapide
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-book me-2"),
                            "Documentation rapide"
                        ], className="d-flex align-items-center")
                    ], className="py-2", style={"backgroundColor": "#4a6fa5", "color": "white"}),
                    dbc.CardBody([
                        dbc.Accordion([
                            dbc.AccordionItem(
                                [
                                    html.P("Le module ETP calcule l'évapotranspiration potentielle à l'aide de plusieurs méthodes standard :", 
                                          className="mb-3"),
                                    html.Ul([
                                        html.Li([
                                            html.I(className="fas fa-star me-2", style={"color": "#f39c12"}),
                                            html.Strong("FAO-56 "), " "
                                        ]),
                                        html.Li([
                                            html.I(className="fas fa-star me-2", style={"color": "#f39c12"}),
                                            html.Strong("Hargreaves"), " (requiert moins de données)"
                                        ]),
                                        html.Li([
                                            html.I(className="fas fa-star me-2", style={"color": "#f39c12"}),
                                            html.Strong("Oudin"), " (méthode simplifiée)"
                                        ]),
                                        html.Li([
                                            html.I(className="fas fa-star me-2", style={"color": "#f39c12"}),
                                            html.Strong("Turc"), " (adaptée climats humides)"
                                        ])
                                    ])
                                ],
                                title="Comment utiliser le module ETP ?",
                                style={"border": "1px solid #eaeaea", "borderRadius": "6px"}
                            ),
                            dbc.AccordionItem(
                                [
                                    html.P("La correction de biais permet d'ajuster les sorties des modèles climatiques aux observations :", 
                                          className="mb-3"),
                                    html.P("Méthodes disponibles :"),
                                    html.Ul([
                                        html.Li([
                                            html.I(className="fas fa-chart-line me-2", style={"color": "#3498db"}),
                                            html.Strong("Quantile Delta Mapping"), " (précision élevée)"
                                        ]),
                                        html.Li([
                                            html.I(className="fas fa-chart-line me-2", style={"color": "#3498db"}),
                                            html.Strong("Linear Scaling"), " (simple et rapide)"
                                        ]),
                                        html.Li([
                                            html.I(className="fas fa-chart-line me-2", style={"color": "#3498db"}),
                                            html.Strong("Delta Change"), " (préservation tendances)"
                                        ]),
                                        html.Li([
                                            html.I(className="fas fa-chart-line me-2", style={"color": "#3498db"}),
                                            html.Strong("Scaled Distribution Mapping"), " (distribution avancée)"
                                        ])
                                    ])
                                ],
                                title="Qu'est-ce que la correction de biais ?",
                                style={"border": "1px solid #eaeaea", "borderRadius": "6px"}
                            ),
                            dbc.AccordionItem(
                                [
                                    html.P("Deux types de modèles hydrologiques sont disponibles :", 
                                          className="mb-3"),
                                    html.Ul([
                                        html.Li([
                                            html.I(className="fas fa-cogs me-2", style={"color": "#e74c3c"}),
                                            html.Strong("ModHyPMA"), " - Modèle à base physique pour une simulation réaliste"
                                        ]),
                                        html.Li([
                                            html.I(className="fas fa-brain me-2", style={"color": "#9b59b6"}),
                                            html.Strong("LSTM"), " - Réseau de neurones récurrent pour la prédiction"
                                        ]),
                                    ]),
                                    html.P("Chaque modèle nécessite des périodes spécifiques de calage et validation.", 
                                          className="mt-3 small text-muted")
                                ],
                                title="Quels modèles sont disponibles ?",
                                style={"border": "1px solid #eaeaea", "borderRadius": "6px"}
                            ),
                        ], flush=True, start_collapsed=True, 
                           style={"border": "none"})
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "10px"})
            ])
        ], className="mb-4"),
        
        # Pied de page
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Hr(className="my-4", style={"borderColor": "#eaeaea"}),
                    html.Div([
                        html.Small(
                            f"© 2026 {settings.APP_NAME} - Version {settings.APP_VERSION}",
                            className="text-muted me-3"
                        ),
                        html.Small("•", className="text-muted me-3"),
                        html.Small(
                            "Développé avec Dash & FastAPI",
                            className="text-muted me-3"
                        ),
                        html.Small("•", className="text-muted me-3"),
                        html.Small(
                            "Contact: support@hydrologie.fr",
                            className="text-muted"
                        ),
                        html.Small("•", className="text-muted me-3"),
                        html.Span("Dernière mise à jour: ", className="text-muted small"),
                        html.Span(datetime.now().strftime("%d/%m/%Y"),
                                  className="text-muted small fw-bold"),
                        
                    ], className="d-flex justify-content-center align-items-center flex-wrap")
                ])
            ])
        ], className="mt-4"),
        
        # Styles CSS supplémentaires
        
    ], fluid=False, className="py-4", style={'backgroundColor': '#f8f9fa', "marginLeft": "210px"})