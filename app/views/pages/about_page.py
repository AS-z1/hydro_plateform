"""
Page À propos - Présentation du développeur et de la plateforme
Design professionnel et  mettant en valeur le travail d'un développeur passionné
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
from config import settings

def create_about_page():
    """Crée la page À propos avec une interface élégante et personnelle"""
    
    return dbc.Container([
        # Header avec effet de vague (comme la page d'aide)
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Animation vague décorative
                    html.Div([
                        html.Div(className="wave"),
                        html.Div(className="wave wave2"),
                        html.Div(className="wave wave3"),
                    ], className="wave-container"),
                    
                    # Contenu header
                    html.Div([
                        html.I(className="fas fa-code fa-3x mb-3", 
                              style={"color": "#4a6fa5", "opacity": "0.9"}),
                        html.H1("À propos de la plateforme", 
                               className="display-4 fw-bold",
                               style={"color": "#2c3e50", "letterSpacing": "-0.5px"}),
                        html.P("Une application hydrologique conçue pour vous. ",
                              className="lead text-muted",
                              style={"fontSize": "1.2rem"}),
                    ], className="text-center py-5")
                ], className="position-relative")
            ])
        ], className="mb-5", style={"backgroundColor": "white", "borderRadius": "0 0 50px 50px",
                                    "boxShadow": "0 4px 20px rgba(0,0,0,0.02)"}),

        # Section : La plateforme
        dbc.Row([
            dbc.Col([
                html.H4(" La Plateforme Hydrologique", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-location-arrow fa-3x mb-3", style={"color": "#4a6fa5"}),
                            html.H5("Vision", className="card-title fw-bold mb-3"),
                            html.P("Offrir aux hydrologues et chercheurs une suite d'outils "
                                  "professionnels, accessibles et performants pour l'analyse, "
                                  "la correction de biais climatique et la modélisation des débits.",
                                  className="text-muted"),
                        ], className="text-center")
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0", style={"borderRadius": "15px"})
            ], md=4, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-bolt fa-3x mb-3", style={"color": "#e67e22"}),
                            html.H5("Performance", className="card-title fw-bold mb-3"),
                            html.P("Algorithmes optimisés, calculs parallélisés et interface "
                                  "réactive pour traiter efficacement de grands volumes de "
                                  "données climatiques et hydrologiques.",
                                  className="text-muted"),
                        ], className="text-center")
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0", style={"borderRadius": "15px"})
            ], md=4, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-magic fa-3x mb-3", style={"color": "#2ecc71"}),
                            html.H5("Innovation", className="card-title fw-bold mb-3"),
                            html.P("Intégration de méthodes de pointe (Quantile Mapping, NSGA-II, "
                                  "LSTM) et d'une interface utilisateur moderne pour une "
                                  "expérience fluide.",
                                  className="text-muted"),
                        ], className="text-center")
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0", style={"borderRadius": "15px"})
            ], md=4, className="mb-4")
        ], className="mb-5"),

        # Section : Le développeur
        dbc.Row([
            dbc.Col([
                html.H4("👨‍💻 Le Développeur", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Img(
                                        src="/assets/Dev.PNG",  # Remplacer par une vraie photo si disponible
                                        className="rounded-circle mb-3",
                                        style={"width": "150px", "height": "150px", 
                                               "objectFit": "cover", "border": "5px solid #4a6fa5"}
                                    ),
                                    html.H4("Xavy AFFOGNON", className="fw-bold mb-1"),
                                    html.P("Développeur indépendant & Hydrologue", 
                                          className="text-muted mb-3"),
                                    html.Div([
                                        dbc.Button(
                                            html.I(className="fab fa-linkedin-in"),
                                            href="https://linkedin.com/in/...",
                                            color="primary",
                                            outline=True,
                                            size="sm",
                                            className="me-2",
                                            style={"borderRadius": "50%", "width": "40px", "height": "40px"}
                                        ),
                                        dbc.Button(
                                            html.I(className="fab fa-github"),
                                            href="https://github.com/AS-z1",
                                            color="dark",
                                            outline=True,
                                            size="sm",
                                            className="me-2",
                                            style={"borderRadius": "50%", "width": "40px", "height": "40px"}
                                        ),
                                        dbc.Button(
                                            html.I(className="fas fa-envelope"),
                                            href="mailto:asxyzt1@gmail.com",
                                            color="danger",
                                            outline=True,
                                            size="sm",
                                            style={"borderRadius": "50%", "width": "40px", "height": "40px"}
                                        ),
                                    ], className="d-flex justify-content-center")
                                ], className="text-center")
                            ], md=4, className="mb-3 mb-md-0"),

                            dbc.Col([
                                html.Div([
                                    html.P([
                                        "Passionné par l'hydrologie et le développement logiciel, "
                                        "j'ai conçu cette plateforme pour répondre aux besoins "
                                        "des professionnels de l'eau. Fort de quelques années d'expérience "
                                        "en modélisation hydrologique et en data science, "
                                        "j'ai voulu créer un outil à la fois puissant et intuitif."
                                    ], className="lead mb-4", style={"fontSize": "1.1rem"}),

                                    html.H6("Compétences clés", className="fw-bold mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.I(className="fas fa-check-circle me-2", style={"color": "#2ecc71"}),
                                                "Python, R & Java",
                                            ], className="mb-2"),
                                            html.Div([
                                                html.I(className="fas fa-check-circle me-2", style={"color": "#2ecc71"}),
                                                "Hydrologie & Climatologie",
                                            ], className="mb-2"),
                                        ], md=6),
                                        dbc.Col([
                                            html.Div([
                                                html.I(className="fas fa-check-circle me-2", style={"color": "#2ecc71"}),
                                                "Machine Learning ",
                                            ], className="mb-2"),
                                            html.Div([
                                                html.I(className="fas fa-check-circle me-2", style={"color": "#2ecc71"}),
                                                "IoT",
                                            ], className="mb-2"),
                                        ], md=6),
                                    ], className="mb-4"),

                                    html.P([
                                        "Cette plateforme est le fruit d'un travail passionné et "
                                        "continu. N'hésitez pas à me contacter pour toute question, "
                                        "collaboration ou suggestion d'amélioration."
                                    ], className="fst-italic text-muted"),
                                ])
                            ], md=8, className="d-flex align-items-center")
                        ])
                    ], className="p-5")
                ], className="shadow-sm border-0", style={"borderRadius": "20px"})
            ], width=12, className="mb-5")
        ]),

        # Section : Technologies utilisées
        dbc.Row([
            dbc.Col([
                html.H4("🛠️ Technologies & Outils", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fab fa-python fa-3x mb-3", style={"color": "#3776ab"}),
                                    html.H6("Python", className="fw-bold"),
                                    html.P("3.9+", className="small text-muted")
                                ], className="text-center")
                            ], md=2, xs=2, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-chart-line fa-3x mb-3", style={"color": "#4a6fa5"}),
                                    html.H6("Dash / Plotly", className="fw-bold"),
                                    html.P("2.9+", className="small text-muted")
                                ], className="text-center")
                            ], md=2, xs=2, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-database fa-3x mb-3", style={"color": "#47A248"}),
                                    html.H6("Pandas / NumPy", className="fw-bold"),
                                    html.P("1.5+", className="small text-muted")
                                ], className="text-center")
                            ], md=2, xs=2, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-brain fa-3x mb-3", style={"color": "#9b59b6"}),
                                    html.H6("TensorFlow", className="fw-bold"),
                                    html.P("2.10+", className="small text-muted")
                                ], className="text-center")
                            ], md=2, xs=2, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-cogs fa-3x mb-3", style={"color": "#e67e22"}),
                                    html.H6("PyMOO", className="fw-bold"),
                                    html.P("0.6+", className="small text-muted")
                                ], className="text-center")
                            ], md=2, xs=4, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-leaf fa-2x mb-3", style={"color": "#3998db"}),
                                    html.H6("pyet", className="fw-bold"),
                                    html.P("0.95+", className="small text-muted")
                                ], className="text-center")
                            ], md=2, xs=2, className="mb-3"),

                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-spa fa-2x mb-3", style={"color": "#ae8a27"}),
                                    html.H6("Ibicus", className="fw-bold"),
                                    html.P("1.0+", className="small text-muted")
                                ], className="text-center")
                            ], md=2, xs=2, className="mb-3"),
                        ])
                    ], className="p-4")
                ], className="shadow-sm border-0", style={"borderRadius": "15px"})
            ], width=12, className="mb-5")
        ]),

        # Section : Contact
        dbc.Row([
            dbc.Col([
                html.H4("📬 Me contacter", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-envelope fa-2x mb-3", style={"color": "#4a6fa5"}),
                                    html.H6("Email", className="fw-bold"),
                                    html.P("asxyzt1@gmail.com", className="text-muted small"),
                                    dbc.Button("Envoyer un message", color="primary", size="sm",
                                              href="mailto:asxyzt1@gmail.com",
                                              style={"borderRadius": "20px"})
                                ], className="text-center p-3")
                            ], md=4, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fab fa-github fa-2x mb-3", style={"color": "#333"}),
                                    html.H6("GitHub", className="fw-bold"),
                                    html.P("Consultez le code source", className="text-muted small"),
                                    dbc.Button("Voir le projet", color="dark", size="sm",
                                              href="https://github.com/...",
                                              style={"borderRadius": "20px"})
                                ], className="text-center p-3")
                            ], md=4, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fab fa-linkedin fa-2x mb-3", style={"color": "#0e76a8"}),
                                    html.H6("LinkedIn", className="fw-bold"),
                                    html.P("Réseau professionnel", className="text-muted small"),
                                    dbc.Button("Me suivre", color="primary", size="sm",
                                              href="https://linkedin.com/in/...",
                                              style={"borderRadius": "20px"})
                                ], className="text-center p-3")
                            ], md=4, className="mb-3"),
                        ])
                    ], className="p-4")
                ], className="shadow-sm border-0", style={"borderRadius": "15px"})
            ], width=10, className="mb-5")
        ]),

        # Pied de page
        dbc.Row([
            dbc.Col([
                html.Hr(className="my-4", style={"borderColor": "#eaeaea"}),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-heart me-2", style={"color": "#e74c3c"}),
                        html.Span("Développé par Xavy AFFOGNON • ", 
                                 className="text-muted small"),
                        html.Span("Version ", className="text-muted small"),
                        html.Span(settings.APP_VERSION, className="text-muted small fw-bold"),
                        html.Span(" • ", className="text-muted small"),
                        html.Span(datetime.now().strftime("%Y"), className="text-muted small"),
                    ], className="d-flex align-items-center justify-content-center flex-wrap")
                ])
            ])
        ]),

        # Styles CSS pour les vagues (réutilisés de help_page)
        html.Div([
            html.Link(
                rel="stylesheet",
                href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
            )
        ]),

    ], fluid=False, className="py-3", 
       style={'backgroundColor': '#f8f9fa', "marginLeft": "210px", "minHeight": "100vh"})