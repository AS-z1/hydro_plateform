import dash_bootstrap_components as dbc
from dash import html
from config import settings

def create_sidebar():
    """Crée la barre latérale professionnelle"""
    return html.Div(
        [
            # En-tête sidebar
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-water fa-2x",
                                        style={
                                            "color": settings.PRIMARY_COLOR,
                                            "marginBottom": "15px"
                                        }
                                    ),
                                    html.H5(
                                        "Navigation",
                                        style={
                                            "fontWeight": "600",
                                            "color": "#2c3e50",
                                            "marginBottom": "5px",
                                            "fontSize": "15px"
                                        }
                                    ),
                                    html.P(
                                        "Plateforme Hydrologique",
                                        style={
                                            "color": "#6c757d",
                                            "fontSize": "12px",
                                            "marginBottom": "0"
                                        }
                                    ),
                                ],
                                className="text-center"
                            )
                        ],
                        className="p-3"
                    )
                ],
                className="border-0 shadow-sm mb-3"
            ),
            
            # Menu de navigation
            dbc.Nav(
                [
                    # Section principale
                    html.Div(
                        [
                            html.H6(
                                "ANALYSE",
                                style={
                                    "color": "#95a5a6",
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "marginBottom": "10px",
                                    "marginTop": "20px",
                                    "paddingLeft": "15px"
                                }
                            ),
                            
                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-home me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "Accueil",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="/dash/home",
                                id="sidebar-home",
                                active="exact",
                                className="sidebar-link"
                            ),
                            
                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-sun me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "Calcul ETP",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="/dash/eto",
                                id="sidebar-eto",
                                active="exact",
                                className="sidebar-link"
                            ),
                            
                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-adjust me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "Correction Biais",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="/dash/bias",
                                id="sidebar-bias",
                                active="exact",
                                className="sidebar-link"
                            ),
                            
                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-project-diagram me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "Modélisation",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="/dash/modeling",
                                id="sidebar-modeling",
                                active="exact",
                                className="sidebar-link"
                            ),
                            
                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-chart-line me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "Prédiction",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="/dash/prediction",
                                id="sidebar-prediction",
                                active="exact",
                                className="sidebar-link"
                            ),
                        ]
                    ),
                    
                    # Section secondaire
                    html.Div(
                        [
                            html.H6(
                                "SUPPORT",
                                style={
                                    "color": "#95a5a6",
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "marginBottom": "10px",
                                    "marginTop": "30px",
                                    "paddingLeft": "15px"
                                }
                            ),
                            
                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-question-circle me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "Aide & Documentation",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="/dash/help",
                                id="sidebar-help",
                                active="exact",
                                className="sidebar-link"
                            ),
                            
                            
                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-info-circle me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "À propos",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="/dash/about",
                                id="sidebar-about",
                                active="exact",
                                className="sidebar-link"
                            ),


                            dbc.NavLink(
                                [
                                    html.I(
                                        className="fas fa-cog me-3",
                                        style={"width": "20px", "textAlign": "center"}
                                    ),
                                    html.Span(
                                        "Paramètres",
                                        style={"fontSize": "13px"}
                                    )
                                ],
                                href="#",
                                id="sidebar-settings",
                                className="sidebar-link"
                            ),

                        ]
                    ),
                ],
                vertical=True,
                pills=True,
                className="flex-column"
            ),
            
            # Status bar
            html.Div(
                [
                    dbc.Badge(
                        "En ligne",
                        color="success",
                        className="me-2",
                        style={"fontSize": "10px", "padding": "2px 6px"}
                    ),
                    html.Small(
                        "v" + settings.APP_VERSION,
                        style={"color": "#95a5a6", "fontSize": "10px"}
                    )
                ],
                className="position-absolute bottom-0 start-0 p-3 w-100",
                style={"borderTop": "1px solid #e9ecef"}
            ),
        ],
        className="sidebar d-flex flex-column",
        style={
            "width": "210px",
            "height": "90vh",
            "position": "fixed",
            "left": "0",
            "top": "60px",
            "backgroundColor": "#ffffff",
            "borderRight": "1px solid #e9ecef",
            "overflowY": "auto",
            "fontFamily": settings.FONT_FAMILY,
            "zIndex": "1000"
        }
    )

# CSS pour la sidebar
sidebar_style = """
.sidebar-link {
    color: #495057 !important;
    padding: 10px 15px !important;
    margin: 2px 10px !important;
    border-radius: 6px !important;
    font-weight: 400 !important;
    transition: all 0.2s ease !important;
    text-decoration: none !important;
    display: flex !important;
    align-items: center !important;
}

.sidebar-link:hover {
    color: #3498db !important;
    background-color: rgba(52, 152, 219, 0.1) !important;
    transform: translateX(3px);
}

.sidebar-link.active {
    color: #3498db !important;
    background-color: rgba(52, 152, 219, 0.1) !important;
    font-weight: 500 !important;
    border-left: 3px solid #3498db;
}

.sidebar-link i {
    font-size: 14px;
}

/* Scrollbar personnalisée */
.sidebar::-webkit-scrollbar {
    width: 6px;
}

.sidebar::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.sidebar::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

.sidebar::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}
"""