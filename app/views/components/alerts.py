import dash_bootstrap_components as dbc
from dash import html
import dash

def create_alert(type: str, message: str, title: str = None, dismissable: bool = True, 
                 duration: int = None, id: str = None):
    """
    Crée une alerte professionnelle
    
    Args:
        type: 'success', 'danger', 'warning', 'info'
        message: Message de l'alerte
        title: Titre optionnel
        dismissable: Si l'alerte peut être fermée
        duration: Durée en secondes avant disparition automatique
        id: ID unique pour l'alerte
    """
    
    colors = {
        "success": {"bg": "#d1f7c4", "border": "#27ae60", "icon": "check-circle", "color": "#155724"},
        "danger": {"bg": "#ffe6e6", "border": "#e74c3c", "icon": "exclamation-circle", "color": "#721c24"},
        "warning": {"bg": "#fff3cd", "border": "#f39c12", "icon": "exclamation-triangle", "color": "#856404"},
        "info": {"bg": "#e8f4fd", "border": "#3498db", "icon": "info-circle", "color": "#0c5460"}
    }
    
    alert_type = colors.get(type, colors["info"])
    
    alert_content = []
    
    # Titre
    if title:
        alert_content.append(
            html.H6(
                title,
                className="mb-2",
                style={
                    "fontWeight": "600",
                    "fontSize": "14px",
                    "color": alert_type["color"]
                }
            )
        )
    
    # Message
    alert_content.append(
        html.P(
            message,
            className="mb-0",
            style={
                "fontSize": "13px",
                "color": alert_type["color"]
            }
        )
    )
    
    # Bouton de fermeture
    close_button = None
    if dismissable:
        close_button = dbc.Button(
            html.I(className="fas fa-times"),
            color="link",
            className="p-0",
            style={
                "position": "absolute",
                "top": "10px",
                "right": "10px",
                "color": alert_type["color"],
                "opacity": "0.6"
            }
        )
    
    # Créer les arguments pour html.Div
    div_args = {
        'children': [
            # Icone
            html.Div(
                html.I(
                    className=f"fas fa-{alert_type['icon']} fa-lg",
                    style={"color": alert_type["border"]}
                ),
                className="me-3"
            ),
            
            # Contenu
            html.Div(
                alert_content,
                className="flex-grow-1"
            ),
            
            close_button
        ],
        'className': "d-flex align-items-start position-relative p-3 rounded",
        'style': {
            "backgroundColor": alert_type["bg"],
            "border": f"1px solid {alert_type['border']}",
            "borderLeft": f"4px solid {alert_type['border']}",
            "fontFamily": "'Inter', sans-serif"
        }
    }
    
    # Ajouter l'ID seulement s'il est fourni
    if id:
        div_args['id'] = id
    
    alert = html.Div(**div_args)
    
    # Ajouter un timer pour la disparition automatique
    if duration and id:
        dash.clientside_callback(
            f"""
            function(n_clicks) {{
                setTimeout(function() {{
                    var element = document.getElementById('{id}');
                    if (element) {{
                        element.style.opacity = '0';
                        element.style.transition = 'opacity 0.5s';
                        setTimeout(function() {{
                            if (element) element.style.display = 'none';
                        }}, 500);
                    }}
                }}, {duration * 1000});
                return dash_clientside.no_update;
            }}
            """,
            f"{id}-timer",
            [dash.dependencies.Input(f"{id}", "n_clicks")],
            prevent_initial_call=False
        )
    
    return alert

def create_loading_overlay(message: str = "Chargement..."):
    """Crée un overlay de chargement professionnel"""
    return html.Div(
        [
            html.Div(
                [
                    dbc.Spinner(
                        color="primary",
                        size="lg",
                        className="mb-3"
                    ),
                    html.P(
                        message,
                        className="mb-0",
                        style={
                            "color": "#6c757d",
                            "fontSize": "14px",
                            "fontWeight": "500"
                        }
                    )
                ],
                className="text-center"
            )
        ],
        className="d-flex justify-content-center align-items-center",
        style={
            "position": "fixed",
            "top": "60px",
            "left": "250px",
            "right": "0",
            "bottom": "0",
            "backgroundColor": "rgba(255, 255, 255, 0.9)",
            "zIndex": "2000",
            "backdropFilter": "blur(3px)"
        }
    )

def create_error_card(error_message: str, error_details: str = None, 
                     suggestion: str = None, error_code: str = None):
    """Crée une carte d'erreur détaillée"""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div(
                    [
                        html.I(
                            className="fas fa-exclamation-triangle me-2",
                            style={"color": "#e74c3c"}
                        ),
                        html.H5(
                            "Erreur rencontrée",
                            className="mb-0",
                            style={"color": "#2c3e50"}
                        )
                    ],
                    className="d-flex align-items-center"
                ),
                className="border-bottom"
            ),
            
            dbc.CardBody(
                [
                    # Message d'erreur principal
                    html.Div(
                        [
                            html.H6(
                                "Message :",
                                style={
                                    "color": "#6c757d",
                                    "fontSize": "12px",
                                    "textTransform": "uppercase",
                                    "marginBottom": "5px"
                                }
                            ),
                            html.P(
                                error_message,
                                className="mb-4",
                                style={
                                    "color": "#2c3e50",
                                    "fontSize": "14px",
                                    "fontWeight": "500"
                                }
                            )
                        ]
                    ),
                    
                    # Détails de l'erreur (si disponibles)
                    html.Div(
                        [
                            html.H6(
                                "Détails :",
                                style={
                                    "color": "#6c757d",
                                    "fontSize": "12px",
                                    "textTransform": "uppercase",
                                    "marginBottom": "5px"
                                }
                            ),
                            html.Pre(
                                error_details if error_details else "Aucun détail supplémentaire",
                                className="mb-4 p-3 rounded",
                                style={
                                    "backgroundColor": "#f8f9fa",
                                    "border": "1px solid #e9ecef",
                                    "fontSize": "11px",
                                    "whiteSpace": "pre-wrap",
                                    "maxHeight": "200px",
                                    "overflowY": "auto"
                                }
                            )
                        ]
                    ) if error_details else None,
                    
                    # Suggestion (si disponible)
                    html.Div(
                        [
                            html.H6(
                                "Suggestions :",
                                style={
                                    "color": "#6c757d",
                                    "fontSize": "12px",
                                    "textTransform": "uppercase",
                                    "marginBottom": "5px"
                                }
                            ),
                            html.P(
                                suggestion if suggestion else "Veuillez réessayer ou contacter le support.",
                                className="mb-0",
                                style={
                                    "color": "#2c3e50",
                                    "fontSize": "13px"
                                }
                            )
                        ]
                    ) if suggestion else None,
                    
                    # Code d'erreur (si disponible)
                    html.Div(
                        [
                            html.H6(
                                "Code d'erreur :",
                                style={
                                    "color": "#6c757d",
                                    "fontSize": "12px",
                                    "textTransform": "uppercase",
                                    "marginBottom": "5px"
                                }
                            ),
                            dbc.Badge(
                                error_code,
                                color="secondary",
                                className="px-3 py-2",
                                style={"fontSize": "11px"}
                            )
                        ]
                    ) if error_code else None,
                ]
            ),
            
            dbc.CardFooter(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            [
                                html.I(className="fas fa-redo me-2"),
                                "Réessayer"
                            ],
                            color="primary",
                            size="sm"
                        ),
                        dbc.Button(
                            [
                                html.I(className="fas fa-question-circle me-2"),
                                "Aide"
                            ],
                            color="light",
                            size="sm",
                            href="/dash/help"
                        ),
                        dbc.Button(
                            [
                                html.I(className="fas fa-bug me-2"),
                                "Signaler un bug"
                            ],
                            color="light",
                            size="sm"
                        ),
                    ],
                    className="w-100"
                )
            )
        ],
        className="shadow-sm border-danger",
        style={"borderLeft": "4px solid #e74c3c"}
    )