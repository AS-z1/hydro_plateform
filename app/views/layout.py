import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from config import settings
from app.views.components.header import create_header
from app.views.components.sidebar import create_sidebar
from app.views.pages import (
    home,
    eto_page,
    bias_page,
    modeling_page,
    prediction_page,
    help_page,
    about_page
)

def init_dash_callbacks(app):
    """Initialise tous les callbacks Dash"""
    
    # Callback pour la navigation
    @app.callback(
        [Output('page-content', 'children'),
         Output('global-alerts', 'children'),
         Output('nav-home', 'className'),
         Output('nav-eto', 'className'),
         Output('nav-bias', 'className'),
         Output('nav-modeling', 'className'),
         Output('nav-prediction', 'className'),
         Output('sidebar-home', 'active'),
         Output('sidebar-eto', 'active'),
         Output('sidebar-bias', 'active'),
         Output('sidebar-modeling', 'active'),
         Output('sidebar-prediction', 'active'),
         Output('sidebar-help', 'active'),
         Output('sidebar-about', 'active')],  # ✅ AJOUTÉ
        [Input('url', 'pathname')]
    )
    def display_page(pathname):
        """Gère l'affichage des pages et la navigation active"""
        if pathname is None:
            return dash.no_update
        
        # Navigation active - initialisation
        nav_classes = ['nav-link-custom'] * 5
        sidebar_active = [''] * 7  # home, eto, bias, modeling, prediction, help, about
        
        # Déterminer la page
        if pathname.endswith('/home') or pathname == '/dash/' or pathname == '/dash':
            page = home.create_home_page()
            nav_classes[0] = 'nav-link-custom active'
            sidebar_active[0] = 'exact'
            
        elif pathname.endswith('/eto'):
            page = eto_page.create_etp_page()
            nav_classes[1] = 'nav-link-custom active'
            sidebar_active[1] = 'exact'
            
        elif pathname.endswith('/bias'):
            page = bias_page.create_bias_page()
            nav_classes[2] = 'nav-link-custom active'
            sidebar_active[2] = 'exact'
            
        elif pathname.endswith('/modeling'):
            page = modeling_page.create_modeling_page()
            nav_classes[3] = 'nav-link-custom active'
            sidebar_active[3] = 'exact'
            
        elif pathname.endswith('/prediction'):
            page = prediction_page.create_prediction_page()
            nav_classes[4] = 'nav-link-custom active'
            sidebar_active[4] = 'exact'
            
        elif pathname.endswith('/help'):
            page = help_page.create_help_page()
            sidebar_active[5] = 'exact'

        elif pathname.endswith('/about'):
            page = about_page.create_about_page()
            sidebar_active[6] = 'exact'
            
        else:
            page = html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle fa-4x mb-4", 
                                  style={"color": "#f39c12"}),
                            html.H1("404", className="display-1 fw-bold", 
                                   style={"color": "#2c3e50"}),
                            html.H4("Page non trouvée", className="mb-4",
                                   style={"color": "#7f8c8d"}),
                            html.P(f"La page '{pathname}' n'existe pas ou a été déplacée.", 
                                  className="text-muted mb-4"),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-home me-2"),
                                    "Retour à l'accueil"
                                ],
                                href="/dash/home",
                                color="primary",
                                className="px-5 py-2",
                                style={"backgroundColor": "#4a6fa5", 
                                       "border": "none",
                                       "borderRadius": "8px"}
                            )
                        ], className="text-center p-5")
                    ], className="p-0")
                ], className="shadow-sm border-0 mx-auto", 
                   style={"maxWidth": "600px", "borderRadius": "12px"})
            ])
        
        return page, None, *nav_classes, *sidebar_active  # 1 + 1 + 5 + 7 = 14 sorties
    
    # Callback pour afficher les erreurs
    @app.callback(
        Output('global-alerts', 'children', allow_duplicate=True),
        Input('error-store', 'data'),
        prevent_initial_call=True
    )
    def show_error(error_data):
        if error_data and 'type' in error_data:
            from app.views.components.alerts import create_alert
            return create_alert(
                type=error_data['type'],
                message=error_data['message'],
                title=error_data.get('title'),
                dismissable=True
            )
        return None
    
    # Callback pour gérer le chargement
    @app.callback(
        Output('loading-overlay', 'style'),
        [Input('url', 'pathname')]
    )
    def show_loading(pathname):
        if pathname and pathname != '/dash/':
            return {
                'display': 'flex',
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'width': '100%',
                'height': '100%',
                'backgroundColor': 'rgba(255,255,255,0.9)',
                'zIndex': '9999',
                'justifyContent': 'center',
                'alignItems': 'center',
                'flexDirection': 'column'
            }
        return {'display': 'none'}
    
    # Callback client-side amélioré pour cacher le loading avec animation
    app.clientside_callback(
        """
        function(pathname) {
            setTimeout(function() {
                var overlay = document.getElementById('loading-overlay');
                if (overlay) {
                    overlay.style.opacity = '0';
                    setTimeout(function() {
                        overlay.style.display = 'none';
                    }, 300);
                }
            }, 800);
            return window.dash_clientside.no_update;
        }
        """,
        Output('loading-overlay', 'children', allow_duplicate=True),
        Input('url', 'pathname'),
        prevent_initial_call=True
    )
    
    # CALLBACK POUR LA RECHERCHE DANS LA PAGE D'AIDE (optionnel)
    @app.callback(
        Output('help-search-store', 'data'),
        Input('search-btn', 'n_clicks'),
        State('help-search', 'value'),
        prevent_initial_call=True
    )
    def handle_help_search(n_clicks, search_term):
        """Gère la recherche dans la page d'aide"""
        if n_clicks and search_term:
            # Stocker le terme de recherche
            return {
                'term': search_term,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            }
        return None
    
    return app