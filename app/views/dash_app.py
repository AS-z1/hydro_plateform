import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH (correction : parents[2] au lieu de 3)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import settings
from app.views.components.header import create_header
from app.views.components.sidebar import create_sidebar
from app.views.layout import init_dash_callbacks

def create_dash_application():
    """Crée et configure l'application Dash"""
    
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css",
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
        ],
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        ]
    )
    
    app.title = settings.APP_NAME
    
    app.layout = html.Div(
        [
            dcc.Location(id='url', refresh=False),
            dcc.Store(id='session-store'),
            dcc.Store(id='error-store'),
            create_header(),
            create_sidebar(),
            html.Div(
                [
                    html.Div(id='global-alerts', className="container-fluid mt-3"),
                    html.Div(id='page-content', className="container-fluid py-4"),
                    html.Div(
                        id='loading-overlay',
                        className="loading-overlay",
                        style={'display': 'none'}
                    ),
                ],
                className="main-content"
            ),
        ],
        className="app-layout"
    )
    
    init_dash_callbacks(app)
    
    return app

# Instance pour Gunicorn (Render)
app = create_dash_application()
server = app.server

# Pour exécution locale
if __name__ == "__main__":
    app.run(debug=True, port=8050)