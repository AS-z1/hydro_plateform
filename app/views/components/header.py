import dash_bootstrap_components as dbc
from dash import html
import dash
from config import settings

def create_header():
    """Crée l'en-tête professionnel"""
    return dbc.Navbar(
        dbc.Container(
            [
                # Logo et titre
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.I(
                                        className="fas fa-water fa-2x",
                                        style={
                                            "color": settings.PRIMARY_COLOR,
                                            "marginRight": "10px"
                                        }
                                    ),
                                    html.H4(
                                        settings.APP_NAME,
                                        className="mb-0",
                                        style={
                                            "fontWeight": "600",
                                            "letterSpacing": "-0.5px",
                                            "color": "#2c3e50",
                                            "fontSize": "18px"
                                        }
                                    ),
                                ],
                                className="d-flex align-items-center"
                            ),
                            width="auto"
                        ),
                    ],
                    align="center",
                    className="g-0"
                ),
                
                # Navigation principale
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavLink(
                                        [
                                            html.I(className="fas fa-home me-2"),
                                            "Accueil"
                                        ],
                                        href="/dash/home",
                                        id="nav-home",
                                        className="nav-link-custom"
                                    ),
                                    dbc.NavLink(
                                        [
                                            html.I(className="fas fa-sun me-2"),
                                            "ETP"
                                        ],
                                        href="/dash/eto",
                                        id="nav-eto",
                                        active="exact",
                                        className="nav-link-custom"
                                    ),
                                    dbc.NavLink(
                                        [
                                            html.I(className="fas fa-adjust me-2"),
                                            "Correction Biais"
                                        ],
                                        href="/dash/bias",
                                        id="nav-bias",
                                        className="nav-link-custom"
                                    ),
                                    dbc.NavLink(
                                        [
                                            html.I(className="fas fa-project-diagram me-2"),
                                            "Modélisation"
                                        ],
                                        href="/dash/modeling",
                                        id="nav-modeling",
                                        className="nav-link-custom"
                                    ),
                                    dbc.NavLink(
                                        [
                                            html.I(className="fas fa-chart-line me-2"),
                                            "Prédiction"
                                        ],
                                        href="/dash/prediction",
                                        id="nav-prediction",
                                        className="nav-link-custom"
                                    ),
                                ],
                                className="mx-auto",
                                navbar=True
                            ),
                            width="auto"
                        ),
                    ],
                    align="center",
                    className="g-0"
                ),
                
                # Actions utilisateur
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        html.I(className="fas fa-bell"),
                                        id="btn-notifications",
                                        color="light",
                                        className="position-relative"
                                    ),
                                    dbc.Button(
                                        html.I(className="fas fa-question-circle"),
                                        href="/dash/help",
                                        color="light"
                                    ),
                                    dbc.DropdownMenu(
                                        [
                                            dbc.DropdownMenuItem(
                                                [
                                                    html.I(className="fas fa-user me-2"),
                                                    "Profil"
                                                ],
                                                href="#"
                                            ),
                                            dbc.DropdownMenuItem(
                                                [
                                                    html.I(className="fas fa-cog me-2"),
                                                    "Paramètres"
                                                ],
                                                href="#"
                                            ),
                                            dbc.DropdownMenuItem(divider=True),
                                            dbc.DropdownMenuItem(
                                                [
                                                    html.I(className="fas fa-sign-out-alt me-2"),
                                                    "Déconnexion"
                                                ],
                                                href="#"
                                            ),
                                        ],
                                        label=html.I(className="fas fa-user-circle fa-lg"),
                                        align_end=True,
                                        color="light",
                                        className="dropdown-custom"
                                    ),
                                ],
                                className="ms-auto"
                            ),
                            width="auto"
                        ),
                    ],
                    align="center",
                    className="g-0"
                ),
            ],
            fluid=True,
            className="px-4"
        ),
        color="white",
        dark=False,
        sticky="top",
        className="shadow-sm border-bottom",
        style={
            "height": "60px",
            "fontFamily": settings.FONT_FAMILY,
            "fontSize": settings.FONT_SIZE_BASE
        }
    )

# CSS personnalisé pour les liens de navigation
custom_nav_style = """
.nav-link-custom {
    font-weight: 500;
    color: #6c757d !important;
    padding: 8px 16px !important;
    border-radius: 6px;
    transition: all 0.2s ease;
    margin: 0 4px;
    font-size: 14px;
}

.nav-link-custom:hover {
    color: #3498db !important;
    background-color: rgba(52, 152, 219, 0.1);
    transform: translateY(-1px);
}

.nav-link-custom.active {
    color: #e67e22 !important;
    background-color: rgba(52, 152, 219, 0.1);
    font-weight: 600;
}

.dropdown-custom .dropdown-toggle {
    background: none !important;
    border: none !important;
    color: #6c757d !important;
    padding: 8px;
}

.dropdown-custom .dropdown-toggle:hover {
    color: #3498db !important;
}

.dropdown-custom .dropdown-menu {
    border: 1px solid #e9ecef;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border-radius: 8px;
    padding: 8px 0;
    font-size: 13px;
}

.dropdown-custom .dropdown-item {
    padding: 8px 16px;
    color: #495057;
    font-weight: 400;
}

.dropdown-custom .dropdown-item:hover {
    background-color: rgba(52, 152, 219, 0.1);
    color: #3498db;
}
"""