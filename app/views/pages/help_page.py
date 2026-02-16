"""
Page d'aide et support utilisateur
Design professionnel et intuitif pour une expérience optimale
"""

from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from config import settings

def create_help_page():
    """Crée la page d'aide avec une interface moderne et intuitive"""
    
    return dbc.Container([
        # Header avec effet de vague
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
                        html.I(className="fas fa-life-ring fa-3x mb-3", 
                              style={"color": "#4a6fa5", "opacity": "0.9"}),
                        html.H1("Centre d'Aide", 
                               className="display-4 fw-bold",
                               style={"color": "#2c3e50", "letterSpacing": "-0.5px"}),
                        html.P("Documentation, tutoriels et support technique",
                              className="lead text-muted",
                              style={"fontSize": "1.2rem"}),
                        
                        # Barre de recherche
                        dbc.InputGroup([
                            dbc.Input(
                                id="help-search",
                                type="text",
                                placeholder="Rechercher une aide, une fonctionnalité...",
                                size="lg",
                                className="border-end-0",
                                style={"borderRadius": "30px 0 0 30px", 
                                       "border": "2px solid #e9ecef",
                                       "borderRight": "none",
                                       "padding": "0.8rem 1.5rem"}
                            ),
                            dbc.Button(
                                html.I(className="fas fa-search"),
                                id="search-btn",
                                color="primary",
                                className="px-4",
                                style={"borderRadius": "0 30px 30px 0",
                                       "backgroundColor": "#4a6fa5",
                                       "border": "2px solid #4a6fa5"}
                            )
                        ], className="w-50 mx-auto mt-4")
                    ], className="text-center py-5")
                ], className="position-relative")
            ])
        ], className="mb-5", style={"backgroundColor": "white", "borderRadius": "0 0 50px 50px",
                                    "boxShadow": "0 4px 20px rgba(0,0,0,0.02)"}),
        
        # Cartes d'accès rapide
        dbc.Row([
            dbc.Col([
                html.H4("🚀 Accès Rapide", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),
        
        dbc.Row([
            # Documentation
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-book-open fa-3x",
                                      style={"color": "#4a6fa5"})
                            ], className="text-center mb-4"),
                            html.H5("Documentation Technique", 
                                   className="card-title text-center fw-bold mb-3"),
                            html.P("Guides complets, API, algorithmes et spécifications",
                                  className="text-muted text-center small mb-4"),
                            dbc.Button([
                                html.I(className="fas fa-arrow-right me-2"),
                                "Accéder"
                            ], color="outline-primary", size="sm",
                               className="w-100",
                               style={"borderRadius": "20px"})
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "15px", "transition": "transform 0.3s",
                          "cursor": "pointer"},
                   id="doc-card")
            ], md=3, className="mb-4"),
            
            # Tutoriels
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-video fa-3x",
                                      style={"color": "#2ecc71"})
                            ], className="text-center mb-4"),
                            html.H5("Tutoriels Vidéo", 
                                   className="card-title text-center fw-bold mb-3"),
                            html.P("Formations interactives et démonstrations pas à pas",
                                  className="text-muted text-center small mb-4"),
                            dbc.Button([
                                html.I(className="fas fa-arrow-right me-2"),
                                "Voir"
                            ], color="outline-success", size="sm",
                               className="w-100",
                               style={"borderRadius": "20px"})
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "15px", "transition": "transform 0.3s",
                          "cursor": "pointer"})
            ], md=3, className="mb-4"),
            
            # FAQ
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-question-circle fa-3x",
                                      style={"color": "#f39c12"})
                            ], className="text-center mb-4"),
                            html.H5("FAQ Interactive", 
                                   className="card-title text-center fw-bold mb-3"),
                            html.P("Questions fréquentes et résolution de problèmes",
                                  className="text-muted text-center small mb-4"),
                            dbc.Button([
                                html.I(className="fas fa-arrow-right me-2"),
                                "Consulter"
                            ], color="outline-warning", size="sm",
                               className="w-100",
                               style={"borderRadius": "20px"})
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "15px", "transition": "transform 0.3s",
                          "cursor": "pointer"})
            ], md=3, className="mb-4"),
            
            # Support
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-headset fa-3x",
                                      style={"color": "#e74c3c"})
                            ], className="text-center mb-4"),
                            html.H5("Support Technique", 
                                   className="card-title text-center fw-bold mb-3"),
                            html.P("Assistance personnalisée par nos experts",
                                  className="text-muted text-center small mb-4"),
                            dbc.Button([
                                html.I(className="fas fa-arrow-right me-2"),
                                "Contacter"
                            ], color="outline-danger", size="sm",
                               className="w-100",
                               style={"borderRadius": "20px"})
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "15px", "transition": "transform 0.3s",
                          "cursor": "pointer"})
            ], md=3, className="mb-4")
        ], className="mb-5"),
        
        # Section documentation par module
        dbc.Row([
            dbc.Col([
                html.H4("📘 Documentation par Module", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),
        
        dbc.Row([
            # Module ETP
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-sun me-2"),
                            "Calcul ETP",
                            html.Span("v2.1", 
                                     className="badge bg-info ms-2",
                                     style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="py-3", 
                       style={"backgroundColor": "rgba(74, 111, 165, 0.1)", 
                              "borderBottom": "3px solid #4a6fa5"}),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Strong("Méthodes supportées:"),
                                html.Ul([
                                    html.Li("FAO-56"),
                                    html.Li("Hargreaves "),
                                    html.Li("Oudin "),
                                    html.Li("Turc "),
                                    html.Li("Hamon")
                                ], className="mt-2 small")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Strong("Prérequis données:"),
                                html.P("Température min, max, moyenne, humidité, vent, radiation",
                                      className="small text-muted mt-1")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Span("Précision:", className="fw-bold me-2"),
                                html.Span("⭐ 4.8/5", className="badge bg-success")
                            ]),
                            
                            html.Hr(className="my-3"),
                            
                            dbc.Button(
                                "Guide complet",
                                color="link",
                                size="sm",
                                className="p-0",
                                style={"color": "#4a6fa5"}
                            )
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "12px"})
            ], md=6, lg=3, className="mb-4"),
            
            # Module Correction de biais
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-adjust me-2"),
                            "Correction Biais",
                            html.Span("v1.5", 
                                     className="badge bg-info ms-2",
                                     style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="py-3", 
                       style={"backgroundColor": "rgba(46, 204, 113, 0.1)", 
                              "borderBottom": "3px solid #2ecc71"}),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Strong("Algorithmes:"),
                                html.Ul([
                                    html.Li("Quantile Delta Mapping"),
                                    html.Li("ISIMIP"),
                                    html.Li("Linear Scaling"),
                                    html.Li("Delta Change"),
                                    html.Li("Scaled Distribution Mapping")
                                ], className="mt-2 small")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Strong("Performance:"),
                                html.Div([
                                    html.Span("Réduction biais: ", className="text-muted"),
                                    html.Strong("70-85%", className="ms-1")
                                ]),
                                html.Div([
                                    html.Span("Temps calcul: ", className="text-muted"),
                                    html.Strong("< 1s/1000 pts", className="ms-1")
                                ])
                            ], className="mb-3"),
                            
                            dbc.Button(
                                "Documentation",
                                color="link",
                                size="sm",
                                className="p-0",
                                style={"color": "#2ecc71"}
                            )
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "12px"})
            ], md=6, lg=3, className="mb-4"),
            
            # Module Modélisation
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-project-diagram me-2"),
                            "Modélisation",
                            html.Span("v3.0", 
                                     className="badge bg-info ms-2",
                                     style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="py-3", 
                       style={"backgroundColor": "rgba(155, 89, 182, 0.1)", 
                              "borderBottom": "3px solid #9b59b6"}),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Strong("Modèles disponibles:"),
                                html.Div([
                                    html.I(className="fas fa-cog me-1", style={"color": "#9b59b6"}),
                                    "ModHyPMA - 4 paramètres optimisés"
                                ], className="mt-2"),
                                html.Div([
                                    html.I(className="fas fa-brain me-1", style={"color": "#9b59b6"}),
                                    "LSTM - Deep Learning"
                                ], className="mt-1"),
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Strong("Optimisation:"),
                                html.Div([
                                    html.Span("NSGA-II multi-objectifs", className="badge bg-primary me-2"),
                                    html.Span("Validation croisée", className="badge bg-secondary")
                                ], className="mt-2")
                            ], className="mb-3"),
                            
                            dbc.Button(
                                "Architecture",
                                color="link",
                                size="sm",
                                className="p-0",
                                style={"color": "#9b59b6"}
                            )
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "12px"})
            ], md=6, lg=3, className="mb-4"),
            
            # Module Prédiction
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-line me-2"),
                            "Prédiction",
                            html.Span("βeta", 
                                     className="badge bg-warning ms-2",
                                     style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="py-3", 
                       style={"backgroundColor": "rgba(231, 76, 60, 0.1)", 
                              "borderBottom": "3px solid #e74c3c"}),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Strong("Fonctionnalités:"),
                                html.Ul([
                                    html.Li("Prédiction séquentielle"),
                                    html.Li("Scénarios climatiques"),
                                    html.Li("Analyse d'incertitude"),
                                    html.Li("Seuils critiques")
                                ], className="mt-2 small")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Strong("Modèle pré-entraîné:"),
                                html.Div([
                                    html.Span("LSTM - 3 couches, 50 unités", className="d-block small"),
                                    html.Span("Dataset: 20 ans historiques", className="d-block small text-muted")
                                ], className="mt-1")
                            ], className="mb-3"),
                            
                            dbc.Button(
                                "Performance",
                                color="link",
                                size="sm",
                                className="p-0",
                                style={"color": "#e74c3c"}
                            )
                        ])
                    ], className="p-4")
                ], className="h-100 shadow-sm border-0",
                   style={"borderRadius": "12px"})
            ], md=6, lg=3, className="mb-4")
        ], className="mb-5"),
        
        # Section FAQ interactive
        dbc.Row([
            dbc.Col([
                html.H4("❓ Foire Aux Questions", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Accordion([
                            dbc.AccordionItem([
                                html.P([
                                    "1. ", html.Strong("Préparez votre fichier:"), 
                                    " CSV ou Excel avec en-têtes",
                                    html.Br(),
                                    "2. ", html.Strong("Colonnes requises:"), 
                                    " date, Qobs, Pluie, ETP",
                                    html.Br(),
                                    "3. ", html.Strong("Format date:"), 
                                    " YYYY-MM-DD ou DD/MM/YYYY",
                                    html.Br(),
                                    "4. ", html.Strong("Période:"), 
                                    " Minimum 365 jours pour calage/validation"
                                ]),
                                dbc.Alert([
                                    html.I(className="fas fa-download me-2"),
                                    "Téléchargez notre fichier modèle"
                                ], color="primary", className="mt-3 p-2", style={"borderRadius": "8px"})
                            ], title="📁 Comment préparer mes données ?", 
                               style={"border": "none"}),
                            
                            dbc.AccordionItem([
                                html.P([
                                    html.Strong("ModHyPMA:"), " Modèle conceptuel pluie-débit",
                                    html.Br(),
                                    "• Paramètres: m (sol), l (temps), P2 (infiltration), TX (seuil)",
                                    html.Br(),
                                    "• Plages: m[0.9-1.45], l[26-150], P2[2.2-10], TX[0.00001-0.8]",
                                    html.Br(),
                                    html.Br(),
                                    html.Strong("LSTM:"), " Réseau de neurones récurrent",
                                    html.Br(),
                                    "• Architecture: Entrée → LSTM(units) → Dense(1)",
                                    html.Br(),
                                    "• Hyperparamètres: epochs, lr, batch_size, seq_length",
                                ])
                            ], title="🧠 Différence ModHyPMA vs LSTM ?", 
                               style={"border": "none"}),
                            
                            dbc.AccordionItem([
                                html.P([
                                    "Le module utilise l'algorithme NSGA-II pour optimiser simultanément:",
                                    html.Br(),
                                    html.Br(),
                                    "🎯 ", html.Strong("Objectif 1:"), " Maximiser le NSE (Nash-Sutcliffe)",
                                    html.Br(),
                                    "🎯 ", html.Strong("Objectif 2:"), " Minimiser le RMSE",
                                    html.Br(),
                                    html.Br(),
                                    "La population (50 individus) évolue sur 30 générations pour trouver le "
                                    "meilleur compromis entre ces objectifs."
                                ]),
                                dbc.Progress(value=85, label="Performance", className="mt-3",
                                           style={"height": "20px", "borderRadius": "10px"})
                            ], title="⚡ Comment fonctionne l'optimisation NSGA-II ?", 
                               style={"border": "none"}),
                            
                            dbc.AccordionItem([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Métriques d'évaluation", className="mb-3"),
                                        html.Div([
                                            html.Div([
                                                html.Span("NSE > 0.75: ", className="fw-bold"),
                                                html.Span("Excellent", className="badge bg-success ms-2")
                                            ], className="mb-2"),
                                            html.Div([
                                                html.Span("NSE 0.65-0.75: ", className="fw-bold"),
                                                html.Span("Bon", className="badge bg-primary ms-2")
                                            ], className="mb-2"),
                                            html.Div([
                                                html.Span("NSE 0.5-0.65: ", className="fw-bold"),
                                                html.Span("Satisfaisant", className="badge bg-warning ms-2")
                                            ], className="mb-2"),
                                            html.Div([
                                                html.Span("NSE < 0.5: ", className="fw-bold"),
                                                html.Span("À améliorer", className="badge bg-danger ms-2")
                                            ])
                                        ])
                                    ], md=6),
                                    dbc.Col([
                                        html.H6("KGE (Kling-Gupta)", className="mb-3"),
                                        html.P("Le KGE combine corrélation, biais et variabilité. "
                                              "Un KGE > 0.7 indique une très bonne performance.",
                                              className="small")
                                    ], md=6)
                                ])
                            ], title="📊 Comment interpréter les métriques ?", 
                               style={"border": "none"}),
                            
                            dbc.AccordionItem([
                                html.P([
                                    "🔧 ", html.Strong("Erreur 'PyMOO non installé':"), 
                                    " pip install pymoo",
                                    html.Br(),
                                    "🔧 ", html.Strong("Module LSTM non disponible:"), 
                                    " pip install tensorflow",
                                    html.Br(),
                                    "🔧 ", html.Strong("Erreur de mémoire:"), 
                                    " Réduisez la taille des données",
                                    html.Br(),
                                    "🔧 ", html.Strong("Graphique vide:"), 
                                    " Vérifiez les colonnes sélectionnées"
                                ])
                            ], title="⚠️ Résolution des problèmes courants", 
                               style={"border": "none"}),
                        ], start_collapsed=True, flush=True,
                           style={"--bs-accordion-btn-focus-box-shadow": "none"})
                    ], className="p-3")
                ], className="shadow-sm border-0", style={"borderRadius": "15px"})
            ], width=12, className="mb-5")
        ]),
        
        # Section Support & Contact
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-clock fa-2x mb-3",
                                          style={"color": "#4a6fa5"}),
                                    html.H6("Disponibilité", className="fw-bold"),
                                    html.P("Lun-Ven: 7h-22h", className="small text-muted mb-0"),
                                    html.P("Sam-Dim: Urgences", className="small text-muted")
                                ], className="text-center")
                            ], md=3),
                            
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-envelope fa-2x mb-3",
                                          style={"color": "#2ecc71"}),
                                    html.H6("Email", className="fw-bold"),
                                    html.P("support@hydrologie.fr", className="small mb-0"),
                                    html.P("contact@hydrologie.fr", className="small text-muted")
                                ], className="text-center")
                            ], md=3),
                            
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-phone-alt fa-2x mb-3",
                                          style={"color": "#f39c12"}),
                                    html.H6("Téléphone", className="fw-bold"),
                                    html.P("+2290167322179", className="small mb-0"),
                                    html.P("Numéro vert: -------", className="small text-muted")
                                ], className="text-center")
                            ], md=3),
                            
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-comments fa-2x mb-3",
                                          style={"color": "#e74c3c"}),
                                    html.H6("Chat direct", className="fw-bold"),
                                    html.P("Réponse < 3 min", className="small text-success mb-1"),
                                    dbc.Button("Démarrer", color="danger", size="sm",
                                              style={"borderRadius": "20px"})
                                ], className="text-center")
                            ], md=3)
                        ])
                    ], className="p-4")
                ], className="shadow border-0",
                   style={"borderRadius": "15px", 
                          "background": "linear-gradient(145deg, #ffffff, #f8f9fa)"})
            ], width=12, className="mb-5")
        ]),
        
        # Section Ressources & Liens utiles
        dbc.Row([
            dbc.Col([
                html.H4("📚 Ressources Complémentaires", 
                       className="mb-4 fw-bold",
                       style={"color": "#2c3e50", "borderLeft": "5px solid #4a6fa5", 
                              "paddingLeft": "15px"})
            ], width=12, className="mb-3")
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Div([
                            html.I(className="fas fa-file-pdf me-3", style={"color": "#e74c3c"}),
                            html.Span("Guide utilisateur complet - Version 3.2", 
                                     className="fw-medium"),
                            html.Small(" • 15 pages", className="text-muted ms-2"),
                            html.Span("PDF", className="badge bg-danger ms-2", style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="border-0 py-3", style={"backgroundColor": "transparent"}),
                    
                    dbc.ListGroupItem([
                        html.Div([
                            html.I(className="fas fa-file-excel me-3", style={"color": "#2ecc71"}),
                            html.Span("Fichier exemple - Données météo complètes", 
                                     className="fw-medium"),
                            html.Small(" • 5 colonnes, 1095 lignes", className="text-muted ms-2"),
                            html.Span("XLSX", className="badge bg-success ms-2", style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="border-0 py-3", style={"backgroundColor": "transparent"}),
                    
                    dbc.ListGroupItem([
                        html.Div([
                            html.I(className="fas fa-video me-3", style={"color": "#3498db"}),
                            html.Span("Webinaire: Introduction à la modélisation hydrologique", 
                                     className="fw-medium"),
                            html.Small(" • 45 min", className="text-muted ms-2"),
                            html.Span("Replay", className="badge bg-info ms-2", style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="border-0 py-3", style={"backgroundColor": "transparent"}),
                    
                    dbc.ListGroupItem([
                        html.Div([
                            html.I(className="fas fa-graduation-cap me-3", style={"color": "#9b59b6"}),
                            html.Span("Formation certifiante - Hydrologie opérationnelle", 
                                     className="fw-medium"),
                            html.Small(" • 4 modules", className="text-muted ms-2"),
                            html.Span("Inscription", className="badge bg-warning ms-2", style={"fontSize": "10px"})
                        ], className="d-flex align-items-center")
                    ], className="border-0 py-3", style={"backgroundColor": "transparent"})
                ], flush=True)
            ], width=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("📊 Statistiques d'utilisation", className="mb-3 fw-bold"),
                        html.Div([
                            html.Div([
                                html.Span("Sessions actives", className="text-muted small d-block"),
                                html.H3("247", className="fw-bold", style={"color": "#2c3e50"})
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Tickets résolus", className="text-muted small d-block"),
                                html.H3("1,284", className="fw-bold", style={"color": "#2c3e50"})
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Satisfaction", className="text-muted small d-block"),
                                html.H3("98%", className="fw-bold", style={"color": "#2ecc71"})
                            ])
                        ])
                    ], className="p-4")
                ], className="shadow-sm border-0 h-100", style={"borderRadius": "15px"})
            ], width=4)
        ], className="mb-5"),
        
        # Pied de page
        dbc.Row([
            dbc.Col([
                html.Hr(className="my-4", style={"borderColor": "#eaeaea"}),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-code-branch me-2", style={"color": "#95a5a6"}),
                        html.Span(f"Version {settings.APP_VERSION} • ", 
                                 className="text-muted small"),
                        html.Span("Dernière mise à jour: ", className="text-muted small"),
                        html.Span(datetime.now().strftime("%d/%m/%Y"), 
                                 className="text-muted small fw-bold"),
                    ], className="d-flex align-items-center justify-content-center")
                ])
            ])
        ]),
        
        # Stockage pour recherche (optionnel)
        dcc.Store(id="help-search-store"),
        
        # Styles CSS intégrés pour les vagues
        html.Div([
            html.Link(
                rel="stylesheet",
                href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
            )
        ]),
        
    ], fluid=False, className="py-3", 
       style={'backgroundColor': '#f8f9fa', "marginLeft": "210px", "minHeight": "100vh"})