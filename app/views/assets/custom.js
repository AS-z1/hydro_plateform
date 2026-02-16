/**
 * Scripts JavaScript personnalisés pour la plateforme hydrologique
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('zHydro Platform - Version 1.0.0');
    
    // Initialisation des tooltips Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Gestion des uploads
    initFileUploads();
    
    // Gestion des alertes automatiques
    initAutoDismissAlerts();
    
    // Amélioration des dropdowns
    enhanceDropdowns();
    
    // Personnalisation des graphiques Plotly
    customizePlotlyCharts();
});

/**
 * Initialise les zones d'upload de fichiers
 */
function initFileUploads() {
    const uploadAreas = document.querySelectorAll('.dash-upload, [id*="upload"]');
    
    uploadAreas.forEach(area => {
        area.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#3498db';
            this.style.backgroundColor = '#e8f4fd';
        });
        
        area.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#dee2e6';
            this.style.backgroundColor = '#f8f9fa';
        });
        
        area.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#27ae60';
            this.style.backgroundColor = '#d1f7c4';
            
            // Réinitialiser après 1 seconde
            setTimeout(() => {
                this.style.borderColor = '#dee2e6';
                this.style.backgroundColor = '#f8f9fa';
            }, 1000);
        });
    });
}

/**
 * Initialise la disparition automatique des alertes
 */
function initAutoDismissAlerts() {
    const autoDismissAlerts = document.querySelectorAll('.alert-dismissible[data-auto-dismiss]');
    
    autoDismissAlerts.forEach(alert => {
        const delay = alert.getAttribute('data-auto-dismiss') || '5000';
        
        setTimeout(() => {
            if (alert.parentNode) {
                alert.style.opacity = '0';
                alert.style.transition = 'opacity 0.5s';
                
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.parentNode.removeChild(alert);
                    }
                }, 500);
            }
        }, parseInt(delay));
    });
}

/**
 * Améliore les dropdowns
 */
function enhanceDropdowns() {
    const dropdowns = document.querySelectorAll('.dropdown');
    
    dropdowns.forEach(dropdown => {
        dropdown.addEventListener('show.bs.dropdown', function() {
            this.classList.add('show');
        });
        
        dropdown.addEventListener('hide.bs.dropdown', function() {
            this.classList.remove('show');
        });
    });
}

/**
 * Personnalise les graphiques Plotly
 */
function customizePlotlyCharts() {
    // Configuration globale Plotly
    if (window.Plotly) {
        Plotly.setPlotConfig({
            modeBarButtonsToRemove: ['sendDataToCloud', 'autoScale2d', 'resetScale2d'],
            displaylogo: false,
            responsive: true,
            displayModeBar: true,
            modeBarButtons: [['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toImage']]
        });
    }
    
    // Écouteur pour les nouveaux graphiques
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.classList && node.classList.contains('js-plotly-plot')) {
                        enhancePlotlyChart(node);
                    }
                });
            }
        });
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
}

/**
 * Améliore un graphique Plotly spécifique
 */
function enhancePlotlyChart(chartElement) {
    // Attendre que Plotly soit chargé
    setTimeout(() => {
        const plot = chartElement;
        if (plot && plot.layout) {
            // Appliquer des styles cohérents
            const update = {
                'font': { size: 10, family: 'Inter, sans-serif' },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': { t: 40, r: 20, b: 40, l: 60 }
            };
            
            Plotly.relayout(plot, update);
        }
    }, 100);
}

/**
 * Affiche une notification toast
 */
function showToast(message, type = 'info', duration = 3000) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        createToastContainer();
    }
    
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast align-items-center text-bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    document.getElementById('toast-container').appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast, {
        animation: true,
        autohide: true,
        delay: duration
    });
    
    bsToast.show();
    
    // Nettoyer après disparition
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

/**
 * Crée le conteneur pour les toasts
 */
function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'position-fixed bottom-0 end-0 p-3';
    container.style.zIndex = '9999';
    document.body.appendChild(container);
}

/**
 * Formate les nombres avec séparateurs
 */
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return 'N/A';
    
    return parseFloat(num).toLocaleString('fr-FR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/**
 * Télécharge des données au format CSV
 */
function downloadCSV(data, filename = 'data.csv') {
    const csvContent = typeof data === 'string' ? data : arrayToCSV(data);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

/**
 * Convertit un tableau en CSV
 */
function arrayToCSV(array) {
    if (!Array.isArray(array) || array.length === 0) return '';
    
    const headers = Object.keys(array[0]);
    const rows = array.map(row => 
        headers.map(header => {
            const cell = row[header];
            return typeof cell === 'string' && cell.includes(',') ? `"${cell}"` : cell;
        }).join(',')
    );
    
    return [headers.join(','), ...rows].join('\n');
}

/**
 * Gestion des erreurs Ajax
 */
function handleAjaxError(xhr, status, error) {
    let errorMessage = 'Une erreur est survenue';
    
    if (xhr.responseJSON && xhr.responseJSON.message) {
        errorMessage = xhr.responseJSON.message;
    } else if (error) {
        errorMessage = error;
    }
    
    showToast(errorMessage, 'danger');
    console.error('Erreur Ajax:', status, error, xhr);
}

/**
 * Met en surbrillance un élément
 */
function highlightElement(elementId, duration = 2000) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('highlight');
        
        setTimeout(() => {
            element.classList.remove('highlight');
        }, duration);
    }
}

// Styles CSS supplémentaires pour les fonctionnalités JS
const style = document.createElement('style');
style.textContent = `
    .highlight {
        animation: highlight-pulse 2s ease;
    }
    
    @keyframes highlight-pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }
    }
    
    .toast {
        font-size: 0.8125rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        backdrop-filter: blur(2px);
    }
    
    .spinner-large {
        width: 3rem;
        height: 3rem;
    }
`;
document.head.appendChild(style);