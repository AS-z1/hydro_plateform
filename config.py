import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration de l'application"""
    
    # Application
    APP_NAME: str = "zHydro Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # FastAPI
    #API_HOST: str = "0.0.0.0"
    #API_PORT: int = 8000
    #API_PREFIX: str = "/api/v1"
    
    # Dash
    DASH_HOST: str = "0.0.0.0"
    DASH_PORT: int = 8050
    DASH_PREFIX: str = "/dash"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALLOWED_ORIGINS: list = ["*"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: list = [".csv", ".xlsx", ".xls", ".json", ".txt"]
    
    # Model Paths
    MODEL_DIR: Path = BASE_DIR / "models"
    LSTM_MODELS: Dict[str, str] = {
        "max": "lstm_qmax_q90.h5",
        "mean": "lstm_mean.h5"
    }
    
    # Theme
    THEME: str = "light"
    PRIMARY_COLOR: str = "#3498db"
    SECONDARY_COLOR: str = "#2ecc71"
    ACCENT_COLOR: str = "#e74c3c"
    
    # Font
    FONT_FAMILY: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    FONT_SIZE_BASE: str = "14px"
    FONT_SIZE_SMALL: str = "12px"
    FONT_SIZE_LARGE: str = "16px"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def create_directories(self):
        """Créer les répertoires nécessaires"""
        directories = [self.DATA_DIR, self.LOGS_DIR, self.TEMP_DIR, self.MODEL_DIR]
        for directory in directories:
            directory.mkdir(exist_ok=True)

settings = Settings()
settings.create_directories()