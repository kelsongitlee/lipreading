"""
Route blueprints for the lip reading web application
"""
from .upload import upload_bp
from .webcam import webcam_bp

__all__ = ['upload_bp', 'webcam_bp']
