"""
Model loading and initialization utilities
"""
import os
import sys
from pathlib import Path

class ModelLoader:
    """Handles loading and initialization of the lip reading model"""
    
    def __init__(self):
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the lip reading pipeline"""
        try:
            print("🚀 Initializing lip reading model...")
            
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            
            # Add project root to Python path
            sys.path.insert(0, str(project_root))
            
            # Import required modules
            from pipelines.pipeline import InferencePipeline
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            # Initialize Hydra config
            config_dir = str(project_root / "configs")
            config_name = "LRS3_V_WER19.1.ini"
            
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=config_name)
                self.pipeline = InferencePipeline(cfg)
            
            print("✅ Model initialized successfully!")
            
        except Exception as e:
            print(f"❌ Model initialization error: {e}")
            raise
    
    def get_pipeline(self):
        """Get the initialized pipeline"""
        return self.pipeline
    
    def is_ready(self):
        """Check if model is ready"""
        return self.pipeline is not None
