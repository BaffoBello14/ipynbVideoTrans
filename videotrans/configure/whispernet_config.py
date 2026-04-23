'\nWhisper.NET configuration management\n'

from pathlib import Path
from typing import Optional

from videotrans.configure.config import ROOT_DIR


class WhisperNetConfig:
    'Whisper.NET Configuration Class'
    
    def __init__(self):
        self.deps_dir = Path(ROOT_DIR) / "deps"
        self.native_dir = self.deps_dir / "native"
        self.models_dir = Path(ROOT_DIR) / "models"
        
        #Default settings
        self.default_use_gpu = True
        self.default_gpu_device = 0
        self.default_no_context = True
        self.default_no_speech_threshold = -0.8
        self.default_logprob_threshold = -1.0
        
    def get_runtime_files(self):
        'Get Whisper.NET runtime files'
        runtime_files = {}
        
        # Check main DLL files
        whisper_dll = self.deps_dir / "Whisper.net.dll"
        runtime_files['whisper_dll'] = whisper_dll if whisper_dll.exists() else None
        
        # Check native DLL files
        if self.native_dir.exists():
            native_dlls = []
            for dll_file in self.native_dir.glob("*.dll"):
                native_dlls.append(dll_file)
            runtime_files['native_dlls'] = native_dlls
        else:
            runtime_files['native_dlls'] = []
            
        return runtime_files
        
    def validate_setup(self) -> tuple[bool, str]:
        'Verify Whisper.NET setup is complete'
        errors = []
        
        # Check DLL files
        whisper_dll = self.deps_dir / "Whisper.net.dll"
        if not whisper_dll.exists():
            errors.append(f"Whisper.net.dll not found: {whisper_dll}")
            
        # Check the native directory
        if not self.native_dir.exists():
            errors.append(f"Native directory not found: {self.native_dir}")
        else:
            native_dlls = list(self.native_dir.glob("*.dll"))
            if not native_dlls:
                errors.append(f"No native DLLs found in: {self.native_dir}")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Setup is valid"
        
    def get_available_models(self) -> list[str]:
        'Get a list of available models'
        models = []
        if self.models_dir.exists():
            # Find model files in ggml format
            for model_file in self.models_dir.glob("*.bin"):
                if "ggml" in model_file.name.lower():
                    models.append(model_file.name)
        
        return sorted(models)
        
    def get_model_path(self, model_name: str) -> Optional[Path]:
        'Get the full path to the model file'
        # First check if it is an absolute path
        if Path(model_name).is_absolute() and Path(model_name).exists():
            return Path(model_name)
            
        # Then check the relative path
        model_path = self.models_dir / model_name
        if model_path.exists():
            return model_path
            
        # Check if only the file name is provided and search in the models directory
        for model_file in self.models_dir.rglob(model_name):
            if model_file.is_file():
                return model_file
                
        return None


#Create global configuration instance
whispernet_config = WhisperNetConfig()