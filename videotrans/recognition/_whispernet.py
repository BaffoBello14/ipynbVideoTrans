#Whisper.net Identification
import json
import logging
import os
import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

from videotrans.configure import config
from videotrans.configure.config import ROOT_DIR, tr, app_cfg, settings, params, TEMP_DIR, logger, defaulelang
from videotrans.configure.whispernet_config import whispernet_config
from videotrans.recognition._base import BaseRecogn
from videotrans.util import tools

# Module level initialization state
_initialized = False
_clr = None

def _init_pythonnet():
    'Initialize pythonnet and CLR'
    global _initialized, _clr
    if _initialized:
        return _clr
    
    try:
        import pythonnet
        pythonnet.load("coreclr")
        import clr
        _clr = clr
        logger.info("Pythonnet and CLR initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pythonnet: {e}")
        _clr = None
    finally:
        _initialized = True
    
    return _clr


@dataclass
class WhisperNetRecogn(BaseRecogn):
    # Class-level DLL directory handle
    _dll_dir_handles: list = None
    
    def __post_init__(self):
        super().__post_init__()
        self.whisper_factory = None
        if WhisperNetRecogn._dll_dir_handles is None:
            WhisperNetRecogn._dll_dir_handles = []

    def _add_dll_search_dir(self, path: str) -> None:
        'Add DLL search directory'
        if not os.path.isdir(path):
            return
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(path)
            WhisperNetRecogn._dll_dir_handles.append(handle)

    def _preload_native_library(self, native_dir: str) -> None:
        'Preload native libraries'
        if not sys.platform.startswith("win"):
            return
        # Try to load according to priority
        for name in ("ggml-vulkan-whisper.dll", "ggml-cpu-whisper.dll", "ggml-whisper.dll", "whisper.dll"):
            lib_path = os.path.join(native_dir, name)
            if os.path.isfile(lib_path):
                try:
                    ctypes.WinDLL(lib_path)
                    logger.info(f"Loaded native library: {lib_path}")
                    return
                except OSError as exc:
                    logger.warning(f"Failed to load {lib_path}: {exc}")
        
        raise FileNotFoundError(f"No native whisper DLL found in {native_dir}")

    def _exec(self) -> Union[List[Dict], None]:
        if self._exit(): 
            return
        
        # Verify Whisper.NET settings
        is_valid, error_msg = whispernet_config.validate_setup()
        if not is_valid:
            raise RuntimeError(f"Whisper.NET setup invalid: {error_msg}")
        
        #Initialize pythonnet
        clr = _init_pythonnet()
        if clr is None:
            raise RuntimeError("Failed to initialize pythonnet")
        
        try:
            #Set dependency directory
            deps_dir = os.path.join(ROOT_DIR, "deps")
            whisper_dll = os.path.join(deps_dir, "Whisper.net.dll")
            native_dir = os.path.join(deps_dir, "native")
            
            if not os.path.isfile(whisper_dll):
                raise FileNotFoundError(f"Whisper.net.dll not found at {whisper_dll}")
            if not os.path.isdir(native_dir):
                raise FileNotFoundError(f"Native directory not found at {native_dir}")

            #Add DLL search path
            self._add_dll_search_dir(native_dir)
            
            # Preload native library
            self._preload_native_library(native_dir)
            
            # Add managed dependencies
            managed_deps = [
                os.path.join(deps_dir, "Microsoft.Extensions.AI.Abstractions.dll"),
                os.path.join(deps_dir, "Microsoft.Bcl.AsyncInterfaces.dll"),
                os.path.join(deps_dir, "System.Memory.dll"),
                os.path.join(deps_dir, "System.Buffers.dll"),
                os.path.join(deps_dir, "System.Runtime.CompilerServices.Unsafe.dll"),
                os.path.join(deps_dir, "System.Numerics.Vectors.dll"),
            ]
            for dll in managed_deps:
                if os.path.isfile(dll):
                    try:
                        clr.AddReference(dll)
                    except Exception as e:
                        logger.warning(f"Could not add managed dependency {dll}: {e}")
            
            # Add Whisper.net.dll reference
            clr.AddReference(whisper_dll)
            
            # Configure Whisper.NET runtime options - directly using class attributes
            try:
                from Whisper.net.LibraryLoader import RuntimeOptions, RuntimeLibrary
                from System.Collections.Generic import List
                
                # Set library path
                try:
                    RuntimeOptions.LibraryPath = native_dir
                    logger.info(f"RuntimeOptions.LibraryPath={RuntimeOptions.LibraryPath}")
                except Exception as e:
                    logger.warning(f"Failed to set LibraryPath: {e}")
                
                # Set the runtime order
                try:
                    order = List[RuntimeLibrary]()
                    order.Add(RuntimeLibrary.Vulkan)
                    order.Add(RuntimeLibrary.Cpu)
                    RuntimeOptions.RuntimeLibraryOrder = order
                except Exception as e:
                    logger.warning(f"Failed to set RuntimeLibraryOrder: {e}")
                
                # Force the loaded library to be set
                try:
                    RuntimeOptions.LoadedLibrary = RuntimeLibrary.Vulkan
                    logger.info(f"RuntimeOptions.LoadedLibrary={RuntimeOptions.LoadedLibrary}")
                except Exception:
                    try:
                        RuntimeOptions.LoadedLibrary = RuntimeLibrary.Cpu
                    except Exception as e:
                        logger.warning(f"Failed to set LoadedLibrary: {e}")
                


            except ImportError:
                logger.warning("Runtime options not available")
            except Exception as e:
                logger.warning(f"Could not configure runtime options: {e}")
            
            #Import Whisper.NET class
            from Whisper.net import WhisperFactoryOptions, WhisperFactory

            #Create factory options
            factory_options = WhisperFactoryOptions()
            # Note: Vulkan also requires UseGpu=True, not just CUDA
            # For AMD graphics cards, use the GPU via the Vulkan backend
            factory_options.UseGpu = True  # Force GPU to be enabled


            factory_options.GpuDevice = 0
            factory_options.DelayInitialization = False
            logger.info(f"WhisperFactoryOptions: UseGpu={factory_options.UseGpu}")

            # Find model
            model_path = whispernet_config.get_model_path(self.model_name)
            if model_path is None:
                if not self.model_name.endswith('.bin'):
                    model_path = whispernet_config.get_model_path(f"{self.model_name}.bin")
            if model_path is None:
                model_path = whispernet_config.get_model_path(f"ggml-{self.model_name}")
            if model_path is None:
                model_path = whispernet_config.get_model_path(f"ggml-{self.model_name}.bin")
            if model_path is None:
                raise FileNotFoundError(f"Model not found: {self.model_name}")
            
            model_path_str = str(model_path.absolute())
            logger.info(f"Using model: {model_path_str}")

            factory = WhisperFactory.FromPath(model_path_str, factory_options)
            
            # Record runtime information
            try:
                runtime_info = factory.GetRuntimeInfo()
                logger.info(f"Whisper.NET RuntimeInfo: {runtime_info}")
            except Exception as e:
                logger.warning(f"Could not get RuntimeInfo: {e}")
            #Create processor
            builder = factory.CreateBuilder()

            # Set language
            if self.detect_language and self.detect_language != 'auto':
                lang_code = self.detect_language.split('-')[0]
                builder = builder.WithLanguage(lang_code)
            else:
                builder = builder.WithLanguageDetection()

            #Set other parameters
            if settings.get('condition_on_previous_text', False):
                builder = builder.WithNoContext()
            builder = builder.WithNoSpeechThreshold(float(settings.get('no_speech_threshold', -0.8)))
            builder = builder.WithLogProbThreshold(float(settings.get('logprob_threshold', -1.0)))

            #Set progress callback
            segments = []
            
            def on_segment(segment):
                # Process TimeSpan type
                start = getattr(segment, "Start", None)
                if start is not None:
                    if hasattr(start, 'TotalSeconds'):
                        start = float(start.TotalSeconds)
                    else:
                        start = float(start)
                else:
                    start = 0.0
                
                end = getattr(segment, "End", None)
                if end is not None:
                    if hasattr(end, 'TotalSeconds'):
                        end = float(end.TotalSeconds)
                    else:
                        end = float(end)
                else:
                    end = 0.0
                    
                text = str(getattr(segment, "Text", "")).strip()
                if text:
                    segments.append({
                        "line": len(segments) + 1,
                        "start_time": int(start * 1000),
                        "end_time": int(end * 1000),
                        "text": text,
                        "time": tools.ms_to_time_string(ms=int(start * 1000)) + ' --> ' + tools.ms_to_time_string(ms=int(end * 1000)),
                    })
                    self._signal(text=f"Processing: {len(segments)}")

            #Add event handler
            try:
                from Whisper.net import OnSegmentEventHandler
                handler = OnSegmentEventHandler(on_segment)
            except:
                from System import Action
                from Whisper.net import SegmentData
                handler = Action[SegmentData](on_segment)
                
            builder = builder.WithSegmentEventHandler(handler)

            processor = builder.Build()

            # Process audio files
            from System.IO import File
            audio_stream = File.OpenRead(self.audio_file)
            try:
                processor.Process(audio_stream)
            finally:
                audio_stream.Dispose()

            # Clean up resources
            processor.Dispose()
            factory.Dispose()

            return segments

        except ImportError as e:
            error_msg = f"Whisper.net not properly installed: {str(e)}"
            logger.exception(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            logger.exception(f"Whisper.net recognition failed: {str(e)}")
            raise RuntimeError(f"Whisper.net recognition failed: {str(e)}")