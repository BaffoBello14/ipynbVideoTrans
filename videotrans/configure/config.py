# -*- coding: utf-8 -*-
import datetime
import json
import logging
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from queue import Queue
from dataclasses import dataclass, field
from typing import Dict, Any, List


from videotrans.util.contants import (
    no_proxy, DEFAULT_GEMINI_MODEL, OPENAITTS_ROLES, ChatTTS_VOICE, Qwentts_Models,
    Whisper_Models, Zijiehuoshan_Model, Zhipuai_Model, Localllm_Model, Azure_Model,
    Chatgpt_Model, Openairecognapi_Model, Qpenaitts_Model, Qwenmt_Model, Ai302_Models,
    Whisper_cpp_models, Deepseek_Model, Openrouter_Model, Guiji_Model
)

IS_FROZEN = True if getattr(sys, 'frozen', False) else False
SYS_TMP = Path(tempfile.gettempdir()).as_posix()
#Program root directory
ROOT_DIR = Path(sys.executable).parent.as_posix() if IS_FROZEN else Path(__file__).parent.parent.parent.as_posix()
TEMP_ROOT = f'{ROOT_DIR}/tmp'
LOGS_DIR = f'{ROOT_DIR}/logs'
TEMP_DIR= f'{TEMP_ROOT}/_temp'

Path(f"{ROOT_DIR}/logs").mkdir(parents=True, exist_ok=True)
Path(f"{TEMP_ROOT}").mkdir(parents=True, exist_ok=True)

def _set_env():
    # Environment variable settings
    if IS_FROZEN:
        os.environ['TQDM_DISABLE'] = '1'
    os.environ['no_proxy'] = no_proxy
    os.environ['NO_PROXY'] = no_proxy
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CT2_VERBOSE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ['QT_API'] = 'pyside6'
    os.environ['SOFT_NAME'] = 'pyvideotrans'
    os.environ['MODELSCOPE_CACHE'] = ROOT_DIR + "/models"
    os.environ['HF_HOME'] = ROOT_DIR + "/models"
    os.environ['HF_HUB_CACHE'] = ROOT_DIR + "/models"
    os.environ['HF_TOKEN_PATH'] = ROOT_DIR + "/models/hf_token.txt"
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'true'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = "3600"
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    if sys.platform == 'win32' and IS_FROZEN:
        os.environ['PATH'] = f'{ROOT_DIR}/_internal/torch/lib;' + os.environ.get("PATH", "")
    os.environ['PATH'] = ROOT_DIR + os.pathsep + f'{ROOT_DIR}/ffmpeg' + os.pathsep + f'{ROOT_DIR}/ffmpeg/sox' + os.pathsep + os.environ.get(
        "PATH", "")

def _set_logs():
    # Log initialization
    logger = logging.getLogger('VideoTrans')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    _file_handler = logging.FileHandler(f'{ROOT_DIR}/logs/{datetime.datetime.now().strftime("%Y%m%d")}.log',
                                        encoding='utf-8')
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(formatter)
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(logging.WARNING)
    _console_handler.setFormatter(formatter)
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)

    logging.getLogger("transformers").setLevel(logging.DEBUG)
    logging.getLogger("filelock").setLevel(logging.DEBUG)
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
    return logger



@lru_cache(maxsize=None)
def _get_langjson_list():
    lang_dir = Path(f'{ROOT_DIR}/videotrans/language')
    _SUPPORT_LANG={}
    if lang_dir.exists():
        for it in lang_dir.glob('*.json'):
            if it.stat().st_size > 0:
                _SUPPORT_LANG[it.stem] = it.as_posix()
    return _SUPPORT_LANG

# The main process initializes the language and translation dictionary, using settings data
def _init_language():
    SUPPORT_LANG=_get_langjson_list()
    try:
        _defaulelang = os.environ.get('PYVIDEOTRANS_LANG', settings.lang)
        if not _defaulelang:
            import locale
            _defaulelang = locale.getdefaultlocale()[0][:2].lower() if locale.getdefaultlocale()[0] else 'en'
    except:
        _defaulelang = "en"

    if _defaulelang not in SUPPORT_LANG:
        _defaulelang = "en"
    if not settings.lang:
        settings.lang = _defaulelang
        settings.save()
    _transobj=_get_transobj(_defaulelang)
    return _defaulelang,_transobj

@lru_cache()
def _get_transobj(lang):
    SUPPORT_LANG=_get_langjson_list()
    try:
        _transobj = json.loads(Path(SUPPORT_LANG.get(lang)).read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError, TypeError):
        _transobj = None
    return _transobj

@dataclass
class AppCfg:
    '\n    Stores runtime properties (formerly global variables) that belong directly to config.py.\n    '
    # Graphics card related
    NVIDIA_GPU_NUMS: int = -1

    #Queue and control
    stoped_uuid_set: set = field(default_factory=set)
    uuid_logs_queue: Dict = field(default_factory=dict)
    global_msg: List = field(default_factory=list)
    exit_soft: bool = False

    # Window and UI
    child_forms: Dict = field(default_factory=dict)
    INFO_WIN: Dict = field(default_factory=lambda: {"data": {}, "win": None})

    #Task status
    queue_novice: Dict = field(default_factory=dict)
    current_status: str = "stop"
    task_countdown: int = 0

    # Thread queue
    prepare_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    regcon_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    diariz_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    trans_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    dubb_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    align_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    regcon2_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    assemb_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))
    taskdone_queue: Queue = field(default_factory=lambda: Queue(maxsize=0))

    #Run parameters
    exec_mode: str = "gui"
    video_codec: Any = None
    codec_cache: Dict = field(default_factory=dict)
    line_roles: Dict = field(default_factory=dict)
    onlyone_source_sub: Any = None
    onlyone_target_sub: Any = None
    onlyone_trans: bool = False
    dubbing_role: Dict = field(default_factory=dict)
    SUPPORT_LANG: Dict = field(default_factory=dict)
    proxy: str = ''

    def __post_init__(self):
        self.SUPPORT_LANG=_get_langjson_list()


@dataclass
class AppSettings:
    '\n    AppSettings: corresponds to cfg.json, including parse_init function\n    '
    homedir: str = ROOT_DIR + "/output"
    lang: str = ""
    # Note: Python attribute names cannot contain hyphens. They are replaced by underscores here. The mapping is processed during save/load.
    initial_prompt_zh_cn: str = ""
    initial_prompt_zh_tw: str = ""
    # Other properties are loaded dynamically, only some are listed here to get IDE prompts
    hf_token: str = ""
    proxy:str=''

    _json_path: str = f"{ROOT_DIR}/videotrans/cfg.json"
    WHISPER_MODEL_LIST:List=field(default_factory=list,repr=False)
    ChatTTS_voicelist:List=field(default_factory=list,repr=False)
    Whisper_CPP_MODEL_LIST:List=field(default_factory=list,repr=False)
    Whisper_NET_MODEL_LIST:List=field(default_factory=list,repr=False)

    def __post_init__(self):
        self.parse_init()

    def save(self):
        self._save_to_disk()


    def parse_init(self, update_data: Dict = None) -> Dict:
        'Corresponds to the original parse_init function, used to initialize or update configuration'
        default = self._get_defaults()

        # 1. If it is an update operation
        if update_data:
            self._apply_dict(update_data)
            self._save_to_disk()
            self.WHISPER_MODEL_LIST = re.split(r'[,，]', update_data.get('model_list', ''))
            return self.to_dict()

        # 2. If it is an initialization operation
        if not Path(self._json_path).exists():
            self._apply_dict(default)
            self._save_to_disk()
            self.WHISPER_MODEL_LIST = re.split(r'[,，]', default.get('model_list', ''))
            return default

        try:
            temp_json = json.loads(Path(self._json_path).read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            self._apply_dict(default)
            return default

        # Merge logic
        merged_settings = {}
        # Handle special hyphen keys
        hyphen_map = {"initial_prompt_zh-cn": "initial_prompt_zh_cn", "initial_prompt_zh-tw": "initial_prompt_zh_tw"}

        for key, val in temp_json.items():
            py_key = hyphen_map.get(key, key) # Convert to python attribute name

            # If it is not in the default value and is not a special key, skip
            if key not in default and py_key not in default:
                continue

            value = str(val).strip()
            if re.match(r'^\d+$', value):
                merged_settings[py_key] = int(value)
            elif re.match(r'^\d*\.\d+$', value):
                merged_settings[py_key] = float(value)
            elif value.lower() == 'true':
                merged_settings[py_key] = True
            elif value.lower() == 'false':
                merged_settings[py_key] = False
            elif value:
                merged_settings[py_key] = value

        #Extended model list processing
        _extend_models = ['localllm_model','zhipuai_model','deepseek_model','openrouter_model',
                          'guiji_model','zijiehuoshan_model','model_list','qwentts_models',
                          'gemini_model','chattts_voice']

        for m in _extend_models:
            # Get default value (as string)
            def_val = str(default.get(m, ''))
            # Get the value in the current json
            curr_val = str(merged_settings.get(m, def_val))

            _de = def_val.split(',')
            _cache = curr_val.split(',')
            _new = [str(it) for it in _cache if it and it not in _de]
            if _new:
                _de.extend(_new)
            merged_settings[m] = ",".join(_de)


        # update to self
        default.update(merged_settings) # Here default is the python attribute dictionary
        
        self.WHISPER_MODEL_LIST = re.split(r'[,，]', default.get('model_list', ''))
        self.ChatTTS_voicelist = re.split(r'[,，]', str(default.get('chattts_voice', '')))
        self.Whisper_CPP_MODEL_LIST = str(default.get('Whisper_cpp_models', 'ggml-tiny')).strip().split(',')
        self.Whisper_NET_MODEL_LIST = str(default.get('Whisper_net_models', 'ggml-tiny.bin')).strip().split(',')
        
        self._apply_dict(default)

        # Save and process hf_token
        self._save_to_disk()
        self._handle_hf_token()

        return self.to_dict()

        #Post-logic: Set environment variables according to settings (original code logic)

    def _get_defaults(self) -> Dict:
        return {
            "homedir": ROOT_DIR + "/output",
            "lang": "",
            "Faster_Whisper_XXL": "",
            "Whisper_cpp": "",
            "Whisper_cpp_models": Whisper_cpp_models,
            "Whisper_net_models": Whisper_cpp_models,
            "crf": 23,
            "edgetts_max_concurrent_tasks": 10,
            "edgetts_retry_nums": 3,
            "force_lib": False,
            "hw_decode":False,# ffmpeg attempts to hardware decode the video
            "preset": "medium",
            "ffmpeg_cmd": "",
            "aisendsrt": True,
            "dont_notify": False,
            "video_codec": 264,
            "noise_separate_nums": 4,
            "aitrans_temperature": 0.2,
            "aitrans_context": False,
            "batch_single": False,
            "ai302_models": Ai302_Models,
            'qwenmt_model': Qwenmt_Model,
            "openaitts_model": Qpenaitts_Model,
            "openairecognapi_model": Openairecognapi_Model,
            "chatgpt_model": Chatgpt_Model,
            "azure_model": Azure_Model,
            "localllm_model": Localllm_Model,
            "zhipuai_model": Zhipuai_Model,
            "deepseek_model": Deepseek_Model,
            "openrouter_model": Openrouter_Model,
            "guiji_model": Guiji_Model,
            "zijiehuoshan_model": Zijiehuoshan_Model,
            "model_list": Whisper_Models,
            "max_audio_speed_rate": 100,
            "max_video_pts_rate": 10,
            "threshold": 0.5,
            "min_speech_duration_ms": 1000,
            "max_speech_duration_s": 5,
            "min_silence_duration_ms": 140,
            "no_speech_threshold": 0.5,
            "whisper_prepare":False,
            "merge_short_sub": False,
            "vad_type": "silero",
            "trans_thread": 10,
            "aitrans_thread": 50,
            "translation_wait": 0,
            "dubbing_wait": 1,
            "dubbing_thread": 1,
            "normal_text":False,
            "remove_dubb_silence": True,
            "save_segment_audio": False,
            "countdown_sec": 30,
            "backaudio_volume": 0.8,
            "loop_backaudio": 1,
            "cuda_com_type": "default",
            "initial_prompt_zh-cn": "",  # Note: It will be mapped to _zh_cn in the object
            "initial_prompt_zh-tw": "",
            "initial_prompt_en": "",
            "initial_prompt_fr": "",
            "initial_prompt_de": "",
            "initial_prompt_ja": "",
            "initial_prompt_ko": "",
            "initial_prompt_ru": "",
            "initial_prompt_es": "",
            "initial_prompt_th": "",
            "initial_prompt_it": "",
            "initial_prompt_el": "",
            "initial_prompt_nb": "",
            "initial_prompt_pt": "",
            "initial_prompt_vi": "",
            "initial_prompt_ar": "",
            "initial_prompt_tr": "",
            "initial_prompt_hi": "",
            "initial_prompt_hu": "",
            "initial_prompt_uk": "",
            "initial_prompt_id": "",
            "initial_prompt_ms": "",
            "initial_prompt_kk": "",
            "initial_prompt_cs": "",
            "initial_prompt_pl": "",
            "initial_prompt_nl": "",
            "initial_prompt_sv": "",
            "initial_prompt_he": "",
            "initial_prompt_bn": "",
            "initial_prompt_fil": "",
            "initial_prompt_fa": "",
            "initial_prompt_ur": "",
            "initial_prompt_yue": "",
            "beam_size": 5,
            "best_of": 5,
            "uvr_models":"UVR-MDX-NET-Inst_HQ_4",
            "condition_on_previous_text": False,
            "temperature": "0.0",
            "repetition_penalty": 1.0,
            "compression_ratio_threshold": 2.2,
            "qwentts_role": '',
            "qwentts_models": Qwentts_Models,
            "show_more_settings": False,
            "speaker_type": "built",
            "hf_token": "",
            "cjk_len": 22,
            "other_len": 46,
            "gemini_model": DEFAULT_GEMINI_MODEL,
            "llm_chunk_size": 50,
            "llm_ai_type": "openai",
            "gemini_recogn_chunk": 50,
            "zh_hant_s": True,
            "process_max": 1,
            "process_max_gpu": 1,
            "multi_gpus": False,
            "azure_lines": 1,
            "chattts_voice": ChatTTS_VOICE,
            "proxy": ""
        }

    def _apply_dict(self, data: Dict):
        'Assign dictionary values to instance properties'
        for k, v in data.items():
            # Also handle hyphen mapping
            attr_name = k
            if k == "initial_prompt_zh-cn": attr_name = "initial_prompt_zh_cn"
            if k == "initial_prompt_zh-tw": attr_name = "initial_prompt_zh_tw"
            setattr(self, attr_name, v)

    def to_dict(self) -> Dict:
        'Convert to dictionary, handle hyphens'
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        if "initial_prompt_zh_cn" in data:
            data["initial_prompt_zh-cn"] = data.pop("initial_prompt_zh_cn")
        if "initial_prompt_zh_tw" in data:
            data["initial_prompt_zh-tw"] = data.pop("initial_prompt_zh_tw")
        return data

    def _save_to_disk(self):
        try:
            with open(self._json_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(self.to_dict(), ensure_ascii=False))
        except Exception as e:
            logger.exception(f'Failed to save settings locally:{e}',exc_info=True)

    def _handle_hf_token(self):
        p = Path(ROOT_DIR + "/models/hf_token.txt")
        if p.is_file():
            tk = p.read_text().strip()
            if tk:
                self.hf_token = tk
            else:
                p.unlink(missing_ok=True)
        if not p.is_file() and self.hf_token:
            p.write_text(self.hf_token)

    # Compatible with settings['key'] access
    def __getitem__(self, key):
        attr = key
        if key == "initial_prompt_zh-cn": attr = "initial_prompt_zh_cn"
        if key == "initial_prompt_zh-tw": attr = "initial_prompt_zh_tw"
        return getattr(self, attr)

    def __setitem__(self, key, value):
        attr = key
        if key == "initial_prompt_zh-cn": attr = "initial_prompt_zh_cn"
        if key == "initial_prompt_zh-tw": attr = "initial_prompt_zh_tw"
        setattr(self, attr, value)

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default


@dataclass
class AppParams:
    '\n    AppParams: corresponds to params.json, including getset_params function\n    '
    _json_path: str = f"{ROOT_DIR}/videotrans/params.json"

    def __post_init__(self):
        self.getset_params()

    def save(self):
        self._save_to_disk()


    def getset_params(self, update_data: Dict = None) -> Dict:
        'Corresponds to the original getset_params function'
        if update_data:
            self._apply_dict(update_data)
            self._save_to_disk()
            return self.to_dict()

        default = self._get_defaults()
        if Path(self._json_path).exists():
            try:
                loaded = json.loads(Path(self._json_path).read_text(encoding='utf-8'))
                default.update(loaded)
            except (OSError, json.JSONDecodeError):
                pass
        else:
            # First save
            self._apply_dict(default)
            self._save_to_disk()

        self._apply_dict(default)
        return self.to_dict()

    def _get_defaults(self):
        # The default dictionary in the original getset_params
        # Depends on the value in settings
        return {
            "last_opendir": os.path.expanduser("~"),
            "is_cuda": False,
            "line_roles": {},
            "rephrase": 0,
            "is_separate": False,
            "embed_bgm": True,
            "remove_noise": False,
            "enable_diariz": False,
            "nums_diariz": 0,
            "target_dir": "",
            "source_language": "en",
            "target_language": "zh-cn",
            "subtitle_language": "chi",
            "translate_type": 0,
            "subtitle_type": 1,
            "tts_type": 0,
            "model_name": "large-v3-turbo",
            "recogn_type": 0,
            "fix_punc": False,
            "stt_fix_punc": False,
            "voice_autorate": True,
            "video_autorate": False,
            "align_sub_audio": True,
            "voice_role": "No",
            "voice_rate": "0",
            "deepl_authkey": "",
            "deepl_api": "",
            "deepl_gid": "",
            "deeplx_address": "",
            "deeplx_key": "",
            "libre_address": "",
            "libre_key": "",
            "ott_address": "",
            "tencent_SecretId": "",
            "tencent_SecretKey": "",
            "tencent_termlist": "",
            "gcloud_credential_json": "",
            "gcloud_language_code": "",
            "gcloud_voice_name": "",
            "gcloud_audio_encoding": "",
            "gcloud_ssml_gender": "",
            "ali_id": "",
            "ali_key": "",
            "baidu_appid": "",
            "baidu_miyue": "",
            "chatgpt_api": "",
            "chatgpt_key": "",
            "chatgpt_max_token": "8192",
            "chatgpt_model": str(settings.get('chatgpt_model', '-')).strip().split(',')[0],
            "claude_api": "",
            "claude_key": "",
            "claude_model": str(settings.get('claude_model', '-')).strip().split(',')[0],
            "azure_api": "",
            "azure_key": "",
            "azure_version": "2025-04-01-preview",
            "azure_model": str(settings.get('azure_model', '-')).strip().split(',')[0],
            "gemini_key": "",
            "gemini_model": "gemini-2.5-flash",
            "gemini_maxtoken": 18192,
            "gemini_thinking_budget": 24576,
            "gemini_ttsstyle": "",
            "gemini_ttsmodel": "gemini-2.5-flash-preview-tts",
            "localllm_api": "",
            "localllm_key": "",
            "localllm_model": str(settings.get('localllm_model', '-')).strip().split(',')[0],
            "localllm_max_token": "4096",
            "zhipu_key": "",
            "zhipu_model": str(settings.get('zhipuai_model', '-')).strip().split(',')[0],
            "zhipu_max_token": "98304",
            "guiji_key": "",
            "guiji_model": str(settings.get('guiji_model', '-')).strip().split(',')[0],
            "guiji_max_token": "8192",
            "deepseek_key": "",
            "deepseek_model": str(settings.get('deepseek_model', '-')).strip().split(',')[0],
            "deepseek_max_token": "8192",
            "openrouter_key": "",
            "openrouter_model": str(settings.get('openrouter_model', '-')).strip().split(',')[0],
            "openrouter_max_token": "8192",
            "zijiehuoshan_key": "",
            "zijiehuoshan_model": str(settings.get('zijiehuoshan_model', '-')).strip().split(',')[0],
            "qwenmt_key": "",
            "qwenmt_domains": "",
            "qwenmt_model": "qwen-mt-turbo",
            "qwenmt_asr_model": "qwen3-asr-flash",
            "qwenttslocal_refaudio": "nverguo.wav#You say the four elements are empty, but why? Close your eyes tightly. If you open your eyes and look at me, I won't believe you. Your eyes are empty.",
            "qwenttslocal_url": "",
            "qwenttslocal_prompt": "",
            "ai302_key": "",
            "ai302_model": "",
            "ai302_model_recogn": "whisper-1",
            "whipserx_api": "http://127.0.0.1:9092",
            "trans_api_url": "",
            "trans_secret": "",
            "coquitts_role": "",
            "coquitts_key": "",
            "elevenlabstts_role": [],
            "elevenlabstts_key": "",
            "elevenlabstts_models": "eleven_flash_v2_5",
            "openaitts_api": "",
            "openaitts_key": "",
            "openaitts_model": "tts-1",
            "openaitts_role":"",
            
            "xaitts_key":"",
            "mitts_key":"",

            "openaitts_instructions": "",
            "qwentts_key": "",
            "qwentts_model": "qwen-tts-latest",
            "qwentts_role": "Chelsie",
            "kokoro_api": "",
            "openairecognapi_url": "",
            "openairecognapi_key": "",
            "openairecognapi_prompt": "",
            "openairecognapi_model": "whisper-1",
            "parakeet_address": "",
            "clone_api": "",
            "clone_voicelist": ["clone"],
            "moss_tts_url": "",
            "moss_tts_role": ["No", "clone"],
            "moss_tts_local_role": "",
            "moss_tts_preset_test_text": "",
            "zh_recogn_api": "",
            "recognapi_url": "",
            "recognapi_key": "",
            "stt_url": "",
            "stt_model": "large-v3-turbo",
            "sense_url": "",
            "ttsapi_url": "",
            "ttsapi_voice_role": "",
            "ttsapi_extra": "pyvideotrans",
            "ttsapi_language_boost": "auto",
            "ttsapi_emotion": "happy",
            "minimaxi_apikey": "",
            "minimaxi_emotion": "",
            "minimaxi_apiurl": "api.minimaxi.com",
            "minimaxi_model": "speech-02-turbo",
            
            "minimax_key":"",
            "minimax_model":"MiniMax-M2.7",
            "minimax_max_tokens":8192,
            "minimax_api":"https://api.minimaxi.com/v1",
            
            "ai302tts_key": "",
            "ai302tts_model": "",
            "ai302tts_role": OPENAITTS_ROLES,
            "azure_speech_region": "",
            "azure_speech_key": "",
            "chatterbox_url": "",
            "chatterbox_role": "nverguo.wav#You say the four elements are empty, but why? Close your eyes tightly. If you open your eyes and look at me, I won't believe you. Your eyes are empty.",
            "chatterbox_cfg_weight": 0.5,
            "chatterbox_exaggeration": 0.5,
            "gptsovits_url": "",
            "gptsovits_role": "",
            "gptsovits_isv2": True,
            "gptsovits_extra": "pyvideotrans",
            "cosyvoice_url": "",
            "cosyvoice_role": "nverguo.wav#You say the four elements are empty, but why? Close your eyes tightly. If you open your eyes and look at me, I won't believe you. Your eyes are empty.",
            "omnivoice_url": "",
            "omnivoice_role": "nverguo.wav#You say the four elements are empty, but why? Close your eyes tightly. If you open your eyes and look at me, I won't believe you. Your eyes are empty.",
            "fishtts_url": "",
            "fishtts_role": "nverguo.wav#You say the four elements are empty, but why? Close your eyes tightly. If you open your eyes and look at me, I won't believe you. Your eyes are empty.",
            "f5tts_url": "",
            "f5tts_model": "",
            "f5tts_ttstype": "F5-TTS",
            "f5tts_role": "nverguo.wav#You say the four elements are empty, but why? Close your eyes tightly. If you open your eyes and look at me, I won't believe you. Your eyes are empty.",
            "index_tts_version": 1,
            "f5tts_is_whisper": False,
            "indextts_url": "",
            "voxcpmtts_url": "",
            "voxcpmtts_version": "v2",
            "sparktts_url": "",
            "diatts_url": "",
            "doubao_appid": "",
            "doubao_access": "",
            "volcenginetts_appid": "",
            "volcenginetts_access": "",
            "volcenginetts_cluster": "",
            "doubao2_appid": "",
            "doubao2_access": "",
            "zijierecognmodel_appid": "",
            "zijierecognmodel_token": "",
            "chattts_api": "",
            "app_mode": "biaozhun",
            "stt_source_language": 0,
            "stt_recogn_type": 0,
            "stt_model_name": "large-v3-turbo",
            "stt_cuda": False,
            "stt_remove_noise": False,
            "stt_enable_diariz": False,
            "stt_rephrase": 0,
            "stt_nums_diariz": 0,
            "subtitlecover_outformat": "srt",
            "deepgram_apikey": "",
            "trans_translate_type": 0,
            "trans_source_language": 0,
            "trans_target_language": 1,
            "trans_out_format": 0,
            "dubb_source_language": 0,
            "dubb_tts_type": 0,
            "dubb_role": 0,
            "dubb_out_format": 0,
            "dubb_voice_autorate": False,
            "dubb_hecheng_rate": 0,
            "dubb_pitch_rate": 0,
            "dubb_volume_rate": 0,
            "recogn2pass": True
        }

    def _apply_dict(self, data: Dict):
        for k, v in data.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def _save_to_disk(self):
        try:
            with open(self._json_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(self.to_dict(), ensure_ascii=False))
        except Exception as e:
            logger.exception(f'Failed to save params to local:{e}',exc_info=True)
    # Compatible with params['key']
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

def tr(lang_key, *kw):
    global _transobj
    'translation function'
    if not _transobj:
        _transobj=_get_transobj(defaulelang)
    if not _transobj:
        return lang_key

    if isinstance(lang_key, list):
        str_list = [t for t in [_transobj.get(it) for it in lang_key] if t]
        return ",".join(str_list)
    lang = _transobj.get(lang_key)
    if not lang:
        return lang_key
    if not kw:
        return lang
    try:
        return lang.format(*kw)
    except IndexError:
        return lang


def push_queue(uuid, jsondata):
    'Compatible with old push_queue'
    if app_cfg.exit_soft or uuid in app_cfg.stoped_uuid_set:
        return
    if uuid not in app_cfg.uuid_logs_queue:
        app_cfg.uuid_logs_queue[uuid] = Queue(maxsize=0)
    try:
        if isinstance(app_cfg.uuid_logs_queue[uuid], Queue):
            app_cfg.uuid_logs_queue[uuid].put_nowait(jsondata)
    except Exception as e:
        logger.exception(f'push_queue error:{e}', exc_info=True)


def update_logging_level(new_level_str):
    'Dynamically modify log levels'
    new_level = getattr(logging, new_level_str.upper(), logging.INFO)
    _logger = logging.getLogger('VideoTrans')
    _logger.setLevel(new_level)
    for handler in _logger.handlers:
        if isinstance(handler, (logging.StreamHandler, logging.FileHandler)):
            handler.setLevel(new_level)
    print(f"The system log level has been dynamically switched to:{new_level_str}")

@lru_cache()
def __getattr__(name):
    'Implement config.xxx compatibility.\n    Search order:\n    1. Current module (handled by Python by default)\n    2. app_cfg (original global variable)\n    3. settings (key in the original settings dictionary)\n    4. params (key in the original params dictionary)'

    # 2. Try to get it from settings
    # Note: settings has many attributes. Use getattr here to avoid throwing errors.
    if name == 'settings':
        return settings
    if name == 'params':
        return params
    if name.startswith('settings.'):
        try:
            return getattr(settings, name)
        except AttributeError:
            pass
    if name.startswith('params.'):
        # 3. Try to get from params
        try:
            return getattr(params, name)
        except AttributeError:
            pass

    # 1. Try to get (original global variable) from AppCfg
    if hasattr(app_cfg, name):
        return getattr(app_cfg, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


_set_env()

logger=_set_logs()

app_cfg: AppCfg = AppCfg()
settings: AppSettings = AppSettings()
params: AppParams = AppParams()

HOME_DIR = settings.homedir  # Update global HOME_DIR
Path(HOME_DIR).mkdir(parents=True, exist_ok=True)

defaulelang,_transobj=_init_language()

_proxy = settings.proxy or os.environ.get('HTTPS_PROXY', '')
if _proxy:
    app_cfg.proxy=_proxy
    os.environ['HTTPS_PROXY'] = app_cfg.proxy
    os.environ['HTTP_PROXY'] = app_cfg.proxy


# Main process execution
def init_run():
    global TEMP_DIR
    TEMP_DIR = f'{TEMP_ROOT}/{os.getpid()}'
    Path(f"{TEMP_DIR}").mkdir(parents=True, exist_ok=True)
    # Directory creation
    Path(f'{TEMP_ROOT}/translate_cache').mkdir(exist_ok=True, parents=True)
    Path(f'{ROOT_DIR}/models').mkdir(exist_ok=True, parents=True)
    Path(f'{ROOT_DIR}/f5-tts').mkdir(exist_ok=True, parents=True)
    
