from typing import Union, List, Dict
from pathlib import Path
from videotrans.configure.config import tr,settings,params,app_cfg,logger,ROOT_DIR


FASTER_WHISPER = 0
OPENAI_WHISPER = 1
QWENASR = 2
FUNASR_CN = 3
HUGGINGFACE_ASR = 4

OPENAI_API = 5
GEMINI_SPEECH = 6
QWEN3ASR = 7
ZIJIE_RECOGN_MODEL=8

ZHIPU_API = 9


Deepgram = 10
DOUBAO_API = 11


PARAKEET = 12
Whisper_CPP=13
Faster_Whisper_XXL = 14
WHISPERX_API = 15

AI_302 = 16
ElevenLabs = 17


GOOGLE_SPEECH = 18


STT_API = 19
CUSTOM_API = 20
WHISPER_NET = 21
CAMB_ASR = 22

_ID_NAME_DICT = {
    FASTER_WHISPER:tr("Faster-whisper"),
    OPENAI_WHISPER:tr("OpenAI-whisper"),
    QWENASR:f'Qwen-ASR({tr("Local")})',
    FUNASR_CN:tr("FunASR-Chinese"),
    HUGGINGFACE_ASR:'Huggingface_ASR',
    
    OPENAI_API:tr("OpenAI Speech to Text"),
    GEMINI_SPEECH:tr("Gemini AI"),
    
    QWEN3ASR:tr("Ali Qwen3-ASR"),
    ZIJIE_RECOGN_MODEL:tr("VolcEngine STT"),
    ZHIPU_API:f'{tr("Zhipu AI")} GLM-ASR',
    
    Deepgram:"Deepgram.com",
    DOUBAO_API:tr("VolcEngine Subtitle API"),
    
    PARAKEET:"Parakeet-tdt",
    
    Whisper_CPP:"Whisper.cpp",
    Faster_Whisper_XXL:"Faster-Whisper-XXL.exe",
    WHISPERX_API:"WhisperX",

    AI_302:"302.AI",
    ElevenLabs:"ElevenLabs.io",
    GOOGLE_SPEECH:tr("Google Speech to Text"),
    STT_API:tr("STT Speech API"),
    CUSTOM_API:tr("Custom API"),
    WHISPER_NET:"Whisper.NET",
    CAMB_ASR:"CAMB AI",
}
RECOGN_NAME_LIST=list(_ID_NAME_DICT.values())

HUGGINGFACE_ASR_MODELS={
"nvidia/parakeet-ctc-1.1b":['en'],

# hub
"reazon-research/japanese-wav2vec2-large-rs35kh":['ja'],
# pipeline whisper
"kotoba-tech/kotoba-whisper-v2.0":['ja'],

# pipeline whisper
"biodatlab/whisper-th-large-v3":['th'],
"vinai/Phowhisper-large":['vi'],

"openai/whisper-large-v3":[],
# "openai/whisper-tiny":[],
# "Systran/faster-whisper-tiny":[]

}
try:
    if Path(f'{ROOT_DIR}/huggingface_models.txt').exists():
        for it in Path(f'{ROOT_DIR}/huggingface_models.txt').read_text(encoding='utf-8').strip().split("\n"):
            HUGGINGFACE_ASR_MODELS[it]=[]
except Exception as e:
    logger.waring(f'Adding custom Huggingface_ASR model failed:{e}')

# Determine whether the channel and model used support speech recognition in this language
#langcode=language code,recognin_type=recognition channel,model_name=model name
def is_allow_lang(langcode: str = None, recogn_type: int = None, model_name=None):
    # faster-whisper/openai-whisper supports all languages
    if recogn_type in [FASTER_WHISPER,OPENAI_WHISPER,WHISPERX_API,Faster_Whisper_XXL,Whisper_CPP,OPENAI_API,AI_302,GEMINI_SPEECH,WHISPER_NET]:
        return True
    # openai and Systran models in the huggingface_asr channel also support all languages
    if recogn_type == HUGGINGFACE_ASR and not HUGGINGFACE_ASR_MODELS.get(model_name):
        return True
    if recogn_type==HUGGINGFACE_ASR and HUGGINGFACE_ASR_MODELS.get(model_name):
        if langcode not in HUGGINGFACE_ASR_MODELS[model_name]:
            return tr("Only support")+tr(HUGGINGFACE_ASR_MODELS[model_name])
        return True
    if (langcode == 'auto' or not langcode) and recogn_type not in [FASTER_WHISPER, OPENAI_WHISPER, GEMINI_SPEECH, ElevenLabs,Faster_Whisper_XXL,Whisper_CPP,WHISPERX_API,AI_302,OPENAI_API,WHISPER_NET]:
        return tr("Recognition language is only supported in faster-whisper or openai-whisper or Gemini  modes.")

    return True


# Custom recognition, openai-api recognition, zh_recogn recognition whether relevant information and sk, etc. have been filled in
# Return True if correct, False if failed, and pop up window
def is_input_api(recogn_type: int = None, return_str=False):
    if recogn_type == STT_API and not params.get('stt_url',''):
        if return_str:
            return "Please configure the api and key information of the stt channel first."
        return False

    if recogn_type == PARAKEET and not params.get('parakeet_address',''):
        if return_str:
            return "Please configure the url address."
        return False
    if recogn_type == QWEN3ASR and not params.get('qwenmt_key',''):
        if return_str:
            return "Please configure the api key ."
        return False


    if recogn_type == CUSTOM_API and not params.get('recognapi_url',''):
        if return_str:
            return "Please configure the api and key information of the CUSTOM_API channel first."
        return False

    if recogn_type == OPENAI_API and not params.get('openairecognapi_key',''):
        if return_str:
            return "Please configure the api and key information of the OPENAI_API channel first."
        return False
    if recogn_type == DOUBAO_API and not params.get('doubao_appid',''):
        if return_str:
            return "Please configure the api and key information of the DOUBAO_API channel first."
        return False
    if recogn_type == ZIJIE_RECOGN_MODEL and not params.get('zijierecognmodel_appid',''):
        if return_str:
            return "Please configure the api and key information of the Volcengine channel first."
        return False
    if recogn_type == Deepgram and not params.get('deepgram_apikey',''):
        if return_str:
            return "Please configure the API Key information of the Deepgram channel first."
        return False
    if recogn_type == GEMINI_SPEECH and not params.get('gemini_key',''):
        if return_str:
            return "Please configure the API Key information of the Gemini channel first."
        return False
    if recogn_type == AI_302 and not params.get('ai302_key',''):
        if return_str:
            return "Please configure the API Key information of the Gemini channel first."
        ai302.openwin()
        return False
    # ElevenLabs
    if recogn_type == ElevenLabs and not params.get('elevenlabstts_key',''):
        if return_str:
            return "Please configure the API Key information of the ElevenLabs channel first."
        return False
    if recogn_type == ZHIPU_API and not params.get('zhipu_key',''):
        if return_str:
            return "Please configure the API Key information of the Zhipu AI channel first."
        return False
    if recogn_type == CAMB_ASR and not params.get('camb_api_key',''):
        if return_str:
            return "Please configure the API Key information of the CAMB AI channel first."
        return False
    return True


# Unified entrance
def run(*,
        detect_language=None,
        audio_file=None,
        cache_folder=None,
        model_name=None,
        uuid=None,
        recogn_type: int = 0,
        is_cuda=None,
        subtitle_type=0,
        max_speakers=-1, # -1 Do not enable speaker recognition, 0 = no limit on the number, >0 maximum number
        llm_post=False,
        recogn2pass=False#Second recognition of dubbing files to generate short subtitles

        ) -> Union[List[Dict], None]:

    if app_cfg.exit_soft or (uuid and uuid in app_cfg.stoped_uuid_set):
        return
    kwargs = {
        "detect_language": detect_language,
        "audio_file": audio_file,
        "cache_folder": cache_folder,
        "model_name": model_name,
        "uuid": uuid,
        "is_cuda": is_cuda,
        "subtitle_type": subtitle_type,
        "recogn_type":recogn_type,
        "max_speakers":max_speakers,
        "llm_post":llm_post,
        "recogn2pass":recogn2pass
    }
    logger.debug(f'[recognition]__init__:{kwargs=}')

    if recogn_type == GOOGLE_SPEECH:
        from videotrans.recognition._google import GoogleRecogn
        return GoogleRecogn(**kwargs).run()

    if recogn_type == DOUBAO_API:
        from videotrans.recognition._doubao import DoubaoRecogn
        return DoubaoRecogn(**kwargs).run()
    if recogn_type == ZIJIE_RECOGN_MODEL:
        from videotrans.recognition._zijiemodel import ZijieRecogn
        return ZijieRecogn(**kwargs).run()
    if recogn_type == CUSTOM_API:
        from videotrans.recognition._recognapi import APIRecogn
        return APIRecogn(**kwargs).run()
    if recogn_type == STT_API:
        from videotrans.recognition._stt import SttAPIRecogn
        return SttAPIRecogn(**kwargs).run()

    if recogn_type == OPENAI_API:
        from videotrans.recognition._openairecognapi import OpenaiAPIRecogn
        return OpenaiAPIRecogn(**kwargs).run()
    if recogn_type == WHISPERX_API:
        from videotrans.recognition._whisperx import WhisperXRecogn
        return WhisperXRecogn(**kwargs).run()
    if recogn_type == QWEN3ASR:
        from videotrans.recognition._qwen3asr import Qwen3ASRRecogn
        return Qwen3ASRRecogn(**kwargs).run()
    if recogn_type == QWENASR:
        from videotrans.recognition._qwenasrlocal import QwenasrlocalRecogn
        return QwenasrlocalRecogn(**kwargs).run()
    if recogn_type == FUNASR_CN:
        from videotrans.recognition._funasr import FunasrRecogn
        return FunasrRecogn(**kwargs).run()
    if recogn_type == Deepgram:
        from videotrans.recognition._deepgram import DeepgramRecogn
        return DeepgramRecogn(**kwargs).run()
    if recogn_type == GEMINI_SPEECH:
        from videotrans.recognition._gemini import GeminiRecogn
        return GeminiRecogn(**kwargs).run()
    if recogn_type == PARAKEET:
        from videotrans.recognition._parakeet import ParaketRecogn
        return ParaketRecogn(**kwargs).run()
    if recogn_type == AI_302:
        from videotrans.recognition._ai302 import AI302Recogn
        return AI302Recogn(**kwargs).run()
    if recogn_type == ElevenLabs:
        from videotrans.recognition._elevenlabs import ElevenLabsRecogn
        return ElevenLabsRecogn(**kwargs).run()
    if recogn_type == HUGGINGFACE_ASR:
        from videotrans.recognition._huggingface import HuggingfaceRecogn
        return HuggingfaceRecogn(**kwargs).run()
    if recogn_type == ZHIPU_API:
        from videotrans.recognition._glmasr import GLMASRRecogn
        return GLMASRRecogn(**kwargs).run()

    if recogn_type == WHISPER_NET:
        from videotrans.recognition._whispernet import WhisperNetRecogn
        return WhisperNetRecogn(**kwargs).run()

    if recogn_type == CAMB_ASR:
        from videotrans.recognition._camb import CambRecogn
        return CambRecogn(**kwargs).run()

    from videotrans.recognition._overall import FasterAll
    return FasterAll(**kwargs).run()