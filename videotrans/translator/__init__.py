# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, List

# Numbers represent display order
from videotrans.configure.config import tr,settings,params,app_cfg,logger,ROOT_DIR
from videotrans.util import tools




GOOGLE_INDEX = 0
MICROSOFT_INDEX = 1
M2M100_INDEX=2


CHATGPT_INDEX = 3
DEEPSEEK_INDEX = 4
GEMINI_INDEX = 5
ZHIPUAI_INDEX = 6
AZUREGPT_INDEX = 7
LOCALLLM_INDEX = 8


OPENROUTER_INDEX = 9
SILICONFLOW_INDEX = 10
AI302_INDEX = 11

QWENMT_INDEX = 12
ZIJIE_INDEX = 13


TENCENT_INDEX = 14
BAIDU_INDEX = 15
DEEPL_INDEX = 16
DEEPLX_INDEX = 17
ALI_INDEX = 18

OTT_INDEX = 19
LIBRE_INDEX = 20

MyMemoryAPI_INDEX = 21
TRANSAPI_INDEX = 22
MINIMAX_INDEX = 23
CAMB_INDEX = 24



# AI translation channel to facilitate judgment
AI_TRANS_CHANNELS=[
    CHATGPT_INDEX,
    LOCALLLM_INDEX,
    ZIJIE_INDEX,
    AZUREGPT_INDEX,
    GEMINI_INDEX,
    QWENMT_INDEX,
    AI302_INDEX,
    ZHIPUAI_INDEX,
    SILICONFLOW_INDEX,
    DEEPSEEK_INDEX,
    OPENROUTER_INDEX,
    MINIMAX_INDEX,
    CAMB_INDEX
]
# Translation channel name list, displayed in the interface
_ID_NAME_DICT = {
    GOOGLE_INDEX:tr('Google'),
    MICROSOFT_INDEX:tr('Microsoft'),
    M2M100_INDEX:f'M2M100({tr("Local")})',
    
    
    CHATGPT_INDEX:tr('OpenAI ChatGPT'),
    DEEPSEEK_INDEX:"DeepSeek",
    GEMINI_INDEX:"Gemini AI",
    ZHIPUAI_INDEX:tr('Zhipu AI'),
    AZUREGPT_INDEX:"AzureGPT AI",
    LOCALLLM_INDEX:tr('Local LLM'),
    
    OPENROUTER_INDEX:"OpenRouter",
    SILICONFLOW_INDEX:tr('SiliconFlow'),
    AI302_INDEX:"302.AI",
    
    QWENMT_INDEX:tr('Ali-Bailian'),
    ZIJIE_INDEX:tr('VolcEngine LLM'),

    TENCENT_INDEX:tr('Tencent'),
    BAIDU_INDEX:tr('Baidu'),
    DEEPL_INDEX:"DeepL",
    DEEPLX_INDEX:"DeepLx",
    ALI_INDEX:tr('Alibaba Machine Translation'),

    OTT_INDEX:tr('OTT'),
    LIBRE_INDEX:tr('LibreTranslate'),
    MyMemoryAPI_INDEX:tr('MyMemoryAPI'),
    TRANSAPI_INDEX:tr('Customized API'),
    MINIMAX_INDEX:"MiniMax AI",
    CAMB_INDEX:"CAMB AI",
}
TRANSLASTE_NAME_LIST=list(_ID_NAME_DICT.values())

# subtitles language code https://zh.wikipedia.org/wiki/ISO_639-2%E4%BB%A3%E7%A0%81%E5%88%97%E8%A1%A8
#  https://www.loc.gov/standards/iso639-2/php/code_list.php
# Tencent Translation https://cloud.tencent.com/document/api/551/15619
# google translate https://translate.google.com/
# Baidu Translation https://fanyi.baidu.com/
# deepl  https://deepl.com/
# microsoft https://www.bing.com/translator?mkt=zh-CN
# Ali Machine Translation https://help.aliyun.com/zh/machine-translation/developer-reference/machine-translation-language-code-list?spm=a2c4g.11186623.help-menu-30396.d_4_4.4bda2b009oye8y
# qwen-mt https://help.aliyun.com/zh/model-studio/machine-translation?spm=5176.30275541.J_ZGek9Blx07Hclc3Ddt9dg.1.69bf2f3dfuEVHs&scm=20140722.S_help@@%E6%96%87%E6%A1%A3@@2860790._.ID_help@@%E6%96%87%E6%A1%A3@@2860790-RL_qwen~DAS~mt-LOC_2024SPHelpResult-OR_ser-PAR1_0bc3b4ad17766086921897050e02b4-V_4-PAR3_o-RE_new5-P0_0-P1_0#038d2865bbydc
# m2m100  https://github.com/ymoslem/DesktopTranslator/blob/main/utils/m2m_languages.json
LANGNAME_DICT = {
    "en": tr("English"),
    "zh-cn": tr("Simplified Chinese"),
    "zh-tw": tr("Traditional Chinese"),
    "fr": tr("French"),
    "de": tr("German"),
    "ja": tr("Japanese"),
    "ko": tr("Korean"),
    "ru": tr("Russian"),
    "es": tr("Spanish"),
    "th": tr("Thai"),
    "it": tr("Italian"),
    "el": tr("Greek"),
    "pt": tr("Portuguese"),
    "vi": tr("Vietnamese"),
    "ar": tr("Arabic"),
    "tr": tr("Turkish"),
    "hi": tr("Hindi"),
    "hu": tr("Hungarian"),
    "uk": tr("Ukrainian"),
    "id": tr("Indonesian"),
    "ms": tr("Malay"),
    "kk": tr("Kazakh"),
    "cs": tr("Czech"),
    "pl": tr("Polish"),
    "nl": tr("Dutch"),
    "sv": tr("Swedish"),
    "he": tr("Hebrew"),
    "bn": tr("Bengali"),
    "fa": tr("Persian"),
    "fil": tr("Filipino"),
    "ur": tr("Urdu"),
    "nb": tr("Norway"),# written norwegian
    "yue": tr("Cantonese")
}

# If there is a new
try:
    if Path(ROOT_DIR+f'/videotrans/newlang.txt').exists():
        _new_lang=Path(ROOT_DIR+f'/videotrans/newlang.txt').read_text().strip().split("\n")
        for nl in _new_lang:
            LANGNAME_DICT[nl]=nl
except Exception as e:
    logger.exception(f'An error occurred while reading the custom new language code newlang.txt{e}', exc_info=True)
# Reversely search for the language code by display name
LANGNAME_DICT_REV={v:k for k,v in LANGNAME_DICT.items()}
# Find the code list corresponding to each translation channel based on the language code
LANG_CODE = {
    "zh-cn": [
        "zh-cn",  #googlechannel
        "chi",  # Subtitle embedding language
        "zh",  # Baidu channel
        "ZH-HANS",  # deepl deeplx channel
        "zh",  # Tencent Channel
        "zh",  #OTTchannel
        "zh-Hans",  #MicrosoftTranslator
        "Simplified Chinese",  #AITranslation
        "zh",  # ali
        "Chinese", # qwen-mt qwen-tts qwen-asr
        "zh" # m2m100
    ],
    "zh-tw": [
        "zh-tw",
        "chi",
        "cht",
        "ZH-HANT",
        "zh-TW",
        "zt",
        "zh-Hant",
        "Traditional Chinese",
        "zh-tw",
        "Traditional Chinese",
        "zh" # m2m100
    ],
    "ur": [
        "ur",  #googlechannel
        "urd",  # Subtitle embedding language
        "ur",  # Baidu channel
        "No",  # deepl deeplx channel
        "No",  # Tencent Channel
        "No",  #OTTchannel
        "ur",  #MicrosoftTranslator
        "Urdu",  #AITranslation
        "ur",  # ali
        "Urdu",
        "ur" # m2m100
    ],
    "yue": [
        "yue",  #googlechannel
        "chi",  # Subtitle embedding language
        "yue",  # Baidu channel
        "No",  # deepl deeplx channel
        "No",  # Tencent Channel
        "No",  #OTTchannel
        "yue",  #MicrosoftTranslator
        "Cantonese",  #AITranslation
        "yue",  # ali
        "Cantonese",
        "zh" # m2m100
    ],

    "fil": [
        "tl",  #googlechannel
        "fil",  # Subtitle embedding language
        "fil",  # Baidu channel
        "No",  # deepl deeplx channel
        "No",  # Tencent Channel
        "No",  #OTTchannel
        "fil",  #MicrosoftTranslator
        "Filipino",  #AITranslation
        "fil",  # ali
        "Filipino",
        "No"
    ],
    

    
    "en": [
        "en",
        "eng",
        "en",
        "EN-US",
        "en",
        "en",
        "en",
        "English",
        "en",
        "English",
        "en" # m2m100
    ],
    "fr": [
        "fr",
        "fre",
        "fra",
        "FR",
        "fr",
        "fr",
        "fr",
        "French",
        "fr",
        "French",
        "fr" # m2m100
    ],
    "de": [
        "de",
        "ger",
        "de",
        "DE",
        "de",
        "de",
        "de",
        "German",
        "de",
        "German",
        "de" # m2m100
    ],
    "ja": [
        "ja",
        "jpn",
        "jp",
        "JA",
        "ja",
        "ja",
        "ja",
        "Japanese",
        "ja",
        "Japanese",
        "ja" # m2m100
    ],
    "ko": [
        "ko",
        "kor",
        "kor",
        "KO",
        "ko",
        "ko",
        "ko",
        "Korean",
        "ko",
        "Korean",
        "ko" # m2m100
    ],
    "ru": [
        "ru",
        "rus",
        "ru",
        "RU",
        "ru",
        "ru",
        "ru",
        "Russian",
        "ru",
        "Russian",
        "ru" # m2m100
    ],
    "es": [
        "es",
        "spa",
        "spa",
        "ES",
        "es",
        "es",
        "es",
        "Spanish",
        "es",
        "Spanish",
        "es" # m2m100
    ],
    "th": [
        "th",
        "tha",
        "th",
        "No",
        "th",
        "th",
        "th",
        "Thai",
        "th",
        "Thai",
        "th" # m2m100
    ],
    "it": [
        "it",
        "ita",
        "it",
        "IT",
        "it",
        "it",
        "it",
        "Italian",
        "it",
        "Italian",
        "it" # m2m100
    ],
    "el": [
        "el",          # google
        "gre",         # subtitle embed (ISO 639-2/B)
        "el",          # baidu
        "EL",          # deepl / deeplx
        "el",          # tencent
        "el",          # OTT
        "el",          # microsoft / bing
        "Greek",       # AI (LLM)
        "el",          # alibaba
        "Greek",       # qwen-mt / qwen-tts / qwen-asr
        "el"           # m2m100
    ],
    "nb": [
        "no",          # google
        "nob",         # subtitle embed (ISO 639-2/B)
        "nob",          # baidu
        "NB",          # deepl / deeplx
        "No",          # tencent is not supported
        "No",          # OTT not supported
        "nb",          # microsoft / bing
        "Norwegian Bokmål",       # AI (LLM) Written Norwegian
        "no",          # alibaba
        "Norwegian Bokmål",       # qwen-mt / qwen-tts / qwen-asr
        "no"           # m2m100
    ],
    "pt": [
        "pt",  # pt-PT
        "por",
        "pt",
        "PT-PT",
        "PT-PT",
        "pt",
        "pt",
        "Portuguese",
        "pt",
        "Portuguese",
        "pt" # m2m100
    ],
    "vi": [
        "vi",
        "vie",
        "vie",
        "vi",
        "vi",
        "vi",
        "vi",
        "Vietnamese",
        "vi",
        "Vietnamese",
        "vi" # m2m100
    ],
    "ar": [
        "ar",
        "are",
        "ara",
        "AR",
        "ar",
        "ar",
        "ar",
        "Arabic",
        "ar",
        "Arabic",
        "ar" # m2m100
    ],
    "tr": [
        "tr",
        "tur",
        "tr",
        "TR",
        "tr",
        "tr",
        "tr",
        "Turkish",
        "tr",
        "Turkish",
        "tr" # m2m100
    ],
    "hi": [
        "hi",
        "hin",
        "hi",
        "No",
        "hi",
        "hi",
        "hi",
        "Hindi",
        "hi",
        "Hindi",
        "hi" # m2m100
    ],
    "hu": [
        "hu",
        "hun",
        "hu",
        "HU",
        "No",
        "hu",
        "hu",
        "Hungarian",
        "hu",
        "Hungarian",
        "hu" # m2m100
    ],
    "uk": [
        "uk",
        "ukr",
        "ukr",  #baidu
        "UK",  # deepl
        "No",  #tencent
        "uk",  # ott
        "uk",  #Microsoft
        "Ukrainian",
        "No",
        "Ukrainian",
        "uk" # m2m100
    ],
    "id": [
        "id",
        "ind",
        "id",
        "ID",
        "id",
        "id",
        "id",
        "Indonesian",
        "id",
        "Indonesian",
        "id" # m2m100
    ],
    "ms": [
        "ms",
        "may",
        "may",
        "No",
        "ms",
        "ms",
        "ms",
        "Malay",
        "ms",
        "Malay",
        "ms" # m2m100
    ],
    "kk": [
        "kk",
        "kaz",
        "No",
        "No",
        "No",
        "No",
        "kk",
        "Kazakh",
        "kk",
        "Kazakh",
        "kk" # m2m100
    ],
    "cs": [
        "cs",
        "ces",
        "cs",
        "CS",
        "No",
        "cs",
        "cs",
        "Czech",
        "cs",
        "Czech",
        "cs" # m2m100
    ],
    "pl": [
        "pl",
        "pol",
        "pl",
        "PL",
        "No",
        "pl",
        "pl",
        "Polish",
        "pl",
        "Polish",
        "pl" # m2m100
    ],
    "nl": [
        "nl",  #googlechannel
        "dut",  # Subtitle embedding language
        "nl",  # Baidu channel
        "NL",  # deepl deeplx channel
        "No",  # Tencent Channel
        "nl",  #OTTchannel
        "nl",  #MicrosoftTranslator
        "Dutch",  #AITranslation
        "nl",
        "Dutch",
        "nl" # m2m100
    ],
    "sv": [
        "sv",  #googlechannel
        "swe",  # Subtitle embedding language
        "swe",  # Baidu channel
        "SV",  # deepl deeplx channel
        "No",  # Tencent Channel
        "sv",  #OTTchannel
        "sv",  #MicrosoftTranslator
        "Swedish",  #AITranslation
        "sv",
        "Swedish",
        "sv" # m2m100
    ],
    "he": [
        "he",  #googlechannel
        "heb",  # Subtitle embedding language
        "heb",  # Baidu channel
        "HE",  # deepl deeplx channel
        "No",  # Tencent Channel
        "No",  #OTTchannel
        "he",  #MicrosoftTranslator
        "Hebrew",  #AITranslation
        "he",
        "Hebrew",
        "he" # m2m100
    ],
    "bn": [
        "bn",  #googlechannel
        "ben",  # Subtitle embedding language
        "ben",  # Baidu channel
        "No",  # deepl deeplx channel
        "No",  # Tencent Channel
        "No",  #OTTchannel
        "bn",  #MicrosoftTranslator
        "Bengali",  #AITranslation,
        "bn",
        "Bengali",
        "bn" # m2m100
    ],
    "fa": [
        "fa",  #googlechannel
        "per",  # Subtitle embedding language
        "per",  # Baidu channel
        "No",  # deepl deeplx channel
        "No",  # Tencent Channel
        "No",  #OTTchannel
        "fa",  #MicrosoftTranslator
        "Persian",  #AITranslation
        "fa",  # ali
        "Western Persian",
        "fa" # m2m100
    ],
    "auto": [
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto"
    ]
}


#According to the language name displayed on the interface, such as "Simplified Chinese, English", obtain the language code in the configuration file, such as zh-cn en, etc., if it is cli, it is the language code directly
def get_code(show_text=None):
    # - None means that if no language is selected, None is returned. The caller needs to judge based on the return result.
    # If not found in LANG CODE, return it as is
    if not show_text or show_text in ['-','No']:
        return None
    if show_text=='zh':
        return 'zh-cn'
    if show_text in LANG_CODE:
        return show_text
    return LANGNAME_DICT_REV.get(show_text,show_text)


#According to the displayed language and translation channel, obtain the source language code and target language code required by the translation channel
# translate_type translation channel index
# show_source The original language name displayed or - or language code
# show_target Displayed target language name or - or language code
# If it is an AI channel, return the natural language name of the language
# The newly added language code is returned directly
# - No is compatible with early irregular writing methods
def get_source_target_code(*, show_source=None, show_target=None, translate_type=None):
    source_list = None
    target_list = None

    if show_source and show_source not in ['-','No']:
        if show_source in LANG_CODE:# is the language code
            source_list = LANG_CODE[show_source] 
        elif LANGNAME_DICT_REV.get(show_source):# is the language display name
            source_list=LANG_CODE.get(LANGNAME_DICT_REV.get(show_source))
        elif show_source=='zh':#Special compatible zh
            source_list=LANG_CODE['zh-cn']

    if show_target and show_target not in ['-','No']:
        if show_target in LANG_CODE:# is the language code
            target_list = LANG_CODE[show_target] 
        elif LANGNAME_DICT_REV.get(show_target):#languagename
            target_list=LANG_CODE.get(LANGNAME_DICT_REV.get(show_target))
        elif show_target=='zh':
            #Specially compatible with zh
            target_list=LANG_CODE['zh-cn']

    # None found, it may be a new language code
    if not source_list and not target_list:
        return show_source,show_target#Return to original input

    # If no channel is set, use Google
    if not translate_type or translate_type in [GOOGLE_INDEX,MyMemoryAPI_INDEX, TRANSAPI_INDEX,CAMB_INDEX]:
        return source_list[0] if source_list else show_source, target_list[0] if target_list else show_target

    # qwenmt translation channel language code
    if translate_type == QWENMT_INDEX:
        if params.get('qwenmt_model', 'qwen-mt-turbo').startswith('qwen-mt'):
            return 'auto',target_list[9] if target_list else show_target
        return source_list[7] if source_list else show_source, target_list[7] if target_list else show_target

    #AIchannel
    if translate_type in AI_TRANS_CHANNELS:
        return source_list[7] if source_list else show_source, target_list[7] if target_list else show_target

    if translate_type == BAIDU_INDEX:
        return source_list[2] if source_list else show_source, target_list[2] if target_list else show_target

    if translate_type in [DEEPLX_INDEX, DEEPL_INDEX]:
        return source_list[3] if source_list else show_source, target_list[3] if target_list else show_target

    if translate_type == TENCENT_INDEX:
        return source_list[4] if source_list else show_source, target_list[4] if target_list else show_target

    if translate_type in [OTT_INDEX, LIBRE_INDEX]:
        return source_list[5] if source_list else show_source, target_list[5] if target_list else show_target
    if translate_type == MICROSOFT_INDEX:
        return source_list[6] if source_list else show_source, target_list[6] if target_list else show_target
    if translate_type == ALI_INDEX:
        return source_list[8] if source_list else show_source, target_list[8] if target_list else show_target
    if translate_type == M2M100_INDEX:
        return source_list[10] if source_list else show_source, target_list[10] if target_list else show_target
    return show_source,show_target

# Return the language name required by qwen-mt qwen-tts qwen-asr separately
def get_language_qwen(langcode=None):
    if not langcode:
        return None
    if langcode=='zh':
        langcode='zh-cn'
    _lang_list=LANG_CODE.get(langcode)
    if not _lang_list:
        return langcode
    return _lang_list[9]


# Determine whether the current translation channel and target language allow translation
# For example, deepl does not allow translation to certain target languages, whether to fill in the api key for certain channels, etc.
# translate_type translation channel
# show_target The target language name displayed after translation
# only_key=True only detects key and api, not the target language
def is_allow_translate(*, translate_type=None, show_target=None, only_key=False,  return_str=False):
    if not translate_type:
        return True
    if translate_type in [GOOGLE_INDEX, MyMemoryAPI_INDEX, MICROSOFT_INDEX]:
        return True

    if translate_type == CHATGPT_INDEX and not params.get('chatgpt_key',''):
        if return_str:
            return "Please configure the api and key information of the OpenAI ChatGPT channel first."
        return False
    if translate_type == ZHIPUAI_INDEX and not params.get('zhipu_key',''):
        if return_str:
            return 'Please fill in the api key of Zhipu AI in the menu-Zhipu AI'
        return False
    if translate_type == DEEPSEEK_INDEX and not params.get('deepseek_key',''):
        if return_str:
            return 'Please fill in the api key in the menu-DeepSeek'
        return False
    if translate_type == OPENROUTER_INDEX and not params.get('openrouter_key',''):
        if return_str:
            return 'Please fill in the api key in the menu-OpenRouter'
        return False

    if translate_type == SILICONFLOW_INDEX and not params.get('guiji_key',''):
        if return_str:
            return 'Please fill in the api key of silicon-based flow in the menu - silicon-based flow'
        return False
    if translate_type == AI302_INDEX and not params.get('ai302_key',''):
        if return_str:
            return "Please configure the api and key information of the 302.AI channel first."
        return False

    if translate_type == TRANSAPI_INDEX and not params.get('trans_api_url',''):
        if return_str:
            return "Please configure the api and key information of the Trans_API channel first."
        return False

    if translate_type == LOCALLLM_INDEX and not params.get('localllm_api',''):
        if return_str:
            return "Please configure the api and key information of the LocalLLM channel first."
        return False
    if translate_type == ZIJIE_INDEX and (
            not params.get('zijiehuoshan_model','').strip() or not params.get('zijiehuoshan_key','').strip()):
        if return_str:
            return "Please configure the api and key information of the ZiJie channel first."
        return False

    if translate_type == GEMINI_INDEX and not params.get('gemini_key',''):
        if return_str:
            return "Please configure the api and key information of the Gemini channel first."
        return False
    if translate_type == QWENMT_INDEX and not params.get('qwenmt_key',''):
        if return_str:
            return "Please configure the api and key information of the QwenMT channel first."
        return False
    if translate_type == AZUREGPT_INDEX and (
            not params.get('azure_key','') or not params.get('azure_api','')):
        if return_str:
            return "Please configure the api and key information of the Azure GPT channel first."
        return False

    if translate_type == BAIDU_INDEX and (
            not params.get("baidu_appid",'') or not params.get("baidu_miyue",'')):
        if return_str:
            return "Please configure the api and key information of the Baidu channel first."
        return False
    if translate_type == TENCENT_INDEX and (
            not params.get("tencent_SecretId",'') or not params.get("tencent_SecretKey",'')):
        if return_str:
            return "Please configure the appid and key information of the Tencent channel first."
        return False
    if translate_type == ALI_INDEX and (
            not params.get("ali_id",'') or not params.get("ali_key",'')):
        if return_str:
            return "Please configure the appid and key information of the Alibaba translate channel first."
        return False
    if translate_type == DEEPL_INDEX and not params.get("deepl_authkey",''):
        if return_str:
            return "Please configure the api and key information of the DeepL channel first."
        return False
    if translate_type == DEEPLX_INDEX and not params.get("deeplx_address",''):
        if return_str:
            return "Please configure the api and key information of the DeepLx channel first."
        return False
    if translate_type == LIBRE_INDEX and not params.get("libre_address",''):
        if return_str:
            return "Please configure the api and key information of the LibreTranslate channel first."
        return False

    if translate_type == MINIMAX_INDEX and not params.get('minimax_key',''):
        if return_str:
            return "Please configure the api and key information of the MiniMax channel first."
        return False
    if translate_type == CAMB_INDEX and not params.get('camb_api_key',''):
        if return_str:
            return "Please configure the API key information of the CAMB AI channel first."
        return False
    if translate_type == TRANSAPI_INDEX and not params.get("trans_api_url",''):
        if return_str:
            return "Please configure the api and key information of the TransAPI channel first."
        return False
    if translate_type == OTT_INDEX and not params.get("ott_address",''):
        if return_str:
            return "Please configure the api and key information of the OTT channel first."
        return False
    # If you only need to determine whether the api key and other information have been filled in, return here.
    if only_key:
        return True
    # Then check whether it is No, that is, it is not supported.
    index = 0
    if translate_type == BAIDU_INDEX:
        index = 2
    elif translate_type in [DEEPLX_INDEX, DEEPL_INDEX]:
        index = 3
    elif translate_type == TENCENT_INDEX:
        index = 4
    elif translate_type == MICROSOFT_INDEX:
        index = 6
    elif translate_type == ALI_INDEX:
        index = 8
    elif translate_type == M2M100_INDEX:
        index = 10

    if show_target:
        target_list=None
        if show_target in LANG_CODE:
            target_list = LANG_CODE[show_target]
        elif LANGNAME_DICT_REV.get(show_target):
            target_list=LANG_CODE.get(LANGNAME_DICT_REV.get(show_target))
        elif show_target=='zh':
            #Specially compatible with zh
            target_list=LANG_CODE['zh-cn']
        if target_list and target_list[index] == 'No':
            if return_str:
                return tr('deepl_nosupport') + f':{show_target}'
            tools.show_error(tr('deepl_nosupport') + f':{show_target}')
            return False
    return True


# Get the default language for speech recognition, such as English pronunciation or Chinese pronunciation
# Judge based on the original language, which is basically the same as Google, but only retains the part before _
def get_audio_code(*, show_source=None):
    if not show_source or show_source in ['auto','-']:
        return 'auto'
    source_list = LANG_CODE[show_source] if show_source in LANG_CODE else LANG_CODE.get(LANGNAME_DICT_REV.get(show_source))
    return source_list[0] if source_list else "auto"


# Get the 3-digit alphabetical language code of embedded soft subtitles, determined according to the target language
def get_subtitle_code(*, show_target=None):
    try:
        if show_target in LANG_CODE:
            return LANG_CODE[show_target][1]
        if show_target in LANGNAME_DICT_REV:
            return LANG_CODE[LANGNAME_DICT_REV[show_target]][1]
    except Exception as e:
        logger.error(f'Getting subtitle embed 3 as language code error:{e}')
    return 'eng'

def _check_google():
    import requests
    try:
        requests.head(f"https://translate.google.com",timeout=5)
    except Exception as e:
        logger.exception(f'Detection of google translation failed{e}', exc_info=True)
        return False
    
    return True
    


# For translation, first extract the target language code based on the translation channel and target language.
def run(*, translate_type=0,
        text_list=None,
        is_test=False,
        source_code=None,
        target_code=None,
        uuid=None) -> Union[List, str, None]:
    translate_type = int(translate_type)
    # Under the ai channel, target_language_name is the language name
    # Under other channels is the language code
    # source_code is the original language code
    target_language_name = target_code
    if translate_type in AI_TRANS_CHANNELS:
        # For AI channels, return natural language expressions in the target language
        _, target_language_name = get_source_target_code(show_target=target_code, translate_type=translate_type)
    kwargs = {
        "text_list": text_list,
        "target_language_name": target_language_name,
        "source_code": source_code if source_code and source_code not in ['-', 'No'] else 'auto',
        "target_code": target_code,
        "uuid": uuid,
        "is_test": is_test,
        "translate_type":translate_type
    }
    
    
    
    
    

    # If no proxy is set and Google fails to be detected, use Microsoft Translator.
    if translate_type == GOOGLE_INDEX:
        from videotrans.translator._google import Google
        if app_cfg.proxy or _check_google() is True:
            return Google(**kwargs).run()
        logger.warning('== No proxy is set and detection of google fails, use Microsoft Translator')
        translate_type = MICROSOFT_INDEX
        
    if translate_type == MyMemoryAPI_INDEX:
        from videotrans.translator._mymemory import MyMemory
        return MyMemory(**kwargs).run()
    if translate_type == QWENMT_INDEX:
        from videotrans.translator._qwenmt import QwenMT
        return QwenMT(**kwargs).run()

    if translate_type == MICROSOFT_INDEX:
        from videotrans.translator._microsoft import Microsoft
        return Microsoft(**kwargs).run()

    if translate_type == TENCENT_INDEX:
        from videotrans.translator._tencent import Tencent
        return Tencent(**kwargs).run()

    if translate_type == BAIDU_INDEX:
        from videotrans.translator._baidu import Baidu
        return Baidu(**kwargs).run()

    if translate_type == OTT_INDEX:
        from videotrans.translator._ott import OTT
        return OTT(**kwargs).run()

    if translate_type == TRANSAPI_INDEX:
        from videotrans.translator._transapi import TransAPI
        return TransAPI(**kwargs).run()

    if translate_type == DEEPL_INDEX:
        from videotrans.translator._deepl import DeepL
        return DeepL(**kwargs).run()

    if translate_type == DEEPLX_INDEX:
        from videotrans.translator._deeplx import DeepLX
        return DeepLX(**kwargs).run()

    if translate_type == AI302_INDEX:
        from videotrans.translator._ai302 import AI302
        return AI302(**kwargs).run()

    if translate_type == LOCALLLM_INDEX:
        from videotrans.translator._localllm import LocalLLM
        return LocalLLM(**kwargs).run()
    
    if translate_type == ZIJIE_INDEX:
        from videotrans.translator._huoshan import HuoShan
        return HuoShan(**kwargs).run()

    if translate_type == CHATGPT_INDEX:
        from videotrans.translator._chatgpt import ChatGPT
        return ChatGPT(**kwargs).run()
    if translate_type == ZHIPUAI_INDEX:
        from videotrans.translator._zhipuai import ZhipuAI
        return ZhipuAI(**kwargs).run()
    if translate_type == OPENROUTER_INDEX:
        from videotrans.translator._openrouter import OpenRouter
        return OpenRouter(**kwargs).run()
    if translate_type == DEEPSEEK_INDEX:
        from videotrans.translator._deepseek import DeepSeek
        return DeepSeek(**kwargs).run()

    if translate_type == SILICONFLOW_INDEX:
        from videotrans.translator._siliconflow import SILICONFLOW
        return SILICONFLOW(**kwargs).run()

    if translate_type == AZUREGPT_INDEX:
        from videotrans.translator._azure import AzureGPT
        return AzureGPT(**kwargs).run()



    if translate_type == GEMINI_INDEX:
        from videotrans.translator._gemini import Gemini
        return Gemini(**kwargs).run()
    if translate_type == LIBRE_INDEX:
        from videotrans.translator._libre import Libre    
        return Libre(**kwargs).run()
    if translate_type == ALI_INDEX:
        from videotrans.translator._ali import Ali
        return Ali(**kwargs).run()
    if translate_type == MINIMAX_INDEX:
        from videotrans.translator._minimax import MiniMax
        return MiniMax(**kwargs).run()
    if translate_type == M2M100_INDEX:
        from videotrans.translator._m2m100 import M2M100Trans
        return M2M100Trans(**kwargs).run()
    if translate_type == CAMB_INDEX:
        from videotrans.translator._camb import CambTranslator
        return CambTranslator(**kwargs).run()

    raise RuntimeError('No translation channels selected')
