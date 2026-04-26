from dataclasses import dataclass

# The video translation process uses all attributes
@dataclass
class TaskCfgBase:
    # General area
    is_cuda:bool=False#Whether to use cuda acceleration
    uuid:str=None #Default unique task id
    cache_folder: str=None  # Temporary folder of the current file, used to store temporary process files
    target_dir: str=None  # Output folder, target video output folder
    source_language: str=None  #Original language name or code
    source_language_code: str=None  # Original language code
    source_sub: str=None  # Absolute path of original subtitle file
    source_wav: str=None  # Original language audio, exists in the temporary folder
    source_wav_output: str=None  #Original language audio output, exists in the target folder
    target_language: str=None  # Target language name or code
    target_language_code: str=None  # Target language code
    target_sub: str=None  #Absolute path of target subtitle file
    target_wav: str=None  # Target language audio, exists in the temporary folder
    target_wav_output: str=None  # Target language audio output, exists in the target folder
    name:str=None # Normalized absolute path of the original file D:/XXX/1.MP4
    noextname: str=None  # Remove the original video name from the extension
    basename:str=None # noextname + ext name 1.mp4
    ext:str=None # extension mp4
    dirname:str=None # The directory where the original file is located D:/XXX
    shound_del_name:str=None # If moved after normalization, the absolute path of the temporary file that needs to be deleted

   

# Speech recognition
@dataclass
class TaskCfgSTT(TaskCfgBase):
    ####### Speech recognition related
    detect_language: str = None  # Subtitle detection language code
    recogn_type: int = None  # Speech recognition channel
    model_name: str = None  # Model name
    shibie_audio: str = None  # Convert to pcm_s16le 16k as audio file for speech recognition
    remove_noise: bool = False  # Whether to remove noise
    enable_diariz: bool = False  # Whether to perform speaker recognition
    nums_diariz: int = 0  # Whether to perform speaker recognition
    rephrase: int = 2  # 0 Default sentence segmentation is not processed 1=LLM re-segmentation 2=Automatic correction
    fix_punc: bool = False  # Whether to restore punctuation marks

#dubbing
@dataclass
class TaskCfgTTS(TaskCfgBase):
    ######## Dubbing related
    tts_type:int=None # speech synthesis channel
    volume: str="+0%"  # volume
    pitch: str="+0Hz"  # pitch
    voice_rate: str="+0%"  # speaking speed
    voice_role: str=None  #voice character
    voice_autorate:bool=False #Whether audio is automatically accelerated
    video_autorate:bool=False #Whether the video automatically slows down
    remove_silent_mid:bool=False # Whether to remove the gaps between subtitles
    align_sub_audio:bool=True # Whether to force subtitles and sounds to be aligned

# subtitle translation
@dataclass
class TaskCfgSTS(TaskCfgBase):
    ######## Subtitle translation related
    translate_type:int=None # Subtitle translation channel

#VideoTranslationAll
@dataclass
class TaskCfgVTT(TaskCfgSTT,TaskCfgTTS,TaskCfgSTS):
    ############## Unique to video translation
    subtitle_language: str=None  # Soft subtitle embedded language code, 3 digits
    app_mode: str="biaozhun"  # Working mode biaohzun tiqu
    subtitles: str=""  # Existing subtitle text, such as pre-imported
    targetdir_mp4: str=None  #Finally output the synthesized mp4
    novoice_mp4: str=None  # Silent video separated from original video
    is_separate: bool=False  # Whether to separate vocals and background sounds
    embed_bgm: bool=True  # Do you need to re-embed the background sound?
    instrument: str=None  # Isolated background audio
    vocal: str=None  # Isolated vocal audio
    back_audio: str=None  # Manually added original background sound audio
    clear_cache:bool=False # Whether to clean existing files
    background_music: str=None  # Manually added background audio, complete path after sorting
    subtitle_type: int=0  # Soft and hard subtitle embedding type 0=no embedding, 1=hard subtitle, 2=soft subtitle, 3=double hard, 4=double soft
    only_out_mp4:bool=False# Whether to only output mp4 and only use it for video translation
    recogn2pass:bool=False# Re-identify dubbing audio
    output_srt:int=0# Transcribe and translate. The mode output subtitles is similar, 0=single subtitles, 1=target language online double subtitles, 2=target language online double subtitles
    copysrt_rawvideo:bool=False# Whether to copy the generated subtitles to the video directory