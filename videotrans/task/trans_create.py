import copy, json, threading
import subprocess
import platform,glob,sys
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

from videotrans import translator
from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR
from videotrans.recognition import run as run_recogn, Faster_Whisper_XXL, Whisper_CPP, \
    is_allow_lang as recogn_allow_lang, FASTER_WHISPER
from videotrans.translator import run as run_trans, get_audio_code
from videotrans.tts import run as run_tts, EDGE_TTS, AZURE_TTS, SUPPORT_CLONE
from videotrans.task.simple_runnable_qt import run_in_threadpool
from videotrans.util import tools, contants
from ._base import BaseTask
from videotrans.util.help_ffmpeg import get_video_codec
from videotrans.task._rate import SpeedRate

@dataclass
class TransCreate(BaseTask):
    # Store original language subtitles
    source_srt_list: List = field(default_factory=list)
    # Store target language subtitles
    target_srt_list: List = field(default_factory=list)
    # The original video duration is updated to this point after slow processing and merging.
    video_time: float = 0.0
    # video information
    """
    {
        "video_fps":0,
        "video_codec_name":"h264",
        "audio_codec_name":"aac",
        "width":0,
        "height":0,
        "time":0
    }
    """
    video_info: Dict = field(default_factory=dict, repr=False)
    # Whether to perform c:v copy operation on the video
    is_copy_video: bool = False
    # MP4 encoding type to be output 264 265
    video_codec_num: int = 264
    # Whether to ignore audio and video alignment
    ignore_align: bool = False

    # Whether it is an audio translation task, if so, it will end when the dubbing is completed, no need to merge
    is_audio_trans: bool = False
    queue_tts: List = field(default_factory=list, repr=False)

    def __post_init__(self):
        # First, process the default configuration of this class
        super().__post_init__()
        if self.cfg.clear_cache and Path(self.cfg.target_dir).is_dir():
            shutil.rmtree(self.cfg.target_dir, ignore_errors=True)
        self._signal(text=tr('kaishichuli'))
        # -1=Do not enable speakers, 0=Enable and do not limit the number of speakers, >0+1 is the maximum number of speakers
        self.max_speakers = self.cfg.nums_diariz if self.cfg.enable_diariz else -1
        if self.max_speakers > 0:
            self.max_speakers += 1
        self.shoud_recogn = True
        # Output encoding, 264 or 265
        self.video_codec_num = int(settings.get('video_codec', 264))
        # Whether there is manually added background audio
        if tools.vail_file(self.cfg.back_audio):
            self.cfg.background_music = Path(self.cfg.back_audio).as_posix()

        # Temporary folder
        if not self.cfg.cache_folder:
            self.cfg.cache_folder = f"{TEMP_DIR}/{self.uuid}"
        # Output folder, remove possible double slashes
        self.cfg.target_dir = re.sub(r'/{2,}', '/', self.cfg.target_dir, flags=re.I | re.S)
        # Detect original language of subtitles
        self.cfg.detect_language = get_audio_code(show_source=self.cfg.source_language_code)

        # Store the separated silent mp4 to a temporary folder
        self.cfg.novoice_mp4 = f"{self.cfg.cache_folder}/novoice.mp4"

        # Original language subtitle file: output folder
        self.cfg.source_sub = f"{self.cfg.target_dir}/{self.cfg.source_language_code}.srt"
        #Original language audio files: output folder
        self.cfg.source_wav_output = f"{self.cfg.target_dir}/{self.cfg.source_language_code}.m4a"
        # Original language audio files: temporary folder
        self.cfg.source_wav = f"{self.cfg.cache_folder}/{self.cfg.source_language_code}.wav"

        # Target language subtitles: output folder
        self.cfg.target_sub = f"{self.cfg.target_dir}/{self.cfg.target_language_code}.srt"
        # Target audio file after dubbing: output folder
        self.cfg.target_wav_output = f"{self.cfg.target_dir}/{self.cfg.target_language_code}.m4a"
        # Target audio file after dubbing: temporary folder
        self.cfg.target_wav = f"{self.cfg.cache_folder}/target.wav"

        # The final mp4 video that needs to be output
        self.cfg.targetdir_mp4 = f"{self.cfg.target_dir}/{self.cfg.noextname}.mp4"

        # If the dubbing role is not No, dubbing is required
        if self.cfg.voice_role and self.cfg.voice_role != 'No' and self.cfg.target_language_code:
            self.shoud_dubbing = True

        # If it is not tiqu, both video and audio subtitles need to be merged
        if self.cfg.app_mode != 'tiqu' and (self.shoud_dubbing or self.cfg.subtitle_type > 0):
            self.shoud_hebing = True

        # Whether translation is required: if the target language code exists and is not equal to the original language, translation is required
        if self.cfg.target_language_code and self.cfg.target_language_code != self.cfg.source_language_code:
            self.shoud_trans = True

        # If the original language and target language are equal and a dubbing character exists, replace the dubbing
        if self.cfg.voice_role and self.cfg.voice_role != 'No' and self.cfg.source_language_code == self.cfg.target_language_code:
            self.cfg.target_wav_output = f"{self.cfg.target_dir}/{self.cfg.target_language_code}-dubbing.m4a"
            self.cfg.target_wav = f"{self.cfg.cache_folder}/target-dubbing.wav"
            self.shoud_dubbing = True

        # Determine if it is audio, it will end when the audio is generated, no need to merge, no need to separate the video, and no need to process the background sound
        if self.cfg.ext in contants.AUDIO_EXITS:
            self.is_audio_trans = True
            #self.cfg.is_separate = False
            self.shoud_hebing = False

        # No target language is set, no dubbing or translation
        if not self.cfg.target_language_code:
            self.shoud_dubbing = False
            self.shoud_trans = False

        if self.cfg.voice_role == 'No':
            self.shoud_dubbing = False

        if self.cfg.app_mode == 'tiqu':
            #self.cfg.is_separate = False
            self.cfg.enable_diariz = False
            self.shoud_dubbing = False

        # Record the final configuration information used
        logger.debug(f"Final configuration information:{self.cfg=}")
        # Disable modification of subtitles
        self._signal(text="forbid", type="disabled_edit")

        # Start a thread to display progress
        def runing():
            t = time.time()
            while not self.hasend:
                if self._exit(): return
                time.sleep(1)
                self._signal(text=f"{int(time.time() - t)}???{self.precent}", type="set_precent")
        if app_cfg.exec_mode != 'cli':
            threading.Thread(target=runing, daemon=True).start()

    # 1. Preprocessing, separating audio and video, separating human voices, etc.
    def prepare(self) -> None:
        if self._exit(): return
        self._signal(text=tr("Hold on a monment..."))
        Path(self.cfg.cache_folder).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.target_dir).mkdir(parents=True, exist_ok=True)
        # Delete any invalid files that may exist
        self._unlink_size0(self.cfg.source_sub)
        self._unlink_size0(self.cfg.target_sub)
        self._unlink_size0(self.cfg.targetdir_mp4)

        try:
            # Delete existing ones, which may fail.
            Path(self.cfg.source_wav).unlink(missing_ok=True)
            Path(self.cfg.source_wav_output).unlink(missing_ok=True)
            Path(self.cfg.target_wav).unlink(missing_ok=True)
            Path(self.cfg.target_wav_output).unlink(missing_ok=True)
        except Exception as e:
            logger.exception(f'Failed to delete existing file:{e}', exc_info=True)

        self.video_info = tools.get_video_info(self.cfg.name)
        # milliseconds
        self.video_time = self.video_info['time']
        audio_stream_len = self.video_info.get('streams_audio', 0)

        # No video stream, not audio, and not extracted, error reported
        if self.video_info.get('video_streams', 0) < 1 and not self.is_audio_trans and self.cfg.app_mode != 'tiqu':
            self.hasend = True
            raise RuntimeError(
                tr('The video file {} does not contain valid video data and cannot be processed.', self.cfg.name))

        # No audio stream, no original language subtitles, error. There is a silent video stream
        if audio_stream_len < 1 and not tools.vail_file(self.cfg.source_sub):
            self.hasend = True
            raise RuntimeError(
                tr('There is no valid audio in the file {} and it cannot be processed. Please play it manually to confirm that there is sound.',
                   self.cfg.name))

        # If the original video encoding format is h264 and the color is yuv420p, copy the video stream directly is_copy_video=True
        if self.video_info['video_codec_name'] == 'h264' and self.video_info['color'] == 'yuv420p':
            self.is_copy_video = True

        # If subtitle text exists, it will be regarded as the original language subtitle and will no longer be recognized.
        if self.cfg.subtitles.strip():
            with open(self.cfg.source_sub, 'w', encoding="utf-8", errors="ignore") as f:
                txt = re.sub(r':\d+\.\d+', lambda m: m.group().replace('.', ','),
                             self.cfg.subtitles.strip(), flags=re.I | re.S)
                f.write(txt)
            self.shoud_recogn = False

        # Determine whether a human voice file already exists. As long as it exists, this file will be used as the raw material for speech recognition.
        self.cfg.vocal = f"{self.cfg.cache_folder}/vocal.wav"
        raw_vocal = f"{self.cfg.target_dir}/vocal.wav"

        if tools.vail_file(raw_vocal):
            shutil.copy2(raw_vocal, self.cfg.vocal)
        
        #Need background sound separation
        if self.cfg.is_separate:
            raw_instrument = f"{self.cfg.target_dir}/instrument.wav"
            self.cfg.instrument = f"{self.cfg.cache_folder}/instrument.wav"
            
            if tools.vail_file(raw_instrument):
                shutil.copy2(raw_instrument, self.cfg.instrument)
            self.shoud_separate = True

        # Separate original video into silent video
        if not self.is_audio_trans and self.cfg.app_mode != 'tiqu':
            app_cfg.queue_novice[self.uuid] = 'ing'
            if not self.is_copy_video:
                self._signal(text=tr("Video needs transcoded and take a long time.."))
            run_in_threadpool(self._split_novoice_byraw)
        else:
            app_cfg.queue_novice[self.uuid] = 'end'

        # It is necessary to separate the vocal background sound, and there is no separated file
        if audio_stream_len > 0 and self.cfg.is_separate and (not tools.vail_file(self.cfg.vocal) or not tools.vail_file(self.cfg.instrument)):
                self._signal(text=tr('Separating background music'))
                try:
                    self._split_audio_byraw(True)
                except Exception as e:
                    logger.exception(f'Failed to separate vocal background sound', exc_info=True)
                finally:
                    if not tools.vail_file(self.cfg.vocal) or not tools.vail_file(self.cfg.instrument):
                        # Detachment failed
                        self.cfg.instrument = None
                        self.cfg.vocal = None
                        self.cfg.is_separate = False
                        self.shoud_separate = False
            
        

        if audio_stream_len > 0 and not tools.vail_file(self.cfg.source_wav) and tools.vail_file(self.cfg.vocal):
            # If a human voice file exists (maybe only the successful human voice is separated, or the human voice separated by other tools is put into the target folder separately), then use this file as the speech recognition file
            cmd = [
                "-y",
                "-i",
                self.cfg.vocal,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                self.cfg.source_wav
            ]
            try:
                logger.debug(f'There is a separate vocal file vocal.wav, use this as the original audio for speech recognition')
                tools.runffmpeg(cmd)
            except Exception as e:
                logger.error(f'Failed to convert vocal file to 16000 source_wav:{e}')

        # If the original audio does not exist yet self.cfg.source_wav, it indicates failure and is forced to be extracted from the original video.
        if audio_stream_len > 0 and not tools.vail_file(self.cfg.source_wav):
            self._split_audio_byraw()

        self._signal(text=tr('endfenliyinpin'))

    # Start recognition
    def recogn(self) -> None:
        if self._exit(): return
        if not self.shoud_recogn: return
        self.precent += 3
        self._signal(text=tr("kaishishibie"))
        if tools.vail_file(self.cfg.source_sub):
            self.source_srt_list = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
            if Path(self.cfg.target_dir + "/speaker.json").exists():
                shutil.copy2(self.cfg.target_dir + "/speaker.json", self.cfg.cache_folder + "/speaker.json")
            self._recogn_succeed()
            return

        if not tools.vail_file(self.cfg.source_wav):
            error = tr("Failed to separate audio, please check the log or retry")
            self.hasend = True
            raise RuntimeError(error)

        # If background vocal separation has been performed, noise reduction will no longer be performed.
        if not self.cfg.is_separate and self.cfg.remove_noise:
            title = tr("Starting to process speech noise reduction, which may take a long time, please be patient")
            _remove_noise_wav=f"{self.cfg.cache_folder}/remove_noise.wav"
            _cache_noise_wav=f"{self.cfg.target_dir}/removed_noise.wav"
            if not Path(_cache_noise_wav).exists():
                tools.check_and_down_ms(model_id='iic/speech_frcrn_ans_cirm_16k', callback=self._process_callback)
                from videotrans.process.prepare_audio import remove_noise
                kw = {
                    "input_file": self.cfg.source_wav,
                    "output_file": _remove_noise_wav,
                    "is_cuda": self.cfg.is_cuda
                }
                try:
                    _rs = self._new_process(callback=remove_noise, title=title, is_cuda=self.cfg.is_cuda, kwargs=kw)
                    if _rs:
                        self.cfg.source_wav = _rs
                        shutil.copy2(_rs,_cache_noise_wav)
                    self._signal(text='remove noise end')
                except:
                    pass
            else:
                shutil.copy2(_cache_noise_wav,_remove_noise_wav)
                self.cfg.source_wav = _remove_noise_wav

        self._signal(text=tr("Speech Recognition to Word Processing"))

        if self.cfg.recogn_type == Faster_Whisper_XXL:
            xxl_path = settings.get('Faster_Whisper_XXL', 'Faster_Whisper_XXL.exe')
            cmd = [
                xxl_path,
                self.cfg.source_wav,
                "-pp",
                "-f", "srt"
            ]
            cmd.extend(['-l', self.cfg.detect_language.split('-')[0]])
            prompt = None
            prompt = settings.get(f'initial_prompt_{self.cfg.detect_language}')
            if prompt:
                cmd += ['--initial_prompt', prompt]
            cmd.extend(['--model', self.cfg.model_name, '--output_dir', self.cfg.target_dir])

            txt_file = Path(xxl_path).parent.resolve().as_posix() + '/pyvideotrans.txt'

            if Path(txt_file).exists():
                cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))

            cmdstr = " ".join(cmd)
            outsrt_file = self.cfg.target_dir + '/' + Path(self.cfg.source_wav).stem + ".srt"
            logger.debug(f'Faster_Whisper_XXL: {cmdstr=}\n{outsrt_file=}\n{self.cfg.source_sub=}')

            self._external_cmd_with_wrapper(cmd)

            try:
                shutil.copy2(outsrt_file, self.cfg.source_sub)
            except shutil.SameFileError:
                pass
            self.source_srt_list = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
        elif self.cfg.recogn_type == Whisper_CPP:
            cpp_path = settings.get('Whisper_cpp', 'whisper-cli')
            cmd = [
                cpp_path,
                "-f",
                self.cfg.source_wav,
                "-osrt",
                "-np"

            ]
            cmd += ["-l", self.cfg.detect_language.split('-')[0]]
            prompt = None
            prompt = settings.get(f'initial_prompt_{self.cfg.detect_language}')
            if prompt:
                cmd += ['--prompt', prompt]
            cpp_folder = Path(cpp_path).parent.resolve().as_posix()
            if not Path(f'{cpp_folder}/models/{self.cfg.model_name}').is_file():
                raise RuntimeError(tr('The model does not exist. Please download the model to the {} directory first.',
                                      f'{cpp_folder}/models'))
            txt_file = cpp_folder + '/pyvideotrans.txt'

            if Path(txt_file).exists():
                cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))

            cmd.extend(['-m', f'models/{self.cfg.model_name}', '-of', self.cfg.source_sub[:-4]])

            logger.debug(f'Whisper.cpp: {cmd=}')

            self._external_cmd_with_wrapper(cmd)
            self.source_srt_list = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
        else:
            # -1 is not enabled, 0 does not limit the number, >0 plus 1 is the specified number of speakers
            logger.debug(f'[trans_create]:run_recogn() {time.time()=}')
            raw_subtitles = run_recogn(
                recogn_type=self.cfg.recogn_type,
                uuid=self.uuid,
                model_name=self.cfg.model_name,
                audio_file=self.cfg.source_wav,
                detect_language=self.cfg.detect_language,
                cache_folder=self.cfg.cache_folder,
                is_cuda=self.cfg.is_cuda,
                subtitle_type=self.cfg.subtitle_type,
                max_speakers=self.max_speakers,
                llm_post=self.cfg.rephrase == 1
            )
            if self._exit(): return
            if not raw_subtitles:
                raise RuntimeError(self.cfg.basename + tr('recogn result is empty'))
            self._save_srt_target(raw_subtitles, self.cfg.source_sub)
            self.source_srt_list = raw_subtitles

        # Restore punctuation marks in Chinese and English
        if self.cfg.fix_punc and self.cfg.detect_language[:2] in ['zh', 'en']:
            tools.check_and_down_ms(model_id='iic/punc_ct-transformer_cn-en-common-vocab471067-large',
                                    callback=self._process_callback)
            from videotrans.process.prepare_audio import fix_punc
            # Delete existing punctuation in advance
            text_dict = {f'{it["line"]}': re.sub(r'[,.?!，。？！]', ' ', it["text"]) for it in self.source_srt_list}
            kw = {"text_dict": text_dict, "is_cuda": self.cfg.is_cuda}
            try:
                _rs = self._new_process(callback=fix_punc, title=tr("Restoring punct"), is_cuda=self.cfg.is_cuda,
                                        kwargs=kw)
                if _rs:
                    for it in self.source_srt_list:
                        it['text'] = _rs.get(f'{it["line"]}', it['text'])
                        if self.cfg.detect_language[:2] == 'en':
                            it['text'] = it['text'].replace('，', ',').replace('。', '. ').replace('？', '?').replace('！',
                                                                                                                   '!')
                    self._save_srt_target(self.source_srt_list, self.cfg.source_sub)
            except:
                pass

        self._signal(text=Path(self.cfg.source_sub).read_text(encoding='utf-8'), type='replace_subtitle')
        # whisperx-api
        # openairecogn and the model is gpt-4o-transcribe-diarize
        # funasr and the model is paraformer-zh
        # deepgram
        # The above have already been speaker identified. If there are already speaker identification results, the sentence will not be re-segmented.
        if Path(self.cfg.cache_folder + "/speaker.json").exists():
            self._recogn_succeed()
            self._signal(text=tr('endtiquzimu'))
            return

        if self.cfg.rephrase == 1:
            # LLM re-segmentation
            try:
                from videotrans.translator._chatgpt import ChatGPT

                ob = ChatGPT(uuid=self.uuid)
                self._signal(text=tr("Re-segmenting..."))
                srt_list = ob.llm_segment(self.source_srt_list, settings.get('llm_ai_type', 'openai'))
                if srt_list and len(srt_list) > len(self.source_srt_list) / 2:
                    self.source_srt_list = srt_list
                    shutil.copy2(self.cfg.source_sub, f'{self.cfg.source_sub}-No-{tr("LLM Rephrase")}.srt')
                    self._save_srt_target(self.source_srt_list, self.cfg.source_sub)
                else:
                    raise
            except Exception as e:
                self._signal(text=tr("Re-segmenting Error"))
                logger.warning(f"Re-segmentation failed [except] and has been restored to its original state{e}")

        self._recogn_succeed()
        self._signal(text=tr('endtiquzimu'))

    def _recogn_succeed(self) -> None:
        self.precent += 5
        if self.cfg.app_mode == 'tiqu':
            dest_name = f"{self.cfg.target_dir}/{self.cfg.noextname}"
            if not self.shoud_trans:
                self.hasend = True
                self.precent = 100
                dest_name += '.srt'
                shutil.copy2(self.cfg.source_sub, dest_name)
                Path(self.cfg.source_sub).unlink(missing_ok=True)
            else:
                dest_name += f"-{self.cfg.source_language_code}.srt"
                shutil.copy2(self.cfg.source_sub, dest_name)
        self._signal(text=tr('endtiquzimu'))

    # After dubbing, identify the dubbing file again to generate short subtitles.
    # Start recognition
    def recogn2pass(self) -> None:
        if not self.shoud_dubbing or not self.cfg.recogn2pass or self._exit(): 
            return
        # If no subtitles are embedded, or double subtitles are embedded, skip
        if self.cfg.subtitle_type > 2 and (self.cfg.source_language_code != self.cfg.target_language_code):
            logger.debug(f'Skip the secondary recognition. Because dual subtitles are embedded, the timestamps of the dual subtitles will not be consistent after the secondary recognition, so skip:{self.cfg.subtitle_type=}')
            return

        if not tools.vail_file(self.cfg.target_wav):
            logger.debug(f'Skip secondary recognition because there is no dubbing audio file')
            return
            
        self.precent += 3
        self._signal(text=tr("Secondary speech recognition of dubbing files"))
        logger.debug(f'Enter secondary identification')

        shibie_audio = f'{self.cfg.cache_folder}/recogn2pass-{time.time()}.wav'
        outsrt_file = f'{self.cfg.cache_folder}/recogn2pass-{time.time()}.srt'
        try:
            tools.conver_to_16k(self.cfg.target_wav, shibie_audio)
        except Exception as e:
            logger.exception(f'When the secondary recognition of dubbing audio is used to generate subtitles, the preprocessing of the audio fails and is silently skipped:{e}', exc_info=True)
            return
        finally:
            if not tools.vail_file(shibie_audio):
                logger.exception(f'When the secondary recognition of dubbing audio is used to generate subtitles, the preprocessing of the audio fails and is silently skipped:{e}', exc_info=True)
                return
        
        try:
            # Determine whether the original channel supports target language recognition self.cfg.target_language_code
            recogn_type = self.cfg.recogn_type
            model_name = self.cfg.model_name
            detect_language = self.cfg.target_language_code.split('-')[0]

            if recogn_allow_lang(langcode=self.cfg.target_language_code, recogn_type=recogn_type,
                                 model_name=model_name) is not True:
                recogn_type = FASTER_WHISPER
                model_name = 'large-v3-turbo'

            if recogn_type == Faster_Whisper_XXL:
                xxl_path = settings.get('Faster_Whisper_XXL', 'Faster_Whisper_XXL.exe')
                cmd = [
                    xxl_path,
                    shibie_audio,
                    "-pp",
                    "-f", "srt"
                ]
                cmd.extend(['-l', detect_language.split('-')[0]])
                prompt = settings.get(f'initial_prompt_{detect_language}')
                if prompt:
                    cmd += ['--initial_prompt', prompt]
                cmd.extend(['--model', model_name, '--output_dir', self.cfg.cache_folder])

                txt_file = Path(xxl_path).parent.resolve().as_posix() + '/pyvideotrans.txt'

                if Path(txt_file).exists():
                    cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))

                cmdstr = " ".join(cmd)
                logger.debug(f'Faster_Whisper_XXL: {cmdstr=}\n{outsrt_file=}')
                self._external_cmd_with_wrapper(cmd)
            elif recogn_type == Whisper_CPP:
                cpp_path = settings.get('Whisper_cpp', 'whisper-cli')
                cmd = [
                    cpp_path,
                    "-f",
                    shibie_audio,
                    "-osrt",
                    "-np"

                ]
                cmd += ["-l", detect_language]
                prompt = settings.get(f'initial_prompt_{detect_language}')
                if prompt:
                    cmd += ['--prompt', prompt]
                cpp_folder = Path(cpp_path).parent.resolve().as_posix()
                if not Path(f'{cpp_folder}/models/{model_name}').is_file():
                    logger.error(tr('The model does not exist. Please download the model to the {} directory first.',
                                           f'{cpp_folder}/models'))
                    return
                txt_file = cpp_folder + '/pyvideotrans.txt'
                if Path(txt_file).exists():
                    cmd.extend(Path(txt_file).read_text(encoding='utf-8').strip().split(' '))
                cmd.extend(['-m', f'models/{model_name}', '-of', outsrt_file[:-4]])
                logger.debug(f'Whisper.cpp: {cmd=}')
                self._external_cmd_with_wrapper(cmd)
            else:
                # -1 is not enabled, 0 does not limit the number, >0 plus 1 is the specified number of speakers
                logger.debug(f'[trans_create]: Secondary identification')
                raw_subtitles = run_recogn(
                    recogn_type=recogn_type,
                    uuid=self.uuid,
                    model_name=model_name,
                    audio_file=shibie_audio,
                    detect_language=detect_language,
                    cache_folder=self.cfg.cache_folder,
                    is_cuda=self.cfg.is_cuda,
                    recogn2pass=True  # Secondary identification
                )
                if self._exit(): return
                if not raw_subtitles:
                    logger.error('Secondary recognition error:' + tr('recogn result is empty'))
                self._save_srt_target(raw_subtitles, outsrt_file)

            if not tools.vail_file(outsrt_file):
                logger.error(f'Secondary recognition of dubbing files failed, the reason is unknown')
                return
            # override
            shutil.copy2(outsrt_file, self.cfg.target_sub)
            self._signal(text='STT 2 pass end')
            logger.debug('Secondary identification completed successfully')
        
        except Exception as e:
            logger.exception(f'When the secondary recognition of dubbing audio is used to generate subtitles, the preprocessing of the audio fails and is silently skipped:{e}', exc_info=True)
            return

        return True

    def diariz(self):
        #Speaker is set to 1, no separation is performed
        if self._exit() or not self.cfg.enable_diariz or self.max_speakers == 1 or Path(
                self.cfg.cache_folder + "/speaker.json").exists():
            return
        # built pyannote reverb ali_CAM
        speaker_type = settings.get('speaker_type', 'built')
        hf_token = settings.get('hf_token')
        if speaker_type == 'built' and self.cfg.detect_language[:2] not in ['zh', 'en']:
            logger.error(f'The built speaker separation model is currently selected, but is not supported for the current language:{self.cfg.detect_language}')
            return
        if speaker_type in ['pyannote', 'reverb'] and not hf_token:
            logger.error(f'The pyannote speaker separation model is currently selected, but the token of huggingface.co is not set:{self.cfg.detect_language}')
            return
        if speaker_type in ['pyannote', 'reverb']:
            # Determine whether huggingface.co is accessible
            # First test whether you can connect to huggingface.co. It is not accessible in mainland China unless you use a VPN.
            try:
                import requests
                requests.head('https://huggingface.co', timeout=5)
            except Exception:
                logger.error(f'Current selection{speaker_type} Speaker separation model, but cannot connect to https://huggingface.co, may fail')

        try:
            self.precent += 3
            title = tr(f'Begin separating the speakers') + f':{speaker_type}'
            spk_list = None
            kw = {
                "input_file": self.cfg.source_wav,
                "subtitles": [[it['start_time'], it['end_time']] for it in self.source_srt_list],
                "num_speakers": self.max_speakers,
                "is_cuda": self.cfg.is_cuda
            }
            if speaker_type == 'built':
                tools.down_file_from_ms(f'{ROOT_DIR}/models/onnx', [
                    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/seg_model.onnx",
                    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/nemo_en_titanet_small.onnx",
                    "https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
                ], callback=self._process_callback)
                from videotrans.process.prepare_audio import built_speakers as _run_speakers
                del kw['is_cuda']
                kw['num_speakers'] = -1 if self.max_speakers < 1 else self.max_speakers
                kw['language'] = self.cfg.detect_language
            elif speaker_type == 'ali_CAM':
                tools.check_and_down_ms(model_id='iic/speech_campplus_speaker-diarization_common',
                                        callback=self._process_callback)
                from videotrans.process.prepare_audio import cam_speakers as _run_speakers
            elif speaker_type == 'pyannote':
                from videotrans.process.prepare_audio import pyannote_speakers as _run_speakers
            elif speaker_type == 'reverb':
                from videotrans.process.prepare_audio import reverb_speakers as _run_speakers
            else:
                logger.error(f'The currently selected speaker separation model does not support:{speaker_type=}')
                return
            if speaker_type in ['pyannote', 'reverb']:
                self._signal(text='Downloading speakers models')
                from huggingface_hub import snapshot_download
                print(f'Download token:{speaker_type},{hf_token=}')
                snapshot_download(
                    repo_id="pyannote/speaker-diarization-3.1" if speaker_type == 'pyannote' else "Revai/reverb-diarization-v1",
                    token=hf_token
                )

            spk_list = self._new_process(callback=_run_speakers, title=title,
                                         is_cuda=self.cfg.is_cuda and speaker_type != 'built', kwargs=kw)

            if spk_list:
                Path(self.cfg.cache_folder + "/speaker.json").write_text(json.dumps(spk_list), encoding='utf-8')
                logger.debug('Split speaker completed successfully')
                shutil.copy2(self.cfg.cache_folder + "/speaker.json", self.cfg.target_dir + "/speaker.json")
            self._signal(text=tr('separating speakers end'))
        except:
            pass

    # Translate subtitle files
    def trans(self) -> None:
        if self._exit(): return
        if not self.shoud_trans: return
        self.precent += 3
        self._signal(text=tr('starttrans'))

        # If there are subtitles in the target language, there is no need to continue translation. The front desk will directly use the subtitles to replace them.
        if self._srt_vail(self.cfg.target_sub):
            self._signal(
                text=Path(self.cfg.target_sub).read_text(encoding="utf-8", errors="ignore"),
                type='replace_subtitle'
            )
            return
        try:
            rawsrt = tools.get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
            self._signal(text=tr('kaishitiquhefanyi'))

            target_srt = run_trans(
                translate_type=self.cfg.translate_type,
                text_list=copy.deepcopy(rawsrt),
                uuid=self.uuid,
                source_code=self.cfg.source_language_code,
                target_code=self.cfg.target_language_code
            )
            if self._exit():
                return
            # Check each subtitle one by one
            target_srt = self._check_target_sub(rawsrt, target_srt)

            # Only extract and bilingual output
            if self.cfg.app_mode == 'tiqu' and self.cfg.output_srt > 0 and self.cfg.source_language_code != self.cfg.target_language_code:
                _source_srt_len = len(rawsrt)
                for i, it in enumerate(target_srt):
                    if i < _source_srt_len and self.cfg.output_srt == 1:
                        # The target language is below
                        it['text'] = ("\n".join([rawsrt[i]['text'].strip(), it['text'].strip()])).strip()
                    elif i < _source_srt_len and self.cfg.output_srt == 2:
                        it['text'] = ("\n".join([it['text'].strip(), rawsrt[i]['text'].strip()])).strip()

            self._save_srt_target(target_srt, self.cfg.target_sub)

            if self.cfg.app_mode == 'tiqu':
                _output_file = f"{self.cfg.target_dir}/{self.cfg.noextname}.srt"
                if self.cfg.copysrt_rawvideo:
                    p = Path(self.cfg.name)
                    _output_file = f'{p.parent.as_posix()}/{p.stem}.srt'
                if not Path(_output_file).exists() or not Path(_output_file).samefile(Path(self.cfg.target_sub)):
                    shutil.copy2(self.cfg.target_sub, _output_file)
                    self._del_sub()

                self.hasend = True
                self.precent = 100
        except Exception as e:
            self.hasend = True
            raise
        self._signal(text=tr('endtrans'))

    def _del_sub(self):
        try:
            Path(self.cfg.source_sub).unlink(missing_ok=True)
            Path(self.cfg.target_sub).unlink(missing_ok=True)
        except:
            pass

    # Dubbing subtitles
    def dubbing(self) -> None:
        if self._exit():
            return
        if self.cfg.app_mode == 'tiqu' or not self.shoud_dubbing:
            return

        self._signal(text=tr('kaishipeiyin'))
        self.precent += 3
        try:
            self._tts()
            # Determine the next step to readjust subtitles
        except Exception as e:
            self.hasend = True
            raise
        self._signal(text=tr('The dubbing is finished'))

    # Align audio and video subtitles
    def align(self) -> None:
        if self._exit():
            return
        if self.cfg.app_mode == 'tiqu' or not self.shoud_dubbing or self.ignore_align:
            return

        self._signal(text=tr('duiqicaozuo'))
        self.precent += 3
        if self.cfg.voice_autorate or self.cfg.video_autorate:
            self._signal(text=tr("Sound & video speed alignment stage"))
        try:
            # If the video needs to be slow, determine whether the silent video has been separated.
            if self.cfg.video_autorate:
                tools.is_novoice_mp4(self.cfg.novoice_mp4, self.uuid)
            # If there is a video, the length of the video will prevail.
            if tools.vail_file(self.cfg.novoice_mp4):
                self.video_time = tools.get_video_duration(self.cfg.novoice_mp4)

            print(f'speedrate===')
            rate_inst = SpeedRate(
                queue_tts=self.queue_tts,
                uuid=self.uuid,
                shoud_audiorate=self.cfg.voice_autorate,
                # Whether the video needs to be slow, process novoice_mp4 if necessary
                shoud_videorate=self.cfg.video_autorate if not self.is_audio_trans else False,
                novoice_mp4=self.cfg.novoice_mp4 if not self.is_audio_trans else None,
                # Original total duration
                raw_total_time=self.video_time,

                target_audio=self.cfg.target_wav,
                cache_folder=self.cfg.cache_folder,
                align_sub_audio=self.cfg.align_sub_audio,  # Both work when audio acceleration and video slowdown are not enabled
                remove_silent_mid=self.cfg.remove_silent_mid,  # Both work when audio acceleration and video slowdown are not enabled
                stretch_short_max_ms=getattr(self.cfg, 'stretch_short_max_ms', 0) or 0,
                stretch_short_max_ratio=getattr(self.cfg, 'stretch_short_max_ratio', 1.15) or 1.15,
            )
            self.queue_tts = rate_inst.run()
            # After slow processing, update the total duration of the new video for audio and video alignment.
            if tools.vail_file(self.cfg.novoice_mp4):
                self.video_time = tools.get_video_duration(self.cfg.novoice_mp4)

            # Align subtitles
            if self.cfg.voice_autorate or self.cfg.video_autorate or self.cfg.align_sub_audio:
                srt = ""
                for (idx, it) in enumerate(self.queue_tts):
                    startraw = tools.ms_to_time_string(ms=it['start_time'])
                    endraw = tools.ms_to_time_string(ms=it['end_time'])
                    srt += f"{idx + 1}\n{startraw} --> {endraw}\n{it['text']}\n\n"
                # Save subtitles to the target folder
                with  Path(self.cfg.target_sub).open('w', encoding="utf-8") as f:
                    f.write(srt.strip())
        except Exception as e:
            self.hasend = True
            raise

        # After success, if the volume exists, adjust the volume
        if self.cfg.tts_type not in [EDGE_TTS, AZURE_TTS] and self.cfg.volume != '+0%' and tools.vail_file(
                self.cfg.target_wav):
            volume = self.cfg.volume.replace('%', '').strip()
            try:
                volume = 1 + float(volume) / 100
                if volume != 1.0:
                    tmp_name = self.cfg.cache_folder + f'/volume-{volume}-{Path(self.cfg.target_wav).name}'
                    tools.runffmpeg(['-y', '-i', os.path.basename(self.cfg.target_wav), '-af', f"volume={volume}",
                                     os.path.basename(tmp_name)], cmd_dir=self.cfg.cache_folder)
                    shutil.copy2(tmp_name, self.cfg.target_wav)
            except:
                pass

        self._signal(text=tr('Alignment phase complete, awaiting the next step'))

    # Combine video, audio, and subtitles
    def assembling(self) -> None:
        if self._exit(): return
        # Audio translation, extraction mode without merging
        if self.is_audio_trans or self.cfg.app_mode == 'tiqu' or not self.shoud_hebing:
            return
        if self.precent < 95:
            self.precent += 3
        self._signal(text=tr('kaishihebing'))
        try:
            self._join_video_audio_srt()
        except Exception as e:
            self.hasend = True
            raise

    #Finishing, depending on whether output and linshi_output are the same, if not, move
    def task_done(self) -> None:
        # Normal completion is still ing, manual stop becomes stop
        if self._exit(): return
        self.precent = 99

        # When extracting, delete
        if self.cfg.app_mode == 'tiqu':
            try:
                Path(f"{self.cfg.target_dir}/{self.cfg.source_language_code}.srt").unlink(
                    missing_ok=True)
                Path(f"{self.cfg.target_dir}/{self.cfg.target_language_code}.srt").unlink(
                    missing_ok=True)
            except:
                pass  # Ignore deletion failures
        else:    
            if self.is_audio_trans and tools.vail_file(self.cfg.target_wav):
                try:
                    shutil.copy2(self.cfg.target_wav, f"{self.cfg.target_dir}/{self.cfg.target_language_code}-{self.cfg.noextname}.wav")
                except shutil.SameFileError:
                    pass

            try:
                if self.cfg.shound_del_name:
                    Path(self.cfg.shound_del_name).unlink(missing_ok=True)
                if self.cfg.only_out_mp4:
                    shutil.move(self.cfg.targetdir_mp4, Path(self.cfg.target_dir).parent / f'{self.cfg.noextname}.mp4')
                    shutil.rmtree(self.cfg.target_dir, ignore_errors=True)
            except Exception as e:
                logger.exception(e, exc_info=True)
        self.hasend = True
        self.precent = 100
        try:
            shutil.rmtree(self.cfg.cache_folder, ignore_errors=True)
        except:
            pass
        self._signal(text=f"{self.cfg.name}", type='succeed')
        tools.send_notification(tr('Succeed'), f"{self.cfg.basename}")

    # Separate the silent video from the original video
    def _split_novoice_byraw(self):
        cmd = [
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            self.cfg.name,
            "-an",
            "-c:v",
            "copy" if self.is_copy_video else f"libx264"
        ]
        _name=os.path.basename(self.cfg.novoice_mp4)
        enc_qua=[] if self.is_copy_video else ['-crf','18']
        if self.is_copy_video or settings.get('force_lib'):
            return tools.runffmpeg(cmd+enc_qua+[_name], noextname=self.uuid, cmd_dir=self.cfg.cache_folder)
        
        try:
            hw_decode_args,_,vcodec,enc_args=self._get_hard_cfg(codec="264")
            cmd = [
                "-y",
                "-fflags",
                "+genpts",
            ]
            cmd+=hw_decode_args

            cmd+=[
                "-i",
                self.cfg.name,
                "-an",
                "-c:v",
                vcodec,
                _name
            ]
            self._subprocess(cmd)
            app_cfg.queue_novice[self.uuid] = 'end'
        except Exception as e:
            logger.exception(f'Hardware separation of silent video failed:{e}',exc_info=True)
            return tools.runffmpeg([
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                self.cfg.name,
                "-an",
                "-c:v",
                "libx264",
                _name
            ], noextname=self.uuid, cmd_dir=self.cfg.cache_folder,force_cpu=True)
            

    # Separate audio from original video
    def _split_audio_byraw(self, is_separate=False):
        cmd = [
            "-y",
            "-i",
            self.cfg.name,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            self.cfg.source_wav
        ]
        rs = tools.runffmpeg(cmd)
        if not is_separate:
            return rs

        # Continue vocal separation
        tmpfile = self.cfg.cache_folder + "/441000_ac2_raw.wav"
        tools.runffmpeg([
            "-y",
            "-i",
            self.cfg.name,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            tmpfile
        ])

        if tools.vail_file(self.cfg.vocal) and tools.vail_file(self.cfg.instrument):
            return
        title = tr('Separating vocals and background music, which may take a longer time')
        uvr_models=settings.get('uvr_models')
        tools.down_file_from_ms(f'{ROOT_DIR}/models/onnx', [
            f"https://www.modelscope.cn/models/himyworld/videotrans/resolve/master/onnx/{uvr_models}.onnx"
        ], callback=self._process_callback)
        from videotrans.process.prepare_audio import vocal_bgm
        # Return False None on failure
        kw = {"input_file": tmpfile, "vocal_file": self.cfg.vocal, "instr_file": self.cfg.instrument,"uvr_models":uvr_models}
        try:
            rs = self._new_process(callback=vocal_bgm, title=title, is_cuda=False, kwargs=kw)
            if rs and tools.vail_file(self.cfg.vocal) and tools.vail_file(self.cfg.instrument):
                cmd = [
                    "-y",
                    "-i",
                    self.cfg.vocal,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    '-af',
                    "volume=1.5",
                    self.cfg.source_wav
                ]
                tools.runffmpeg(cmd)
                shutil.copy2(self.cfg.vocal, f'{self.cfg.target_dir}/vocal.wav')
                shutil.copy2(self.cfg.instrument, f'{self.cfg.target_dir}/instrument.wav')
        except Exception as e:
            logger.exception(f'Vocal background sound separation failed:{e}', exc_info=True)

    # Preprocessing of dubbing, removing invalid characters, and sorting out the start time
    def _tts(self, daz_json=None) -> None:
        queue_tts = []
        subs = tools.get_subtitle_from_srt(self.cfg.target_sub)
        source_subs = tools.get_subtitle_from_srt(self.cfg.source_sub)
        if len(subs) < 1:
            raise RuntimeError(f"SRT file error:{self.cfg.target_sub}")
        try:
            rate = int(str(self.cfg.voice_rate).replace('%', ''))
        except:
            rate = 0
        if rate >= 0:
            rate = f"+{rate}%"
        else:
            rate = f"{rate}%"
        # Take out each row of roles set
        line_roles = app_cfg.line_roles
        voice_role = self.cfg.voice_role
        force_clone = str(voice_role).strip().lower() == 'clone' and self.cfg.tts_type in SUPPORT_CLONE

        # Take out each subtitle, line number\nStart time --> End time\nContent
        for i, it in enumerate(subs):
            if it['end_time'] < it['start_time'] or not it['text'].strip():
                continue
            # Determine whether there is a separately set row role, if not, use the global one
            voice = 'clone' if force_clone else line_roles.get(f'{it["line"]}', voice_role)

            tmp_dict = {
                "text": it['text'],
                "line": it['line'],
                "start_time": it['start_time'],
                "end_time": it['end_time'],
                "startraw": it['startraw'],
                "endraw": it['endraw'],
                "ref_text": source_subs[i]['text'] if source_subs and i < len(source_subs) else '',
                "start_time_source": source_subs[i]['start_time'] if source_subs and i < len(source_subs) else it[
                    'start_time'],
                "end_time_source": source_subs[i]['end_time'] if source_subs and i < len(source_subs) else it[
                    'end_time'],
                "role": voice,
                "rate": rate,
                "volume": self.cfg.volume,
                "pitch": self.cfg.pitch,
                "tts_type": self.cfg.tts_type,
                "filename": f"{self.cfg.cache_folder}/dubb-{i}.wav"
            }
            # If it is a clone-voice type, the corresponding fragment needs to be intercepted
            # is a clone
            if str(voice).strip().lower() == 'clone' and self.cfg.tts_type in SUPPORT_CLONE:
                tmp_dict['ref_wav'] = f"{self.cfg.cache_folder}/clone-{i}.wav"
                tmp_dict['ref_language'] = self.cfg.detect_language[:2]
            queue_tts.append(tmp_dict)

        self.queue_tts = copy.deepcopy(queue_tts)

        if not self.queue_tts or len(self.queue_tts) < 1:
            raise RuntimeError(f'Queue tts length is 0')

        # If there is a ref_wav, it needs to be cloned, and there is a reference audio
        if len([it.get("ref_wav") for it in self.queue_tts if it.get("ref_wav")]) > 0:
            self._create_ref_from_vocal()

        # Specific dubbing operations
        run_tts(
            queue_tts=self.queue_tts,
            language=self.cfg.target_language_code,
            uuid=self.uuid,
            tts_type=self.cfg.tts_type,
            is_cuda=self.cfg.is_cuda
        )
        if settings.get('save_segment_audio', False):
            outname = self.cfg.target_dir + f'/segment_audio_{self.cfg.noextname}'
            Path(outname).mkdir(parents=True, exist_ok=True)
            for it in self.queue_tts:
                text = re.sub(r'["\'*?\\/\|:<>\r\n\t]+', '', it['text'], flags=re.I | re.S)
                name = f'{outname}/{it["line"]}-{text[:60]}.wav'
                if Path(it['filename']).exists():
                    shutil.copy2(it['filename'], name)

    #Multi-threading implementation of cropping reference audio
    def _create_ref_from_vocal(self):
        # If the background separation of vocals fails, the original audio will be used directly.
        vocal = self.cfg.source_wav

        #Cut the corresponding clip into reference audio
        def _cutaudio_from_vocal(it):
            try:
                logger.debug(f"Cut the corresponding clip into reference audio:{it['startraw']}->{it['endraw']}\nCurrent{it=}")
                tools.cut_from_audio(
                    audio_file=vocal,
                    ss=it['startraw'],
                    to=it['endraw'],
                    out_file=it['ref_wav']
                )
            except Exception as e:
                logger.exception(f'Failed to crop reference audio:{e}', exc_info=True)

        all_task = []
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(12, len(self.queue_tts), os.cpu_count())) as pool:
            for item in self.queue_tts:
                if item.get('ref_wav'):
                    all_task.append(pool.submit(_cutaudio_from_vocal, item))
            if len(all_task) > 0:
                _ = [i.result() for i in all_task]

    # Add background music
    def _back_music(self) -> None:
        if self._exit() or not self.shoud_dubbing:
            return

        if not tools.vail_file(self.cfg.target_wav) or not tools.vail_file(self.cfg.background_music):
            return
        try:
            self._signal(text=tr("Adding background audio"))
            # Get video length
            vtime = tools.get_audio_time(self.cfg.target_wav)
            # Get the background audio length
            atime = tools.get_audio_time(self.cfg.background_music)
            bgm_file = self.cfg.cache_folder + f'/bgm_file.wav'
            self.convert_to_wav(self.cfg.background_music, bgm_file)
            self.cfg.background_music = bgm_file
            beishu = math.ceil(vtime / atime)
            if settings.get('loop_backaudio') and beishu > 1 and vtime - 1000 > atime:
                # Get the extended segment
                file_list = [self.cfg.background_music for n in range(beishu + 1)]
                concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                tools.create_concat_txt(file_list, concat_txt=concat_txt)
                tools.concat_multi_audio(
                    concat_txt=concat_txt,
                    out=self.cfg.cache_folder + "/bgm_file_extend.wav")
                self.cfg.background_music = self.cfg.cache_folder + "/bgm_file_extend.wav"
            # Reduce the volume of background audio
            tools.runffmpeg(
                ['-y',
                 '-i', self.cfg.background_music,
                 "-filter:a", f"volume={settings.get('backaudio_volume', 0.8)}",
                 '-c:a', 'pcm_s16le',
                 self.cfg.cache_folder + f"/bgm_file_extend_volume.wav"
                 ])
            # Background audio and dubbing merged
            cmd = ['-y',
                   '-i', os.path.basename(self.cfg.target_wav),
                   '-i', "bgm_file_extend_volume.wav",
                   '-filter_complex', "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2",
                   '-ac', '2',
                   '-c:a', 'pcm_s16le',
                   "lastend.wav"
                   ]
            tools.runffmpeg(cmd, cmd_dir=self.cfg.cache_folder)
            self.cfg.target_wav = self.cfg.cache_folder + f"/lastend.wav"
        except Exception as e:
            logger.exception(f'Failed to add background music:{str(e)}', exc_info=True)

    def _separate(self) -> None:
        if self._exit() or not self.shoud_separate or not self.cfg.embed_bgm:
            return
        # If background audio separation fails, return silently
        if not tools.vail_file(self.cfg.instrument):
            return
        if not tools.vail_file(self.cfg.target_wav):
            return
        try:
            self._signal(text=tr("Re-embedded background sounds"))
            vtime = tools.get_audio_time(self.cfg.target_wav)
            atime = tools.get_audio_time(self.cfg.instrument)
            beishu = math.ceil(vtime / atime)

            instrument_file = self.cfg.instrument
            logger.debug(f'Merge background sounds{beishu=},{atime=},{vtime=}')
            if atime + 1000 < vtime:
                if int(settings.get('loop_backaudio'))==1:
                    # Background sound connection extension clip
                    file_list = [instrument_file for n in range(beishu + 1)]
                    concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                    tools.create_concat_txt(file_list, concat_txt=concat_txt)
                    tools.concat_multi_audio(concat_txt=concat_txt,
                                             out=self.cfg.cache_folder + "/instrument-concat.wav")
                else:
                    # Extend background sound
                    tools.change_speed_rubberband(instrument_file, self.cfg.cache_folder + f"/instrument-concat.wav", vtime)
                instrument_file=self.cfg.cache_folder + f"/instrument-concat.wav"
            # Background sound merged dubbing
            self._backandvocal(instrument_file, self.cfg.target_wav)
        except Exception as e:
            logger.exception(e, exc_info=True)

    # After merging, the final file is still a vocal file, and the duration needs to be equal to the vocal file.
    def _backandvocal(self, backwav, peiyinm4a):
        backwav = Path(backwav).as_posix()
        tmpdir = self.cfg.cache_folder
        tmpwav = Path(tmpdir + f'/{time.time()}-1.wav').as_posix()
        tmpm4a = Path(tmpdir + f'/{time.time()}.wav').as_posix()
        #Convert the background to an m4a file and reduce the volume to 0.8
        self.convert_to_wav(backwav, tmpm4a, ["-filter:a", f"volume={settings.get('backaudio_volume', 0.8)}"])
        tools.runffmpeg(['-y', '-i', os.path.basename(peiyinm4a), '-i', os.path.basename(tmpm4a), '-filter_complex',
                         "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2", '-ac', '2', "-b:a", "128k",
                         '-c:a', 'pcm_s16le', os.path.basename(tmpwav)], cmd_dir=self.cfg.cache_folder)
        shutil.copy2(tmpwav, peiyinm4a)

    # Process required subtitles
    def _process_subtitles(self) -> Union[tuple[str, str], None]:
        logger.debug(f"\n======Prepare subtitles to be embedded:{self.cfg.subtitle_type=}=====")
        if not Path(self.cfg.target_sub).exists():
            logger.error(tr("No valid subtitle file exists"))
            return

        # If the original language and target language are the same, or there are no subtitles in the original language, force single subtitles
        if not Path(self.cfg.source_sub).exists() or (self.cfg.source_language_code == self.cfg.target_language_code):
            if self.cfg.subtitle_type == 3:
                self.cfg.subtitle_type = 1
            elif self.cfg.subtitle_type == 4:
                self.cfg.subtitle_type = 2

        process_end_subtitle = self.cfg.cache_folder + f'/end.srt'
        # Number of characters in a single line
        maxlen = int(
            settings.get('cjk_len', 15) if self.cfg.target_language_code[:2] in ["zh", "ja", "jp", "ko",
                                                                                        'yu'] else
            settings.get('other_len', 60))
        target_sub_list = tools.get_subtitle_from_srt(self.cfg.target_sub)
        
        srt_string = ""
        # Two language subtitle separators for dual hard subtitles, used to define different styles
        _join_flag=''
        # Dual Hard Dual Soft Subtitle Assembly
        if self.cfg.subtitle_type in [3, 4]:
            source_sub_list = tools.get_subtitle_from_srt(self.cfg.source_sub)
            source_length = len(source_sub_list)
            # Original language single line character length
            source_maxlen = int(
                settings.get('cjk_len', 15) if self.cfg.source_language_code[:2] in ["zh", "ja", "jp", "ko",
                                                                                            'yu'] else
                settings.get('other_len', 60))

            # bilingual subtitles
            # Determine whether dual hard subtitles exist and the ass.json file exists and (Bottom_Fontsize != Fontsize or PrimaryColour!=Bottom_PrimaryColour) It is necessary to set different colors and sizes for the second line of bilingual subtitles
            _join_flag=self._get_join_flag()
                    
            for i, it in enumerate(target_sub_list):
                # newline
                _text = tools.simple_wrap(it['text'].strip(), maxlen, self.cfg.target_language_code)
                srt_string += f"{it['line']}\n{it['time']}\n"
                if source_length > 0 and i < source_length:
                    _text_source=tools.simple_wrap(source_sub_list[i]['text'], source_maxlen, self.cfg.source_language_code)
                    _text=f'{_text_source}\n{_join_flag}{_text}' if self.cfg.output_srt==1 else f'{_text}\n{_join_flag}{_text_source}'
                srt_string += f"{_text}\n\n"
            srt_string=srt_string.strip()
            process_end_subtitle = f"{self.cfg.cache_folder}/shuang.srt"
            Path(process_end_subtitle).write_text(srt_string, encoding='utf-8')
            Path(self.cfg.target_dir + "/shuang.srt").write_text(srt_string.replace('###','') if _join_flag=='###' else srt_string, encoding='utf-8')
        else:
            # Single subtitles, need to process character number and line wrapping
            for i, it in enumerate(target_sub_list):
                tmp = tools.simple_wrap(it['text'].strip(), maxlen, self.cfg.target_language_code)
                srt_string += f"{it['line']}\n{it['time']}\n{tmp.strip()}\n\n"
            with Path(process_end_subtitle).open('w', encoding='utf-8') as f:
                f.write(srt_string)

        # Target subtitle language
        subtitle_langcode = translator.get_subtitle_code(show_target=self.cfg.target_language)
        logger.debug(
            f'Finalize subtitle embedding type:{self.cfg.subtitle_type},Target subtitle language:{subtitle_langcode}, subtitle file:{process_end_subtitle}\n')
        # Single soft or double soft
        if self.cfg.subtitle_type in [2, 4]:
            return os.path.basename(process_end_subtitle), subtitle_langcode

        # Convert hard subtitles to ass format and set style
        process_end_subtitle_ass = tools.set_ass_font(process_end_subtitle)
        basename = os.path.basename(process_end_subtitle_ass)
        return basename, subtitle_langcode


    def _get_join_flag(self):
        _join_flag=""
        if self.cfg.subtitle_type!=3 or not Path(f'{ROOT_DIR}/videotrans/ass.json').exists():
            return _join_flag
        try:
            assjson=json.loads(Path(f'{ROOT_DIR}/videotrans/ass.json').read_text(encoding='utf-8'))
        except Exception as e:
            logger.debug(f'Error reading ass.json, ignored:{e}')
            return _join_flag
        else:
            for k,v in assjson.items():
                if k.startswith('Bottom_') and v!= assjson.get(k[7:]):
                    _join_flag='###'
                    break
        return _join_flag



    # Freeze the last frame of the video
    def _video_extend(self, duration_ms=1000):
        sec = duration_ms / 1000.0
        final_video_path = Path(f'{self.cfg.cache_folder}/final_video_with_freeze_lastend.mp4').as_posix()
        cmd = ['-y', '-i', os.path.basename(self.cfg.novoice_mp4),
               '-vf', f'tpad=stop_mode=clone:stop_duration={sec:.3f}',
               '-c:v', 'libx264',
               '-crf', f'{settings.get("crf", 23)}',
               '-preset', settings.get('preset', 'veryfast'),
               '-an', 'final_video_with_freeze_lastend.mp4']

        if tools.runffmpeg(cmd, force_cpu=True, cmd_dir=self.cfg.cache_folder) and Path(final_video_path).exists():
            shutil.copy2(final_video_path, self.cfg.novoice_mp4)
            logger.debug(f"Video freeze frame should be extended{duration_ms}ms, actual extension in seconds rounded up{sec}s, the operation was successful.")
        else:
            logger.warning('Video freeze extension operation failed!')

    #Final composite video
    def _join_video_audio_srt(self) -> None:
        if self._exit():
            return
        if not self.shoud_hebing:
            return True

        # Determine whether novoice_mp4 is completed
        tools.is_novoice_mp4(self.cfg.novoice_mp4, self.uuid)
        if not Path(self.cfg.novoice_mp4).exists():
            raise RuntimeError(f'{self.cfg.novoice_mp4} does not exist')
        
        # Need dubbing but no dubbing file
        if self.shoud_dubbing and not tools.vail_file(self.cfg.target_wav):
            raise RuntimeError(f"{tr('Dubbing')}{tr('anerror')}:{self.cfg.target_wav}")

        self.precent = min(max(90, self.precent), 95)


        target_m4a = self.cfg.cache_folder + "/origin_audio.m4a"
        # Used to determine whether the output of the original audio has ended. is True means the end.
        output_source_output = True
        if not self.shoud_dubbing:
            # Use original audio without dubbing
            self._get_origin_audio(target_m4a)
            shutil.copy2(target_m4a, self.cfg.source_wav_output)
        else:
            try:
                output_source_output = False
                # High quality original audio output to the target directory, executed in a separate thread, does not affect continued operation
                cmd = [
                    "-y",
                    "-i",
                    self.cfg.name,
                    "-vn",
                    "-b:a", "128k",
                    "-c:a",
                    "aac",
                    self.cfg.source_wav_output
                ]

                def _output():
                    nonlocal output_source_output
                    try:
                        tools.runffmpeg(cmd)
                    except Exception:
                        pass
                    finally:
                        output_source_output = True
                threading.Thread(target=_output, daemon=True).start()
            except Exception:
                pass

            # Add background music
            self._back_music()
            # Re-embed the separated background sound
            self._separate()
            
            tools.runffmpeg([
                "-y",
                "-i",
                os.path.basename(self.cfg.target_wav),
                "-ac", "2", "-b:a", "128k", "-c:a", "aac",
                os.path.basename(target_m4a)
            ], cmd_dir=self.cfg.cache_folder)
            shutil.copy2(target_m4a, self.cfg.target_wav_output)

        self.precent = min(max(95, self.precent), 98)
        
        
        # Process required subtitles
        subtitles_file, subtitle_langcode = None, None
        if self.cfg.subtitle_type > 0:
            subtitles_file, subtitle_langcode = self._process_subtitles()

        # Enter the video directory when embedding subtitles
        os.chdir(self.cfg.cache_folder)

        # Align end
        duration_ms = int(tools.get_video_duration(self.cfg.novoice_mp4))
        duration_s = f'{duration_ms / 1000.0:.6f}'
        audio_ms = tools.get_audio_time(target_m4a)
        if duration_ms < audio_ms:
            self._video_extend(audio_ms - duration_ms)
            duration_ms = int(tools.get_video_duration(self.cfg.novoice_mp4))
            duration_s = f'{duration_ms / 1000.0:.6f}'

        # Export the generated video to a temporary directory first to prevent targetdir_mp4 containing various strange symbols from causing ffmpeg to fail.
        tmp_target_mp4 = self.cfg.cache_folder + f"/laste_target.mp4"
        self._signal(text=tr("Video + Subtitles + Dubbing in merge"))

        try:
            protxt = self.cfg.cache_folder + f"/compose{time.time()}.txt"
            protxt_basename = os.path.basename(protxt)
            threading.Thread(target=self._hebing_pro, args=(protxt, self.video_time), daemon=True).start()
            
            # If the video to be output is 264 encoded, since the beginning and middle encoding are both 264, you can consider using copy (if there are no hard subtitles embedded)
            is_copy_mode = (str(self.video_codec_num) == '264')
            # No audio video stream
            novoice_mp4_basename = os.path.basename(self.cfg.novoice_mp4)
            # Need to embed audio
            target_m4a_basename = os.path.basename(target_m4a)
            #Synthesized result video
            tmp_target_mp4_basename = os.path.basename(tmp_target_mp4)

            # Get available hardware
            if not app_cfg.video_codec:
                app_cfg.video_codec = tools.get_video_codec()                        


            cmd0 = [
                "-y",
                "-progress",
                protxt_basename
            ]
            
            cmd1=[
                "-i",
                novoice_mp4_basename,
                "-i",
                target_m4a_basename
            ]
            enc_qua=['-crf', f'{settings.get("crf", 23)}','-preset', settings.get('preset','medium')]
            
            # No subtitles or soft subtitles
            if self.cfg.subtitle_type not in [1,3]:               
                #softsubtitles
                if self.cfg.subtitle_type in [2, 4]:
                    cmd1.extend(["-i",subtitles_file])               
                cmd1.extend([
                    '-map', 
                    '0:v',
                    '-map', 
                    '1:a'
                ])
                if self.cfg.subtitle_type in [2, 4]:
                    cmd1.extend(['-map', '2:s'])
                
                cmd1.extend([
                    "-c:v",
                    f"libx{self.video_codec_num}",
                    "-c:a",
                    "copy",
                ])
                if self.cfg.subtitle_type in [2, 4]:
                    cmd1.extend([
                        "-c:s",
                        "mov_text",
                        "-metadata:s:s:0",
                        f"language={subtitle_langcode}"
                    ])
                
                cmd2=[
                    "-movflags",
                    "+faststart",
                ]
                if self.cfg.video_autorate:
                    cmd2.extend(tools.ffmpeg_vfr_output_args())
                
                cmd2.extend(["-t", str(duration_s),  tmp_target_mp4_basename])
                if is_copy_mode:
                    cmd1[cmd1.index('-c:v')+1]='copy'
                    logger.debug(f'[Final video composition] copy mode, no re-encoding required:\n{cmd0+cmd1+cmd2}')
                    tools.runffmpeg(cmd0+cmd1+cmd2, cmd_dir=self.cfg.cache_folder, force_cpu=True)
                elif app_cfg.video_codec.startswith('libx') or settings.get('force_lib'):
                    # If hardware encoding is not supported, there is no need to try the hardware.
                    logger.debug(f'[Final video composition] Hardware encoding is not supported or a forced soft codec is specified:\n{cmd0+cmd1+cmd2}')
                    tools.runffmpeg(cmd0+cmd1+enc_qua+cmd2, cmd_dir=self.cfg.cache_folder, force_cpu=True)                    
                else:
                    # Try to use hardware codec
                    hw_decode_args,_,vcodec,enc_args=self._get_hard_cfg()
                    cmd1[cmd1.index('-c:v')+1]=vcodec
                    # If hardware processing fails, fall back to soft programming
                    try:
                        self._subprocess(cmd0+hw_decode_args+cmd1+enc_args+cmd2)
                    except:
                        cmd1[cmd1.index('-c:v')+1]=f'libx{self.video_codec_num}'
                        logger.warning(f'Hardware processing video synthesis failed, fallback to soft editing')
                        tools.runffmpeg(cmd0+cmd1+enc_qua+cmd2, cmd_dir=self.cfg.cache_folder, force_cpu=True)
                   
            #hard subtitles
            else:
                cmd1.append('-filter_complex')          
                subtitle_filter=[f"[0:v]subtitles=filename='{subtitles_file}'[v_out]"]
                cmd2=[
                    "-map", 
                    "[v_out]",
                    "-map", 
                    "1:a",
                    "-c:v",
                    f'libx{self.video_codec_num}',
                    '-c:a',
                    'copy',
                ]                 
                cmd3=["-movflags", "+faststart"]
                
                if self.cfg.video_autorate:
                    cmd3.extend(tools.ffmpeg_vfr_output_args())
                    
                cmd3.extend(["-t", str(duration_s), tmp_target_mp4_basename])
                if app_cfg.video_codec.startswith('libx')  or settings.get('force_lib'):
                    logger.debug(f'[Final video composition] Hardware codec is not supported or forced soft codec is specified:\n{cmd0+cmd1+cmd2}')
                    tools.runffmpeg(cmd0+cmd1+subtitle_filter+cmd2+enc_qua+cmd3, cmd_dir=self.cfg.cache_folder, force_cpu=True)
                else:
                    # If hardware processing fails, fall back to soft programming
                    try:
                        hw_decode_args,vf_string,vcodec,enc_args=self._get_hard_cfg(subtitles_file)
                        cmd2[cmd2.index('-c:v')+1]=vcodec
                        self._subprocess(cmd0+hw_decode_args+cmd1+[vf_string]+cmd2+enc_args+cmd3)
                    except:
                        cmd2[cmd2.index('-c:v')+1]=f'libx{self.video_codec_num}'
                        logger.warning(f'Hardware processing video synthesis failed, fallback to soft editing')
                        tools.runffmpeg(cmd0+cmd1+subtitle_filter+cmd2+enc_qua+cmd3,  cmd_dir=self.cfg.cache_folder, force_cpu=True)
        except Exception as e:
            msg = tr('Error in embedding the final step of the subtitle dubbing')
            raise RuntimeError(msg)

        os.chdir(ROOT_DIR)
        if Path(tmp_target_mp4).exists():
            try:
                shutil.copy2(tmp_target_mp4, self.cfg.targetdir_mp4)
            except Exception as e:
                raise RuntimeError(tr('Translation successful but transfer failed. ', tmp_target_mp4))

        self._create_txt()
        time.sleep(1)
        # It is possible that the program that outputs the original audio to the target folder is still executing, but it does not affect
        while output_source_output is not True:
            print(f'{output_source_output=}')
            time.sleep(1)
        return True

    def _get_origin_audio(self, output):
        # Take out the original audio in scenes where dubbing is not required
        if self.video_info.get('streams_audio', 0) == 0:
            # No audio stream
            return
        cmd = [
            "-y",
            "-i",
            self.cfg.name,
            "-vn"
        ]
        if self.video_info['audio_codec_name'] == 'aac':
            cmd.extend(['-c:a', 'copy'])
        else:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        cmd.append(output)
        return tools.runffmpeg(cmd)

    # ffmpeg progress log
    def _hebing_pro(self, protxt, video_time=0) -> None:
        while 1:
            if app_cfg.exit_soft or self.hasend or self.precent >= 100: return

            content = tools.read_last_n_lines(protxt)
            if not content:
                time.sleep(0.5)
                continue

            if content[-1] == 'progress=end':
                return
            idx = len(content) - 1
            end_time = "00:00:00"
            while idx > 0:
                if content[idx].startswith('out_time='):
                    end_time = content[idx].split('=')[1].strip()
                    break
                idx -= 1
            self._signal(text=tr('kaishihebing') + f' {end_time}')
            time.sleep(0.5)

    #Create description txt
    def _create_txt(self) -> None:
        try:

            with open(self.cfg.target_dir + f'/{tr("readme")}.txt',
                      'w', encoding="utf-8", errors="ignore") as f:
                f.write(f"""The following are all files that may be generated. Depending on the options configured during execution, some files may not be generated. The reason why these files and materials are generated is to facilitate users in need and further use other software for processing, without having to perform repeated tasks such as voice export, audio and video separation, and subtitle recognition.

        *.mp4 = final target video file
        {self.cfg.source_language_code}.m4a = audio file from the original video
        {self.cfg.target_language_code}.m4a = dubbed audio file
        removed_noise.wav = original audio file after noise reduction
        {self.cfg.source_language_code}.srt = subtitle file recognized based on sound in the original video
        {self.cfg.target_language_code}.srt = subtitle file translated into target language
        speaker.json = speaker flag
        -Noxxx.srt = subtitles before re-segmentation
        shuang.srt = bilingual subtitles
        vocal.wav = vocal audio file separated from the original video
        instrument.wav = background music audio file separated from the original video


        If you feel that this project is valuable to you and hope that the project can be maintained stably and continuously, you are welcome to sponsor small amounts. With certain financial support, I will be able to continue to invest more time and energy.
        Donation address: https://pvt9.com/about

        ====

        Here are the descriptions of all possible files that might exist. Depending on the configuration options when executing, some files may not be generated.

        *.mp4 = The final completed target video file
        {self.cfg.source_language_code}.m4a = The audio file in the original video
        {self.cfg.target_language_code}.m4a = dubbing audio
        removed_noise.wav = original video after removed noise
        {self.cfg.source_language_code}.srt = Subtitles recognized in the original video
        {self.cfg.target_language_code}.srt = Subtitles translated into the target language
        shuang.srt = Source language and target language subtitles srt 
        vocal.wav = The vocal audio file separated from the original video
        instrument.wav = The background music audio file separated from the original video


        If you feel that this project is valuable to you and hope that it can be maintained consistently, we welcome small sponsorships. With some financial support, I will be able to continue to invest more time and energy
        Donation address: https://ko-fi.com/jianchang512


        ====

        Github: https://github.com/jianchang512/pyvideotrans
        Docs: https://pvt9.com

                        """)
        except:
            pass





    # During video synthesis, return available hardware decoding parameters, subtitle embedding parameters, video encoding parameters, and quality-related parameters.
    def _get_hard_cfg(self, subtitles_file=None,codec=None):
        os_name = platform.system()
        if not app_cfg.video_codec:
            app_cfg.video_codec = get_video_codec()
        # Only used to determine the encoder part, the specific 264 or 265 is determined by codec
        hw_type=app_cfg.video_codec
        logger.debug(f'original{hw_type=}')
        
        if '_' in hw_type:
            _hw_type_list = hw_type.lower().split('_')
            if _hw_type_list[0]=='vaapi':
                hw_type='vaapi'
            else:
                hw_type=_hw_type_list[1]
        
        
        logger.debug(f'After finishing{hw_type=}')
        
        
        # Since hard subtitles are soft filtered, they must be suppressed in memory first.
        # Different hardware encoders may need to re-upload the image to the video memory (hwupload) after soft filtering
        
        # Default fallback to soft encoding
        codec=f'{self.video_codec_num}' if not codec else codec
        vcodec = f"libx{codec}"
        _crf=f'{settings.get("crf", 23)}'

        # Global parameters, related to hardware decoding
        global_args = []
        # Hard subtitle embedding parameters, soft subtitles are ignored
        vf_string = f"[0:v]subtitles=filename='{subtitles_file}'[v_out]"
        
        # Hardware compatibility is limited to prevent errors
        _preset=settings.get('preset','medium')
        if 'fast' in _preset:
            _preset='fast'
        elif 'slow' in _preset:
            _preset='slow'
        
        if _preset not in ['fast','slow','medium']:
            _preset='medium'
        enc_args = ['-crf', _crf,'-preset', _preset]
        
        
        # --- Parameter mapping table ---
        PRESET_MAP = {
            # NVENC: p1 (fastest) - p7 (slowest/best quality)
            'nvenc': {'fast': 'p2', 'medium': 'p4', 'slow': 'p7'}, 
            # QSV: veryfast, faster, fast, medium, slow, slower, veryslow
            'qsv': {'fast': 'fast', 'medium': 'medium', 'slow': 'slow'},
            # AMF: speed, balanced, quality
            'amf': {'fast': 'speed', 'medium': 'balanced', 'slow': 'quality'},
            # VAAPI: usually also accepts standard presets
            'vaapi': {'fast': 'fast', 'medium': 'medium', 'slow': 'slow'},
            # VideoToolbox: -preset parameter is generally not supported, leave blank to skip processing
            'videotoolbox': None 
        }
        
        # --- Nvidia (NVENC) ---
        if hw_type in ['nvenc']:
            vcodec = "h264_nvenc" if codec == '264' else "hevc_nvenc"
            # nvenc uses -cq (Constant Quality) instead of crf. The p4 default has a better balance between speed and quality.
            enc_args = ['-cq', _crf, '-preset', PRESET_MAP.get('nvenc').get(_preset,'p4')]
            # Prioritize hardware decoding
            if settings.get('hw_decode'):
                global_args=['-hwaccel','cuda','-hwaccel_output_format', 'cuda']
                vf_string = f"[0:v]hwdownload,format=nv12,subtitles=filename='{subtitles_file}',hwupload_cuda[v_out]"
            else:
                vf_string = f"[0:v]subtitles=filename='{subtitles_file}'[v_out]"

            return global_args,vf_string,vcodec,enc_args
        # --- Mac (VideoToolbox) ---
        if hw_type in ['videotoolbox']:
            vcodec = "h264_videotoolbox" if codec == '264' else "hevc_videotoolbox"
            # videotoolbox quality control, usually -q:v (range is about 40-60 visually lossless)
            quality = 100 - (int(_crf) * 1.4)
            enc_args = ['-q:v', f'{int(max(1, min(quality, 100)))}']
            return global_args,vf_string,vcodec,enc_args
            

        # --- Intel (QSV) & AMD (AMF) ---
        if hw_type in ['qsv', 'amf', 'vaapi']:
            if os_name == 'Linux':
                # [Linux special processing]
                # Under Linux, Intel and AMD open source drivers usually use the VAAPI interface.
                devices = glob.glob('/dev/dri/renderD*')
                device= devices[0] if devices else '/dev/dri/renderD128'
                if settings.get('hw_decode'):
                    global_args = ['-hwaccel', 'vaapi', '-hwaccel_device', device, '-hwaccel_output_format', 'vaapi']
                    vf_string = f"[0:v]hwdownload,format=nv12,subtitles=filename='{subtitles_file}',format=nv12,hwupload[v_out]"                
                else:
                    global_args = [
                        '-init_hw_device', f'vaapi=vaapi:{device}'
                    ]
                    vf_string = f"[0:v]subtitles=filename='{subtitles_file}',format=nv12,hwupload[v_out]"                
                vcodec = "h264_vaapi" if codec == '264' else "hevc_vaapi"
                enc_args = ['-qp', _crf,'-preset', PRESET_MAP.get('vaapi').get(_preset,'medium')]
                return global_args,vf_string,vcodec,enc_args
                # VAAPI requires that after the soft filter (subtitle) is processed, the pixel format should be converted and uploaded to the video memory
            
            # Windows environment
            if hw_type in ['qsv']:
                vcodec = "h264_qsv" if codec == '264' else "hevc_qsv"
                # QSV uses ICQ mode (Intelligent Constant Quality)
                enc_args = ['-global_quality', _crf, '-preset', PRESET_MAP.get('qsv').get(_preset,'medium')]
            else:
                vcodec = "h264_amf" if codec == '264' else "hevc_amf"
                # AMF uses constant quality parameters (CQP)
                enc_args = ['-rc', 'cqp', '-qp_p', _crf, '-qp_i', _crf, '-quality', PRESET_MAP.get('amf').get(_preset,'balanced')]
            return global_args,vf_string,vcodec,enc_args
        
        return global_args,vf_string,vcodec,enc_args

    
    def _subprocess(self,cmd):
        logger.debug(f'[Try hardware codec execution command]\n{" ".join(cmd)}\n')
        try:
            creationflags = 0
            if sys.platform == 'win32':
                creationflags = subprocess.CREATE_NO_WINDOW
            if app_cfg.exit_soft:
                return
            cmd=["ffmpeg",'-nostdin']+cmd
            subprocess.run(
                cmd,
                #stdout=subprocess.PIPE,
                #stderr=subprocess.PIPE,
                encoding="utf-8",
                errors='ignore',
                check=True,
                text=True,
                capture_output=True,
                creationflags=creationflags,
                cwd=self.cfg.cache_folder
            )
            return True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr or ""
            logger.error(f"An error occurred while trying to execute the command using hardware [CalledProcessError]:{error_message}\n{e.stdout}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while trying to execute the command using hardware [Exception]:{e}")
            raise
