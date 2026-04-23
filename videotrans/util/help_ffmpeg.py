import copy
import json
import re
import platform
import subprocess
import sys
import time
from pathlib import Path

from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,TEMP_DIR,logger
from videotrans.util import contants


def extract_concise_error(stderr_text: str, max_lines=3, max_length=250) -> str:
    if not stderr_text:
        return "Unknown error (empty stderr)"

    lines = stderr_text.strip().splitlines()
    if not lines:
        return "Unknown error (empty stderr lines)"
    
    result=re.findall(r'Error\s(.*?)\n',stderr_text)
    if not result:
        return " ".join(lines[-10:])
    return " ".join(result)

# Simplify the function to remove hardware support to avoid complexity and compatibility errors. It only supports hardware acceleration in the final merge stage and is implemented separately in trans_create.py
def runffmpeg(arg, *, noextname=None, uuid=None, force_cpu=True,cmd_dir=None):
    '\n    Execute ffmpeg command\n    '
    if settings.get('force_lib'):
        force_cpu=True

    final_args = arg

    cmd = ['ffmpeg', "-hide_banner", "-nostdin","-ignore_unknown",'-threads','0']
    if "-y" not in final_args:
        cmd.append("-y")
    cmd.extend(final_args)

    if cmd and Path(cmd[-1]).suffix:
        cmd[-1] = Path(cmd[-1]).as_posix()

    if settings.get('ffmpeg_cmd'):
        custom_params = [p for p in settings.get('ffmpeg_cmd','').split(' ') if p]
        cmd = cmd[:-1] + custom_params + cmd[-1:]

    try:
        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NO_WINDOW
        if app_cfg.exit_soft:
            return
        if cmd[-1].lower().endswith('.mp4'):
            logger.debug(f'[FFMPEG-CMD]:\n{" ".join(cmd)}\n')
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors='replace',
            check=True,
            text=True,
            creationflags=creationflags,
            cwd=cmd_dir
        )
        if noextname:
            app_cfg.queue_novice[noextname] = "end"
        return True
    except FileNotFoundError as e:
        logger.warning(f"Command not found:{cmd[0]}. Please make sure ffmpeg is installed and in your system PATH.")
        if noextname:
            app_cfg.queue_novice[noextname] = f"error:{e}"
        raise
    except subprocess.CalledProcessError as e:
        error_message = e.stderr or ""
        logger.warning(f"FFmpeg command execution failed (force_cpu={force_cpu}). \nCommand:{' '.join(cmd)}\nError:{error_message} {e.stdout}")
        err=extract_concise_error(e.stderr)
        if noextname:
            app_cfg.queue_novice[noextname] = f"error:{err}"
        # Separate prompts for path and name issues on win
        if sys.platform=='win32' and 'No such file or directory' in str(e):
            _err=get_filepath_from_cmd(cmd)
            err=_err or err
        raise RuntimeError(err)
    except Exception as e:
        if noextname:
            app_cfg.queue_novice[noextname] = f"error:{e}"
        logger.debug(f"An unknown error occurred while executing ffmpeg:{e}")
        raise

# Get the path after -i and the last path from the cmd list to determine whether the file name is regular.

def get_filepath_from_cmd(cmd:list):
    file_list=[cmd[i+1] for i,param in enumerate(cmd) if param=='-i']
    file_list.append(cmd[-1])
    special=['"',"'","`","*","?",":",">","<","|","\n","\r"]
    for file in file_list:
        if len(file)>=255:
            return  tr('The file path and file name may be too long. Please move the file to a flat and short directory and rename the file to a shorter name, ensuring that the length from the drive letter to the end of the file name does not exceed 255 characters: {},For example D:/videotrans/1.mp4 D:/videotrans/2.wav',file)
        for flag in special:
            if flag in file:
                return tr('There may be special characters in the file name or path. Please move the file to a simple directory consisting of English and numerical characters, rename the file to a simple name, and try again to avoid errors: {},For example D:/videotrans/1.mp4 D:/videotrans/2.wav',file)
    return None

def check_hw_on_start(_compat=None):
    get_video_codec(264)
    get_video_codec(265)

def get_video_codec(compat=None) -> str:
    "Test to determine the best available hardware-accelerated H.264/H.265 encoder.\n\n    Prefer hardware encoders based on platform. If the hardware test fails, fall back to software encoding.\n    Results will be cached. This release optimizes structure and efficiency through data-driven design and advance inspection.\n\n    Depends on 'config' module for settings and paths. Assuming 'ffmpeg' is in your system PATH,\n    The test input file exists and TEMP_DIR is writable.\n\n\n    Returns:\n        str: Recommended ffmpeg video encoder name (e.g. 'h264_nvenc', 'libx264')."
    import torch
    _codec_cache = app_cfg.codec_cache  # Use cache in config
    try:
        if not _codec_cache and Path(f'{ROOT_DIR}/videotrans/codec.json').exists():
            _codec_cache=json.loads(Path(f'{ROOT_DIR}/videotrans/codec.json').read_text(encoding='utf-8'))
    except Exception as e:
        logger.debug(f'parse codec.json error:{e}')
        
    
    plat = platform.system()
    if compat and compat in [264,265]:
        video_codec_pref=compat
    else:
        try:
            video_codec_pref = int(settings.get('video_codec', 264))
        except (ValueError, TypeError):
            logger.warning("'video_codec' in configuration is invalid. H.264 (264) will be used by default.")
            video_codec_pref = 264

    cache_key = f'{plat}-{video_codec_pref}'
    if cache_key in _codec_cache:
        logger.debug(f"Return cached codecs{cache_key}: {_codec_cache[cache_key]}")
        return _codec_cache[cache_key]

    h_prefix, default_codec = ('hevc', 'libx265') if video_codec_pref == 265 else ('h264', 'libx264')
    if video_codec_pref not in [264, 265]:
        logger.warning(f"Unexpected video_codec value '{video_codec_pref}'. Treated as H.264.")

    ENCODER_PRIORITY = {
        'Darwin': ['videotoolbox'],
        'Windows': ['nvenc', 'qsv', 'amf'],
        'Linux': ['nvenc', 'vaapi', 'qsv']
    }

    try:
        test_input_file = Path(ROOT_DIR) / "videotrans/styles/no-remove.mp4"
        temp_dir = Path(TEMP_DIR)
    except Exception as e:
        logger.warning(f"An error occurred while preparing to test the hardware encoder:{e}. Will use software encoding{default_codec}。")
        _codec_cache[cache_key] = default_codec
        return default_codec

    def test_encoder_internal(encoder_to_test: str, timeout: int = 10) -> bool:
        timestamp = int(time.time() * 1000)
        output_file = temp_dir / f"test_{encoder_to_test}_{timestamp}.mp4"
        command = [
            "ffmpeg", "-y", "-hide_banner",
            "-t", "1", "-i", str(test_input_file),
            "-c:v", encoder_to_test, "-f", "mp4", str(output_file)
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

        logger.debug(f"Testing whether the encoder is available:{encoder_to_test}...")
        success = False
        try:
            subprocess.run(
                command, check=True, capture_output=True, text=True,
                encoding='utf-8', errors='ignore', creationflags=creationflags, timeout=timeout
            )
            logger.debug(f"hardware encoder '{encoder_to_test}' available.")
            success = True
        except FileNotFoundError:
            logger.debug("'ffmpeg' command not found in PATH. Encoder testing is not possible.")
            raise  # Rethrow the exception and let the upper logic catch and terminate the test
        except subprocess.CalledProcessError as e:
            logger.debug(f"hardware encoder '{encoder_to_test}' Not available")
            raise
        except PermissionError:
            logger.debug(f"Failed while testing hardware encoder: writing{output_file} Permission is denied. {command=}")
            raise
        except subprocess.TimeoutExpired:
            logger.debug(f"hardware encoder '{encoder_to_test}'Test in{timeout} Timeout after seconds.{command=}")
            raise
        except Exception as e:
            logger.debug(f"Test hardware encoder{encoder_to_test} An unexpected error occurred:{e} {command=}")
            raise
        finally:
            try:
                if output_file.exists():
                    output_file.unlink(missing_ok=True)
            except OSError:
                pass
            return success

    selected_codec = default_codec  #Initialize as fallback option

    encoders_to_test = ENCODER_PRIORITY.get(plat, [])
    if not encoders_to_test:
        logger.debug(f"Unsupported platforms:{plat}. Will use software encoder{default_codec}。")
    else:
        logger.debug(f"Platform:{plat}. Detecting best ' by priority{h_prefix}'Encoder:{encoders_to_test}")
        try:
            for encoder_suffix in encoders_to_test:
                if encoder_suffix == 'nvenc':
                    try:
                        if not torch.cuda.is_available():
                            logger.debug('CUDA is not available, skipping nvenc tests.')
                            continue  # Skip the current loop and test the next encoder
                    except ImportError:
                        logger.debug('The torch module is not found, the nvenc test will be tried directly.')

                full_encoder_name = f"{h_prefix}_{encoder_suffix}"
                if test_encoder_internal(full_encoder_name):
                    selected_codec = full_encoder_name
                    logger.debug(f"Hardware encoder selected:{selected_codec}")
                    break
            else:  # for-else loop ends normally (no break)
                logger.debug(f"All hardware accelerators failed the test. Will use software encoder:{selected_codec}")

            _codec_cache[cache_key] = selected_codec
            # Save cache locally
            Path(f"{ROOT_DIR}/videotrans/codec.json").write_text(json.dumps(_codec_cache))
        except Exception as e:
            # Do not cache when an exception occurs
            logger.exception(f"In the event of an incident during encoder testing, software encoding will be used:{e}", exc_info=True)
            selected_codec = default_codec
    # 
    _codec_cache[cache_key] = selected_codec
    
    logger.debug(f"Finalized encoder to use:{selected_codec}")
    return selected_codec


class _FFprobeInternalError(Exception):
    'Custom exception for internal error delivery.'
    pass


def _run_ffprobe_internal(cmd: list[str]) -> str:
    '\n    (Internal function) Execute the ffprobe command and return its standard output.'
    # Ensure file path parameters are converted to POSIX style strings for better compatibility
    if Path(cmd[-1]).is_file():
        cmd[-1] = Path(cmd[-1]).as_posix()

    command = ['ffprobe'] + [str(arg) for arg in cmd]
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    # print(command)
    try:
        p = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors='replace',
            check=True,
            creationflags=creationflags
        )
        return p.stdout.strip()
    except FileNotFoundError as e:
        msg = f"Command not found: ffmpeg. Ensure FFmpeg is installed and in your PATH."
        logger.warning(msg)
        raise _FFprobeInternalError(msg) from e
    except subprocess.CalledProcessError as e:
        concise_error = extract_concise_error(e.stderr)
        logger.exception(f"ffprobe command failed: {concise_error}", exc_info=True)
        raise _FFprobeInternalError(concise_error) from e
    except (PermissionError, OSError) as e:
        logger.exception(e, exc_info=True)
        raise _FFprobeInternalError(e) from e


def runffprobe(cmd):
    try:
        stdout_result = _run_ffprobe_internal(cmd)
        if stdout_result:
            return stdout_result

        # If stdout is empty but the process did not error (uncommon), emulate the old error path
        # In _run_ffprobe_internal, if stderr has content and the return code is non-0,
        # will throw an exception directly, so this logic is mainly to cover extreme edge cases.
        logger.warning("ffprobe ran successfully but produced no output.")
        raise Exception("ffprobe ran successfully but produced no output.")

    except _FFprobeInternalError as e:
        raise
    except Exception as e:
        #Catch other unexpected errors and re-raise them to maintain consistent behavior
        logger.exception(e, exc_info=True)
        raise



def get_video_info(mp4_file, *, video_fps=False, video_scale=False, video_time=False, get_codec=False):
    '\n    Get video information.\n    '


    if not Path(mp4_file).exists():
        raise Exception(f'{mp4_file} is not exists')
    try:
        out_json = runffprobe(
            ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', mp4_file]
        )
        if not out_json:
            raise RuntimeError(tr('Failed to parse {} information, please confirm that the file can be played normally',mp4_file))
    except Exception as e:
        raise

    try:
        out = json.loads(out_json)
    except json.JSONDecodeError:
        raise RuntimeError(tr('Failed to parse {} information, please confirm that the file can be played normally',mp4_file))


    if "streams" not in out or not out["streams"]:
        raise RuntimeError(tr('The original file {} does not contain any audio or video data. The file may be damaged. Please confirm that the file can be played.',mp4_file))

    result = {
        "video_fps": 30,
        "r_frame_rate":30,
        "video_codec_name": "",
        "audio_codec_name": "",
        "width": 0,
        "height": 0,
        "time": 0,
        "streams_len": len(out['streams']),
        "streams_audio": 0,
        "video_streams":0,
        "color": "yuv420p"
    }
    try:
        # The duration in the first stream shall prevail, but there may be some formats, such as mkv, there is no duration field in the first stream or it is always 0
        result['time']=int(float(out['streams'][0]['duration'])*1000)#The length of the first stream shall prevail
        if result['time']<=0:
            result['time']=int(float(out['format']['duration'])*1000)
    except:
        result['time']=int(float(out['format']['duration'])*1000)
    
        

    video_stream = next((s for s in out['streams'] if s.get('codec_type') == 'video'), None)
    audio_streams = [s for s in out['streams'] if s.get('codec_type') == 'audio']

    result['streams_audio'] = len(audio_streams)
    if audio_streams:
        result['audio_codec_name'] = audio_streams[0].get('codec_name', "")

    if video_stream:
        result['video_streams']=1
        result['video_codec_name'] = video_stream.get('codec_name', "")
        result['width'] = int(video_stream.get('width', 0))
        result['height'] = int(video_stream.get('height', 0))
        result['color'] = video_stream.get('pix_fmt', 'yuv420p').lower()

        # FPS calculation logic
        def parse_fps(rate_str):
            try:
                num, den = map(int, rate_str.split('/'))
                return num / den if den != 0 else 0
            except:
                return 0

        fps1 = parse_fps(video_stream.get('r_frame_rate'))
        
        if not fps1 or fps1<1:
            fps_avg=parse_fps(video_stream.get('avg_frame_rate'))
        else:
            fps_avg = fps1



        result['video_fps'] = fps_avg if 1 <= fps_avg <= 120 else 30
        result['r_frame_rate'] = video_stream.get('r_frame_rate',result['video_fps'])

    # Ensure backward compatibility
    if video_time:
        return result['time']
    if video_fps:
        return result['video_fps']
    if video_scale:
        return result['width'], result['height']
    if get_codec:
        return result['video_codec_name'], result['audio_codec_name']

    return result


def _get_ms_from_media(file):
    ms=0
    ext=Path(file).suffix.lower()[1:]
    try:
        if ext in contants.VIDEO_EXTS:
            ms=int(float(runffprobe(['-v','error','-select_streams','v:0','-show_entries','stream=duration','-of','default=noprint_wrappers=1:nokey=1',file]))*1000)
        elif ext in contants.AUDIO_EXITS:
            ms=int(float(runffprobe(['-v','error','-select_streams','a:0','-show_entries','stream=duration','-of','default=noprint_wrappers=1:nokey=1',file]))*1000)
    except Exception:
        # Other formats such as mkv may not be able to read duration from the stream
        pass
    if ms==0:
        ms=int(float(runffprobe(['-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',file])))
    return ms


# Get the video stream duration without audio
def get_video_ms_noaudio(mp4):
    return _get_ms_from_media(mp4)


# Get the duration of a video in ms
def get_video_duration(file_path):
    return _get_ms_from_media(file_path)
# Get the audio duration and return ms
def get_audio_time(audio_file):
    return _get_ms_from_media(audio_file)



def conver_to_16k(audio, target_audio):
    cmd=[
            "-y",
            "-i",
            Path(audio).as_posix(),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            '-af', "volume=2.0,alimiter=limit=1.0",
            Path(target_audio).as_posix()
    ]
    return runffmpeg(cmd)

# Convert wav to m4a cuda + h264_cuvid
def wav2m4a(wavfile, m4afile, extra=None):
    cmd = [
        "-y",
        "-i",
        Path(wavfile).as_posix(),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        Path(m4afile).as_posix()
    ]
    if extra:
        cmd = cmd[:3] + extra + cmd[3:]
    return runffmpeg(cmd)


def create_concat_txt(filelist, concat_txt=None):

    '\n    Create a connection file for use by FFmpeg concat.\n    Make sure to write an absolute path to avoid FFmpeg not being able to find the file due to working directory problems.\n    '
    txt = []
    for it in filelist:
        path_obj = Path(it)
        if not path_obj.exists() or path_obj.stat().st_size == 0:
            continue
        # Store names to avoid command line truncation errors on Windows due to too many fragments.
        txt.append(f"file '{path_obj.name}'")
    if not txt:
        # If there is no valid file, creating an empty concat file may cause errors. It is better to throw an exception directly.
        raise ValueError("Cannot create concat txt from an empty or invalid file list.")

    logger.debug(f'{concat_txt=},{filelist[0]=}')
    with open(concat_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt))
    return concat_txt


# Multiple audio clip connections
def concat_multi_audio(*, out=None, concat_txt=None):
    if out:
        out = Path(out).as_posix()


    cmd = ['-y', '-f', 'concat', '-safe', '0', '-i', concat_txt, "-b:a", "128k"]
    if out.endswith('.m4a'):
        cmd += ['-c:a', 'aac']
    elif out.endswith('.wav'):
        cmd += ['-c:a', 'pcm_s16le']
    runffmpeg(cmd + [out],cmd_dir=Path(concat_txt).parent.as_posix())
    return True


# Currently only used to extend the background sound after video translation
def change_speed_rubberband(input_path,out_file, target_duration):
    '\n    Audio speed shifting using Rubber Band\n    '
    try:
        import pyrubberband as pyrb
    except Exception as e:
        logger.warning('Failed when doing audio speed shifting because rubberband library is installed')
        return
        
    import soundfile as sf
    import numpy as np  # Add numpy for channel processing
    try:
        y, sr = sf.read(input_path)
        if len(y) == 0:
            logger.warning(f"[Audio-RB] Empty audio file:{input_path}")
            return
            
        current_duration = int((len(y) / sr) * 1000)
        
        if target_duration <= 0: target_duration = 1
        
        # [Logic Optimization] If the target duration is longer than the current time, it means that the audio needs to be slowed down.
        # But in the current alignment strategy, audio is usually only compressed (accelerated).
        # If target > current does occur, it usually means we should pad the silence instead of stretching the audio.
        # For safety reasons, if the difference is too large, it will not be processed.
        #if target_duration > current_duration:
        # # Allow small errors, or be handled by subsequent silent padding
        # logger.debug(f"[Audio-RB] target duration ({target_duration}) > current duration ({current_duration}), skip speed change, fill in with silence.")
        #     return

        time_stretch_rate = current_duration / target_duration
        
        # Limit range
        time_stretch_rate = max(0.2, min(time_stretch_rate, 50.0))
        
        logger.debug(f"[Audio-RB] {input_path} Original length:{current_duration}ms -> target:{target_duration}ms magnification:{time_stretch_rate:.2f}")

        y_stretched = pyrb.time_stretch(y, sr, time_stretch_rate)
        
        # [Key Correction] Make sure the output is Stereo (2 channels) to prevent subsequent ffmpeg concat errors.
        # If it is mono (ndim=1), copy it to dual channel
        if y_stretched.ndim == 1:
            y_stretched = np.column_stack((y_stretched, y_stretched))
        
        sf.write(out_file, y_stretched, sr)
        
    except Exception as e:
        logger.error(f"[Audio-RB] Audio processing failed{input_path}: {e}")
        return


def precise_speed_up_audio(*, file_path=None, out=None, target_duration_ms=None):
    from pydub import AudioSegment
    ext = file_path[-3:].lower()
    out_ext=ext
    if out:
        out_ext=out[-3:].lower()
    codecs = {"m4a": "aac", "mp3": "libmp3lame", "wav": "pcm_s16le"}
    audio = AudioSegment.from_file(file_path, format='mp4' if ext == 'm4a' else ext)

    current_duration_ms = len(audio)

    # Complete acceleration using atempo filter
    # Construct atempo filter chain
    # atempo restriction: parameters must be between [0.5, 2.0]
    atempo_list = []
    speed_factor = current_duration_ms / target_duration_ms

    # Handle acceleration situations (> 2.0)
    while speed_factor > 2.0:
        atempo_list.append("atempo=2.0")
        speed_factor /= 2.0

    #Put in the remaining magnification
    atempo_list.append(f"atempo={speed_factor}")

    # Use commas to connect filters to form a series effect, such as "atempo=2.0,atempo=1.5"
    filter_str = ",".join(atempo_list)
    rubberband_filter_str = f"rubberband=tempo={current_duration_ms / target_duration_ms}"
    if not out:
        Path(file_path).rename(file_path+f".{ext}")
        file_path=file_path+f".{ext}"
        out=file_path
    cmd = [
        '-y',
        '-i',
        file_path,
        '-filter:a',
        rubberband_filter_str,
        '-t', f"{target_duration_ms/1000.0}",  # Force cutting to the target duration to prevent accuracy errors
        '-ar', "48000",
        '-ac', "2",
        '-c:a', codecs.get(out_ext,'pcm_s16le'),
        out
    ]
    try:
        runffmpeg(cmd, force_cpu=True)
    except Exception as e:
        cmd[4]=filter_str
        runffmpeg(cmd, force_cpu=True)


# Extract a segment from the audio
def cut_from_audio(*, ss, to, audio_file, out_file):
    from . import help_srt
    from pathlib import Path
    if not Path(audio_file).exists():
        return False
    Path(out_file).parent.mkdir(exist_ok=True,parents=True)
    cmd = [
        "-y",
        "-i",
        audio_file,
        "-ss",
        help_srt.format_time(ss, '.'),
        "-to",
        help_srt.format_time(to, '.'),
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        out_file
    ]
    return runffmpeg(cmd)


def send_notification(title, message):
    if app_cfg.exec_mode=='cli':
        print(f'\n*****[{title}]: {message}*****\n')
        return
    if app_cfg.exit_soft or settings.get('dont_notify',False):
        return
    from plyer import notification
    try:
        notification.notify(
            title=title[:60],
            message=message[:120],
            ticker="pyVideoTrans",
            app_name="pyVideoTrans",
            app_icon=ROOT_DIR + '/videotrans/styles/icon.ico',
            timeout=10  # Display duration in seconds
        )
    except Exception:
        pass



def remove_silence_wav(audio_file):
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    
    audio = AudioSegment.from_file(audio_file, format="wav")    
    
    # The mute of TTS is usually very clean. If there is still slight noise in the background, you can turn it up.
    silence_threshold = -40
    
    # As long as the silence lasts for more than 50ms, it will be detected
    min_silence_len = 50

    # 3. Detect non-silent segments
    nonsilent_chunks = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=1
    )

    # 4. Process cutting logic
    if len(nonsilent_chunks) > 0:
        # At the beginning and end of the detected non-silent sound, retain an additional 100 milliseconds of sound to prevent weak consonants or codas from being swallowed
        head_padding_ms = 80   # Header retained for 80 milliseconds
        tail_padding_ms = 180  # The tail sound is usually dragged out longer and is kept for 180 milliseconds.
        
        # Get the start time of the first non-silent block
        raw_start = nonsilent_chunks[0][0]
        # Get the end time of the last non-silent block
        raw_end = nonsilent_chunks[-1][1]
        
        # Calculate the final cropping position, use max and min to prevent the index from exceeding the total length of the audio
        start_trim = max(0, raw_start - head_padding_ms)
        end_trim = min(len(audio), raw_end + tail_padding_ms)
        
        # Trim audio
        trimmed_audio = audio[start_trim:end_trim]
        trimmed_audio.export(audio_file, format="wav")
        return True
    
    return False # If all is mute, return False


# input_file_path may be a string: file path, or audio data
def remove_silence_from_end(input_file_path,is_start=True):
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    # Load the audio file
    format = "wav"
    if isinstance(input_file_path, str):
        format = input_file_path.split('.')[-1].lower()
        if format in ['wav', 'mp3', 'm4a']:
            audio = AudioSegment.from_file(input_file_path, format=format if format in ['wav', 'mp3'] else 'mp4')
        else:
            #Convert to mp3
            try:
                runffmpeg(['-y', '-i', input_file_path, input_file_path + ".mp3"])
                audio = AudioSegment.from_file(input_file_path + ".mp3", format="mp3")
            except Exception:
                return input_file_path

    else:
        audio = input_file_path

    # Detect non-silent chunks
    nonsilent_chunks = detect_nonsilent(
        audio,
        min_silence_len=10,
        silence_thresh=audio.dBFS - 20
    )

    # If we have nonsilent chunks, get the start and end of the last nonsilent chunk
    if nonsilent_chunks:
        start_index, end_index = nonsilent_chunks[-1]
    else:
        # If the whole audio is silent, just return it as is
        return input_file_path

    # Remove the silence from the end by slicing the audio segment
    trimmed_audio = audio[:end_index]
    if isinstance(input_file_path, str):
        if format in ['wav', 'mp3', 'm4a']:
            trimmed_audio.export(input_file_path, format=format if format in ['wav', 'mp3'] else 'mp4')
            return input_file_path
        try:
            trimmed_audio.export(input_file_path + ".mp3", format="mp3")
            runffmpeg(['-y', '-i', input_file_path + ".mp3", input_file_path])
        except Exception:
            pass
        return input_file_path
    return trimmed_audio


def format_video(name, target_dir=None):
    from . import help_misc
    raw_pathlib = Path(name)
    # Original base name, for example `1.mp4`
    raw_basename = raw_pathlib.name
    # Base name without suffix, for example `1`
    raw_noextname = raw_pathlib.stem
    # Suffix, such as `.mp4`
    ext = raw_pathlib.suffix.lower()[1:]
    # directory
    raw_dirname = raw_pathlib.parent.resolve().as_posix()

    obj = {
        # The original file name includes the full path, such as F:/python/1.mp4
        "name": name,
        # Original directory such as F:/python
        "dirname": raw_dirname,
        # Basic name with suffix such as 1.mp4
        "basename": raw_basename,
        #Basic name without suffix, such as 1
        "noextname": raw_noextname,
        # Remove the dot from the extension. Such as mp4
        "ext": ext
        #Final storage target location, save directly here
    }

    # If the target folder exists, generate a subfolder under it with the base name without suffix
    if target_dir:
        obj['target_dir'] = Path(f'{target_dir}/{raw_noextname}-{ext}').as_posix()
    # Unique ID identifier Use name and size instead to use cache, for example, determine whether it already exists before noise reduction
    obj['uuid'] = help_misc.get_md5(f'{name}-{time.time()}')[:10]
    return obj


# When exporting a wav file that may be larger, use this function to avoid audio errors larger than 4G.
def large_wav_export_with_soundfile(audio_segment, output_path: str):
    import numpy as np
    import soundfile as sf

    # audio_segment = AudioSegment.from_file(...)
    "Export pydub's AudioSegment object using the soundfile module to support large files.\n    \n    :param audio_segment: audio object of pydub\n    :param output_path: output .wav file path"

    # 1. Get original PCM data (bytes) from pydub
    raw_data = audio_segment.raw_data

    # 2. Convert the original data to a NumPy array, which is the standard input format of soundfile
    # Need to determine the correct data type (dtype)
    sample_width = audio_segment.sample_width
    if sample_width == 1:
        dtype = np.int8  # 8-bit PCM
    elif sample_width == 2:
        dtype = np.int16  # 16-bit PCM
    elif sample_width == 4:
        dtype = np.int32  # 32-bit PCM
    else:
        raise ValueError(f"Unsupported sample bit widths:{sample_width}")

    # frombuffer creates an array from a byte string, 'C' means in C language order
    numpy_array = np.frombuffer(raw_data, dtype=dtype)

    # 3. If it is multi-channel, the one-dimensional array needs to be reshaped into (n_frames, n_channels)
    num_channels = audio_segment.channels
    if num_channels > 1:
        numpy_array = numpy_array.reshape((-1, num_channels))

    # 4. Use soundfile to write to a file
    # soundfile will automatically handle the file size, switching to RF64 when needed
    sf.write(
        output_path,
        numpy_array,
        audio_segment.frame_rate,
        subtype='PCM_16' if sample_width == 2 else ('PCM_24' if sample_width == 3 else 'PCM_32')  #Select subtypes as needed
    )
