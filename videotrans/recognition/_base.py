import re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union
from tenacity import RetryError

from videotrans.configure.config import tr, params, settings, app_cfg, logger, TEMP_DIR
from videotrans.configure._base import BaseCon
from videotrans.util import tools
from videotrans.task.vad import get_speech_timestamp, get_speech_timestamp_silero
from pydub import AudioSegment


@dataclass
class BaseRecogn(BaseCon):
    recogn_type: int = 0  # Speech recognition type
    # Subtitle detection language
    detect_language: str = None

    # Model name
    model_name: Optional[str] = None
    # 16k wav to be recognized
    audio_file: Optional[str] = None
    # Temporary directory
    cache_folder: Optional[str] = None

    #task id
    uuid: Optional[str] = None
    # Enable cuda acceleration
    is_cuda: bool = False

    # Subtitle embedding type 0 1234
    subtitle_type: int = 0
    # Is it ended?
    has_done: bool = field(default=False, init=False)
    # Error message
    error: str = field(default='', init=False)
    # Identify api address
    api_url: str = field(default='', init=False)
    #Device type cpu cuda
    device: str = field(init=False, default='cpu')
    # punctuation marks
    flag: List[str] = field(init=False, default_factory=list)
    # Store the returned subtitle list
    raws: List = field(default_factory=list, init=False)
    # Connect between words, Chinese, Japanese, Korean and Cantonese are directly connected, other spaces
    join_word_flag: str = field(init=False, default=' ')
    # Do you need to convert to Simplified Chinese?
    jianfan: bool = False
    #Number of characters in subtitle lines
    maxlen: int = 20
    audio_duration: int = 0
    max_speakers: int = -1  # Speaker, -1 does not enable speakers, 0 = no limit on the number, >0 maximum number of speakers
    llm_post: bool = False  #Whether to perform llm re-segmentation? If so, there is no need to make simple corrections after the recognition is completed.
    speech_timestamps: List = field(default_factory=list)  # vad cut data
    recogn2pass: bool = False

    def __post_init__(self):
        super().__post_init__()
        logger.debug(f'BaseRecognition initialization')
        if not tools.vail_file(self.audio_file):
            raise RuntimeError(f'No {self.audio_file}')
        self.device = 'cuda' if self.is_cuda else 'cpu'
        # Common punctuation marks
        self.flag = [",", ".", "?", "!", ";", "，", "。", "？", "；", "！"]
        # Soft punctuation such as commas
        self.half_flag = [",", "，", "-", "、", ":", "："]
        # Sentence termination punctuation
        self.end_flag = [".", "。", "?", "？", "!", "！"]
        # Connecting characters Chinese, Japanese, Korean and Cantonese are connected directly without spaces. Other languages are connected with spaces.
        self.join_word_flag = " "
        # Chinese, Japanese and Korean characters
        self.is_cjk = False

        if self.detect_language and self.detect_language[:2].lower() in ['zh', 'ja', 'ko', 'yu']:
            self.maxlen = int(float(settings.get('cjk_len', 20)))
            self.jianfan = True if self.detect_language[:2] == 'zh' and settings.get('zh_hant_s') else False
            self.flag.append(" ")
            self.join_word_flag = ""
            self.is_cjk = True
        else:
            self.maxlen = int(float(settings.get('other_len', 60)))
            self.jianfan = False

    # Some recognition channels need to use VAD to cut audio clips of appropriate length in advance, and then recognize the clips. Each recognition result is a subtitle.
    # whisper model and pre-split unchecked, no need to cut
    def _vad_split(self):
        _st = time.time()
        _vad_type = settings.get('vad_type', 'tenvad')
        title = f'VAD:{_vad_type} split audio...'
        self._signal(text=title)

        _threshold = float(settings.get('threshold', 0.5))
        _min_speech = max(int(float(settings.get('min_speech_duration_ms', 1000))), 0)
        # ten-vad must not be less than 500ms
        if _vad_type == 'tenvad':
            _min_speech = max(_min_speech, 500)

        # The longest time shall not be greater than 30s, and shall not be less than _min_speech
        _max_speech = max(min(int(float(settings.get('max_speech_duration_s', 6)) * 1000), 30000), _min_speech + 1000)
        # The silence threshold must not be lower than 50ms
        _min_silence = max(int(settings.get('min_silence_duration_ms', 600)), 50)
        if self.recogn2pass:
            # 2 recognitions, both halved, to generate short subtitles
            _min_speech = int(max(500, _min_speech // 2))
            # Cannot be less than _min_speech+1000 and cannot be greater than 3000ms
            _max_speech = int(max(_max_speech // 2, _min_speech + 1000))
            # Cannot be greater than 1000ms, and cannot be less than 50ms
            _min_silence = max(min(1000, _min_silence // 2), 50)

        logger.debug(
            f'[Before VAD {_vad_type}][{self.recogn2pass=}],{_min_speech=}ms,{_max_speech=}ms,{_min_silence=}ms')
        kw = {
            "input_wav": self.audio_file,
            "threshold": _threshold,
            "min_speech_duration_ms": _min_speech,
            "max_speech_duration_ms": _max_speech,
            "min_silent_duration_ms": _min_silence
        }
        try:
            self.speech_timestamps = self._new_process(
                callback=get_speech_timestamp if _vad_type == 'tenvad' else get_speech_timestamp_silero,
                title=title,
                kwargs=kw)
        except Exception:
            if not self.recogn2pass:
                raise

        self._signal(text=f'[VAD] process ended {int(time.time() - _st)}s')

    # run->_exec
    def run(self) -> Union[List[Dict], None]:
        _st = time.time()
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        self._signal(text=f"check model")

        if hasattr(self, '_download'):
            self._download()

        self._signal(text=f"starting transcription")
        try:
            srt_list = []
            res = self._exec()
            if not res:
                raise RuntimeError(tr('No speech was detected, please make sure there is human speech in the selected audio/video and that the language is the same as the selected one.'))
            for i, it in enumerate(res):
                text = it['text'].strip()
                # Remove invalid subtitle lines, lines composed entirely of symbols
                if text and not re.match(r'^[,.?!;\'"_，。？；‘’“”！~@#￥%…&*（【】）｛｝《、》\$\(\)\[\]\{\}=+\<\>\s-]+$', text):
                    it['line'] = len(srt_list) + 1
                    if not it.get('startraw'):
                        it['startraw'] = tools.ms_to_time_string(ms=it['start_time'])
                        it['endraw'] = tools.ms_to_time_string(ms=it['end_time'])
                        it['time'] = f"{it['startraw']} --> {it['endraw']}"
                    srt_list.append(it)

            if not srt_list:
                return []

            # Fix timestamp overlap
            for i, it in enumerate(srt_list):
                if i > 0 and srt_list[i - 1]['end_time'] > it['start_time']:
                    logger.warning(
                        f'Correct subtitle timeline overlap: change the previous subtitle end_time={srt_list[i - 1]["end_time"]}Change to current subtitle start_time,{it=}')
                    srt_list[i - 1]['end_time'] = it['start_time']
                    srt_list[i - 1]['endraw'] = tools.ms_to_time_string(ms=it['start_time'])
                    srt_list[i - 1]['time'] = f"{srt_list[i - 1]['startraw']} --> {srt_list[i - 1]['endraw']}"
            if self.recogn2pass:
                return srt_list
            # LLM re-segments the sentence, merges short subtitles, whisper model and does not pre-segment is not selected, these three cases are returned directly
            if self.llm_post or not settings.get('merge_short_sub', True) or (
                    self.recogn_type < 2 and not settings.get('whisper_prepare')):
                if not self.llm_post:
                    for it in srt_list:
                        it['text'] = it['text'].strip('。').strip()
                return srt_list
            # Merge subtitles that are too short into adjacent subtitles to meet the min_speech_duration_ms requirement. The first and last subtitles are not merged.
            return self._fix_post(srt_list)
        except RetryError as e:
            raise e.last_attempt.exception()
        except Exception:
            raise
        finally:
            self._signal(text=f'STT ended:{int(time.time() - _st)}s')
            logger.debug(f'[Voice Recognition] Channel{self.recogn_type},{self.model_name}:Total time spent:{int(time.time() - _st)}s')

    # If LLM is not selected to re-segment the sentence and Merge short subtitles is selected, the identified subtitles will be simply corrected.
    def _fix_post(self, srt_list):
        post_srt_raws = []
        min_speech = max(300, int(float(settings.get('min_speech_duration_ms', 1000))))
        logger.debug(f'Make simple corrections to the recognized subtitles,{min_speech=}')
        for idx, it in enumerate(srt_list):
            if not it['text'].strip():
                continue
            if idx == 0 or idx == len(srt_list) - 1 or it['end_time'] - it['start_time'] >= min_speech:
                post_srt_raws.append(it)
            else:
                # Less than 1s
                prev_diff = it['start_time'] - post_srt_raws[-1]['end_time']
                next_diff = srt_list[idx + 1]['start_time'] - it['end_time']
                # The previous one is not the end of punctuation, but the current one is the end of punctuation
                # The previous one was the pause punctuation in the middle of the sentence, and the current one is the punctuation mark at the end of the sentence.
                # Closer to the previous one
                if (post_srt_raws[-1]['text'][-1] not in self.flag and it['text'][-1] in self.flag) or (
                        post_srt_raws[-1]['text'][-1] in self.half_flag and it['text'][
                    -1] in self.end_flag) or prev_diff <= next_diff:
                    logger.warning(
                        f'Subtitle duration is less than{min_speech=}, need to be merged into the previous subtitles,{prev_diff=},{next_diff=}, current subtitles ={it}, front subtitle ={post_srt_raws[-1]}')
                    post_srt_raws[-1]['end_time'] = it['end_time']
                    post_srt_raws[-1]['endraw'] = tools.ms_to_time_string(ms=it['end_time'])
                    post_srt_raws[-1]['time'] = f"{post_srt_raws[-1]['startraw']} --> {post_srt_raws[-1]['endraw']}"
                    post_srt_raws[-1]['text'] += ' ' + it['text']
                else:
                    logger.warning(
                        f'Subtitle duration is less than{min_speech=}, need to be merged into the following subtitles,{prev_diff=},{next_diff=}, current subtitles ={it},subtitles behind ={srt_list[idx + 1]}')
                    srt_list[idx + 1]['text'] = it['text'] + ' ' + srt_list[idx + 1]['text']
                    srt_list[idx + 1]['start_time'] = it['start_time']
                    srt_list[idx + 1]['startraw'] = tools.ms_to_time_string(ms=it['start_time'])
                    srt_list[idx + 1]['time'] = f"{srt_list[idx + 1]['startraw']} --> {srt_list[idx + 1]['endraw']}"

        if len(post_srt_raws) < 2:
            return post_srt_raws

        # If the duration of the first subtitle is less than min_speech, and the gap between the first subtitle and the second subtitle is less than 2s, merge the first subtitle into the second one; if the gap is too large, it will be an independent sentence and will not be merged.
        if post_srt_raws[0]['end_time'] - post_srt_raws[0]['start_time'] < min_speech and post_srt_raws[1][
            'start_time'] - post_srt_raws[0]['end_time'] < 2000:
            post_srt_raws[1]['start_time'] = post_srt_raws[0]['start_time']
            post_srt_raws[1]['text'] = post_srt_raws[0]['text'] + self.join_word_flag + post_srt_raws[1]['text']
            del post_srt_raws[0]
        if len(post_srt_raws) < 2:
            return post_srt_raws

        # Then determine that the duration of the last subtitle is shorter than min_speech, and the gap from the previous subtitle is less than 2s, then the last subtitle will be merged into the previous one; if the gap is too large, it will be an independent sentence and will not be merged.
        if post_srt_raws[-1]['end_time'] - post_srt_raws[-1]['start_time'] < min_speech and post_srt_raws[-1][
            'start_time'] - post_srt_raws[-2]['end_time'] < 2000:
            post_srt_raws[-2]['end_time'] = post_srt_raws[-1]['end_time']
            post_srt_raws[-2]['text'] += self.join_word_flag + post_srt_raws[-1]['text']
            del post_srt_raws[-1]
        if len(post_srt_raws) < 2:
            return post_srt_raws

        # If there is punctuation in the middle of the current subtitle, and the number of words before the first punctuation is less than 4, and there is no punctuation at the end of the previous subtitle, give the previous subtitle
        for i, it in enumerate(post_srt_raws):
            if i == 0 or i == len(post_srt_raws) - 1:
                continue
            if post_srt_raws[i - 1]['end_time'] != it['start_time']:
                continue
            t = [t for t in re.split(r'[,.，。]', it['text']) if t.strip()]
            # No valid text
            if not t:
                it['text'] = ''
                continue
            # Only one group
            if len(t) == 1:
                continue
            #Chinese, Japanese and Korean word count>3
            if self.is_cjk and len(t[0].strip()) > 3:
                continue

            # There is punctuation at the end of the previous subtitle
            if post_srt_raws[i - 1]['text'][-1] in self.flag:
                continue
            if not self.is_cjk and len(t[0].strip().split(' ')) > 3:
                continue

            post_srt_raws[i - 1]['text'] += self.join_word_flag + it['text'][:len(t[0]) + 1]
            logger.warning(f'The original text of this subtitle ={it["text"]}, the text merged into the previous subtitle ={it["text"][:len(t[0]) + 1]}')
            it['text'] = it["text"][len(t[0]) + 1:]
            logger.warning(f'Remaining question text{it["text"]}\n')

        # If there is punctuation in the middle of the current subtitle, and the words before the last punctuation are less than 4, give the next subtitle
        for i, it in enumerate(post_srt_raws):
            if i == 0 or i == len(post_srt_raws) - 1:
                continue
            if post_srt_raws[i + 1]['start_time'] != it['end_time']:
                continue
            t = [t for t in re.split(r'[,.，。]', it['text']) if t.strip()]
            # No valid text
            if not t:
                it['text'] = ''
                continue
            # Only one group
            if len(t) == 1:
                continue
            # There is punctuation at the end of the subtitles
            if it['text'][-1] in self.flag:
                continue
            #Chinese, Japanese and Korean word count>3
            if self.is_cjk and len(t[-1].strip()) > 3:
                continue
            if not self.is_cjk and len(t[-1].strip().split(' ')) > 3:
                continue

            post_srt_raws[i + 1]['text'] = it['text'][-len(t[-1]):] + self.join_word_flag + post_srt_raws[i + 1]['text']
            logger.warning(f'The original text of this subtitle ={it["text"]}, merge into next subtitle text ={it["text"][-len(t[-1]):]}')
            it['text'] = it["text"][:-len(t[-1])]
            logger.warning(f'remaining text{it["text"]}\n')

        # Remove all . at the end.
        for it in post_srt_raws:
            it['text'] = it['text'].strip('。').strip()
        return [it for it in post_srt_raws if it['text'].strip()]

    def _exec(self) -> Union[List[Dict], None]:
        pass

    def _padforaudio(self):
        silent_segment = AudioSegment.silent(duration=500)
        silent_segment.set_channels(1).set_frame_rate(16000)
        return silent_segment

    def cut_audio(self):
        dir_name = f"{TEMP_DIR}/clip_{time.time()}"
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        data = []
        if not self.speech_timestamps:
            self._vad_split()
        speech_chunks = self.speech_timestamps
        speech_len = len(speech_chunks)
        audio = AudioSegment.from_wav(self.audio_file)
        # For forced splitting of more than 30 seconds and forced merging of less than 1 second, this prevents errors from being reported if some recognition engines do not support it.
        check_1 = []
        # The minimum trimmed speech duration must meet the min_speech_duration_ms requirement, merge those that are too short
        min_speech_duration_ms = min(25000, max(int(settings.get('min_speech_duration_ms', 1000)), 1000))
        for i, it in enumerate(speech_chunks):
            diff = it[1] - it[0]
            if diff < min_speech_duration_ms:
                # Gap from front
                prev_diff = it[0] - check_1[-1][1] if len(check_1) > 0 else None
                # Distance to next gap
                next_diff = speech_chunks[i + 1][0] - it[1] if i < speech_len - 1 else None
                if prev_diff is None and next_diff is not None:
                    logger.warning(
                        f'cut_audio duration is less than{min_speech_duration_ms}ms requires the start time of the next subtitle left shift,{diff=},{prev_diff=},{next_diff=}')
                    # Insert after
                    speech_chunks[i + 1][0] = it[0]
                elif prev_diff is not None and next_diff is None:
                    # Front extension
                    logger.warning(
                        f'cut_audio duration is less than{min_speech_duration_ms}ms requires the previous subtitles to extend the end time,{diff=},{prev_diff=},{next_diff=}')
                    check_1[-1][1] = it[1]
                elif prev_diff is not None and next_diff is not None:
                    if prev_diff < next_diff:
                        check_1[-1][1] = it[1]
                        logger.warning(
                            f'cut_audio duration is less than{min_speech_duration_ms}ms requires the previous subtitles to extend the end time,{diff=},{prev_diff=},{next_diff=}')
                    else:
                        speech_chunks[i + 1][0] = it[0]
                        logger.warning(
                            f'cut_audio duration is less than{min_speech_duration_ms}ms requires the start time of the next subtitle left shift,{diff=},{prev_diff=},{next_diff=}')
                else:
                    check_1.append(it)
            elif diff < 30000:
                check_1.append(it)
            else:
                # More than 30s, divided into two
                off = diff // 2
                check_1.append([it[0], it[0] + off])
                check_1.append([it[0] + off, it[1]])
                logger.warning(f'cut_audio needs to be split if it exceeds 30 seconds.{diff=}')
        speech_chunks = check_1
        #Padding blanks on both sides
        silent_segment = self._padforaudio()
        for i, it in enumerate(speech_chunks):
            start_ms, end_ms = it[0], it[1]
            startraw, endraw = tools.ms_to_time_string(ms=it[0]), tools.ms_to_time_string(ms=it[1])
            chunk = audio[start_ms:end_ms]
            file_name = f"{dir_name}/audio_{i}.wav"
            (silent_segment + chunk + silent_segment).export(file_name, format="wav")
            data.append({
                "line": i + 1,
                "text": "",
                "start_time": start_ms,
                "end_time": end_ms,
                "startraw": startraw,
                "endraw": endraw,
                "time": f'{startraw} --> {endraw}',
                "file": file_name
            })

        return data

    # True exit
    def _exit(self) -> bool:
        if app_cfg.exit_soft or (self.uuid and self.uuid in app_cfg.stoped_uuid_set):
            return True
        return False
