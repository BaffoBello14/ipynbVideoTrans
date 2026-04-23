
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from videotrans.configure.config import tr, params, settings, app_cfg, logger, ROOT_DIR
from videotrans.configure._base import BaseCon
from videotrans.task.taskcfg import TaskCfgBase
from videotrans.util import tools


@dataclass
class BaseTask(BaseCon):
    # Various configuration information, such as translation, dubbing, recognition channels, etc.
    cfg: TaskCfgBase = field(default_factory=TaskCfgBase, repr=False)
    # Progress record
    precent: int = 1
    # Original subtitle information that needs dubbing List[dict]
    queue_tts: List = field(default_factory=list, repr=False)
    # Is it ended?
    hasend: bool = False

    # File names that should be deleted after name normalization
    shound_del_name: str = None

    # Whether speech recognition is required
    shoud_recogn: bool = False

    # Do you need subtitle translation?
    shoud_trans: bool = False

    # Do you need dubbing?
    shoud_dubbing: bool = False

    # Do you need vocal separation?
    shoud_separate: bool = False

    # Whether dubbing or subtitles need to be embedded
    shoud_hebing: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.cfg.uuid:
            self.uuid = self.cfg.uuid

    # Pre-processing, such as splitting audio from video, vocal background separation, transcoding, etc.
    def prepare(self):
        pass

    # Speech recognition creates original language subtitles
    def recogn(self):
        pass
    
    # Speaker recognition, excluding Funasr/Doubao speech recognition large model/Deepgram, and then determine whether there is a speaker, Gemini/openai gpt4-dia will generate a speaker
    def diariz(self):
        pass

    # Translate original language subtitles to target language subtitles
    def trans(self):
        pass

    # Dubbing based on queue_tts
    def dubbing(self):
        pass

    # Dubbing acceleration, video slow alignment
    def align(self):
        pass

    # Merge video, audio, and subtitles to generate result files
    def assembling(self):
        pass

    # Delete temporary files, move or copy, send success message
    def task_done(self):
        pass

    # Whether the subtitles exist and are valid
    def _srt_vail(self, file):
        if not file:
            return False
        if not tools.vail_file(file):
            return False
        try:
            tools.get_subtitle_from_srt(file)
        except Exception:
            try:
                Path(file).unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True

    # Delete invalid files with size 0
    def _unlink_size0(self, file):
        if not file:
            return
        p = Path(file)
        if p.exists() and p.stat().st_size == 0:
            p.unlink(missing_ok=True)

    # Save subtitle file to target folder
    def _save_srt_target(self, srtstr, file):
        # is in the form of a subtitle list, reassembled
        try:
            txt = tools.get_srt_from_list(srtstr)
            with open(file, "w", encoding="utf-8",errors="ignore") as f:
                f.write(txt)
        except Exception:
            raise
        self._signal(text=Path(file).read_text(encoding='utf-8',errors="ignore"), type='replace_subtitle')
        return True

    def _check_target_sub(self, source_srt_list, target_srt_list):
        import re, copy
        if len(source_srt_list) == 1 or len(target_srt_list) == 1:
            target_srt_list[0]['line'] = 1
            return target_srt_list[:1]
        source_len = len(source_srt_list)
        target_len = len(target_srt_list)
        
        if source_len==target_len:
            for i,it in enumerate(source_srt_list):
                tmp = copy.deepcopy(it)
                tmp['text']=target_srt_list[i]['text']
                target_srt_list[i]=tmp
            return target_srt_list

        if target_len>source_len:
            logger.debug(f'The number of lines in the translation result is greater than the original subtitle lines, so 0-{source_len}')
            return target_srt_list[:source_len]
        
        
        logger.debug(f'The number of lines in the translation result is less than the original subtitle lines, append')
        for i,it in enumerate(source_srt_list):
            if i>=target_len:
                tmp=copy.deepcopy(it)
                tmp['text']=' '
                target_srt_list.append(tmp)
        return target_srt_list
        
    


    async def _edgetts_single(self,target_audio,kwargs):
        from edge_tts import Communicate
        from edge_tts.exceptions import NoAudioReceived
        import aiohttp,asyncio
        from io import BytesIO
        
        useproxy_initial = None if not self.proxy_str or Path(f'{ROOT_DIR}/edgetts-noproxy.txt').exists() else self.proxy_str
        proxies_to_try = [useproxy_initial]
        if useproxy_initial is not None:
            proxies_to_try.append(None)
        last_exception = None
        for proxy in proxies_to_try:
            try:
                audio_buffer = BytesIO()
                communicate_task = Communicate(
                            text=kwargs['text'],
                            voice=kwargs['voice'],
                            rate=kwargs['rate'],
                            volume=kwargs['volume'],
                            proxy=proxy,
                            pitch=kwargs['pitch']
                        )
                idx=0
                async for chunk in communicate_task.stream():
                    if chunk["type"] == "audio":
                        audio_buffer.write(chunk["data"])
                        self._signal(text=f'{idx} segment')
                        idx+=1
                audio_buffer.seek(0)        
                from pydub import AudioSegment
                au=AudioSegment.from_file(audio_buffer,format="mp3")
                au.export(target_audio,format='mp3')
                return
            except (NoAudioReceived, aiohttp.ClientError) as e:
                last_exception = e
            except Exception:
                raise
        raise last_exception if last_exception else RuntimeError(f'Dubbing error')
    # The complete process determines whether to exit, and the sub-function needs to be rewritten.
    def _exit(self):
        if app_cfg.exit_soft or app_cfg.current_status != 'ing':
            self.hasend=True
            return True
        return False
