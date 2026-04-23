import copy
import datetime
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from videotrans import tts
from videotrans.configure.config import tr, settings, params, app_cfg, logger, HOME_DIR
from videotrans.task._base import BaseTask
from videotrans.task._rate import TtsSpeedRate
from videotrans.util import tools

'\nOnly dubbing tasks: corresponding to batch dubbing subtitles panel'


@dataclass
class DubbingSrt(BaseTask):
    # Is it a subtitle multi-character dubbing function?
    out_ext:str="wav"
    is_multi_role: bool = field(init=True,default=False)
    # Fixed to True
    shoud_dubbing: bool = field(default=True, init=False)
    ignore_align:bool=False
    # Use this subtitle information directly when dubbing multiple characters
    subs:List = field(default_factory=list, repr=False)
    def __post_init__(self):
        super().__post_init__()
        # Is it subtitles and multi-character dubbing?
        # Output target location
        if not self.cfg.target_dir:
            self.cfg.target_dir = f"{HOME_DIR}/tts"
        if self.cfg.cache_folder:
            Path(self.cfg.cache_folder).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.target_dir).mkdir(parents=True, exist_ok=True)
        # Subtitle files that require dubbing
        self.cfg.target_sub = self.cfg.name
        # After dubbing, the audio file is saved as
        self.cfg.target_wav = f'{self.cfg.target_dir}/{self.cfg.noextname}.wav'
        self._signal(text=tr("Dubbing from subtitles"))
        logger.debug(f'Dubbing{self.cfg=}')


    def dubbing(self):
        try:
            self._signal(text=Path(self.cfg.target_sub).read_text(encoding='utf-8'), type="replace")
            self._tts()
        except Exception as e:
            self.hasend = True
            raise

    # The subtitles may be gbk gb2312 and other encodings and need to be converted to utf-8
    def _convert_to_utf8_if_needed(self, file_path: str) -> str:
        import tempfile
        try:
            # 1. Try to open and read the file completely in UTF-8 encoding to check its validity
            # 'strict' is the default error handling method. UnicodeDecodeError will be thrown when encountering undecoded bytes.
            with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                f.read()
            return file_path
        except UnicodeDecodeError:

            # 2. If UTF-8 decoding fails, try using other common encodings
            # You can adjust the order or content of this list according to your actual situation
            fallback_encodings = ['gbk', 'gb2312', 'big5', 'latin-1']
            original_content = None
            # Read the file content once in binary mode to avoid repeated IO
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
            except FileNotFoundError:
                raise
            for encoding in fallback_encodings:
                try:
                    original_content = raw_data.decode(encoding)
                    break  # As long as one succeeds, break out of the loop
                except UnicodeDecodeError:
                    continue  # If this encoding also fails, try the next one

            # 3. If all alternative encodings fail, it cannot be processed
            if original_content is None:
                return file_path

            # 4. Create a temporary file with a name to save the converted content
            # - mode='w': write in text mode
            # - encoding='utf-8': Specify the writing encoding as UTF-8
            # - suffix='.txt': keep temporary files with .txt extension
            # - delete=False: Ensure that the file will not be automatically deleted after the with statement block ends.
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as temp_file:
                temp_file.write(original_content)
                temp_file_path = temp_file.name
            return temp_file_path
        except FileNotFoundError:
            raise

    # Preprocessing of dubbing, removing invalid characters, and sorting out the start time
    def _tts(self) -> None:
        queue_tts = []
        # Get subtitles
        try:
            rate = int(str(self.cfg.voice_rate).replace('%', ''))
        except ValueError:
            rate = 0
        if rate >= 0:
            rate = f"+{rate}%"
        else:
            rate = f"{rate}%"
        # If the channel is edge-tts and not multi-character dubbing
        _enter_edgetts_single=self.cfg.tts_type == tts.EDGE_TTS and not self.is_multi_role
        if _enter_edgetts_single:
            # The dubbing file is txt
            if self.cfg.target_sub.endswith('.txt'):
                _enter_edgetts_single=True
            elif not self.cfg.voice_autorate and self.cfg.remove_silent_mid:
                # Or it is not automatically accelerated and the silence between subtitles is removed.
                _enter_edgetts_single=True
            else:
                _enter_edgetts_single=False
        

        if _enter_edgetts_single:
            from edge_tts import Communicate
            import asyncio
            # Ignore alignment
            self.ignore_align=True
            self.cfg.target_sub = self._convert_to_utf8_if_needed(self.cfg.target_sub)
            
            tmp_name = self.cfg.target_wav if self.cfg.target_wav.endswith(
                '.mp3') else f"{self.cfg.cache_folder}/{self.cfg.noextname}-edgetts-txt-{time.time()}.mp3"
            if self.cfg.target_sub.endswith('.txt'):
                text=Path(self.cfg.target_sub).read_text(encoding='utf-8')
            else:
                text=""
                self.queue_tts=tools.get_subtitle_from_srt(self.cfg.target_sub)
                for it in self.queue_tts:
                    text+=it["text"]+"\n"
                self.queue_tts=self.queue_tts[:1]

            asyncio.run(self._edgetts_single(
                tmp_name,
                dict(text=text,
                    voice=tools.get_edge_rolelist(self.cfg.voice_role,locale=self.cfg.target_language_code),
                    rate=rate,
                    volume=self.cfg.volume,
                    pitch=self.cfg.pitch
                )
            ))
            logger.debug(f'edge-tts dubbing, no audio acceleration, no video slowdown, no forced alignment, deleted mute between subtitles, use separate text dubbing')
            if not self.cfg.target_wav.endswith('.mp3'):
                tools.runffmpeg(['-y', '-i', tmp_name, '-b:a', '128k', self.cfg.target_wav])
            return
        
        # If the dubbing file is txt, it will be converted into a single subtitle format for unified processing.
        if self.cfg.target_sub.endswith('.txt'):
            text = Path(self.cfg.target_sub).read_text(encoding='utf-8').strip()
            text = re.sub(r"(\s*?\r?\n\s*?){2,}", "\n", text,flags=re.I | re.S)
            text = re.sub(r"(\s*?\r?\n\s*?)", "\n", text,flags=re.I | re.S)

            text_list=re.split(r'(\r?\n)+?',text)
            subs=[]
            for i,it in enumerate(text_list):
                if not it.strip():
                    continue
                subs.append({
                    "line": i+1,
                    "start_time": i*1000,
                    "end_time": i*1000+1000,
                    "startraw": f"00:00:00,000",
                    "endraw": "00:00:01,000",
                    "text": it
                })
        elif self.subs:
            subs=self.subs
        else:
            subs = tools.get_subtitle_from_srt(self.cfg.target_sub)

        # Take out each subtitle, line number\nStart time --> End time\nContent
        for i, it in enumerate(subs):
            if it['end_time'] < it['start_time'] or not it['text'].strip():
                continue
            try:
                spec_role = app_cfg.dubbing_role.get(int(it.get('line', 1))) if self.is_multi_role else None
            except (ValueError,LookupError):
                spec_role = None
            voice_role = spec_role if spec_role else self.cfg.voice_role

            tmp_dict = {
                "line": it['line'],
                "text": it['text'],
                "role": voice_role,
                "start_time": it['start_time'],
                "end_time": it['end_time'],
                "rate": rate,
                "volume": self.cfg.volume,
                "pitch": self.cfg.pitch,
                "tts_type": int(self.cfg.tts_type),
                "filename": f"{self.cfg.cache_folder}/dubb-{i}.wav"}
            queue_tts.append(tmp_dict)

        self.queue_tts = queue_tts

        if not self.queue_tts or len(self.queue_tts) < 1:
            raise RuntimeError(tr('No subtitles required'))
        # Specific dubbing operations
        tts.run(
            queue_tts=self.queue_tts,
            language=self.cfg.target_language_code,
            uuid=self.uuid,
            tts_type=self.cfg.tts_type,
            is_cuda=self.cfg.is_cuda
        )
        # If you need to save the dubbing of each subtitle separately
        if settings.get('save_segment_audio', False):
            outname = self.cfg.target_dir + f'/segment_audio_{self.cfg.noextname}'
            Path(outname).mkdir(parents=True, exist_ok=True)
            for it in self.queue_tts:
                if Path(it['filename']).exists():
                    text = re.sub(r'["\'*?\\/\|:<>\r\n\t]+', '', it['text'],flags=re.I | re.S)
                    name = f'{outname}/{it["start_time"]}-{text[:60]}.wav'
                    try:
                        shutil.copy2(it['filename'], name)
                    except shutil.SameFileError:
                        pass

    
    
    #Audio acceleration aligned subtitles
    def align(self) -> None:
        # txt dubbing and edgetts, ended
        if self.ignore_align:
            return
        # Only one line
        if len(self.queue_tts) == 1:
            if self.cfg.tts_type != tts.EDGE_TTS:
                tools.runffmpeg(['-y', '-i', self.queue_tts[0]['filename'], '-b:a', '128k', self.cfg.target_wav])
            return

        if self.cfg.voice_autorate:
            self._signal(text=tr("Sound speed alignment stage"))
        try:
            target_path = Path(self.cfg.target_wav)
            # If the current folder has the same name, add the time suffix
            if target_path.is_file() and target_path.stat().st_size > 0:
                self.cfg.target_wav = self.cfg.target_wav[:-4] + f'-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}{target_path.suffix}'
            # txt dubbing without audio acceleration, removing the mute between subtitles and ignoring any gaps, directly connecting the dubbing files
            # The separate dubbing function does not force alignment
            rate_inst = TtsSpeedRate(
                queue_tts=self.queue_tts,
                uuid=self.uuid,
                shoud_audiorate=self.cfg.voice_autorate if not self.cfg.target_sub.endswith('.txt') else False,# txt dubbing prohibits automatic acceleration, you need to remove subtitle mute, that is, connect directly
                raw_total_time=self.queue_tts[-1]['end_time'],
                target_audio=self.cfg.target_wav,
                cache_folder=self.cfg.cache_folder,
                remove_silent_mid=self.cfg.remove_silent_mid if not self.cfg.target_sub.endswith('.txt') else True, # Whether to remove the gap between subtitles. This only works when automatic acceleration is not enabled. It is removed when dubbing txt, that is, the audio file is directly connected.
                align_sub_audio=False # Unaligned subtitles only works when not automatically accelerated
            )
            self.queue_tts = rate_inst.run()


            volume = self.cfg.volume.strip()

            if volume != '+0%':
                try:
                    volume = 1 + float(volume) / 100
                    tmp_name = self.cfg.cache_folder + f'/volume-{volume}-{Path(self.cfg.target_wav).name}'
                    tools.runffmpeg(['-y', '-i', self.cfg.target_wav, '-af', f"volume={volume}", tmp_name])
                except Exception:
                    pass
        except Exception as e:
            self.hasend = True
            raise

    def task_done(self):
        if self._exit():
            return
        self.hasend = True
        self.precent = 100
            
        try:
            if Path(self.cfg.target_wav).is_file():
                # Remove silence at the end
                tools.remove_silence_from_end(self.cfg.target_wav, is_start=False)
                self._signal(text=f"{self.cfg.name}", type='succeed')
                if self.out_ext.lower()!='wav':
                    tools.runffmpeg(['-y', '-i', self.cfg.target_wav, f'{self.cfg.target_dir}/{self.cfg.noextname}.{self.out_ext}'])
                    Path(self.cfg.target_wav).unlink(missing_ok=True)
            if self.cfg.shound_del_name:
                Path(self.cfg.shound_del_name).unlink(missing_ok=True)
        except OSError:
            pass
        tools.send_notification(tr('Succeed'), f"{self.cfg.basename}")    

    def _exit(self):
        if app_cfg.exit_soft:
            self.hasend=True
            return True
        return False
