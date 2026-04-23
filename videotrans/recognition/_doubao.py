# zh_recognition recognition
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Union

import requests

from videotrans.configure.config import params
from videotrans.recognition._base import BaseRecogn
from videotrans.util import tools

RETRY_NUMS = 2
RETRY_DELAY = 10


@dataclass
class DoubaoRecogn(BaseRecogn):

    def __post_init__(self):
        super().__post_init__()

    def _exec(self) -> Union[List[Dict], None]:
        if self._exit():
            return

        base_url = 'https://openspeech.bytedance.com/api/v1/vc'
        appid = params.get('doubao_appid','')
        access_token = params.get('doubao_access','')

        # If the size is greater than 190MB, convert to mp3

        tools.runffmpeg(
            ['-y', '-i', self.audio_file, '-ac', '1', '-ar', '16000', self.cache_folder + '/doubao-tmp.mp3'])
        self.audio_file = self.cache_folder + '/doubao-tmp.mp3'
        with open(self.audio_file, 'rb') as f:
            files = f.read()

        self._signal(text=f"Recognition may take a long time, please wait patiently.")

        languagelist = {"zh": "zh-CN", "en": "en-US", "ja": "ja-JP", "ko": "ko-KR", "es": "es-MX", "ru": "ru-RU",
                        "fr": "fr-FR"}
        langcode = self.detect_language[:2].lower()
        if langcode not in languagelist:
            raise RuntimeError(f'Unsupported language codes:{langcode=}')
        language = languagelist[langcode]

        res = requests.post(
            f'{base_url}/submit',
            data=files,
            params=dict(
                appid=appid,
                language=language,
                use_itn='True',
                caption_type='speech',
                max_lines=1  # Only one line of text is allowed per subtitle
                # words_per_line=15,# Each line of text can have up to 15 characters
            ),
            headers={
                'Content-Type': 'audio/wav',
                'Authorization': 'Bearer; {}'.format(access_token)
            },
            timeout=3600
        )
        res.raise_for_status()
        res = res.json()
        if res['code'] != 0:
            raise RuntimeError(f'Request failed:{res["message"]}')

        job_id = res['id']
        delay = 0
        while 1:
            if self._exit():
                return
            delay += 1
            # Get progress
            response = requests.get(
                '{base_url}/query'.format(base_url=base_url),
                params=dict(
                    appid=appid,
                    id=job_id,
                    blocking=0
                ),
                headers={
                    'Authorization': 'Bearer; {}'.format(access_token)
                }
            )
            response.raise_for_status()

            result = response.json()
            if result['code'] == 2000:
                self._signal(text=f"Task is being processed, please wait{delay}s..")
                time.sleep(1)
            elif result['code'] > 0:
                raise RuntimeError(result['message'])
            else:
                break

        for i, it in enumerate(result['utterances']):
            if self._exit():
                return
            srt = {
                "line": i + 1,
                "start_time": it['start_time'],
                "end_time": it['end_time'],
                "endraw": tools.ms_to_time_string(ms=it["end_time"]),
                "startraw": tools.ms_to_time_string(ms=it["start_time"]),
                "text": it['text']
            }
            srt['time'] = f'{srt["startraw"]} --> {srt["endraw"]}'
            self._signal(
                text=f'{srt["line"]}\n{srt["time"]}\n{srt["text"]}\n\n',
                type='subtitle'
            )
            self.raws.append(srt)
        return self.raws
