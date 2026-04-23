import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from typing import Union, Set

import requests

from pydub import AudioSegment
from videotrans.configure._except import StopRetry
from videotrans.configure.config import tr,params,settings,app_cfg,logger
from videotrans.tts._base import BaseTTS
from videotrans.util import tools

RETRY_NUMS = 2
RETRY_DELAY = 5


@dataclass
class GPTSoVITS(BaseTTS):
    splits: Set[str] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        # 2. Process and set api_url (also override the value of the parent class)
        api_url = params.get('gptsovits_url','').strip().rstrip('/').lower()
        self.api_url = 'http://' + api_url.replace('http://', '')
        self._add_internal_host_noproxy(self.api_url)
        # 3. Initialize the new attributes of this class
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
        self.pad_audio=None
        self.speed = float(float(self.rate.replace('+', '').replace('-', '').replace('%', '')) / 100)
    def _exec(self):
        self._local_mul_thread()

    def _item_task(self, data_item: Union[Dict, List, None],idx:int=-1):
        try:
            if self._exit() or  not data_item.get('text','').strip()  or tools.vail_file(data_item['filename']):
                return
            role = data_item['role']
            data = {
                "text": data_item['text'],
                "text_language": "zh" if self.language.startswith('zh') else self.language,
                "extra": params.get('gptsovits_extra',''),
                "ostype": sys.platform,

            }

            roledict = tools.get_gptsovits_role()
            keys=list(roledict.keys())
            ref_wav=data_item.get('ref_wav','')

            if role !='clone' and roledict and role in roledict:
                data.update(roledict[role])
            elif role == 'clone':#clone original audio clip
                if ref_wav and Path(ref_wav).exists():
                    data['prompt_text'] = data_item.get('ref_text').strip()
                    data['prompt_language'] = data_item.get('ref_language','')
                    data['refer_wav_path'] = ref_wav
                    ref_wav_audio=AudioSegment.from_file(ref_wav,format="wav")
                    ms_ref=len(ref_wav_audio)
                    if ms_ref>9990:#Truncation greater than 10s
                        logger.warning(f'The reference audio is longer than 10s and needs to be truncated:{ref_wav=}')
                        ref_wav_audio[:9990].export(ref_wav,format="wav")
                    elif ms_ref<3000:#Greater than 3s is legal
                        logger.warning(f'The reference audio is less than 3s, fill in the blank at the end:{ref_wav=}')
                        self.pad_audio= self.pad_audio if self.pad_audio else self._padforaudio(3000 if ms_ref<1500 else 1600)
                        
                        (ref_wav_audio+self.pad_audio).export(ref_wav,format="wav")
                elif keys[-1]=='clone':
                    # There is no custom reference audio. The clone original audio duration does not match and fails.
                    raise RuntimeError('No refer audio and origin audio duration not between 3-10s')
                else:
                    # Failed to clone the original audio, use the last reference audio
                    data.update(roledict[keys[-1]])

            if not data.get('refer_wav_path') and role !='clone':
                raise StopRetry(message=tr("Must pass in the reference audio file path"))

            if params.get('gptsovits_isv2',''):
                data = {
                    "text": data_item['text'],
                    "text_lang": data.get('text_language', 'zh'),
                    "ref_audio_path": data.get('refer_wav_path', ''),
                    "prompt_text": data.get('prompt_text', ''),
                    "prompt_lang": data.get('prompt_language', ''),
                    "speed_factor": 1.0+self.speed,
                    "text_split_method":"cut0"
                }

                if not self.api_url.endswith('/tts'):
                    self.api_url += '/tts'
            else:
                data['speed']=1.0+self.speed
            logger.debug(f'GPT-SoVITS currently needs to send dubbing data:{data=}\n{self.api_url=}')
            # clone sound
            response = requests.post(f"{self.api_url}", json=data,  timeout=3600,proxies={"https":"","http":""})

            if response.ok:
                # If it is a WAV audio stream, get the original audio data
                with open(data_item['filename'] + ".wav", 'wb') as f:
                    f.write(response.content)
                time.sleep(1)
                self.convert_to_wav(data_item['filename'] + ".wav", data_item['filename'])
                self.error=None
            else:
                try:
                    error_data = response.json() # Here you can directly get the JSON of 500
                except:
                    error_data=response.text
                self.error=RuntimeError(error_data)
                logger.error(f'GPT-SoVITS {ref_wav=}\nReturn error:{error_data=}\n')
        except Exception as e:
            self.error=e
            raise