import base64
import datetime
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, ClassVar

import requests
from videotrans.configure.config import params, logger
from videotrans.tts._base import BaseTTS
from videotrans.util import tools

RETRY_NUMS = 2
RETRY_DELAY = 5


@dataclass
class DoubaoTTS(BaseTTS):
    error_status: ClassVar[Dict[str, str]] = {
        "3001": 'Invalid request. If it is the official version, the sound currently used may need to be purchased separately from Byte Huoshan.',
        "3003": 'Concurrency exceeded',
        "3005": 'Backend server load is high',
        "3006": 'service interruption',
        "3010": 'Text length exceeded',
        "3011": 'The parameters are incorrect or the text is empty, the text does not match the language, or the text only contains punctuation.',
        "3030": 'A single request exceeds the maximum service time limit',
        "3031": 'An exception occurred in the backend',
        "3032": 'Timeout waiting to get audio',
        "3040": 'Tone clone link network abnormality',
        "3050": 'Sound clone sound query failed'
    }

    # Keys are the first two characters of dialect role names in the Doubao UI.
    fangyan: ClassVar[Dict[str, str]] = {
        "东北": "zh_dongbei",
        "粤语": "zh_yueyu",
        "上海": "zh_shanghai",
        "西安": "zh_xian",
        "成都": "zh_chengdu",
        "台湾": "zh_taipu",
        "广西": "zh_guangxi",
    }
    voice_type: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self.stop_next_all=False

    def _exec(self):
        # The concurrency limit is 1 to prevent current limiting
        self._local_mul_thread()

    def _item_task(self, data_item: dict = None,idx:int=-1):
        if self.stop_next_all or self._exit() or not data_item.get('text','').strip():
            return
        def _run():
            if self._exit() or tools.vail_file(data_item['filename']):
                return
            appid = params.get('volcenginetts_appid','')
            access_token = params.get('volcenginetts_access','')
            cluster = params.get('volcenginetts_cluster','')
            speed = 1.0
            if self.rate:
                rate = float(self.rate.replace('%', '')) / 100
                speed += rate

            volume = 1.0
            if self.volume:
                volume += float(self.volume.replace('%', '')) / 100

            #The role is the actual name
            role = data_item['role']
            langcode = self.language[:2].lower()
            if not self.voice_type:
                self.voice_type = tools.get_doubao_rolelist(role, self.language)

            if langcode == 'zh':
                langcode = self.fangyan.get(role[:2], "cn")
            host = "openspeech.bytedance.com"
            api_url = f"https://{host}/api/v1/tts"

            header = {"Authorization": f"Bearer;{access_token}"}

            request_json = {
                "app": {
                    "appid": appid,
                    "token": "access_token",
                    "cluster": cluster
                },
                "user": {
                    "uid": datetime.datetime.now().strftime("%Y%m%d")
                },
                "audio": {
                    "voice_type": self.voice_type,
                    "encoding": "wav",
                    "speed_ratio": speed,
                    "volume_ratio": volume,
                    "pitch_ratio": 1.0,
                    "language": langcode
                },
                "request": {
                    "reqid": str(time.time() * 100000),
                    "text": data_item['text'],
                    "text_type": "plain",
                    "silence_duration": 50,
                    "operation": "query",
                    "pure_english_opt": 1

                }
            }
            logger.debug(f'Byte speech synthesis:{request_json=}')
            resp = requests.post(api_url, json.dumps(request_json), headers=header,verify=False)
            resp.raise_for_status()
            resp_json = resp.json()

            if "data" in resp_json:
                data = resp_json["data"]
                with open(data_item['filename']+"-tmp.wav" , "wb") as f:
                    f.write(base64.b64decode(data))
                self.convert_to_wav(data_item['filename']+"-tmp.wav", data_item['filename'])
                return
            if 'authenticate' in resp_json.get('message','') or 'access denied' in resp_json.get('message',''):
                self.stop_next_all=True
                raise RuntimeError(resp_json.get('message'))
            if 'code' in resp_json:
                logger.debug(f'Byte Volcano speech synthesis failed:{resp_json=}')
            raise RuntimeError(self.error_status.get(str(resp_json['code']), resp_json['message']))

        try:
            _run()
        except Exception as e:
            self.error=e
            raise