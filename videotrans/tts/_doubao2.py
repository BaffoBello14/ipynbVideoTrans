from dataclasses import dataclass
from videotrans.configure.config import logger, params
from videotrans.tts._base import BaseTTS
from videotrans.util import tools


import requests
import json
import base64

RETRY_NUMS = 2
RETRY_DELAY = 5


@dataclass
class Doubao2TTS(BaseTTS):

    def __post_init__(self):
        super().__post_init__()
        self.stop_next_all=False
        #self.dub_nums = 1
        # Speech synthesis model 2.0 exclusive role
        self.model2=[
            "zh_female_vv_uranus_bigtts",
            "zh_male_dayi_saturn_bigtts",
            "zh_female_mizai_saturn_bigtts",
            "zh_female_jitangnv_saturn_bigtts",
            "zh_female_meilinvyou_saturn_bigtts",
            "zh_female_santongyongns_saturn_bigtts",
            "zh_male_ruyayichen_saturn_bigtts",
            "saturn_zh_female_keainvsheng_tob",
            "saturn_zh_female_tiaopigongzhu_tob",
            "saturn_zh_male_shuanglangshaonian_tob",
            "saturn_zh_male_tiancaitongzhuo_tob",
            "saturn_zh_female_cancan_tob",
        ]

    def _exec(self):
        # The concurrency limit is 1 to prevent current limiting
        self._local_mul_thread()
    

    def _save_pcm_to_wav(self,audio_data, output_filename: str, 
                        channels: int = 1, sample_rate: int = 48000, sample_width: int = 2):
        import wave
        import struct
        import math
        'Save raw PCM data (bytearray) as WAV file.\n\n        Args:\n            audio_data (byte array): byte array containing raw PCM audio data.\n            output_filename (str): The path and name of the WAV file to be saved.\n            channels (int): Number of channels. Default is 1 (mono).\n            sample_rate (int): sampling rate (Hz). The default is 44100.\n            sample_width (int): Sample width (bytes). The default is 2 (16-bit).\n                                 1 represents 8-bit, 2 represents 16-bit, and 3 represents 24-bit.'
        if not output_filename.lower().endswith('.wav'):
            output_filename += '.wav'

        try:
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(channels)        # Set the number of channels
                wf.setsampwidth(sample_width)    # Set sampling width (2 bytes = 16 bits)
                wf.setframerate(sample_rate)     # Set sampling rate

                wf.writeframes(audio_data)

        except Exception as e:
            logger.exception(f"Error saving WAV file:{e}",exc_info=True)

    
    def _item_task(self, data_item: dict = None,idx:int=-1):
        try:
            if self.stop_next_all or self._exit() or not data_item.get('text','').strip():
                return

            if tools.vail_file(data_item['filename']):
                return
            appid = params.get('doubao2_appid','')
            access_token = params.get('doubao2_access','')
            speed = 0
            if self.rate:
                speed = int(float(self.rate.replace('%', '')))
                speed=min(max(-50,speed),100)
            volume = 0
            if self.volume:
                volume = int(float(self.volume.replace('%', '')))
                volume=min(max(-50,volume),100)

            #The role is the actual name
            role = data_item['role']
            role=tools.get_doubao2_rolelist(role_name=role,langcode=self.language[:2])
            headers = {
                "X-Api-App-Id": appid,
                "X-Api-Access-Key": access_token,
                "X-Api-Resource-Id": 'seed-tts-2.0' if role in self.model2 else 'seed-tts-1.0',
                "Content-Type": "application/json",
                "Connection": "keep-alive"
            }

            payload = {
                "user": {
                    "uid": "123123"
                },
                "req_params":{
                    "text": data_item.get('text',''),
                    "speaker": role,
                    "model":"seed-tts-1.1",
                    "audio_params": {
                        "format": "pcm",
                        "sample_rate": 48000,
                        "enable_timestamp": True,
                        "speech_rate":int(speed),
                        "loudness_rate":int(volume)
                    },
                    "additions": "{\"explicit_language\":\"crosslingual\",\"enable_language_detector\":\"true\",\"disable_markdown_filter\":true}\"}"
                }
            }
            
     
            url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
            
            
            response = requests.post(url, headers=headers, json=payload, stream=True)
            
            if response.status_code in [404,402,401,400]:
                self.stop_next_all=True
                raise RuntimeError('Please check if the appid and access token parameters are correct')
            if response.status_code == 403:
                self.stop_next_all=True
                raise RuntimeError('The official version of this character may need to be purchased separately in the Byte backend.')
            
            
            
            response.raise_for_status()
            logger.debug(f"code: {response.status_code} header: {response.headers}")

            # Used to store audio data
            audio_data = bytearray()
            total_audio_size = 0
            for chunk in response.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                data = json.loads(chunk)

                if data.get("code", 0) == 0 and "data" in data and data["data"]:
                    chunk_audio = base64.b64decode(data["data"])
                    audio_size = len(chunk_audio)
                    total_audio_size += audio_size
                    audio_data.extend(chunk_audio)
                    continue
                if data.get("code", 0) == 0 and "sentence" in data and data["sentence"]:
                    print("sentence_data:", data)
                    continue
                if data.get("code", 0) == 20000000:
                    break
                if data.get("code", 0) > 0:
                    
                    raise RuntimeError(str(data))

            if audio_data:
                self._save_pcm_to_wav(audio_data,data_item['filename'])
        except Exception as e:
            self.error=e
            raise


