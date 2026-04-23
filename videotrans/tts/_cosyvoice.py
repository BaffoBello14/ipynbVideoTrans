import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from gradio_client import Client, handle_file
from videotrans.configure.config import tr, params, settings, app_cfg, logger, ROOT_DIR
from videotrans.configure._except import  StopRetry
from videotrans.tts._base import BaseTTS
from videotrans.util import tools


RETRY_NUMS = 2
RETRY_DELAY = 5


@dataclass
class CosyVoice(BaseTTS):
    def __post_init__(self):
        super().__post_init__()
        self.api_url = params.get('cosyvoice_url','').strip().rstrip('/').lower()
        self._add_internal_host_noproxy(self.api_url)

    def _exec(self):
        self._local_mul_thread()

    def _item_task_cosyvoice2(self, data_item):

        text = data_item['text'].strip()
        if not text:
            return
        speed = 1.0
        try:
            speed = 1 + float(self.rate.replace('%', '')) / 100
        except ValueError:
            pass
        speed=max(0.5,min(2.0,speed))
        role = data_item['role']
        data = {'ref_wav': '','ref_text':data_item.get('ref_text','')}
        
        rolelist = tools.get_cosyvoice_role()

        if role not in rolelist:
            raise StopRetry(tr('The role {} does not exist',role))
        if role == 'clone':
            data['ref_wav'] = data_item.get('ref_wav','')
            data['ref_text'] = data_item.get('ref_text','')
        else:
            data['ref_wav'] = ROOT_DIR+"/f5-tts/"+rolelist[role].get('reference_audio','')
            data['ref_text'] = rolelist[role].get('reference_text','')

        if not Path(data['ref_wav']).exists():
            raise StopRetry(f"{data['ref_wav']} is not exists")
        
        logger.debug(f'cosyvoice-tts {data=}')
        try:
            client = Client(self.api_url, ssl_verify=False)
        except Exception as e:
            raise StopRetry(str(e))
        # Refer to the text content corresponding to the audio
        prompt_text=data.get('ref_text','')
        print(f"{data['ref_wav']=}")
        # Prompt word, put into the reference audio text when cloning
        instruct_text=params.get('cosyvoice_instruct_text','')
        if instruct_text:
            prompt_text=f'You are a helpful assistant.{instruct_text}<|endofprompt|>{prompt_text}'
        result = client.predict(
            tts_text=text,
            mode_checkbox_group='3s extremely fast reproduction',
            prompt_wav_upload=handle_file(data['ref_wav']),
            prompt_wav_record=handle_file(data['ref_wav']),
            prompt_text=prompt_text,
            instruct_text=instruct_text,
            seed=0,
            stream=False,
            speed=speed,
            api_name="/generate_audio"

        )


        logger.debug(f'result={result}')
        wav_file = result[0] if isinstance(result, (list, tuple)) and result else result
        if isinstance(wav_file, dict) and "value" in wav_file:
            wav_file = wav_file['value']
        if isinstance(wav_file, str) and Path(wav_file).is_file():
            self.convert_to_wav(wav_file, data_item['filename'])
        else:
            raise RuntimeError(str(result))


    def _item_cosyvoice_api(self, data_item):
        if not data_item.get('text',''):
            return
        rate = float(self.rate.replace('%', '')) / 100 if self.rate else 0
        role = data_item['role']

        api_url = self.api_url
        data = {
            "text": data_item['text'],
            "lang": "zh" if self.language.startswith('zh') else self.language,
            "speed": 1 + rate
        }
        rolelist = tools.get_cosyvoice_role()
        if role not in rolelist:
            raise StopRetry(tr('The preset role {} was not found in the configuration',role))
        if role == 'clone':
            # clone sound
            # The original project uses the clone_mul cross-language cloning solution. The actual test effect is not as good as the same language. This place is modified to the same language clone /clone_eq
            ref_wav_path = data_item.get("ref_wav",'')
            if not Path(ref_wav_path).exists():
                raise StopRetry(tr('No reference audio {} exists',ref_wav_path))

            data['reference_text'] = data_item.get('ref_text','')
            data['reference_audio'] = self._audio_to_base64(ref_wav_path)
            api_url += '/clone_eq'
            data['encode'] = 'base64'
        else:
            role_info = rolelist[role]
            data['reference_audio'] = ROOT_DIR+"/f5-tts/"+role_info.get('reference_audio','')

            if not data['reference_audio']:
                raise StopRetry(tr('Preset role {} is incorrectly configured, missing clone reference audio',role))

            # Check if reference text exists to decide which clone interface to use
            reference_text = role_info.get('reference_text', '').strip()
            if reference_text:
                # Provide reference audio and text at the same time, using high-quality clones in the same language
                data['reference_text'] = reference_text
                api_url += '/clone_eq'
            else:
                # Only provide reference audio, use cross-language cloning
                api_url += '/clone_mul'

        logger.debug(f'Request data:{api_url=},{data=}')
        # clone sound
        response = requests.post(f"{api_url}", data=data,  timeout=3600)
        response.raise_for_status()

        # If it is a WAV audio stream, get the original audio data
        with open(data_item['filename'] + ".wav", 'wb') as f:
            f.write(response.content)
        time.sleep(1)
        if not os.path.exists(data_item['filename'] + ".wav"):
            raise RuntimeError(tr('CosyVoice synthesis failed -2'))
        self.convert_to_wav(data_item['filename'] + ".wav", data_item['filename'])

    def _item_task(self, data_item,idx:int=-1):
        if self._exit() or  not data_item.get('text','').strip():
            return
        if self._exit() or tools.vail_file(data_item['filename']):
            return
        
        # Compatible with the previous cozyvoice-api interface
        try:
            if ":9233" not in self.api_url:
                self._item_task_cosyvoice2(data_item)
            else:
                self._item_cosyvoice_api(data_item)

        except Exception as e:
            self.error=e
            raise