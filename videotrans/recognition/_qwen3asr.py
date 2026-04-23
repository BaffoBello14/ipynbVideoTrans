# zh_recognition recognition
from dataclasses import dataclass
from typing import List, Dict, Union

try:
    import dashscope
except ImportError:
    dashscope = None  # install 'dashscope' to use this provider
from videotrans.configure.config import params
from videotrans.recognition._base import BaseRecogn

RETRY_NUMS = 2
RETRY_DELAY = 10


@dataclass
class Qwen3ASRRecogn(BaseRecogn):

    def __post_init__(self):
        super().__post_init__()


    def _exec(self) -> Union[List[Dict], None]:
        if self._exit(): return
        # Send request
        raws = self.cut_audio()
        api_key=params.get('qwenmt_key','')
        model=params.get('qwenmt_asr_model','qwen3-asr-flash')
        error=""
        ok_nums=0
        for i, it in enumerate(raws):
            response = dashscope.MultiModalConversation.call(
                # If there is no environment variable configured, please replace the following line with Bailian API Key: api_key = "sk-xxx",
                api_key=api_key,
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"audio": it['file']},
                    ]
                }],
                result_format="message",
                asr_options={
                    "language": self.detect_language[:2].lower(), # Optional, if the language of the audio is known, you can use this parameter to specify the language to be recognized to improve the recognition accuracy.
                    "enable_lid": True,
                    "enable_itn": True
                }
            )
            if not hasattr(response, 'output') or not hasattr(response.output, 'choices'):
                error=f'{response.code}:{response.message}'
                continue
                
            ok_nums+=1
            txt=''
            for t in response.output.choices[0]['message']['content']:
                txt += t['text']
            raws[i]['text'] = txt
        if ok_nums==0:
            raise RuntimeError(error)
        return raws

 
 