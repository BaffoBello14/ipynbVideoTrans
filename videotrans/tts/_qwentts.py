import logging
from dataclasses import dataclass

try:
    import dashscope
except ImportError:
    dashscope = None  # install 'dashscope' to use this provider
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_not_exception_type, before_log, after_log
from videotrans.configure.config import tr,params,settings,app_cfg,logger
from videotrans.configure._except import NO_RETRY_EXCEPT,StopRetry
from videotrans.tts._base import BaseTTS
from videotrans.util import tools

RETRY_NUMS = 2
RETRY_DELAY = 5


# Force single thread to prevent remote restriction errors
@dataclass
class QWENTTS(BaseTTS):

    def __post_init__(self):
        super().__post_init__()
        self.role_dict=tools.get_qwen3tts_rolelist()
        self.model=params.get('qwentts_model', 'qwen3-tts-flash')
        self.api_key=params.get('qwentts_key', '')
        if self.model.startswith('qwen-tts'):
            self.model='qwen3-tts-flash'


    # Force a single thread to execute to prevent frequent concurrent failures
    def _exec(self):
        if not params.get('qwentts_key',''):
            raise StopRetry(
                tr("please input your Qwen TTS  API KEY"))
        self._local_mul_thread()

    def _item_task(self, data_item: dict = None,idx:int=-1):
        if self._exit() or not data_item.get('text','').strip():
            return
        # Main loop for infinite retry connection errors
        @retry(retry=retry_if_not_exception_type(NO_RETRY_EXCEPT), stop=(stop_after_attempt(RETRY_NUMS)),
               wait=wait_fixed(RETRY_DELAY), before=before_log(logger, logging.INFO),
               after=after_log(logger, logging.INFO))
        def _run():
            if self._exit() or tools.vail_file(data_item['filename']):
                return
            role = self.role_dict.get(data_item['role'],'Cherry')
            response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model=self.model,
                api_key=self.api_key,
                text=data_item['text'],
                voice=role,
            )

            if response is None:
                raise RuntimeError("API call returned None response")
            
            if "Access denied" in response.message:
                raise StopRetry(response.message)
            
            if not hasattr(response, 'output') or response.output is None or not hasattr(response.output, 'audio'):
                raise RuntimeError( f"{response.message if hasattr(response, 'message') else str(response)}")

            resurl = requests.get(response.output.audio["url"])
            resurl.raise_for_status()  # Check if the request is successful
            with open(data_item['filename'] + '.wav', 'wb') as f:
                f.write(resurl.content)
            self.convert_to_wav(data_item['filename'] + ".wav", data_item['filename'])


        try:
            _run()
        except Exception as e:
            self.error=e
            raise
