import re
from dataclasses import dataclass
from typing import List, Union
try:
    import dashscope
except ImportError:
    dashscope = None  # install 'dashscope' to use this provider
from videotrans.configure.config import tr,settings,params,app_cfg,logger
from videotrans.translator._base import BaseTrans
from videotrans.util import tools

RETRY_NUMS = 3
RETRY_DELAY = 5


@dataclass
class QwenMT(BaseTrans):
    def __post_init__(self):
        super().__post_init__()

    def _item_task(self, data: Union[List[str], str]) -> str:
        if self._exit(): return
        text = "\n".join([i.strip() for i in data]) if isinstance(data, list) else data
        model_name=params.get('qwenmt_model', 'qwen-mt-turbo')
        if model_name=='qwen-turbo':
            model_name='qwen-mt-turbo'
        if model_name.startswith('qwen-mt'):

            messages = [
                {
                    "role": "user",
                    "content":text
                }
            ]
            logger.debug(f'qwen-mt request:{messages}')

            translation_options = {
                "source_lang": "auto",
                "target_lang": self.target_language_name
            }
            # Glossary
            term=tools.qwenmt_glossary()
            if term:
                translation_options['terms']=term
            if params.get("qwenmt_domains"):
                translation_options['domains']=params.get("qwenmt_domains")


            response = dashscope.Generation.call(
                # If the environment variable is not configured, please replace the following line with Alibaba Cloud Bailian API Key: api_key="sk-xxx",
                api_key=params.get('qwenmt_key',''),
                model=model_name,
                messages=messages,
                result_format='message',
                translation_options=translation_options
            )
            if response.code or not response.output:
                raise RuntimeError(response.message)
            logger.debug(f'qwen-mt returns response:{response.output.choices[0].message.content}')
            return self.clean_srt(response.output.choices[0].message.content)

        self.prompt = tools.get_prompt(ainame='bailian',aisendsrt=self.aisendsrt).replace('{lang}', self.target_language_name)
        message = [
            {
                'role': 'system',
                'content':'You are a top-tier Subtitle Translation Engine.'},
            {
                'role': 'user',
                'content': self.prompt.replace('{batch_input}', f'{text}').replace('{context_block}',self.full_origin_subtitles)
                },
        ]
        response = dashscope.Generation.call(
            # If there is no environment variable configured, please replace the following line with Bailian API Key: api_key="sk-xxx",
            api_key=params.get('qwenmt_key',''),
            model=model_name,
            # Here we take qwen-plus as an example. You can change the model name as needed. Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=message,
            result_format='message'
        )

        if response.code or not response.output:
            raise RuntimeError(response.message)
        logger.debug(f'Alibaba Bailian AI response:{response.output.choices[0].message.content}')
        match = re.search(r'<TRANSLATE_TEXT>(.*?)</TRANSLATE_TEXT>', response.output.choices[0].message.content, re.S)
        if match:
            return match.group(1)
        return ''



    def clean_srt(self, srt):
        #Replace special symbols
        srt = re.sub(r'&gt;', '>', srt,flags=re.I | re.S)
        # :: Replace with :
        srt = re.sub(r'([：:])\s*', ':', srt,flags=re.I | re.S)
        # ,, replace with ,
        srt = re.sub(r'([,，])\s*', ',', srt,flags=re.I | re.S)
        srt = re.sub(r'([`’\'\"])\s*', '', srt,flags=re.I | re.S)

        # Between seconds and milliseconds. Replace with,
        srt = re.sub(r'(:\d+)\.\s*?(\d+)', r'\1,\2', srt,flags=re.I | re.S)
        # Add spaces before and after the time line
        time_line = r'(\s?\d+:\d+:\d+(?:,\d+)?)\s*?-->\s*?(\d+:\d+:\d+(?:,\d+)?\s?)'
        srt = re.sub(time_line, r"\n\1 --> \2\n", srt,flags=re.I | re.S)
        # twenty one\n00:01:18,560 --> 00:01:22,000\n
        srt = re.sub(r'\s?[a-zA-Z ]{3,}\s*?\n?(\d{2}:\d{2}:\d{2}\,\d{3}\s*?\-\->\s*?\d{2}:\d{2}:\d{2}\,\d{3})\s?\n?',
                     "\n" + r'1\n\1\n', srt,flags=re.I | re.S)
        # Remove extra blank lines
        srt = "\n".join([it.strip() for it in srt.splitlines() if it.strip()])

        # Delete multiple time lines connected by spaces or newlines
        time_line2 = r'(\s\d+:\d+:\d+(?:,\d+)?)\s*?-->\s*?(\d+:\d+:\d+(?:,\d+)?\s)(?:\s*\d+:\d+:\d+(?:,\d+)?)\s*?-->\s*?(\d+:\d+:\d+(?:,\d+)?\s*)'
        srt = re.sub(time_line2, r'\n\1 --> \2\n', srt,flags=re.I | re.S)
        srt_list = [it.strip() for it in srt.splitlines() if it.strip()]

        remove_list = []
        for it in srt_list:
            if len(remove_list) > 0 and str(it) == str(remove_list[-1]):
                if re.match(r'^\d{1,4}$', it):
                    continue
                if re.match(r'\d+:\d+:\d+([,.]\d+)? --> \d+:\d+:\d+([,.]\d+)?',it):
                    continue
            remove_list.append(it)

        srt = "\n".join(remove_list)

        # Add a newline character before the line number
        srt = re.sub(r'\s?(\d+)\s+?(\d+:\d+:\d+)', r"\n\n\1\n\2", srt,flags=re.I | re.S)
        return srt.strip().replace('&#39;', '"').replace('&quot;', "'")
