import logging
import re
from dataclasses import dataclass
from typing import List, Union

import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_not_exception_type, before_log, after_log

from videotrans.configure._except import NO_RETRY_EXCEPT
from videotrans.configure.config import tr,logger,app_cfg,settings,params
from videotrans.translator._base import BaseTrans

RETRY_NUMS = 3
RETRY_DELAY = 5


@dataclass
class Google(BaseTrans):

    def __post_init__(self):
        super().__post_init__()
        self.aisendsrt = False


    # Actually make a request to get the result
    @retry( retry=retry_if_not_exception_type(NO_RETRY_EXCEPT),stop=(stop_after_attempt(RETRY_NUMS)),
           wait=wait_fixed(RETRY_DELAY), before=before_log(logger, logging.INFO),
           after=after_log(logger, logging.INFO))
    def _item_task(self, data: Union[List[str], str]) -> str:

        if self._exit(): return
        text = "\n".join([i.strip() for i in data]) if isinstance(data, list) else data
        source_code = 'auto' if not self.source_code else self.source_code
        url = f"https://translate.google.com/m?sl={source_code}&tl={self.target_code}&hl={self.target_code}&q={text}"
        logger.debug(f'[Google] {self.target_code=} {self.source_code=}')
        headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1'
        }
        response = requests.get(url, headers=headers,  verify=False)
        response.raise_for_status()
        logger.debug(f'[Google]Return code:{response.status_code=}')

        re_result = re.search(r'<div\s+class=\Wresult-container\W>([^<]+?)<', response.text)
        if not re_result or len(re_result.groups()) < 1:
            logger.debug(f'[Google]Return data:{response.text=}')
            raise RuntimeError(tr("Google Translate error"))
        return re_result.group(1)

    def clean_srt(self, srt):
        # The translated srt subtitles are most likely to contain various grammatical errors, symbols and formatting confusion.
        try:
            srt = re.sub(r'(\d{2})\s*[:：]\s*(\d{2})[:：]\s*(\d{2})[\s\,，]+(\d{3})', r'\1:\2:\3,\4', srt,flags=re.I | re.S)
        except re.error:
            pass
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
