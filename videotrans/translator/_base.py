import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from videotrans import translator
from videotrans.configure._base import BaseCon
from videotrans.configure.config import tr, app_cfg, settings, params, logger, TEMP_DIR, TEMP_ROOT
from videotrans.util import tools
from tenacity import RetryError

_GLOBAL_CONTEXT="""
# GLOBAL CONTEXT (Background Info)
<GLOBAL_CONTEXT>
{COMPLETE_SRT_TEXT}
</GLOBAL_CONTEXT>
**[CRITICAL WARNING]:** The `<GLOBAL_CONTEXT>` above is strictly for your reading to understand the plot, character relationships, gender, tone, and overall flow. **NEVER TRANSLATE THE GLOBAL CONTEXT.** 

"""
@dataclass
class BaseTrans(BaseCon):
    # translation channel
    translate_type:int=0
    # Store the subtitle list dictionary to be translated, {text, time, line}
    text_list: List[dict] = field(default_factory=list)
    # Unique task id
    uuid: Optional[str] = None
    # Do not use cache when testing
    is_test: bool = False
    # Original language code
    source_code: str = ""
    #Target language code
    target_code: str = ""
    # For AI channels, this is the natural language expression in the target language, for other channels it is equal to target_code
    target_language_name: str = ""

    # Translate API address
    api_url: str = field(default="", init=False)
    #Model name
    model_name: str = field(default="", init=False)

    # Number of subtitle lines translated simultaneously
    trans_thread: int = 5
    # Pause seconds after translation
    wait_sec: float = float(settings.get('translation_wait', 0))
    #Send in srt format
    aisendsrt: bool = False
    # Original complete subtitles, which can be used as contextual background information when translated by AI
    full_origin_subtitles:str=""

    def __post_init__(self):
        super().__post_init__()
        # is the AI translation channel and is selected to send with full subtitles
        if settings.get('aisendsrt', False) and self.translate_type in translator.AI_TRANS_CHANNELS:
            self.aisendsrt=True
        if self.aisendsrt and settings.get('aitrans_context'):
            self.full_origin_subtitles=_GLOBAL_CONTEXT.replace('{COMPLETE_SRT_TEXT}',"\n".join([it["text"] for it in self.text_list]))

        if not self.aisendsrt:
            self.trans_thread = int(settings.get('trans_thread', 5))
        else:
            self.trans_thread = int(settings.get('aitrans_thread', 20))
    # Make a request to get the content data=[text1,text2,text] | text
    # When translating by line, data=[text_str,...]
    # When AI sends complete subtitles data=srt_string
    def _item_task(self, data: Union[List[str], str]) -> str:
        pass

    def _exit(self):
        if app_cfg.exit_soft or (self.uuid and self.uuid in app_cfg.stoped_uuid_set):
            return True
        return False

    # Actual operation run|runsrt -> _item_task
    def run(self) -> Union[List, str, None]:
        # Start processing each group after splitting
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        _st=time.time()
        self._signal(text="")
        if hasattr(self,'_download'):
            self._download()

        # If it is sent in complete subtitle format, form a string list, otherwise form a [dict,dict] list, each dict is subtitle line information
        if not self.aisendsrt:
            # is a text list [text_str,...]
            source_text = [t['text'].replace("\n"," ") for t in self.text_list]
        else:
            # is a list of subtitles in srt format [{text,line,time},...]
            source_text=self.text_list


        split_source_text = [source_text[i:i + self.trans_thread] for i in range(0, len(source_text), self.trans_thread)]
        print(f'{self.trans_thread=},{self.aisendsrt=},{self.translate_type=}')
        try:
            if self.aisendsrt:
                return self._run_srt(split_source_text)
            return self._run_text(split_source_text)
        except RetryError as e:
            raise e.last_attempt.exception()
        except Exception as e:
            raise
        finally:
            if hasattr(self,'_unload'):
                self._unload()
            logger.debug(f'[Subtitle Translation] Channel{self.translate_type},{self.model_name}:Total time spent:{int(time.time()-_st)}s')
            
    @staticmethod
    def _merge_lines_into_sentences(lines: list) -> tuple:
        """
        Merge SRT line fragments into complete sentences before translation.

        Returns:
            sentences      – list of merged sentence strings (one per sentence group)
            group_first    – dict {original_line_idx: True}  marks the FIRST line of each group
            line_to_sent   – list mapping each original line index → sentence index
        """
        import re
        _END = re.compile(r'[.!?;]\s*$')

        sentences, line_to_sent, group_first = [], [], {}
        current, members = [], []

        for i, line in enumerate(lines):
            text = line.strip()
            current.append(text)
            members.append(i)

            is_end = bool(_END.search(text))
            is_last = (i == len(lines) - 1)

            if is_end or is_last:
                sent_idx = len(sentences)
                sentences.append(' '.join(current))
                group_first[members[0]] = True   # first line gets the translation
                for orig_idx in members:
                    line_to_sent.append(sent_idx)
                current, members = [], []

        return sentences, group_first, line_to_sent

    def _run_text(self, split_source_text):
        # Flatten all lines for sentence-aware merging (non-AI providers only)
        all_lines = [line for batch in split_source_text for line in batch]

        # Merge SRT fragments into complete sentences, translate them, then
        # put the translation on the FIRST line of each group and empty string
        # on the continuation lines (empty lines are skipped by TTS).
        sentences, group_first, line_to_sent = self._merge_lines_into_sentences(all_lines)

        # Batch the merged sentences using trans_thread
        sent_batches = [sentences[i:i + self.trans_thread]
                        for i in range(0, len(sentences), self.trans_thread)]

        translated_sentences: list = []
        for i, batch in enumerate(sent_batches):
            if self._exit():
                return
            self._signal(text=tr('starttrans') + f' {i} ')
            result = self._get_cache(batch)
            if not result:
                result = tools.cleartext(self._item_task(batch))
                self._set_cache(batch, result)

            sep_res = result.split("\n")
            for x in range(len(batch)):
                translated_sentences.append(sep_res[x].strip() if x < len(sep_res) else "")
                self._signal(text=(sep_res[x] if x < len(sep_res) else "") + "\n", type='subtitle')

            if len(sep_res) < len(batch):
                translated_sentences += [""] * (len(batch) - len(sep_res))

            time.sleep(self.wait_sec)

        # Redistribute: first line of group → translated sentence, rest → ""
        target_list = []
        for i in range(len(all_lines)):
            if group_first.get(i):
                sent_idx = line_to_sent[i] if i < len(line_to_sent) else len(translated_sentences) - 1
                target_list.append(
                    translated_sentences[sent_idx] if sent_idx < len(translated_sentences) else ""
                )
            else:
                target_list.append("")   # continuation line: TTS will skip it

        max_i = len(target_list)
        logger.debug(f'Line-by-line translation in normal text lines: Original number of lines:{len(self.text_list)}, number of translated lines:{max_i}')
        for i, it in enumerate(self.text_list):
            if i < max_i:
                self.text_list[i]['text'] = target_list[i]
            else:
                self.text_list[i]['text'] = ""
        return self.text_list

    # Send full subtitle format content for translation
    # At this time, _item_task receives a string in srt format
    def _run_srt(self,split_source_text):
        raws_list=[]
        for i, it in enumerate(split_source_text):
            # is the subtitle class table, at this time it=[{text,line,time}]
            if self._exit(): return
            self._signal(text=tr('starttrans') + f' {i} ')
            for j, srt in enumerate(it):
                it[j]['text'] = srt['text'].strip().replace("\n", " ")
            # Form a legal srt format string
            srt_str = "\n\n".join(
                [f"{srt_dict['line']}\n{srt_dict['time']}\n{srt_dict['text'].strip()}" for srt_dict in it])
            result = self._get_cache(srt_str)
            if not result:
                result = self._item_task(srt_str)
                if not result.strip():
                    raise RuntimeError(tr("Translate result is empty"))
                self._set_cache(it, result)


            self._signal(text=result, type='subtitle')
            tmp=tools.get_subtitle_from_srt(result, is_file=False)
            #logger.debug(f'\nOriginal text to be translated: {srt_str=}\nTranslation result: {result=}\nAfter finishing: {tmp=}')
            raws_list.extend(tmp)
            time.sleep(self.wait_sec)

        logger.debug(f'Translated in SRT format, original subtitle lines:{len(self.text_list)}, the number of rows after sorting into list[dict]:{len(raws_list)}')
        for i, it in enumerate(raws_list):
            if i>=len(self.text_list):
                continue
            it['text']=it['text'].strip()
        return raws_list

    def _set_cache(self, it, res_str):
        if not res_str.strip():
            return
        key_cache = self._get_key(it)

        file_cache = TEMP_ROOT + f'/translate_cache/{key_cache}.txt'
        if not Path(TEMP_ROOT + f'/translate_cache').is_dir():
            Path(TEMP_ROOT + f'/translate_cache').mkdir(parents=True, exist_ok=True)
        Path(file_cache).write_text(res_str, encoding='utf-8')

    def _get_cache(self, it):
        if self.is_test: return None
        key_cache = self._get_key(it)
        file_cache = TEMP_ROOT + f'/translate_cache/{key_cache}.txt'
        if Path(file_cache).exists():
            return Path(file_cache).read_text(encoding='utf-8')
        return None

    def _get_key(self, it):
        Path(TEMP_ROOT + '/translate_cache').mkdir(parents=True, exist_ok=True)
        key_str=f'{self.translate_type}-{self.api_url}-{self.aisendsrt}-{self.model_name}-{self.source_code}-{self.target_code}-'+(it if isinstance(it, str) else json.dumps(it))
        return tools.get_md5(key_str)
