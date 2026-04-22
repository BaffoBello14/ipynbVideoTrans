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
    # 翻译渠道
    translate_type:int=0
    # 存放待翻译的字幕列表字典,{text,time,line}
    text_list: List[dict] = field(default_factory=list)
    # 唯一任务id
    uuid: Optional[str] = None
    # 测试时不使用缓存
    is_test: bool = False
    # 原始语言代码
    source_code: str = ""
    #目标语言代码
    target_code: str = ""
    # 对于AI渠道，这是目标语言的自然语言表达，其他渠道等于 target_code
    target_language_name: str = ""

    # 翻译API 地址
    api_url: str = field(default="", init=False)
    # 模型名
    model_name: str = field(default="", init=False)

    # 同时翻译的字幕行数量
    trans_thread: int = 5
    # 翻译后暂停秒
    wait_sec: float = float(settings.get('translation_wait', 0))
    # 以srt格式发送
    aisendsrt: bool = False
    # 原始完整字幕，当ai翻译时可作为上下文背景信息
    full_origin_subtitles:str=""

    def __post_init__(self):
        super().__post_init__()
        #是AI翻译渠道并且选中了以完整字幕发送
        if settings.get('aisendsrt', False) and self.translate_type in translator.AI_TRANS_CHANNELS:
            self.aisendsrt=True
        if self.aisendsrt and settings.get('aitrans_context'):
            self.full_origin_subtitles=_GLOBAL_CONTEXT.replace('{COMPLETE_SRT_TEXT}',"\n".join([it["text"] for it in self.text_list]))

        if not self.aisendsrt:
            self.trans_thread = int(settings.get('trans_thread', 5))
        else:
            self.trans_thread = int(settings.get('aitrans_thread', 20))
    # 发出请求获取内容 data=[text1,text2,text] | text
    # 按行翻译时，data=[text_str,...]
    # AI发送完整字幕时 data=srt_string
    def _item_task(self, data: Union[List[str], str]) -> str:
        pass

    def _exit(self):
        if app_cfg.exit_soft or (self.uuid and self.uuid in app_cfg.stoped_uuid_set):
            return True
        return False

    # 实际操作 run|runsrt -> _item_task
    def run(self) -> Union[List, str, None]:
        # 开始对分割后的每一组进行处理
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        _st=time.time()
        self._signal(text="")
        if hasattr(self,'_download'):
            self._download()

        # 如果是不是以 完整字幕格式发送，则组成字符串列表，否则组成 [dict,dict] 列表，每个dict都是字幕行信息
        if not self.aisendsrt:
            # 是文字列表  [text_str,...]
            source_text = [t['text'].replace("\n"," ") for t in self.text_list]
        else:
            # 是srt格式字幕列表 [{text,line,time},...]
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
            logger.debug(f'[字幕翻译]渠道{self.translate_type},{self.model_name}:共耗时:{int(time.time()-_st)}s')
            
    @staticmethod
    def _merge_lines_into_sentences(lines: list) -> tuple:
        """
        Merge SRT line fragments into complete sentences before translation.

        Returns:
            sentences   – list of merged sentence strings
            mapping     – list of (sentence_idx, char_start, char_end) per original line,
                          used to redistribute translated sentences back to original lines
        """
        import re
        _END = re.compile(r'[.!?;]\s*$')

        sentences, mapping = [], []
        current, members = [], []

        for i, line in enumerate(lines):
            text = line.strip()
            current.append(text)
            members.append(i)

            is_end = bool(_END.search(text))
            is_last = (i == len(lines) - 1)

            if is_end or is_last:
                sentences.append(' '.join(current))
                for orig_idx in members:
                    mapping.append(len(sentences) - 1)
                current, members = [], []

        return sentences, mapping

    def _run_text(self,split_source_text):
        # Flatten all lines for sentence-aware merging (non-AI providers only)
        all_lines = [line for batch in split_source_text for line in batch]

        # Merge SRT fragments into complete sentences, translate them, then
        # redistribute back to the original per-line structure expected downstream.
        sentences, line_to_sentence = self._merge_lines_into_sentences(all_lines)

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

            # Pad if response has fewer lines than batch
            if len(sep_res) < len(batch):
                translated_sentences += [""] * (len(batch) - len(sep_res))

            time.sleep(self.wait_sec)

        # Map translated sentences back to original SRT line count
        target_list = []
        for i in range(len(all_lines)):
            sent_idx = line_to_sentence[i] if i < len(line_to_sentence) else len(translated_sentences) - 1
            target_list.append(
                translated_sentences[sent_idx] if sent_idx < len(translated_sentences) else ""
            )

        max_i = len(target_list)
        logger.debug(f'以普通文本行按行翻译：原始行数:{len(self.text_list)},翻译后行数:{max_i}')
        for i, it in enumerate(self.text_list):
            if i < max_i:
                self.text_list[i]['text'] = target_list[i]
            else:
                self.text_list[i]['text'] = ""
        return self.text_list

    # 发送完整字幕格式内容进行翻译
    # 此时 _item_task 接收的是 srt格式的字符串
    def _run_srt(self,split_source_text):
        raws_list=[]
        for i, it in enumerate(split_source_text):
            # 是字幕类表，此时 it=[{text,line,time}]
            if self._exit(): return
            self._signal(text=tr('starttrans') + f' {i} ')
            for j, srt in enumerate(it):
                it[j]['text'] = srt['text'].strip().replace("\n", " ")
            # 组成合法的srt格式字符串
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
            #logger.debug(f'\n原始待翻译文本:{srt_str=}\n翻译结果:{result=}\n整理后：{tmp=}')
            raws_list.extend(tmp)
            time.sleep(self.wait_sec)

        logger.debug(f'按SRT格式翻译，原始字幕行数：{len(self.text_list)},整理为list[dict]后的行数:{len(raws_list)}')
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
