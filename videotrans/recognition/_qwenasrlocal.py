# zh_recognition recognition
import json
from dataclasses import dataclass
from typing import List, Dict, Union

from pathlib import Path
import re,time

from videotrans.configure.config import tr, params, settings, app_cfg, logger, defaulelang, ROOT_DIR, TEMP_DIR

from videotrans.recognition._base import BaseRecogn
from videotrans.util import tools
from videotrans.process import qwen3asr_fun


@dataclass
class QwenasrlocalRecogn(BaseRecogn):
    def __post_init__(self):
        super().__post_init__()


    def _download(self):
        if defaulelang == 'zh':
            tools.check_and_down_ms(f'Qwen/Qwen3-ASR-{self.model_name}',callback=self._process_callback,local_dir=f'{ROOT_DIR}/models/models--Qwen--Qwen3-ASR-{self.model_name}')
            
            #tools.check_and_down_ms('Qwen/Qwen3-ForcedAligner-0.6B',callback=self._process_callback,local_dir=f'{ROOT_DIR}/models/models--Qwen--Qwen3-ForcedAligner-0.6B')
        else:
            tools.check_and_down_hf(model_id=f'Qwen3-ASR-{self.model_name}',repo_id=f'Qwen/Qwen3-ASR-{self.model_name}',local_dir=f'{ROOT_DIR}/models/models--Qwen--Qwen3-ASR-{self.model_name}',callback=self._process_callback)
            
            #tools.check_and_down_hf(model_id='Qwen3-ForcedAligner-0.6B',repo_id='Qwen/Qwen3-ForcedAligner-0.6B',local_dir=f'{ROOT_DIR}/models/models--Qwen--Qwen3-ForcedAligner-0.6B',callback=self._process_callback)


    def _exec(self) -> Union[List[Dict], None]:
        if self._exit():
            return

        logs_file = f'{TEMP_DIR}/{self.uuid}/qwen3tts-{time.time()}.log'
        title="Qwen3-ASR"
        cut_audio_list_file = f'{TEMP_DIR}/{self.uuid}/cut_audio_list_{time.time()}.json'
        Path(cut_audio_list_file).write_text(json.dumps(self.cut_audio()),encoding='utf-8')
        kwargs = {     
            "cut_audio_list":   cut_audio_list_file,
            "logs_file": logs_file,
            "is_cuda": self.is_cuda,
            "audio_file":self.audio_file,
            "model_name":self.model_name
        }
        jsdata=self._new_process(callback=qwen3asr_fun,title=title,is_cuda=self.is_cuda,kwargs=kwargs)
        #print(f'{jsdata=}')
        logger.debug(f'Word timestamp data returned by Qwen-asr:{jsdata=}')

        return jsdata#self.segmentation_asr_data(jsdata)
        
    
    def segmentation_asr_data(self,asr_data, 
                                min_duration=1.0, 
                                max_pref_duration=6.0, 
                                max_hard_duration=8.0, 
                                silence_threshold=0.4):
        'Reorganize ASR word-level data into 1-6 second sentences.\n        \n        Args:\n            asr_data (list): ASR original json list\n            min_duration (float): Minimum sentence length (seconds), try not to split shorter than this\n            max_pref_duration (float): The expected maximum duration (seconds). If it exceeds this length, it will tend to be split.\n            max_hard_duration (float): absolute maximum duration (seconds), must not be exceeded\n            silence_threshold (float): How many seconds between words are considered as silence breaks\n\n        Returns:\n            list: Formatted sentence dictionary list'
        if not asr_data:
            return []

        # 1. Define multi-language punctuation mark rules (including common punctuation marks in Chinese, English, Japanese, etc.)
        # Coverage: ,.?;!: and corresponding full-width symbols
        punc_pattern = re.compile(r'[。.?？!！;；:：,，、\u3002\uff0c\uff1f\uff01]')
        
        # 2. Determine whether the character is CJK (Chinese, Japanese and Korean), which is used to determine whether to add spaces when splicing.
        def is_cjk(char):
            if not char: return False
            code = ord(char[0])
            # CJK Unified Ideographs scope roughly
            return 0x4E00 <= code <= 0x9FFF or \
                   0x3040 <= code <= 0x309F or \
                   0x30A0 <= code <= 0x30FF

        segments = []
        current_buffer = []
        
        def flush_buffer(buffer):
            'Combine the currently cached word lists into a sentence dictionary'
            if not buffer:
                return None
                
            start_ms = int(buffer[0]['start_time'] * 1000)
            end_ms = int(buffer[-1]['end_time'] * 1000)
            
            # Intelligent splicing of text
            text_parts = []
            for i, token in enumerate(buffer):
                word = token['text']
                if i == 0:
                    text_parts.append(word)
                else:
                    prev_word = buffer[i-1]['text']
                    # If the end of the previous word and the beginning of the current word are both CJK characters, splice them directly, otherwise add spaces.
                    # Note: Here we take prev_word[-1] and word[0] to judge.
                    if prev_word and word and is_cjk(prev_word[-1]) and is_cjk(word[0]):
                        text_parts.append(word)
                    else:
                        # For non-CJK languages (such as English), or mixed Chinese and English, add spaces
                        # Special case: If the current word is just a punctuation mark, no leading space is usually required (depends on the ASR format, simplified processing here)
                        if punc_pattern.match(word) and len(word) == 1:
                            text_parts.append(word)
                        else:
                            text_parts.append(" " + word) # Add spaces by default
                            
            # Clean up possible extra spaces (such as spaces mixed in Chinese)
            full_text = "".join(text_parts).strip()
            
            endraw=tools.ms_to_time_string(ms=end_ms)
            startraw=tools.ms_to_time_string(ms=start_ms)
            
            return {
                "start_time": start_ms,
                "end_time": end_ms,
                "endraw":endraw,
                "startraw":startraw,
                "time":f"{startraw} -> {endraw}",
                "text": full_text
            }

        # 3. Traverse the data and segment it
        for i, token in enumerate(asr_data):
            # Get current token information
            token_text = token.get('text', '')
            token_start = token.get('start_time', 0.0)
            token_end = token.get('end_time', 0.0)
            
            # Calculate the silence gap with the previous word
            silence_gap = 0.0
            if i > 0:
                silence_gap = token_start - asr_data[i-1]['end_time']
            
            # Even if the buffer is empty, we will put it in first and then determine whether we want to end here.
            # But for the sake of clear logic, we first determine whether we want to "settlement" the previous buffer
            
            should_split = False
            
            if current_buffer:
                buf_start = current_buffer[0]['start_time']
                current_duration = token_end - buf_start # Add the total duration after the current word
                prev_duration = current_buffer[-1]['end_time'] - buf_start # Add the duration before the current word
                
                has_punc = bool(punc_pattern.search(current_buffer[-1]['text']))
                is_long_silence = silence_gap >= silence_threshold
                
                # --- Sentence segmentation decision logic ---
                
                # 1. Hard limit: adding the current word will exceed 8s and must be cut off before the current word.
                if current_duration > max_hard_duration:
                    should_split = True
                
                # 2. Ideal interval sentence (1s - 6s): if there is punctuation or long silence
                elif prev_duration >= min_duration:
                    if has_punc:
                        should_split = True
                    elif is_long_silence:
                        should_split = True
                    # 3. Exceeded the expected maximum duration (6s): Start looking for any opportunity to break sentences (even without punctuation)
                    #Here we use silence as a weak segmentation point, and cut as long as there is a weak pause.
                    elif prev_duration >= max_pref_duration:
                        should_split = True
                
            if should_split:
                seg = flush_buffer(current_buffer)
                if seg: segments.append(seg)
                self._signal(text=seg.get('text','')+"\n",type='subtitle')
                current_buffer = []

            current_buffer.append(token)

        # 4. Process the remaining buffer
        if current_buffer:
            seg = flush_buffer(current_buffer)
            if seg: segments.append(seg)
        return segments
