# Speech recognition, new process execution
# Return tuple
# Failure: If the first value is False, it is a failure, and the second value stores the reason for the failure.
# Success, the first value has the required return value, returns True if not needed, and the second value is None
import re

from videotrans.util import gpus
from videotrans.configure.config import logger, ROOT_DIR, defaulelang


def openai_whisper(
        *,
        prompt=None,
        detect_language=None,
        model_name=None,
        logs_file=None,
        is_cuda=False,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
        speech_timestamps=None,
        audio_file=None,
        jianfan=False,
        audio_duration=0,
        temperature=None,
        compression_ratio_threshold=2.2,
        device_index=0,  # gpu index
        max_speech_ms=6000
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    try:
        import whisper
    except ModuleNotFoundError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"])
        import whisper
    from videotrans.util import tools

    import zhconv

    if not Path(f'{ROOT_DIR}/models/{model_name}.pt').exists():
        msg = f"model{model_name} Does not exist, will be downloaded automatically" if defaulelang == 'zh' else f'Model {model_name} does not exist and will be automatically downloaded'
    else:
        msg = f"loading {model_name}"
    _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
    model = None
    raws = []
    try:
        if speech_timestamps and isinstance(speech_timestamps, str):
            speech_timestamps = json.loads(Path(speech_timestamps).read_text(encoding='utf-8'))
        if not temperature:
            temperature = (
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            )
        elif str(temperature).startswith('[') or str(temperature).startswith('('):
            temperature = tuple([float(i) for i in str(temperature)[1:-1].split(',')])
        else:
            temperature = float(temperature)

        model = whisper.load_model(
            model_name,
            device=f"cuda:{device_index}" if is_cuda else 'cpu',
            download_root=ROOT_DIR + "/models"
        )
        msg = f"Loaded {model_name}"
        _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))

        last_end_time = audio_duration / 1000.0 if audio_duration > 0 else speech_timestamps[-1][1] / 1000.0
        speech_timestamps_flat = []
        if detect_language == 'fil':
            detect_language = 'tl'
        if speech_timestamps:
            _write_log(logs_file, json.dumps({"type": "logs", "text": 'Transcribe batch...'}))
            for it in speech_timestamps:
                speech_timestamps_flat.extend([it[0] / 1000.0, it[1] / 1000.0])
            result = model.transcribe(
                audio_file,
                no_speech_threshold=no_speech_threshold,
                language=detect_language.split('-')[0] if detect_language != 'auto' else None,
                clip_timestamps=speech_timestamps_flat,
                initial_prompt=prompt if prompt else None,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                condition_on_previous_text=condition_on_previous_text
            )
            i = 0
            for segment in result['segments']:
                # If the timestamp is greater than the total duration, it will be skipped if an error occurs.
                if segment['end'] > last_end_time:
                    continue
                text = segment['text']
                if not text.strip():
                    continue
                i += 1
                if jianfan:
                    text = zhconv.convert(text, 'zh-hans')
                s, e = int(segment['start'] * 1000), int(segment['end'] * 1000)
                tmp = {
                    'text': text,
                    'start_time': s,
                    'end_time': e
                }
                tmp['startraw'] = tools.ms_to_time_string(ms=tmp['start_time'])
                tmp['endraw'] = tools.ms_to_time_string(ms=tmp['end_time'])
                tmp['time'] = f"{tmp['startraw']} --> {tmp['endraw']}"
                raws.append(tmp)
                _write_log(logs_file, json.dumps({"type": "subtitle", "text": f'[{i}] {text}\n'}))
            logger.debug(f'In openai-whisper mode, use VAD to split the audio in advance and use it directly{model_name}Text results for each segment of audio returned by the model')
        else:
            _write_log(logs_file, json.dumps({"type": "logs", "text": 'Transcribe word_timestamps'}))
            segments = model.transcribe(
                audio_file,
                no_speech_threshold=no_speech_threshold,
                language=detect_language.split('-')[0] if detect_language != 'auto' else None,
                # clip_timestamps=speech_timestamps_flat,
                initial_prompt=prompt if prompt else None,
                temperature=temperature,
                word_timestamps=True,
                compression_ratio_threshold=compression_ratio_threshold,
                condition_on_previous_text=condition_on_previous_text
            )
            logger.debug(f'In openai-whisper mode, yes{model_name}The sentence segmentation results returned by the model are re-corrected.')
            texts = []
            i=0
            for segment in segments['segments']:
                i+=1
                texts.append({
                    "text": segment['text'],
                    "start": segment['start'],
                    "end": segment['end'],
                    "words": [{'word': it['word'], 'start': it['start'], 'end': it['end']} for it in segment['words']]
                })
                _write_log(logs_file, json.dumps({"type": "subtitle", "text": f'[{i}] {segment["text"]}\n'}))
            raws = _resegment(texts, segments['language'], max_speech_ms)
            if jianfan and raws:
                for it in raws:
                    it['text'] = zhconv.convert(it['text'], 'zh-hans')
    except BaseException:
        msg = traceback.format_exc()
        logger.exception(f'Speech recognition failed:{model_name=},{msg}', exc_info=True)
        return False, msg
    else:
        return raws, None




def _resegment(texts, language, max_speech_ms):
    "\n    Only the Whisper recognition results that are too long are re-segmented and formatted into SRT subtitle format.\n    Keep Whisper's original normal short sentences without global leveling.\n    "

    # --- Helper function: Convert milliseconds to SRT standard time format HH:MM:SS,mmm ---
    def format_srt_time(ms_time):
        ms_time = int(ms_time)
        seconds, milliseconds = divmod(ms_time, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    # --- Language connection rules and punctuation judgment ---
    # Oriental, Chinese, Japanese, Korean and other languages usually do not require spaces, while other alphabetic languages require spaces.
    no_space_langs = {'zh', 'ja', 'th', 'yue', 'ko'}
    use_space = language.lower() not in no_space_langs

    end_punc = set('.?!。？！\n')
    comma_punc = set(',;:，；：、')

    def has_punc(text, punc_set):
        if not text:
            return False
        return text[-1] in punc_set

    def build_text(chunk_words):
        if use_space:
            text_str = " ".join(chunk_words)
            # Fix the problem of punctuation leading spaces caused by space connection in alphabetic languages (such as "Hello, world" -> "Hello, world")
            text_str = re.sub(r'\s+([.,?!:;])', r'\1', text_str)
        else:
            text_str = "".join(chunk_words)
        return text_str.strip()

    # --- Core logic ---
    final_segments = []

    for segment in texts:
        seg_start_ms = float(segment.get('start', 0)) * 1000
        seg_end_ms = float(segment.get('end', 0)) * 1000
        seg_duration = seg_end_ms - seg_start_ms
        words = segment.get('words', [])

        # 1. If the sentence length does not exceed max_speech_ms, or there is no words data for segmentation
        # Keep the sentence as it is without destroying Whisper’s original sentence segmentation structure
        if seg_duration <= max_speech_ms or not words:
            final_segments.append({
                'text': segment.get('text', '').strip(),
                'start': seg_start_ms,
                'end': seg_end_ms
            })
            continue

        # 2. If the sentence is too long, you must go inside it and use words to re-segment it locally.
        current_chunk = []
        chunk_start_ms = None
        prev_word_end_ms = None
        prev_word_text = ""

        for w in words:
            w_text = w.get('word', '').strip()
            if not w_text:
                continue

            w_start_ms = float(w.get('start', 0)) * 1000
            w_end_ms = float(w.get('end', 0)) * 1000

            if chunk_start_ms is None:
                chunk_start_ms = w_start_ms

            # Prediction: If the current word is added, what will be the length of the current clause?
            future_duration = w_end_ms - chunk_start_ms

            # --- Determine whether it needs to be cut off ---
            should_split = False

            # Forced cutoff: If not cut, adding this word will directly time out (make sure it is absolutely <= max_speech_ms)
            if future_duration > max_speech_ms and len(current_chunk) > 0:
                should_split = True
            else:
                # Flexible cut-off: Look for punctuation or obvious speech pauses without timeout.
                pause_ms = w_start_ms - prev_word_end_ms if prev_word_end_ms is not None else 0
                current_duration = prev_word_end_ms - chunk_start_ms if prev_word_end_ms else 0

                if len(current_chunk) > 0:
                    #End when encountering strong punctuation
                    if has_punc(prev_word_text, end_punc):
                        should_split = True
                    # Encountering noticeable long silent pauses (>= 800ms)
                    elif pause_ms >= 800:
                        should_split = True
                    # Encounter short pauses (>= 300ms) accompanied by weak punctuation such as commas
                    elif has_punc(prev_word_text, comma_punc) and pause_ms >= 300:
                        should_split = True
                    # In order to prevent some long sentences that have neither punctuation nor large pauses, if the duration is more than half and a medium pause (>=400ms) is encountered, the sentence will be cut off decisively.
                    elif current_duration > (max_speech_ms * 0.5) and pause_ms >= 400:
                        should_split = True

            if should_split:
                # Settle the current clause
                final_segments.append({
                    'text': build_text(current_chunk),
                    'start': chunk_start_ms,
                    'end': prev_word_end_ms
                })
                # Set the current word as the beginning of the next new clause
                current_chunk = [w_text]
                chunk_start_ms = w_start_ms
            else:
                # Do not cut off, absorb the word into the current clause
                current_chunk.append(w_text)

            prev_word_end_ms = w_end_ms
            prev_word_text = w_text

        # After traversing all the words of the sentence, end the remaining phrases
        if current_chunk:
            final_segments.append({
                'text': build_text(current_chunk),
                'start': chunk_start_ms,
                'end': prev_word_end_ms
            })

    # --- 3. Assembly output: encapsulated into the specified SRT dictionary list format ---
    srt_output = []
    for idx, seg in enumerate(final_segments):
        start_ms = int(seg['start'])
        end_ms = int(seg['end'])

        start_raw = format_srt_time(start_ms)
        end_raw = format_srt_time(end_ms)

        srt_output.append({
            "line": idx + 1,
            "text": seg['text'],
            "start_time": start_ms,
            "end_time": end_ms,
            "startraw": start_raw,
            "endraw": end_raw,
            "time": f"{start_raw} --> {end_raw}"
        })

    return srt_output


def faster_whisper(
        *,
        prompt=None,
        detect_language=None,
        model_name=None,
        logs_file=None,
        is_cuda=False,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
        speech_timestamps=None,
        audio_file=None,
        local_dir=None,
        compute_type="default",
        beam_size=5,
        best_of=5,
        jianfan=False,
        audio_duration=0,
        temperature=None,
        hotwords=None,
        repetition_penalty=1.0,
        compression_ratio_threshold=2.2,
        device_index=0,  # gpu index
        max_speech_ms=6000
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    from videotrans.util import tools
    import zhconv

    model = None
    batched_model = None
    raws = []
    if detect_language == 'fil':
        detect_language = 'tl'

    try:
        if speech_timestamps and isinstance(speech_timestamps, str):
            speech_timestamps = json.loads(Path(speech_timestamps).read_text(encoding='utf-8'))
        last_end_time = audio_duration / 1000.0 if audio_duration > 0 else speech_timestamps[-1][1] / 1000.0
        
        # When not forced to specify, cuda takes precedence int8_float16,cpu int8
        # 50x series graphics cards will report an error, fall back to float16
        if compute_type in ['auto','default']:
            logger.debug(f'[faster_whisper][{is_cuda=}]Original auto|defaultDefault precision:{compute_type}')
            compute_type= 'int8_float16' if is_cuda else 'int8'
        logger.debug(f'[faster_whisper][{is_cuda=}]Actual calculation accuracy:{compute_type}')
        try:
            # 1. Load the basic model
            _write_log(logs_file, json.dumps({"type": "logs", "text": 'loading model'}))
            try:
                model = WhisperModel(
                    local_dir,
                    device="cuda" if is_cuda else 'cpu',
                    device_index=device_index if is_cuda else 0,
                    compute_type=compute_type
                )
            except Exception as e:
                # Data type is incompatible, fallback
                if "the target device or backend do not support efficient" in str(e):
                    compute_type='float16' if is_cuda else ('float32' if compute_type=='int8' else 'int8')
                    logger.debug(f'[faster_whisper][{is_cuda=}] The actual use of calculation accuracy fails and falls back to:{compute_type}')
                    model = WhisperModel(
                        local_dir,
                        device="cuda" if is_cuda else 'cpu',
                        device_index=device_index if is_cuda else 0,
                        # Specify using float16 to retry under cuda. If the original int8 error is reported under cpu, fall back to float32, otherwise int8
                        compute_type=compute_type
                    )
                else:
                    raise
        except Exception as e:
            error = traceback.format_exc()
            logger.error(f'[faster_whisper][{is_cuda=}][{compute_type=}] Voice transcription failed:{local_dir=}\n{error}')
            if 'model.bin is incomplete' in error or 'json.exception.parse_error' in error or 'EOF while parsing a value' in error:
                msg = (
                    f'The model download is incomplete, please delete the directory{local_dir}, re-download' if defaulelang == "zh" else f"The model download may be incomplete, please delete the directory {local_dir} and download it again")
            elif "CUBLAS_STATUS_NOT_SUPPORTED" in error:
                msg = f"data type{compute_type} Not compatible...:{error}"
            elif "cudaErrorNoKernelImageForDevice" in error:
                msg = f"The pytorch and cuda versions are incompatible...:{error}"
            else:
                msg = error
            return False, msg

        if not temperature:
            temperature = [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ]
        elif str(temperature).startswith('[') or str(temperature).startswith('('):
            temperature = [float(i) for i in str(temperature)[1:-1].split(',')]
        else:
            temperature = float(temperature)

        if speech_timestamps:
            _write_log(logs_file, json.dumps({"type": "logs", "text": 'Transcribe batch...'}))
            # 4. Perform batch inference
            # Use batched_model.transcribe
            batched_model = BatchedInferencePipeline(model=model)

            # 3. Convert timestamp format
            # BatchedInferencePipeline requires [{'start': start_sec, 'end': end_sec}, ...]
            clip_timestamps_dicts = [
                {"start": it[0] / 1000.0, "end": it[1] / 1000.0}
                for it in speech_timestamps
            ]
            segments, info = batched_model.transcribe(
                audio_file,
                batch_size=4,  #
                beam_size=beam_size,
                best_of=best_of,
                no_speech_threshold=no_speech_threshold,
                # vad_filter must be False, otherwise clip_timestamps may be ignored or conflict,
                vad_filter=False,
                clip_timestamps=clip_timestamps_dicts,  # Custom segmentation
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=False,
                without_timestamps=True,
                temperature=temperature,
                hotwords=hotwords,
                repetition_penalty=repetition_penalty,
                compression_ratio_threshold=compression_ratio_threshold,
                language=detect_language.split('-')[0] if detect_language and detect_language != 'auto' else None,
                initial_prompt=prompt if prompt else None
            )
            i = 0
            logger.debug(f'In faster-whisper mode, VAD is used to split the audio in advance.{model_name}The text results returned by the model are used directly')
            for segment in segments:
                if segment.end > last_end_time:
                    continue
                text = segment.text
                if not text.strip():
                    continue
                i += 1
                s, e = int(segment.start * 1000), int(segment.end * 1000)
                if jianfan:
                    text = zhconv.convert(text, 'zh-hans')
                tmp = {
                    'text': text,
                    'start_time': s,
                    'end_time': e
                }
                tmp['startraw'] = tools.ms_to_time_string(ms=tmp['start_time'])
                tmp['endraw'] = tools.ms_to_time_string(ms=tmp['end_time'])
                tmp['time'] = f"{tmp['startraw']} --> {tmp['endraw']}"
                raws.append(tmp)
                _write_log(logs_file, json.dumps({"type": "subtitle", "text": f'[{i}] {text}\n'}))
        else:
            _write_log(logs_file, json.dumps({"type": "logs", "text": 'Transcribe word_timestamps'}))
            segments, info = model.transcribe(
                audio_file,
                beam_size=beam_size,
                best_of=best_of,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=140,min_speech_duration_ms=0),
                no_speech_threshold=no_speech_threshold,
                # clip_timestamps="0",  # clip_timestamps,
                word_timestamps=True,
                # without_timestamps=False,
                temperature=temperature,
                hotwords=hotwords,
                repetition_penalty=repetition_penalty,
                compression_ratio_threshold=compression_ratio_threshold,
                language=detect_language.split('-')[0] if detect_language and detect_language != 'auto' else None,
                initial_prompt=prompt if prompt else None
            )
            logger.debug(f'In faster-whisper mode, yes{model_name}The sentence segmentation results returned by the model have been revised')
            texts = []
            i=0
            for segment in segments:
                i+=1
                texts.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "words": [{'word': it.word, 'start': it.start, 'end': it.end} for it in segment.words]
                })
                _write_log(logs_file, json.dumps({"type": "subtitle", "text": f'[{i}] {segment.text}\n'}))
            raws = _resegment(texts, info.language, max_speech_ms)
            if jianfan and raws:
                for it in raws:
                    it['text'] = zhconv.convert(it['text'], 'zh-hans')
    except BaseException:
        msg = traceback.format_exc()
        logger.exception(f'Speech recognition failed:{model_name=},{msg}', exc_info=True)
        return False, msg
    else:
        return raws, None




def pipe_asr(
        prompt=None,
        cut_audio_list=None,
        detect_language=None,
        model_name=None,
        logs_file=None,
        is_cuda=False,
        audio_file=None,
        local_dir=None,
        jianfan=False,
        device_index=0  # gpu index
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    from transformers import pipeline
    import zhconv

    # Define the input generator to directly feed the path or audio data to the pipeline
    def inputs_generator():
        for item in raws:
            yield item['file']

    # 2. Initialize Pipeline
    # Use device_map="auto" to automatically allocate, or specify device
    device_arg = device_index if is_cuda else gpus.mps_or_cpu()
    # Note: When using device_map="auto", there is usually no need to pass the device parameter. Choose one of the two.
    # If it is a single-card environment, the efficiency of directly passing device=0 is usually slightly higher than device_map="auto"
    p = None
    msg = f"Loading pipeline from {local_dir}"
    _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
    if detect_language == 'fil':
        detect_language = 'tl'
    try:
        if cut_audio_list and isinstance(cut_audio_list, str):
            cut_audio_list = json.loads(Path(cut_audio_list).read_text(encoding='utf-8'))

        raws = cut_audio_list

        p = pipeline(
            task="automatic-speech-recognition",
            model=local_dir,
            batch_size=4,
            device=device_arg,
            dtype=torch.float16 if is_cuda else torch.float32,
        )

        msg = f'Pipeline loaded on device={(p.model.device)}'
        _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
        # 3. Dynamically build generate_kwargs
        generate_kwargs = {}

        # Get the model type, such as 'whisper', 'wav2vec2', 'huBERT', 'parakeet', etc.
        model_type = p.model.config.model_type
        is_whisper = "whisper" in model_type.lower()

        if is_whisper:
            # === Whisper-specific parameters ===
            lang = detect_language.split('-')[0] if detect_language != 'auto' else None

            generate_kwargs["task"] = "transcribe"
            if lang:
                generate_kwargs["language"] = lang

            # Handle Prompt
            if prompt:
                # Get tokenizer and convert prompt to token IDs
                # Compatible with older version transformers
                if hasattr(p.tokenizer, "get_prompt_ids"):
                    prompt_ids = p.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
                else:
                    # Universal fallback scheme
                    prompt_ids = p.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

                # Make sure the tensor is on the correct device
                if is_cuda:
                    prompt_ids = prompt_ids.to(p.model.device)

                # Note: You need to take [0] or the tensor itself, depending on the pipeline version.
                # Usually just pass in tensor, but some versions require list.
                # For safety reasons, it is usually supported to convert it to tensor, or to list: prompt_ids.tolist()[0]
                generate_kwargs["prompt_ids"] = prompt_ids

        else:
            # === Other architectures (e.g. Parakeet, Wav2Vec2) ===
            # These models usually do not require the language parameter (or are predefined), and do not support prompt_ids
            pass

        # 4. Perform batch inference
        # p(...) here returns an iterator, which will perform batch processing in the background.
        results_iterator = p(
            inputs_generator(),
            generate_kwargs=generate_kwargs
        )

        total = len(raws)

        # 5. Collect results
        # Note: Here we traverse both raws and results_iterator at the same time
        # Because inputs_generator yields in order, results_iterator will also output in order.
        for i, (it, res) in enumerate(zip(raws, results_iterator)):
            _write_log(logs_file, json.dumps({"type": "logs", "text": f"subtitles {i + 1}/{total}..."}))

            text = res.get('text', '')

            # Clean up file path references (if needed)
            if 'file' in it:
                del it['file']

            if text:
                # Clean up special tags
                cleaned_text = re.sub(r'<unk>|</unk>', '', text).strip()
                if jianfan:
                    cleaned_text = zhconv.convert(cleaned_text, 'zh-hans')
                raws[i]['text'] = cleaned_text

                # If the pipeline returns timestamps (depending on chunk_length_s and return_timestamps parameters)
                _write_log(logs_file, json.dumps({"type": "subtitles", "text": f'[{i}] {cleaned_text}\n'}))
        return raws, None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'Speech recognition failed:{model_name=},{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            if p:
                del p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass


def paraformer(
        cut_audio_list=None,
        detect_language=None,
        model_name=None,
        logs_file=None,
        is_cuda=False,
        audio_file=None,
        max_speakers=-1,
        cache_folder=None,
        device_index=0  # gpu index
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    from videotrans.util import tools
    # from funasr import AutoModel
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    msg = f'Load {model_name}'
    _write_log(logs_file, json.dumps({"type": "logs", "text": f'{msg}'}))

    raw_subtitles = []
    model = None
    device = f'cuda:{device_index}' if is_cuda else gpus.mps_or_cpu()
    try:
        model = pipeline(
            task=Tasks.auto_speech_recognition,
            model='iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            # model_revision="v2.0.4",
            vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            # vad_model_revision="v2.0.4",
            punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            # punc_model_revision="v2.0.3",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            # spk_model_revision="v2.0.2",
            disable_update=True,
            disable_progress_bar=True,
            disable_log=True,
            device=device
            # trust_remote_code=True,
        )

        msg = "Model loading is complete, enter recognition"
        _write_log(logs_file, json.dumps({"type": "logs", "text": f'{msg}'}))
        num = 0
        res = model(audio_file)
        speaker_list = []
        i = 0
        for it in res[0]['sentence_info']:
            if not it.get('text', '').strip():
                continue
            i += 1
            if max_speakers > -1:
                speaker_list.append(f"spk{it.get('spk', 0)}")
            tmp = {
                "line": len(raw_subtitles) + 1,
                "text": it['text'].strip(),
                "start_time": it['start'],
                "end_time": it['end'],
                "startraw": f'{tools.ms_to_time_string(ms=it["start"])}',
                "endraw": f'{tools.ms_to_time_string(ms=it["end"])}'
            }
            _write_log(logs_file, json.dumps({"type": "subtitles", "text": f'[{i}] {it["text"]}\n'}))
            tmp['time'] = f"{tmp['startraw']} --> {tmp['endraw']}"
            raw_subtitles.append(tmp)
        if speaker_list:
            Path(f'{cache_folder}/speaker.json').write_text(json.dumps(speaker_list), encoding='utf-8')
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'Speech recognition failed:{model_name=},{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if model:
                del model
            import gc
            gc.collect()
        except Exception:
            pass

    return raw_subtitles, None


def _write_log(file, msg):
    from pathlib import Path

    try:
        Path(file).write_text(msg, encoding='utf-8')
    except Exception as e:
        logger.exception(f'Error writing new process log', exc_info=True)


def _remove_unwanted_characters(text: str) -> str:
    import re
    # Retain Chinese, Japanese, Korean, English, numbers and common symbols, and remove other characters
    allowed_characters = re.compile(r'<\|\w+\|>')
    return re.sub(allowed_characters, '', text)


def qwen3asr_fun0(
        logs_file=None,
        is_cuda=False,
        audio_file=None,
        model_name="1.7B",
        device_index=0  # gpu index
):
    import torch, json
    torch.set_num_threads(1)
    from qwen_asr import Qwen3ASRModel
    atten = None
    if is_cuda:
        device_map = f'cuda:{device_index}'
        dtype = torch.float16
        try:
            import flash_attn
        except ImportError:
            pass
        else:
            atten = 'flash_attention_2'
    else:
        device_map = 'cpu'
        dtype = torch.float32
    model = None
    try:
        _write_log(logs_file, json.dumps({"type": "logs", "text": f'Load Qwen3ASR on {device_map}'}))
        model = Qwen3ASRModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-ASR-{model_name}",
            dtype=dtype,
            device_map=device_map,
            attn_implementation=atten,
            max_inference_batch_size=1,
            # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=2048,  # Maximum number of tokens to generate. Set a larger value for long audio input.
            forced_aligner=f"{ROOT_DIR}/models/models--Qwen--Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device_map,
                attn_implementation=atten
            ),
        )
        results = model.transcribe(
            audio=[audio_file],
            language=None,  # can also be set to None for automatic language detection
            return_time_stamps=True,
        )
        if not results or not hasattr(results[0], 'time_stamps') or not hasattr(results[0].time_stamps, 'items'):
            return False, "No asr results"
        list_dict = []
        _write_log(logs_file, json.dumps({"type": "logs", "text": f"ASR ended,waiting re-segment"}))
        for it in results[0].time_stamps.items:
            list_dict.append({
                "start_time": it.start_time,
                "end_time": it.end_time,
                "text": it.text,
            })

        return list_dict, None
    except Exception as e:
        return False, str(e)
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if model:
                del model
            import gc
            gc.collect()
        except Exception:
            pass


def qwen3asr_fun(
        cut_audio_list=None,
        logs_file=None,
        is_cuda=False,
        audio_file=None,
        model_name="1.7B",
        device_index=0  # gpu index
):
    from pathlib import Path
    import torch, json
    torch.set_num_threads(1)
    from qwen_asr import Qwen3ASRModel
    if is_cuda:
        device_map = f'cuda:{device_index}'
        dtype = torch.float16
    else:
        device_map = 'cpu'
        dtype = torch.float32
    model = None
    try:
        _write_log(logs_file, json.dumps({"type": "logs", "text": f'Load Qwen3ASR on {device_map}'}))
        model = Qwen3ASRModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-ASR-{model_name}",
            dtype=dtype,
            device_map=device_map,
            attn_implementation=None,
            max_inference_batch_size=8,
            # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=2048,  # Maximum number of tokens to generate. Set a larger value for long audio input.
        )
        srts = json.loads(Path(cut_audio_list).read_text(encoding='utf-8'))

        srts_chunk = [srts[i:i + 8] for i in range(0, len(srts), 8)]
        for i, it_list in enumerate(srts_chunk):
            results = model.transcribe(
                audio=[it['file'] for it in it_list],
                language=[None for it in it_list],  # can also be set to None for automatic language detection
                return_time_stamps=False,
            )
            for j, it in enumerate(it_list):
                it['text'] = results[j].text
            srts_chunk[i] = it_list
            _write_log(logs_file, json.dumps({"type": "subtitle", "text": "\n".join([it['text'] for it in it_list])}))

        return srts, None
    except Exception as e:
        return False, str(e)
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if model:
                del model
            import gc
            gc.collect()
        except Exception:
            pass


def funasr_mlt(
        cut_audio_list=None,
        detect_language=None,
        model_name=None,
        logs_file=None,
        is_cuda=False,
        audio_file=None,
        jianfan=False,
        max_speakers=-1,
        cache_folder=None,
        device_index=0  # gpu index
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    from funasr import AutoModel
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    msg = f'Load {model_name}'
    _write_log(logs_file, json.dumps({"type": "logs", "text": f'{msg}'}))

    model = None
    device = f"cuda:{device_index}" if is_cuda else gpus.mps_or_cpu()
    try:
        if cut_audio_list and isinstance(cut_audio_list, str):
            cut_audio_list = json.loads(Path(cut_audio_list).read_text(encoding='utf-8'))

        srts = cut_audio_list
        if model_name == 'iic/SenseVoiceSmall':
            model = pipeline(
                task=Tasks.auto_speech_recognition,
                model='iic/SenseVoiceSmall',
                # model_revision="master",
                disable_update=True,
                disable_progress_bar=True,
                disable_log=True,
                device=device
            )

            res = model([it['file'] for it in cut_audio_list], batch_size=4, disable_pbar=True)
        else:
            model = AutoModel(
                model=model_name,
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                device=device,
                local_dir=ROOT_DIR + "/models",
                disable_update=True,
                disable_progress_bar=True,
                disable_log=True,
                trust_remote_code=True,
                remote_code=f"{ROOT_DIR}/videotrans/codes/model.py",
                hub='ms',
            )

            # vad
            msg = "Recognition starting"
            _write_log(logs_file, json.dumps({"type": "logs", "text": f'{msg}'}))
            num = 0

            def _show_process(ex, dx):
                nonlocal num
                num += 1
                _write_log(logs_file, json.dumps({"type": "logs", "text": f'STT {num}'}))

            res = model.generate(
                input=[it['file'] for it in srts],
                language=detect_language[:2],  # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,
                batch_size=1,
                progress_callback=_show_process,
                disable_pbar=True
            )
        for i, it in enumerate(res):
            text = _remove_unwanted_characters(it['text'])
            srts[i]['text'] = text
            _write_log(logs_file, json.dumps({"type": "subtitles", "text": f'[{i}] {text}\n'}))
        return srts, None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'Speech recognition failed:{model_name=},{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if model:
                del model
            import gc
            gc.collect()
        except Exception:
            pass
