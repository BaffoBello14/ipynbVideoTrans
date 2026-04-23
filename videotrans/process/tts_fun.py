# Speech recognition, new process execution
# Return tuple
# Failure: If the first value is False, it is a failure, and the second value stores the reason for the failure.
# Success, the first value has the required return value, returns True if not needed, and the second value is None
from videotrans.configure.config import logger,ROOT_DIR

def _write_log(file, msg):
    from pathlib import Path
    try:
        Path(file).write_text(msg, encoding='utf-8')
    except Exception as e:
        logger.exception(f'Error writing new process log', exc_info=True)


def qwen3tts_fun(
        queue_tts_file=None,# The dubbing data is stored in the json file and is obtained according to the file path.
        language='Auto',#language
        logs_file=None,
        defaulelang="en",
        is_cuda=False,
        prompt=None,
        model_name='1.7B',
        roledict=None,
        device_index=0 # gpu index
):
    import re, os, traceback, json, time
    import shutil
    from pathlib import Path
    from videotrans.util import tools

    import torch
    torch.set_num_threads(1)
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    
    CUSTOM_VOICE= {"Vivian", "Serena", "Uncle_fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_anna", "Sohee"}

    
    queue_tts=json.loads(Path(queue_tts_file).read_text(encoding='utf-8'))
    
    atten=None
    if is_cuda:
        device_map = f'cuda:{device_index}'
        dtype=torch.float16
        try:
            import flash_attn
        except ImportError:
            pass
        else:
            atten='flash_attention_2'
    else:
        device_map = 'cpu'
        dtype=torch.float32
    
    BASE_OBJ=None
    CUSTOM_OBJ=None
    

    all_roles={ r.get('role') for r in queue_tts}
    if all_roles & CUSTOM_VOICE:
        # There is a custom tone
        CUSTOM_OBJ=Qwen3TTSModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-CustomVoice",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )
    if "clone" in all_roles or all_roles-CUSTOM_VOICE:
        # There is a cloned sound
        BASE_OBJ=Qwen3TTSModel.from_pretrained(
            f"{ROOT_DIR}/models/models--Qwen--Qwen3-TTS-12Hz-{model_name}-Base",
            device_map=device_map,
            dtype=dtype,
            attn_implementation=atten
        )

    try:

        _len=len(queue_tts)
        for i,it in enumerate(queue_tts):
            text=it.get('text')
            if not text:
                continue
            role=it.get('role')
            filename=it.get('filename','')+"-qwen3tts.wav"
            _write_log(logs_file, json.dumps({"type": "logs", "text": f'{i+1}/{_len} {role}'}))
            
            if role in CUSTOM_VOICE and CUSTOM_OBJ:
                wavs, sr = CUSTOM_OBJ.generate_custom_voice(
                    text=text,
                    language=language, # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
                    speaker=role,
                    instruct=prompt
                )
                sf.write(filename, wavs[0], sr)
                continue
            if not BASE_OBJ:
                continue
            if role == 'clone':
                wavfile = it.get('ref_wav', '')
                ref_text = it.get('ref_text', '')
            else:
                # Use the audio in the f5-tts folder
                wavfile = f'{ROOT_DIR}/f5-tts/{role}'
                ref_text = roledict.get(role) if roledict else None
            if not wavfile or not Path(wavfile).is_file():
                # still does not exist, no reference audio is available
                msg = f"Reference audio does not exist and cannot be cloned:{role=},{wavfile=}"
                _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
                continue
            kw={
                "text":text,
                "language":language,
                "ref_audio":wavfile,
            }
            if not ref_text:
                kw['x_vector_only_mode']=True
            else:
                kw['ref_text']=ref_text
            print(f'{kw=}')
            wavs, sr = BASE_OBJ.generate_voice_clone(**kw)
            sf.write(filename, wavs[0], sr)
        return True,None
    except Exception:
        msg = traceback.format_exc()
        logger.exception(f'Qwen3-TTS dubbing failed:{msg}', exc_info=True)
        return False, msg
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if CUSTOM_OBJ:
                del CUSTOM_OBJ
            if BASE_OBJ:
                del BASE_OBJ
            import gc
            gc.collect()
        except Exception:
            pass
