"""
help_misc.py – headless version (no PySide6 / Qt).
GUI-only functions (show_popup, show_error, show_glossary_editor,
hide_show_element) are replaced with no-op / print stubs.
All backend-required functions are kept intact.
"""
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import tqdm

from videotrans.configure import config
from videotrans.configure.config import (
    tr, params, settings, app_cfg, logger, ROOT_DIR, TEMP_DIR, defaulelang
)


# ── tqdm helper ──────────────────────────────────────────────────────────────

def create_tqdm_class(callback):
    """Return a tqdm subclass that also calls *callback* with progress string."""
    class CallbackTqdm(tqdm.tqdm):
        def display(self, msg=None, pos=None):
            super().display(msg, pos)
            _str = str(self).split('%')
            if callback and len(_str) > 0:
                callback(_str[0] + '%')
    return CallbackTqdm


# ── GUI stubs (headless replacements) ────────────────────────────────────────

def show_popup(title, text):
    """No-op in headless mode – just print the message and return True."""
    print(f"[POPUP] {title}: {text}")
    return True


def show_error(tb_str):
    """No-op in headless mode – print the traceback."""
    print(f"[ERROR] {tb_str[:500]}")


def hide_show_element(wrap_layout, show_status):
    """No-op in headless mode."""
    pass


def show_glossary_editor(parent):
    """No-op in headless mode."""
    pass


# ── URL / browser ─────────────────────────────────────────────────────────────

def open_url(url: str = None):
    import webbrowser
    title_url_dict = {
        'bbs':         "https://bbs.pyvideotrans.com",
        'ffmpeg':      "https://www.ffmpeg.org/download.html",
        'git':         "https://github.com/jianchang512/pyvideotrans",
        'issue':       "https://github.com/jianchang512/pyvideotrans/issues",
        'hfmirrorcom': "https://pyvideotrans.com/819",
        'models':      "https://github.com/jianchang512/stt/releases/tag/0.0",
        'stt':         "https://github.com/jianchang512/stt/",
        'gtrans':      "https://pyvideotrans.com/aiocr",
        'cuda':        "https://pyvideotrans.com/gpu.html",
        'website':     "https://pyvideotrans.com",
        'help':        "https://pyvideotrans.com",
        'xinshou':     "https://pyvideotrans.com/getstart",
        'about':       "https://pyvideotrans.com/about",
        'download':    "https://github.com/jianchang512/pyvideotrans/releases",
    }
    if url and url.startswith("http"):
        return webbrowser.open_new_tab(url)
    if url and url in title_url_dict:
        return webbrowser.open_new_tab(title_url_dict[url])


# ── File helpers ──────────────────────────────────────────────────────────────

def vail_file(file=None):
    if not file:
        return False
    p = Path(file)
    if not p.exists() or not p.is_file():
        return False
    if p.stat().st_size == 0:
        return False
    return True


def is_novoice_mp4(novoice_mp4, noextname, uuid=None):
    """Wait until the silent-video extraction is complete."""
    t = 0
    if noextname not in app_cfg.queue_novice and vail_file(novoice_mp4):
        return True
    if noextname in app_cfg.queue_novice and app_cfg.queue_novice[noextname] == 'end':
        return True
    last_size = 0
    while True:
        if app_cfg.current_status != 'ing' or app_cfg.exit_soft:
            return False
        if vail_file(novoice_mp4):
            current_size = os.path.getsize(novoice_mp4)
            if 0 < last_size == current_size and t > 1200:
                return True
            last_size = current_size

        if noextname not in app_cfg.queue_novice:
            raise RuntimeError(f"{noextname} split no voice video error-1")
        if app_cfg.queue_novice[noextname].startswith('error:'):
            raise RuntimeError(f"{noextname} split no voice {app_cfg.queue_novice[noextname]}")
        if app_cfg.queue_novice[noextname] == 'ing':
            size = f'{round(last_size / 1024 / 1024, 2)}MB' if last_size > 0 else ""
            set_process(text=f"{noextname} {tr('spilt audio and video')} {size}", uuid=uuid)
            time.sleep(1)
            t += 1
            continue
        return True


# ── Hashing ───────────────────────────────────────────────────────────────────

def get_md5(input_string: str):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    return md5.hexdigest()


# ── System ────────────────────────────────────────────────────────────────────

def shutdown_system():
    system = platform.system()
    if system == "Windows":
        subprocess.call("shutdown /s /t 1")
    elif system == "Linux":
        subprocess.call("poweroff")
    elif system == "Darwin":
        subprocess.call("sudo shutdown -h now", shell=True)
    else:
        print(f"Unsupported system: {system}")


def pygameaudio(filepath=None):
    try:
        import soundfile as sf
        import sounddevice as sd
        data, fs = sf.read(filepath)
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(e)


def read_last_n_lines(filename, n=100):
    if not Path(filename).exists():
        return []
    from collections import deque
    try:
        with open(filename, 'r', encoding='utf-8') as fh:
            return list(deque(fh, maxlen=n))
    except Exception:
        return []


# ── Prompt helpers ────────────────────────────────────────────────────────────

def get_prompt(ainame, aisendsrt=True):
    prompt_file = get_prompt_file(ainame=ainame, aisendsrt=aisendsrt)
    content = Path(prompt_file).read_text(encoding='utf-8', errors='ignore')
    glossary = ''
    glossary_path = Path(ROOT_DIR) / 'videotrans' / 'glossary.txt'
    if glossary_path.exists():
        glossary = glossary_path.read_text(encoding='utf-8', errors='ignore').strip()
    if glossary:
        glossary = "\n".join(
            ["|" + it.replace("=", '|') + "|" for it in glossary.split('\n')]
        )
        glossary_prompt = (
            "\n\n# Glossary of terms\nTranslations are made strictly according to the "
            "following glossary. If a term appears in a sentence, the corresponding "
            "translation must be used, not a free translation:\n"
            "| Glossary | Translation |\n| --------- | ----- |\n"
        )
        content = content.replace('# ACTUAL TASK', f"{glossary_prompt}{glossary}\n\n# ACTUAL TASK")
    return content


def get_prompt_file(ainame, aisendsrt=True):
    prompt_path = f'{ROOT_DIR}/videotrans/'
    prompt_name = f'{ainame}.txt'
    if aisendsrt:
        prompt_path += 'prompts/srt/'
    else:
        prompt_path += 'prompts/text/'
    return f'{prompt_path}{prompt_name}'


def qwenmt_glossary():
    glossary_path = Path(ROOT_DIR) / 'videotrans' / 'glossary.txt'
    if glossary_path.exists():
        glossary = glossary_path.read_text(encoding='utf-8', errors='ignore').strip()
        if glossary:
            term = []
            for it in glossary.split('\n'):
                tmp = it.split("=")
                if len(tmp) == 2:
                    term.append({"source": tmp[0], "target": tmp[1]})
            return term if term else None
    return None


# ── Logging / process feedback ────────────────────────────────────────────────

def set_process(*, text="", type="logs", uuid=None):
    if app_cfg.exit_soft:
        return
    if uuid and uuid in app_cfg.stoped_uuid_set:
        return
    try:
        if text:
            text = text.replace('\\n', ' ')
        if type == 'logs':
            text = text[:150]
        if app_cfg.exec_mode == 'cli':
            print(text)
            return
        log = {"text": text, "type": type, "uuid": uuid}
        if uuid:
            config.push_queue(uuid, log)
        else:
            app_cfg.global_msg.append(log)
    except Exception as e:
        logger.exception(f'set_process: {e}', exc_info=True)


# ── Proxy helpers ─────────────────────────────────────────────────────────────

def set_proxy(set_val=''):
    if set_val == 'del':
        app_cfg.proxy = ''
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        return

    if set_val:
        set_val = set_val.lower()
        if not set_val.startswith("http") and not set_val.startswith('sock'):
            set_val = f"http://{set_val}"
        app_cfg.proxy = set_val
        os.environ['HTTP_PROXY'] = set_val
        os.environ['HTTPS_PROXY'] = set_val
        return set_val

    http_proxy = app_cfg.proxy or os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
    if http_proxy:
        http_proxy = http_proxy.lower()
        if not http_proxy.startswith("http") and not http_proxy.startswith('sock'):
            http_proxy = f"http://{http_proxy}"
        return http_proxy

    if sys.platform != 'win32':
        return None
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
        ) as key:
            proxy_enable, _ = winreg.QueryValueEx(key, 'ProxyEnable')
            proxy_server, _ = winreg.QueryValueEx(key, 'ProxyServer')
            if proxy_enable == 1 and proxy_server:
                proxy_server = proxy_server.lower()
                if not proxy_server.startswith("http") and not proxy_server.startswith('sock'):
                    proxy_server = "http://" + proxy_server
                return proxy_server
    except Exception:
        pass
    return None
