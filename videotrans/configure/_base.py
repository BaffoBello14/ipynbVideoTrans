import base64
import json
import os,time,traceback
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from videotrans.configure.config import tr, params, settings, app_cfg, logger, IS_FROZEN
from videotrans.util import tools, contants
from videotrans.process.signelobj import GlobalProcessManager
from concurrent.futures.process import BrokenProcessPool

from videotrans.util.gpus import get_cudaX


@dataclass
class BaseCon:
    # Unique uuid for each task
    uuid: Optional[str] = field(default=None, init=False)
    # Used for other objects that require direct proxy strings
    proxy_str: str = ''
    # Host that does not require a proxy
    no_proxy: str = ''

    def __post_init__(self):
        self.no_proxy=contants.no_proxy
        # Get proxy
        self.proxy_str = self._set_proxy(type='set')


    # All window and task information are exchanged through the queue
    def _signal(self, **kwargs):
        if 'uuid' not in kwargs:
            kwargs['uuid'] = self.uuid
        if not app_cfg.exit_soft:
            tools.set_process(**kwargs)
    
    def _process_callback(self,data):
        if isinstance(data,str):
            return self._signal(text=tr('Downloading please wait')+data)
        if not isinstance(data,dict):
            return
        msg_type = data.get("type")
        percent = data.get("percent")
        filename = data.get("filename")

        if msg_type == "file":
            self._signal(text=f"{tr('Downloading please wait')} {filename} {percent:.2f}%")
        else:
            current_file_idx = data.get("current")
            total_files = data.get("total")

            self._signal(text=f"{tr('Downloading please wait')} {current_file_idx}/{total_files} files")
        
    # Set and get proxy
    def _set_proxy(self, type='set'):
        if type == 'del':
            os.environ['bak_proxy'] = app_cfg.proxy or os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
            app_cfg.proxy = ''
            os.environ.pop('HTTPS_PROXY',None)
            os.environ.pop('HTTP_PROXY',None)
            return None

        if type == 'set':
            raw_proxy = app_cfg.proxy or os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if raw_proxy:
                app_cfg.proxy=raw_proxy
                return raw_proxy
            if not raw_proxy:
                proxy = tools.set_proxy() or os.environ.get('bak_proxy')
                if proxy:
                    os.environ['HTTP_PROXY'] = proxy
                    os.environ['HTTPS_PROXY'] = proxy
                app_cfg.proxy=proxy
                return proxy
        return None

    # Call faster-xxl.exe
    def _external_cmd_with_wrapper(self, cmd_list=None):
        if not cmd_list:
            raise ValueError("cmd_list is None")
        try:
            subprocess.run(cmd_list, capture_output=True, text=True, check=True, encoding='utf-8', creationflags=0,
                           cwd=os.path.dirname(cmd_list[0]))
        except subprocess.CalledProcessError as e:
            if os.name == 'nt' and IS_FROZEN:
                raise RuntimeError(tr('Currently Faster-Whisper-XXL cannot be used in the packaged version. Please deploy the source code or use Faster-Whisper-XXL transcription separately.'))
            raise RuntimeError(e.stderr)

    # Convert speech into wav audio after synthesis
    def convert_to_wav(self, mp3_file_path: str, output_wav_file_path: str, extra=None):
        if app_cfg.exit_soft or not tools.vail_file(mp3_file_path):
            return
        cmd = [
            "-y",
            "-i",
            mp3_file_path,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le"
        ]
        if extra:
            cmd += extra
        cmd += [
            output_wav_file_path
        ]
        try:
            tools.runffmpeg(cmd, force_cpu=True)
            if settings.get('remove_dubb_silence',True):
                tools.remove_silence_wav(output_wav_file_path)
        except Exception:
            pass
        return True


    # Determine whether it is an intranet address
    def _get_internal_host(self, url: str):
        from urllib.parse import urlparse
        import ipaddress
        'Check whether the host of the URL is an intranet address.\n    \n        - If it is an intranet address (private, loopback, unspecified), its "host:port" string is returned.\n        - If it is \'localhost\', the "localhost:port" string is also returned.\n        - If it is not an intranet address or the URL is invalid, False is returned.\n    \n        Args:\n            url: The URL string to be checked.\n    \n        Returns:\n            str | bool: If it is an internal network address, return its network location (netloc), otherwise return False.'
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname

            # If there is no hostname in the URL (such as "path/only"), return False directly
            if not hostname:
                return False

            # 1. Prioritize 'localhost' string processing
            if hostname.lower() == 'localhost':
                return parsed_url.netloc  # Return 'localhost:port'

            # 2. Try to resolve hostname to IP address
            ip_addr = ipaddress.ip_address(hostname)

            # 3. Determine IP address type
            # is_private: 10/8, 172.16/12, 192.168/16
            # is_loopback: 127/8
            # is_unspecified: 0.0.0.0
            if ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_unspecified:
                return parsed_url.netloc  # Return 'ip:port'

        except ValueError:
            # If hostname is a domain name (such as www.google.com) rather than an IP,
            # ipaddress.ip_address(hostname) will throw ValueError.
            # In this case we think it is not an intranet address.
            return False

        # If it is a public IP (such as 8.8.8.8), it does not meet any conditions and will eventually return False
        return False

    # Determine if api_url is an intranet address, add host to no_proxy to avoid requests using proxy access
    def _add_internal_host_noproxy(self, api_url=''):
        host = self._get_internal_host(api_url)
        if host is not False:
            self.no_proxy += f',{host}'
            os.environ['no_proxy'] = self.no_proxy

    def _base64_to_audio(self, encoded_str: str, output_path: str) -> None:
        if not encoded_str:
            raise ValueError("Base64 encoded string is empty.")
        # If the data prefix exists, save it in the conversion format according to the audio format contained in the prefix.
        if encoded_str.startswith('data:audio/'):
            output_ext = Path(output_path).suffix.lower()[1:]
            mime_type, encoded_str = encoded_str.split(',', 1)  # Extract Base64 data part
            # Extract audio format (e.g. 'mp3', 'wav')
            audio_format = mime_type.split('/')[1].split(';')[0].lower()
            support_format = {
                "mpeg": "mp3",
                "wav": "wav",
                "ogg": "ogg",
                "aac": "aac"
            }
            base64data_ext = support_format.get(audio_format, "")
            if base64data_ext and base64data_ext != output_ext:
                # Different formats need to be converted.
                # Decode base64 encoded string into bytes
                wav_bytes = base64.b64decode(encoded_str)
                #Write decoded bytes to file
                with open(output_path + f'.{base64data_ext}', "wb") as wav_file:
                    wav_file.write(wav_bytes)

                tools.runffmpeg([
                    "-y", "-i", output_path + f'.{base64data_ext}', "-b:a", "128k", output_path
                ])
                return
        # Decode base64 encoded string into bytes
        wav_bytes = base64.b64decode(encoded_str)
        #Write decoded bytes to file
        with open(output_path, "wb") as wav_file:
            wav_file.write(wav_bytes)

    def _audio_to_base64(self, file_path: str):
        if not file_path or not Path(file_path).exists():
            return None
        with open(file_path, "rb") as wav_file:
            wav_content = wav_file.read()
            base64_encoded = base64.b64encode(wav_content)
            return base64_encoded.decode("utf-8")

    def _signal_of_process(self, logs_file):
        last_mtime = 0
        while 1:
            _p = Path(logs_file)
            # Deleted
            if last_mtime>0 and not _p.exists():
                return
            try:
                if not _p.exists():
                    time.sleep(1)
                    continue
                # Get the last modification time of the log file
                _mtime = _p.stat().st_mtime
                if _mtime == last_mtime:
                    # Not modified since last time
                    time.sleep(1)
                    continue
                last_mtime=_mtime
                _content=_p.read_text(encoding='utf-8')
                if not _content:
                    time.sleep(1)
                    continue
                _tmp = json.loads(_content)
                if _tmp.get('type', '') == 'error':
                    return
                self._signal(text=_tmp.get('text',''), type=_tmp.get('type', 'logs'))
            except Exception as e:
                # There may be an error in reading the log file, which can be ignored.
                logger.warning(f'An error occurred while reading the inter-process log file, which can be ignored:{e}')
            time.sleep(1)

    # Use new process to perform tasks
    def _new_process(self,callback=None,title="",is_cuda=False,kwargs=None):
        _st = time.time()
        self._signal(text=f'[{title}] starting...')
        logger.debug(f'[New process execution task]:{title}')
        # Submit the task and explicitly pass in the parameters to ensure that the child process gets the correct parameters.
        logs_file=kwargs.get('logs_file')
        device_index=0
        try:
            if logs_file:
                Path(logs_file).touch()
                threading.Thread(target=self._signal_of_process,args=(logs_file,),daemon=True).start()
            # Judge again whether cuda is valid to prevent pre-acquisition failure
            if is_cuda:
                import torch
                if not torch.cuda.is_available():
                    is_cuda=False

            # If using gpu, get available device_index
            if is_cuda:
                
                #Multiple graphics card mode enabled
                if settings.get('multi_gpus'):
                    device_index=get_cudaX()
                if device_index==-1:
                    is_cuda=False
                    kwargs['is_cuda']=False
                    logger.error(f'CUDA enabled but no available graphics card detected, forcing CPU use')
                kwargs['device_index']=max(device_index,0)
            future = GlobalProcessManager.submit_task_cpu(
                callback, 
                **kwargs
            ) if not is_cuda else GlobalProcessManager.submit_task_gpu(
                callback,
                **kwargs
            )
            _rs = future.result()
            if isinstance(_rs,tuple) and len(_rs)==2:
                data,err=_rs
                if data is False:
                    raise RuntimeError(err)
            else:
                data=_rs
            self._signal(text=f'[{title}] end: {int(time.time() - _st)}s')
            return data
        except BrokenProcessPool as e:
            err=traceback.format_exc()
            _model=''
            _cuda=''
            if kwargs.get('model_name'):
                _model=' Model:'+kwargs.get('model_name')
            if is_cuda and device_index>-1:
                _cuda=f" GPU{device_index} \n{tr('may be insufficient memory')}"
            logger.exception(f'{_model}{_cuda}',exc_info=True)
            raise RuntimeError(f'{_model}{_cuda}\n{err}')
        except Exception as e:
            msg=traceback.format_exc()
            logger.exception(f'new process:{msg}',exc_info=True)
            raise
        finally:
            try:
                Path(logs_file).unlink(missing_ok=True)
            except:
                pass

