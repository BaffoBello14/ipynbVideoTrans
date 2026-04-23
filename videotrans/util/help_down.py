# Download the model, first test huggingface.co connectivity, if not available, fall back to mirror hf-mirror.com
import tempfile
import time
from pathlib import Path
import shutil, os, requests
import zipfile
from videotrans.configure.config import ROOT_DIR, tr, app_cfg, settings, params, TEMP_DIR, logger, defaulelang, HOME_DIR
from videotrans.util.contants import FASTER_MODELS_DICT
from .help_misc import create_tqdm_class
from urllib.parse import urlparse

'Parse the URL to get the pure file name (remove ?query)'


def get_filename_from_url(url) -> str:
    parsed = urlparse(url)
    return os.path.basename(parsed.path)


# Used to determine whether a file of the specified type exists in a directory. If it exists, it is deemed to exist.
def file_exists(dirname, glob_patter='*.bin') -> bool:
    if isinstance(glob_patter, str):
        glob_patter = [glob_patter]
    for pat in glob_patter:
        for it in Path(dirname).glob(pat):
            return True
    return False


def is_connect_hf():
    try:
        requests.head('https://huggingface.co', timeout=3)
    except Exception as e:
        print(f'Unable to connect to huggingface.co, use mirror instead: hf-mirror.com\n{e}')
        return False
    else:
        print('You can use huggingface.co')
        return True


# Download the complete model from huggingface.co, first download locally, if failed, download online
def check_and_down_hf(model_id, repo_id, local_dir, callback=None) -> bool:
    try:
        if model_id in FASTER_MODELS_DICT and (defaulelang == 'zh' or is_connect_hf() is False):
            if model_id == 'turbo':
                model_id = 'large-v3-turbo'
            elif model_id == 'distil-large-v3.5-ct2':
                model_id = 'distil-large-v3.5'
            elif model_id == 'large':
                model_id = 'large-v3'

            URL_PRE = f'https://modelscope.cn/models/himyworld/fasterwhisper/resolve/master/{model_id}/'

            if model_id in ['large-v3', 'large-v3-turbo', 'distil-large-v3', 'distil-large-v3.5']:
                all_urls = ["vocabulary.json", "preprocessor_config.json"]
            else:
                all_urls = ["vocabulary.txt"]
            all_urls += ["config.json", "tokenizer.json", "model.bin", ]
            return down_file_from_ms(local_dir, urls=[f'{URL_PRE}{u}' for u in all_urls], callback=callback)
        import huggingface_hub
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            huggingface_hub.snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                etag_timeout=5,
                local_files_only=True
            )
        except LocalEntryNotFoundError:
            Path(local_dir).mkdir(exist_ok=True, parents=True)
            MyTqdmClass = None
            if callback:
                MyTqdmClass = create_tqdm_class(callback)

            if is_connect_hf() is False:
                print(f'Unable to connect to huggingface.co, use mirror instead: hf-mirror.com,{model_id=}')
                endpoint = 'https://hf-mirror.com'
            else:
                print('You can use huggingface.co')
                endpoint = 'https://huggingface.co'

            huggingface_hub.snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                endpoint=endpoint,
                etag_timeout=5,
                tqdm_class=MyTqdmClass,
                local_files_only=False,
                max_workers=1,
                ignore_patterns=["*.msgpack", "*.h5", ".git*", "*.md"]
            )
        else:
            return True
    except Exception as e:
        msg = f'Failed to download the model, you can open the following URL and download all files to\n{local_dir}Within the folder\n' if defaulelang == 'zh' else f'The model download failed. You can try opening the following URL and downloading all files to the {local_dir} folder.'
        raise RuntimeError(f'{msg}\n[https://huggingface.co/{repo_id}/tree/main]\n{e}')
    else:
        junk_paths = [
            ".cache",
            "blobs",
            "refs",
            "snapshots",
            ".no_exist"
        ]

        for junk in junk_paths:
            full_path = Path(local_dir) / junk
            if full_path.exists():
                try:
                    if full_path.is_dir():
                        shutil.rmtree(full_path)
                    else:
                        os.remove(full_path)
                    print(f"clear cache: {junk}")
                except Exception as e:
                    print(f"{junk} {e}")
    return True


# Download a single file from huggingface
def down_file_from_hf(local_dir, urls=None, callback=None) -> bool:
    if is_connect_hf() is False:
        print(f'Unable to connect to huggingface.co, use mirror instead: hf-mirror.com')
        endpoint = 'https://hf-mirror.com'
    else:
        print('You can use huggingface.co')
        endpoint = 'https://huggingface.co'
    for index, url in enumerate(urls):
        try:
            if not url.startswith('https://'):
                url = f'{endpoint}{url}'
            filename = get_filename_from_url(url)
            with requests.get(f'{url}', stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                total_length = response.headers.get('content-length')

                dest_file_obj = open(f'{local_dir}/{filename}', 'wb')

                try:
                    if total_length is None:
                        dest_file_obj.write(response.content)
                    else:
                        total_length = max(int(total_length), 1)
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                dest_file_obj.write(chunk)
                                downloaded += len(chunk)

                                # Calculate progress
                                #Single file progress 0-100
                                file_percent = (downloaded / total_length)
                                if callback:
                                    callback(f'{filename}:{file_percent * 100:.2f}%')
                finally:
                    dest_file_obj.close()  # Close the entity file handle
        except Exception as e:
            raise RuntimeError(
                tr("downloading all files", local_dir) + f'\n[https://huggingface.co{url}]\n\n{e}')
    return True


# Download a single file from modelscope.cn
def down_file_from_ms(local_dir, urls=None, callback=None) -> bool:
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(urls):
        filename = get_filename_from_url(url)
        file_abso_path = f'{local_dir}/{filename}'
        if Path(file_abso_path).exists():
            continue
        try:
            with requests.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                total_length = response.headers.get('content-length')
                dest_file_obj = open(file_abso_path, 'wb')
                try:
                    if total_length is None:
                        dest_file_obj.write(response.content)
                    else:
                        total_length = max(int(total_length), 1)
                        downloaded = 0
                        last_send = time.time()
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                dest_file_obj.write(chunk)
                                downloaded += len(chunk)
                                file_percent = min((downloaded / total_length) * 100, 100)
                                if time.time() - last_send > 3:
                                    last_send = time.time()
                                if callback:
                                    callback(
                                        {"type": "file", "percent": file_percent, "filename": f"{filename}",
                                         "current": index + 1, "total": 5})

                finally:
                    dest_file_obj.close()
        except Exception as e:
            raise
    return True


# Download the zip and extract it to the specified directory
def down_zip(local_dir, zip_url, callback=None) -> bool:
    try:
        filename = get_filename_from_url(zip_url)
        with requests.get(zip_url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            total_length = response.headers.get('content-length')

            #Determine the writing target: if it needs to be decompressed, write it to a temporary file; otherwise, write it directly to the target file
            dest_file_obj = tempfile.TemporaryFile()  #Memory/temporary disk, automatically deleted

            if total_length is None:
                dest_file_obj.write(response.content)
            else:
                total_length = max(int(total_length), 1)
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        dest_file_obj.write(chunk)
                        downloaded += len(chunk)

                        # Calculate progress
                        #Single file progress 0-100
                        file_percent = min(99.0, downloaded * 100 / total_length)
                        if callback:
                            callback(f'{filename} {file_percent:.2f}%')

            if callback:
                callback(f'Extracting zip')
            dest_file_obj.seek(0)  # Return to file header
            with zipfile.ZipFile(dest_file_obj) as zf:
                zf.extractall(path=local_dir)
            if callback:
                callback('extract end')
            dest_file_obj.close()
    except Exception as e:
        msg = tr('model is missing. Please download it', local_dir)
        raise RuntimeError(f"{msg}\n[{zip_url}]\n{e}")
    return True


# Download the complete model from modelscope.cn
def check_and_down_ms(model_id, callback=None, local_dir=None) -> bool:
    from modelscope.hub.callback import ProgressCallback, TqdmCallback
    from modelscope.hub.snapshot_download import snapshot_download
    class Pro(TqdmCallback):
        def __init__(self, *args):
            super().__init__(*args)

        def update(self, size):
            super().update(size)
            try:
                _str = str(self.progress).split('%')[0] + '%'
                callback(_str)
            except:
                pass
            '''
            if callback:
                per=f'{size*100/self.file_size:.2f}%' if self.file_size>0 else ''
                callback(f'{self.filename} {per}')
            else:
                print(f'{self.filename=},{self.file_size=},{size=}')
            '''

    try:
        try:
            # If local loading fails, download online
            snapshot_download(model_id=model_id, local_files_only=True, progress_callbacks=[Pro], local_dir=local_dir)
            if callback:
                callback(f'{model_id} exists')
        except ValueError  as e:
            if callback:
                callback(f'{model_id}')
            snapshot_download(model_id=model_id, progress_callbacks=[Pro], local_dir=local_dir, max_workers=1)
        else:
            return True
    except Exception as e:
        local_dir = f'{ROOT_DIR}/models/models/{model_id}/' if not local_dir else local_dir
        msg = f'Failed to download the model, you can open the following URL and download all files to\n{local_dir}Within the folder\n' if defaulelang == 'zh' else f'The model download failed. You can try opening the following URL and downloading all files to the {local_dir} folder.'
        raise RuntimeError(f'{msg}\n[https://modelscope.cn/models/{model_id}/tree/main]\n{e}')
    return True
