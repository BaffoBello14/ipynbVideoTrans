import asyncio
import copy
import inspect
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from tenacity import RetryError

# Jupyter / Colab kernels already run an asyncio event loop.
# nest_asyncio patches it so that run_until_complete() can be called
# from inside a running loop (e.g. from a notebook cell).
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # outside notebooks nest_asyncio is not needed
from videotrans.configure._base import BaseCon
from videotrans.configure._except import StopRetry
from videotrans.configure.config import tr, settings, params, app_cfg, logger, TEMP_DIR

from videotrans.util import tools

'edge-tts async asynchronous tasks in the current thread\nMulti-threaded execution of other channels\nself.error may be an exception object or a string\n\nrun->exec->[local_mutli]->item_task\n\n'


@dataclass
class BaseTTS(BaseCon):
    # dubbing channel
    tts_type: int = 0
    # Store subtitle information queue
    queue_tts: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    #queue_tts quantity
    len: int = field(init=False)
    # language code
    language: Optional[str] = None
    # unique uid
    uuid: Optional[str] = None
    # Whether to play immediately
    play: bool = False
    # Whether to test
    is_test: bool = False

    # Volume, speed of sound, pitch, default edge-tts format
    volume: str = field(default='+0%', init=False)
    rate: str = field(default='+0%', init=False)
    pitch: str = field(default='+0Hz', init=False)

    # Is it completed?
    has_done: int = field(default=0, init=False)

    # Pause time after each task
    wait_sec: float = float(settings.get('dubbing_wait', 0))
    # Number of concurrent threads
    dub_nums: int = int(float(settings.get('dubbing_thread', 1)))
    # Store message
    error: Optional[Any] = None
    # Dubbing api address
    api_url: str = field(default='', init=False)
    # Enable CUDA, only qwen3-tts-local game oh ah
    is_cuda:bool=False

    def __post_init__(self):
        super().__post_init__()
        if not self.queue_tts:
            raise RuntimeError(tr("No subtitles required"))

        self.queue_tts = copy.deepcopy(self.queue_tts)
        self.len = len(self.queue_tts)
        self._cleantts()

    def _cleantts(self):
        normalizer = None
        if settings.get('normal_text'):
            if self.language[:2] == 'zh':
                from videotrans.util.cn_tn import TextNorm
                normalizer = TextNorm(to_banjiao=True)
            elif self.language[:2] == 'en':
                from videotrans.util.en_tn import EnglishNormalizer
                normalizer = EnglishNormalizer()
        
        for i, it in enumerate(self.queue_tts):
            if it['text'].strip() and normalizer:
                try:
                    self.queue_tts[i]['text'] = normalizer(it['text'])
                except:
                    pass

        if "volume" in self.queue_tts[0]:
            self.volume = self.queue_tts[0]['volume']
        if "rate" in self.queue_tts[0]:
            self.rate = self.queue_tts[0]['rate']
        if "pitch" in self.queue_tts[0]:
            self.pitch = self.queue_tts[0]['pitch']

        if re.match(r'^\d+(\.\d+)?%$', self.rate):
            self.rate = f'+{self.rate}'
        if re.match(r'^\d+(\.\d+)?%$', self.volume):
            self.volume = f'+{self.volume}'
        if re.match(r'^\d+(\.\d+)?Hz$', self.pitch, re.I):
            self.pitch = f'+{self.pitch}'

        if not re.match(r'^[+-]\d+(\.\d+)?%$', self.rate):
            self.rate = '+0%'
        if not re.match(r'^[+-]\d+(\.\d+)?%$', self.volume):
            self.volume = '+0%'
        if not re.match(r'^[+-]\d+(\.\d+)?Hz$', self.pitch, re.I):
            self.pitch = '+0Hz'
        self.pitch = self.pitch.replace('%', '')

    # Entry call subclass _exec() and then create a thread pool to call _item_task or implement logic directly in _exec
    # If an exception is caught, throw it directly and send a stop signal when an error occurs.
    # run->exec->_local_mul_thread->item_task
    # run->exec->item_task
    def run(self) -> None:
        if self._exit(): return
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        self._signal(text="")
        _st = time.time()
        if hasattr(self, '_download'):
            self._download()
        try:
            if inspect.iscoroutinefunction(self._exec):
                # Check whether there is already a running loop (Jupyter / Colab kernel).
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                if running_loop is not None:
                    # We are inside a running loop (Jupyter/Colab).
                    # nest_asyncio (applied at import time) allows run_until_complete
                    # to be called re-entrantly on the same loop.
                    running_loop.run_until_complete(self._exec())
                else:
                    # No running loop: create our own, run, then close it.
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._exec())
                    finally:
                        try:
                            tasks = asyncio.all_tasks(loop=loop)
                            for task in tasks:
                                task.cancel()
                            if tasks:
                                loop.run_until_complete(
                                    asyncio.gather(*tasks, return_exceptions=True)
                                )
                            loop.run_until_complete(loop.shutdown_asyncgens())
                        except Exception:
                            pass
                        finally:
                            loop.close()
            else:
                self._exec()
        except RetryError as e:
            raise e.last_attempt.exception()
        except Exception:
            raise
        finally:
            logger.debug(f'[Subtitle Dubbing] Channel{self.tts_type}:Total time spent:{int(time.time() - _st)}s')

        # Play during audition or testing
        if self.play:
            if tools.vail_file(self.queue_tts[0]['filename']):
                tools.pygameaudio(self.queue_tts[0]['filename'])
                return
            if isinstance(self.error, RetryError):
                raise self.error.last_attempt.exception()
            raise self.error if isinstance(self.error, Exception) else RuntimeError(str(self.error))

        # Record the number of successes
        succeed_nums = 0
        for it in self.queue_tts:
            if not it['text'].strip() or tools.vail_file(it['filename']):
                succeed_nums += 1
        # Only if all dubbing fails will it be considered a failure.
        if succeed_nums < 1:
            if app_cfg.exit_soft: return
            
            if isinstance(self.error, Exception):
                raise self.error if not isinstance(self.error,RetryError) else self.error.last_attempt.exception()
            
            raise RuntimeError((tr("Dubbing failed")) + str(self.error))

        self._signal(text=tr("Dubbing succeeded {}，failed {}", succeed_nums, len(self.queue_tts) - succeed_nums))

    # Used for channels other than edge-tts, single or multi-threaded here. Call _item_task
    # exec->_local_mul_thread->item_task
    def _local_mul_thread(self) -> None:
        if self._exit(): return

        # Single subtitle line, no need for multi-threading
        if len(self.queue_tts) == 1 or self.dub_nums == 1:
            for k, item in enumerate(self.queue_tts):
                if not item.get('text'):
                    continue
                try:
                    self._item_task(item,k)
                except StopRetry:
                    # It is a fatal error. There is no need to continue dubbing the next subtitle. For example, the api address is wrong, api_name does not exist, etc.
                    raise
                except RetryError as e:
                    self.error = e.last_attempt.exception()
                except Exception as e:
                    self.error = e
                finally:
                    self._signal(text=f'TTS[{k + 1}/{self.len}]')
                time.sleep(self.wait_sec)
            return

        all_task = []
        pool = ThreadPoolExecutor(max_workers=self.dub_nums)
        try:
            for k, item in enumerate(self.queue_tts):
                if not item.get('text'):
                    continue
                future = pool.submit(self._item_task, item,k)
                all_task.append(future)

            completed_tasks = 0
            for task in as_completed(all_task):
                try:
                    task.result()
                    # It is a fatal error. There is no need to wait for other tasks. All tasks will definitely fail. For example, the api address is wrong, api_name does not exist, etc.
                except StopRetry:
                    # wait=False means that the main thread does not wait for the running thread to end and goes directly down.
                    # This will cancel all tasks that are still queued but have not started running.
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    self.error = e
                finally:
                    completed_tasks += 1
                    self._signal(text=f"TTS: [{completed_tasks}/{self.len}] ...")
        except StopRetry:
            raise
        finally:
            # Ensure that the thread pool is eventually closed
            # Only queued tasks can be canceled and the main thread will no longer wait.
            pool.shutdown(wait=False)

    # Actual business logic subclass implementation Create a thread pool here, or create logic directly when using a single thread
    def _exec(self) -> None:
        pass

    # Each subtitle task is called by the thread pool. data_item is each element in queue_tts.
    def _item_task(self, data_item: Union[Dict, List, None],idx:int=-1) -> Union[bool, None]:
        pass

    # Return blank 16000 sample rate audio
    def _padforaudio(self, duration=1500):
        from pydub import AudioSegment
        silent_segment = AudioSegment.silent(duration=duration)
        silent_segment.set_channels(1).set_frame_rate(16000)
        return silent_segment

    def _exit(self):
        if app_cfg.exit_soft or (self.uuid and self.uuid in app_cfg.stoped_uuid_set):
            return True
        return False
