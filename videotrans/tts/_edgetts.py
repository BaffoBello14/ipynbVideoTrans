import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
import functools
import aiohttp

from videotrans.util import tools
from edge_tts import Communicate
from edge_tts.exceptions import NoAudioReceived
from videotrans.configure.config import tr, params, settings, app_cfg, logger, ROOT_DIR
from videotrans.tts._base import BaseTTS

# edge-tts current limit, may produce a large number of timeouts, 401 and other errors

MAX_CONCURRENT_TASKS = int(settings.get('edgetts_max_concurrent_tasks',10))
RETRY_NUMS = int(settings.get('edgetts_retry_nums',3))+1
RETRY_DELAY = 5
POLL_INTERVAL = 0.1
SIGNAL_TIMEOUT = 2 # The signal sent to the UI interface will time out for 2 seconds to prevent the UI from lagging.
SAVE_TIMEOUT = 30  # edge_tts may limit the current flow and timeout. If it exceeds 30s, it will be considered a failure to prevent unlimited hangs.

    
@dataclass
class EdgeTTS(BaseTTS):
    def __post_init__(self):
        super().__post_init__()
        self._stop_event = asyncio.Event()
        self.ends_counter = 0
        self.lock = asyncio.Lock()
        # By default, the proxy is used according to the settings. If you do not want to use it, create the edgetts-noproxy.txt file in a separate root directory.
        self.useproxy=None if not self.proxy_str or Path(f'{ROOT_DIR}/edgetts-noproxy.txt').exists() else self.proxy_str
        


    async def increment_counter(self):
        async with self.lock:
            self.ends_counter += 1

    async def _create_audio_with_retry(self, item, index, total_tasks, semaphore):
        # Get the real character required for dubbing based on the character name
        task_id = f" [{index + 1}/{total_tasks}]"
        if not item.get('text','').strip() or tools.vail_file(item['filename']):
            await self.increment_counter()
            return
        
        try:
            async with semaphore:
                
                if self._stop_event.is_set():
                    return
                msg=""
                for attempt in range(RETRY_NUMS+1):
                    if self._stop_event.is_set():
                        return
                    
                    try:
                        
                        if attempt>0:
                            msg= f'Retry after {attempt}nd  '
                        communicate = Communicate(
                            item['text'], voice=item['role'], rate=self.rate,
                            volume=self.volume, proxy=self.useproxy, pitch=self.pitch, connect_timeout=5
                        )
                        # Prevent WebSocket connections or data reading from hanging indefinitely
                        await asyncio.wait_for(
                            communicate.save(item['filename'] + ".mp3"),
                            timeout=SAVE_TIMEOUT
                        )
                        

                        if self._stop_event.is_set(): return
                        loop = asyncio.get_running_loop()
                        signal_with_args = functools.partial(
                            self._signal, 
                            text=f'{tr("kaishipeiyin")} {msg}[{self.ends_counter + 1}/{total_tasks}]'
                        )
                        
                        try:
                            await asyncio.wait_for(
                                loop.run_in_executor(None, signal_with_args),
                                timeout=SIGNAL_TIMEOUT
                            )
                        except asyncio.TimeoutError:
                            logger.warning(f"{task_id}: Sending UI signal timed out!")

                        return

                    except asyncio.TimeoutError as e:
                        #print(f'{e}')
                        if attempt < RETRY_NUMS:
                            await asyncio.sleep(RETRY_DELAY)
                        else:
                            logger.error(f"EdgeTTS voiceover: The maximum number of retries has been reached and the task failed (timeout).")
                            self.error=e
                            # Failure is also a completion, return directly
                            return
                    except (NoAudioReceived, aiohttp.ClientError) as e:
                        #print(f'{e}')
                        self.error=e if not self.useproxy else f'proxy={self.useproxy}, {tr("Please turn off the clear proxy and try again")}:{e}'
                        #Force disable proxy retry
                        self.useproxy=None
                        if attempt < RETRY_NUMS:
                            await asyncio.sleep(RETRY_DELAY)
                        else:
                            logger.error(f"{task_id}: The maximum number of retries has been reached and the task failed.")
                            # Failure is also a completion, return directly
                            return

        except asyncio.CancelledError as e:
            self.error=e
        except Exception as e:
            logger.exception(f"{task_id}: An unknown serious error occurred and the task was terminated.", exc_info=True)
            self.error=e
        finally:
            # Regardless of success, failure, cancellation or exception, increase the count here uniformly
            await self.increment_counter()

    async def watchdog(self, tasks):
        'watchdog'
        await self._stop_event.wait()
        for task in tasks:
            task.cancel()

    async def _exit_monitor(self):
        'Exit monitor'
        while not self._exit():
            if self._stop_event.is_set():
                break
            await asyncio.sleep(POLL_INTERVAL)
        if self._exit():
            self._stop_event.set()
    
    async def _exec(self) -> None:

        if not self.queue_tts:
            return
        
        logger.debug(f'This EdgeTTS dubbing: retry delay:{RETRY_DELAY},If an error occurs, try again:{RETRY_NUMS},concurrency:{MAX_CONCURRENT_TASKS}')
        self._stop_event.clear()
        total_tasks = len(self.queue_tts)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        for it in self.queue_tts:
            resolved = tools.get_edge_rolelist(it['role'], self.language)
            # get_edge_rolelist may return None if the voice name is not in the JSON
            # and does not match the pattern. Fall back to the original value so that
            # a real voice name (e.g. "en-US-JennyNeural") is never lost.
            if resolved is None:
                resolved = it['role']
            # Last safety net: if still None or not a str, use a sensible default
            if not isinstance(resolved, str) or not resolved.strip():
                resolved = f"{self.language[:2]}-{'US' if self.language[:2]=='en' else 'default'}-AriaNeural"
                logger.warning(f"EdgeTTS: voice role is None/empty, falling back to {resolved}")
            it['role'] = resolved

        worker_tasks = [
            asyncio.create_task(
                self._create_audio_with_retry(item, i, total_tasks, semaphore)
            )
            for i, item in enumerate(self.queue_tts)
        ]

        if not worker_tasks:
            return

        monitor_task = asyncio.create_task(self._exit_monitor())
        watchdog_task = asyncio.create_task(self.watchdog(worker_tasks))
        
        all_workers_done = asyncio.gather(*worker_tasks, return_exceptions=True)
        
        # Add a timeout to the overall wait to prevent infinite hangs (e.g. all tasks time out but are not cancelled)
        try:
            done, pending = await asyncio.wait(
                [all_workers_done, monitor_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=total_tasks * SAVE_TIMEOUT * 2  #Total tasks * timeout per * 2 (buffered)
            )
        except asyncio.TimeoutError:
            logger.error('The overall execution timed out! Forcefully cancel all tasks.')
            self._stop_event.set()
            done, pending = await asyncio.wait([all_workers_done, monitor_task], return_when=asyncio.ALL_COMPLETED)
        
        try:
            if monitor_task not in done:
                logger.debug('Execution process: All dubbing tasks are completed.')
                monitor_task.cancel()

            watchdog_task.cancel()
            for task in pending:
                task.cancel()
            
            await asyncio.gather(all_workers_done, monitor_task, watchdog_task, return_exceptions=True)
            
            final_count = self.ends_counter
            if final_count != total_tasks:
                logger.error(
                    f"!!!!!!!!!!!!!!! Task count mismatch !!!!!!!!!!!!!!!!!!"
                    f"Expected number of tasks:{total_tasks}, actual number completed:{final_count}."
                    f"lost{total_tasks - final_count} The status of a task."
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )        
            

            ok, err = 0, 0
            for i, item in enumerate(self.queue_tts):
                if app_cfg.exit_soft:
                    return
                mp3_path = item['filename'] + ".mp3"
                if tools.vail_file(mp3_path):
                    ok += 1
                else:
                    err += 1

            if ok>0:
                all_task = []
                from concurrent.futures import ThreadPoolExecutor
                self._signal(text=f'convert wav {total_tasks}')
                with ThreadPoolExecutor(max_workers=min(4,len(self.queue_tts),os.cpu_count())) as pool:
                    for item in self.queue_tts:
                        mp3_path = item['filename'] + ".mp3"
                        if tools.vail_file(mp3_path):
                            all_task.append(pool.submit(self.convert_to_wav, mp3_path,item['filename']))
                    if len(all_task) > 0:
                        _ = [i.result() for i in all_task]

            if err > 0:
                msg=f'[{err}] errors, {ok} succeed'
                self._signal(text=msg)
                logger.debug(f'EdgeTTS dubbing ends:{msg}')
        finally:
            await asyncio.sleep(0.1)
