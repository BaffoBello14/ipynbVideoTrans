import multiprocessing,os
from videotrans.configure.config import app_cfg,settings,logger

# ==========================================
# Add a wrapper class: to be compatible with the caller's habit of Future objects (.result())
# ==========================================
class AsyncResultFutureWrapper:
    def __init__(self, async_result):
        self.async_result = async_result

    def result(self, timeout=None):
        # Disguise Pool’s unique .get() as Future’s .result()
        return self.async_result.get(timeout=timeout)
        
    def done(self):
        return self.async_result.ready()


# ==========================================
# Global singleton manager
# ==========================================


class GlobalProcessManager:
    _executor_cpu = None
    _executor_gpu = None



    @classmethod
    def get_cpu_process_nums(cls):
        cpu_count=int(os.cpu_count())
        try:
            man_set=int(float(settings.get('process_max',0)))
        except:
            man_set=0
        if man_set>0:
            return int(min(man_set,8,cpu_count))

        import psutil
        mem=psutil.virtual_memory()
        # Maximum 8 processes, minimum 2
        return int(max( min( (mem.available/(1024**3))//4 , 8, cpu_count ), 2))

    @classmethod
    def get_gpu_process_nums(cls):
        cpu_count=int(os.cpu_count())
        try:
            process_max_gpu=int(float(settings.get('process_max_gpu',0)))
        except:
            process_max_gpu=0
        # If the number of gpu processes is manually set, the priority will be the highest. For example, although there is only one card, but the video memory is very large, multiple gpu processes can be manually set.
        if process_max_gpu>0:
            return int(min(process_max_gpu,8,cpu_count))
        if app_cfg.NVIDIA_GPU_NUMS<0:
            return 1
        # If there is no graphics card or multiple graphics cards are not enabled, only one gpu process will be started.
        if  app_cfg.NVIDIA_GPU_NUMS<1 or not bool(settings.get('multi_gpus',False)):
            return 1
        
        return int(min(app_cfg.NVIDIA_GPU_NUMS,8,cpu_count))

    @classmethod
    def get_executor_cpu(cls):
        """
        """
        if cls._executor_cpu is None:
            ctx = multiprocessing.get_context('spawn')
            max_workers=cls.get_cpu_process_nums()
            logger.debug(f'CPU process pool:{max_workers=}')
            #cls._executor_cpu = ProcessPoolExecutor(max_workers=int(max_workers), mp_context=ctx)
            cls._executor_cpu = ctx.Pool(
                processes=int(max_workers), 
                maxtasksperchild=1  # <--- The CPU will also let it die after running and completely release the physical memory.
            )
        return cls._executor_cpu

    @classmethod
    def get_executor_gpu(cls):
        '\n        max_workers is set to 1, which means that only one AI task can be run at the same time.\n        '
        if cls._executor_gpu is None:
            ctx = multiprocessing.get_context('spawn')
            max_workers=cls.get_gpu_process_nums()
            logger.debug(f'GPU process pool:{max_workers=}')
            #cls._executor_gpu = ProcessPoolExecutor(max_workers=int(max_workers), mp_context=ctx)
            cls._executor_gpu = ctx.Pool(
                processes=int(max_workers), 
                maxtasksperchild=1
            )

        return cls._executor_gpu

    @classmethod
    def submit_task_cpu(cls, func, **kwargs):
        _executor=cls.get_executor_cpu()
        async_result = _executor.apply_async(func, kwds=kwargs)
        return AsyncResultFutureWrapper(async_result)
        #return _executor.submit(func, **kwargs)

    @classmethod
    def submit_task_gpu(cls, func, **kwargs):
        _executor=cls.get_executor_gpu()
        # The method for Pool to submit tasks is apply_async, and the kwds keyword parameter needs to be specified.
        async_result = _executor.apply_async(func, kwds=kwargs)
        # Return the wrapper class we wrote, so that after getting the main logic, we can still write future.result()
        return AsyncResultFutureWrapper(async_result)
        #return _executor.submit(func, **kwargs)

    @classmethod
    def shutdown(cls):
        if cls._executor_cpu:
            cls._executor_cpu.close()
            cls._executor_cpu.join()
            cls._executor_cpu = None
            #cls._executor_cpu.shutdown(wait=True)
        if cls._executor_gpu:
            cls._executor_gpu.close()
            cls._executor_gpu.join()
            cls._executor_gpu = None
            #cls._executor_gpu.shutdown(wait=True)

