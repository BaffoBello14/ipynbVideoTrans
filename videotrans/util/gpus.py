# 1. Get and cache the number of available gpu
# 2. Get the available cuda number
# 3. Does MacOSX support mps?
import platform
from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR



# Get the number of available GPUs and cache them in config.NVIDIA_GPU_NUMS, 0=no available graphics card
#
# force_cpu: unused parameter
# True forces the use of cpu, that is, it forces the setting to have no graphics card.
def getset_gpu(force_cpu=False) -> int:
    if force_cpu:
        return 0
    # Not obtained yet is -1
    if app_cfg.NVIDIA_GPU_NUMS > -1:
        return app_cfg.NVIDIA_GPU_NUMS
    print('First searching GPU...')
    import torch
    # No graphics card available
    app_cfg.NVIDIA_GPU_NUMS = 0 if not torch.cuda.is_available() else torch.cuda.device_count()
    print(f'NVIDIA_GPU_NUMS={app_cfg.NVIDIA_GPU_NUMS}')
    return app_cfg.NVIDIA_GPU_NUMS


# Get the cuda graphics card index available with current limitations
# return -1 No graphics card available, forcing the caller to use cpu or mps
# >=0 is the graphics card number
def get_cudaX() -> int:
    if platform.system() == 'Darwin':
        return -1
    try:
        # The number of available graphics cards has not been initialized yet
        if app_cfg.NVIDIA_GPU_NUMS == -1:
            getset_gpu()

        if app_cfg.NVIDIA_GPU_NUMS == 0:
            # No graphics card available
            return -1

        if app_cfg.NVIDIA_GPU_NUMS == 1 or not bool(settings.get('multi_gpus', False)):
            # Only one card, no options or multiple cards but multi-graphics cards are not enabled
            return 0

        import torch
        # If there is available video memory greater than 24G, it can be returned to use directly.
        free_g = (1024 ** 3) * 24
        _default_index = 0
        _default_free, _ = torch.cuda.mem_get_info(_default_index)
        if _default_free > free_g:
            return 0

        # Return the ones with more than 24G of available video memory in sequence. If they do not exist, return the one with the largest free video memory.
        for i in range(1, app_cfg.NVIDIA_GPU_NUMS):
            free_bytes, _ = torch.cuda.mem_get_info(i)
            if free_bytes > free_g:
                logger.debug(f'[Use no.{i}block graphics card], the available video memory is{free_bytes / (1024 ** 3)}GB')
                return i
            if free_bytes > _default_free:
                _default_free = free_bytes
                _default_index = i
        logger.debug(f'[Use no.{_default_index}block graphics card], the available video memory is{_default_free / (1024 ** 3)}GB')
        return _default_index
    except Exception as e:
        logger.exception(f'Failed to obtain the currently available graphics card index and returned to the 0th graphics card:{e}', exc_info=True)
        return 0


# MacOSX determines whether mps is supported
# mps: support
# cpu: not supported, must use cpu
def mps_or_cpu() -> str:
    if platform.system() != 'Darwin':
        return 'cpu'
    import torch
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
