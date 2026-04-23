import re

import requests
from requests.exceptions import TooManyRedirects, MissingSchema, InvalidSchema, InvalidURL, ProxyError, SSLError, \
    Timeout, ConnectionError as ReqConnectionError, RetryError, HTTPError
from videotrans.configure.config import tr, params, settings, app_cfg, logger, defaulelang

# Optional heavy dependencies – imported lazily so missing packages only fail
# when the corresponding provider is actually used.
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from elevenlabs.core import ApiError as ApiError_11
except ImportError:
    ApiError_11 = Exception

try:
    from openai import (AuthenticationError, PermissionDeniedError, NotFoundError,
                        BadRequestError, RateLimitError, APIConnectionError, APIError,
                        ContentFilterFinishReasonError, InternalServerError,
                        LengthFinishReasonError)
except ImportError:
    AuthenticationError = PermissionDeniedError = NotFoundError = BadRequestError = Exception
    RateLimitError = APIConnectionError = APIError = Exception
    ContentFilterFinishReasonError = InternalServerError = LengthFinishReasonError = Exception

try:
    from deepgram.clients.common.v1.errors import DeepgramApiError
except ImportError:
    DeepgramApiError = Exception

try:
    import httpx, httpcore
except ImportError:
    httpx = httpcore = None

try:
    from tenacity import RetryError as TenRetryError
except ImportError:
    TenRetryError = Exception


# The exceptions with error message have been sorted out internally, set ex=None, message='{error message}'
class VideoTransError(Exception):
    def __init__(self, message=''):
        super().__init__(message)
        self.message=message
        

    def __str__(self):
        return str(self.message)


class TranslateSrtError(VideoTransError):
    pass


class DubbSrtError(VideoTransError):
    pass


class SpeechToTextError(VideoTransError):
    pass


class StopRetry(VideoTransError):
    pass




# No need to continue retrying exceptions
NO_RETRY_EXCEPT = (
    TooManyRedirects,  # Too many redirects
    MissingSchema,  # URL is missing protocol (such as "http://")
    InvalidSchema,  # URL protocol is invalid
    InvalidURL,  # URL format is invalid
    SSLError,  # SSL certificate verification failed

    # Connection problem, check the network or try to set up a proxy
    RetryError,
    ReqConnectionError,
    ConnectionError,
    ConnectionRefusedError,  # Connection refused
    ConnectionResetError,  # The connection is reset
    ConnectionAbortedError,  #

    httpx.ConnectError,
    httpx.ReadError,

    # proxy error
    ProxyError,

    # Permanent errors for the openai library (usually 4xx status codes)
    AuthenticationError,  # 401 Authentication failed (API Key error)
    PermissionDeniedError,  # 403 No permission to access the model
    NotFoundError,  # 404 Resource not found (e.g. wrong model name)
    BadRequestError,  # 400 Error request (for example, input content is too long, parameters are invalid, etc.)

    LengthFinishReasonError,
    RateLimitError,

    DeepgramApiError,
    StopRetry
)

'Check if the error message contains a local address'


def _is_local_address(url_or_message):
    if not url_or_message:
        return False

    text = str(url_or_message).lower()
    local_indicators = ['127.0.0.1', 'localhost', '0.0.0.0', '::1', '[::1]']

    return any(indicator in text for indicator in local_indicators)


'Try to extract the API address from the error message'


def _extract_api_url_from_error(error):
    error_str = str(error)

    # Find URL pattern
    url_patterns = [
        r'https?://[^\s\'"]+',
        r'www\.[^\s\'"]+\.[a-z]{2,}',
        r'[a-zA-Z0-9.-]+\.[a-z]{2,}',
    ]

    for pattern in url_patterns:
        matches = re.findall(pattern, error_str)
        if matches:
            return matches[0]

    return None


'Details of handling connection errors'
def _handle_connection_error_detail(error, lang):
    error_str = str(error).lower()

    # Check if it is a local address
    is_local = _is_local_address(error_str)
    api_url = _extract_api_url_from_error(error)

    base_message = ""

    if "dns" in error_str or "name or service not known" in error_str:
        base_message = (
            'Domain name resolution failed and the server address cannot be found.' if lang == 'zh'
            else "Domain name resolution failed, cannot find server address"
        )
    elif "ProxyError" in error_str:
        base_message = (
            'The proxy settings are incorrect or the proxy is unavailable. Please check the proxy or close the proxy and delete the content in the proxy text box.' if lang == 'zh'
            else "The proxy address is not available, please check"
        )

    elif "refused" in error_str or "10061" in error_str or 'Actively refuse' in error_str:
        if is_local:
            base_message = (
                'Connection refused, please ensure the local service is up and running' if lang == 'zh'
                else "Connection refused, please ensure the local service is started and running"
            )
        else:
            base_message = (
                'Connection refused, the target service may not be running or the port is wrong' if lang == 'zh'
                else "Connection refused, target service may not be running or wrong port"
            )
    elif "reset" in error_str:
        base_message = (
            'The connection was reset and the network may be unstable' if lang == 'zh'
            else "Connection reset, network may be unstable"
        )
    elif "timeout" in error_str or "timed out" in error_str:
        base_message = (
            'The connection timed out, please check whether the network connection is stable.' if lang == 'zh'
            else "Connection timeout, please check network stability"
        )
    elif "max retries exceeded" in error_str:
        if is_local:
            if "0.0.0.0" in error_str:
                base_message = (
                    'The API address cannot be 0.0.0.0, please change it to 127.0.0.1' if lang == 'zh'
                    else "The API address cannot be 0.0.0.0, please change it to 127.0.0.1"
                )
            else:
                base_message = (
                    'Multiple retries of connection failed, please make sure the local service has been started correctly' if lang == 'zh'
                    else "Multiple connection retries failed, please ensure local service is properly started"
                )
        else:
            base_message = (
                'Multiple attempts to connect failed and the service may be temporarily unavailable' if lang == 'zh'
                else "Multiple connection retries failed, service may be temporarily unavailable"
            )

    else:
        base_message = (
            'Network connection failed' if lang == 'zh'
            else "Network connection failed"
        )

    # Add additional prompts for Chinese users
    if lang == 'zh' and api_url and not is_local:
        if "api.msedgeservices.com" in api_url.lower():
            base_message += '. Frequent use of EdgeTTS may trigger current limiting, please wait for a while and try again.'
            return base_message
        if "edge.microsoft.com" in api_url.lower():
            base_message += '. Frequent use of Microsoft Translator may trigger current limiting, please wait for a while and try again.'
            return base_message
        # Check whether it serves a well-known foreign API
        foreign_apis = ['openai', 'anthropic', 'claude', 'elevenlabs', 'deepgram', 'google', 'aws.amazon']
        if any(api in api_url.lower() for api in foreign_apis):
            base_message += '. Note: Some foreign services require scientific Internet access in order to access them'

    return base_message





# According to the exception type, return the organized readable error message
def get_msg_from_except(ex):
    if isinstance(ex, VideoTransError):
        return str(ex)
        
    lang = defaulelang
    if isinstance(ex, TenRetryError):
        try:
            ex = ex.last_attempt.exception()
        except AttributeError:
            pass

    #Exception handling mapping
    exception_handlers = {
        # === Authentication and permission issues ===
        AuthenticationError: lambda e: (
            f"API key error, please check if the key is correct{e.body.get('message')}" if lang == 'zh'
            else  e.body.get('message')
        ),

        PermissionDeniedError: lambda e: (
            f"The current key does not have access permissions, please check the permission settings{e.body.get('message')}" if lang == 'zh'
            else  e.message
        ),

        # === Frequency limit ===
        RateLimitError: lambda e: (
            f"Too frequent requests or insufficient balance:{e.body.get('message')}" if lang == 'zh'
            else e.body.get('message')
        ),
        # === There is no problem with the resource ===
        # === Request parameter problem ===
        # === Server issues ===
        (InternalServerError,NotFoundError,BadRequestError,APIConnectionError,APIError): lambda e: e.body.get('message') if hasattr(e,'body') and hasattr(e.body,'get') else str(e),


        LengthFinishReasonError: lambda e: (
            f'The content is too long and exceeds the maximum allowed token. Please reduce the content or increase max_token, or reduce the number of subtitle lines sent each time\n{e}' if lang == 'zh' else f'{e}'),
        ContentFilterFinishReasonError: lambda
            e: f"Content triggers AI risk control and is filtered{e}" if lang == 'zh' else f'Content triggers AI risk control and is filtered\n{e}',



        # === Configuration and address issues ===
        (TooManyRedirects, MissingSchema, InvalidSchema, InvalidURL): lambda e: (
            f"The request address format is incorrect, please check the configuration{e.message}" if lang == 'zh'
            else f"Request URL format is incorrect, check configuration {e.message}"
        ),

        (ProxyError, aiohttp.client_exceptions.ClientProxyConnectionError): lambda e: (
            'The proxy settings are incorrect or the proxy is unavailable. Please check the proxy or close the proxy and delete the content in the proxy text box.' if lang == 'zh'
            else "Proxy configuration issue, check settings or disable proxy"
        ),
        SSLError: lambda e: (
            'The secure connection failed. Please check the system time or network settings. If a proxy is used, please close it and try again.' if lang == 'zh'
            else "Secure connection failed, check system time or network settings"
        ),

        Timeout: lambda e: (
            _handle_connection_error_detail(e, lang)
        ),

        HTTPError: lambda e: f'{e}',

        RetryError: lambda e: (
            'Still failed after retrying multiple times, please check the network connection or service status' if lang == 'zh'
            else "Failed after multiple retries, check network connection or service status"
        ),
        # === Network connection problem ===
        (httpcore.ConnectTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadError): lambda e: (
            _handle_connection_error_detail(e, lang)
        ),


        DeepgramApiError: lambda e: e.message if hasattr(e,'message') else str(e),

        ApiError_11: lambda e: e.body.get('detail',{}).get('message',e.body) if hasattr(e,'body') else str(e),


        ConnectionRefusedError: lambda e: (
            _handle_connection_error_detail(e, lang)
        ),

        ConnectionResetError: lambda e: (
            _handle_connection_error_detail(e, lang)
        ),

        ConnectionAbortedError: lambda e: (
            'The connection was interrupted unexpectedly, please check network stability' if lang == 'zh'
            else "Connection aborted unexpectedly, check network stability"
        ),
        (ReqConnectionError, ConnectionError): lambda e: (
            _handle_connection_error_detail(e, lang)
        ),
        requests.exceptions.RequestException:lambda e:f'{e}',

        RuntimeError: lambda e: (f"{e}" if lang == 'zh' else f"{e}" ),

        FileNotFoundError: lambda e: (
            f"File does not exist:{getattr(e, 'filename', '')}" if lang == 'zh'
            else f"File not found: {getattr(e, 'filename', '')}"
        ),

        PermissionError: lambda e: (
            f"Insufficient permissions to access:{getattr(e, 'filename', '')}" if lang == 'zh'
            else f"Permission denied: {getattr(e, 'filename', '')}"
        ),

        FileExistsError: lambda e: (
            f"File already exists:{getattr(e, 'filename', '')}" if lang == 'zh'
            else f"File already exists: {getattr(e, 'filename', '')}"
        ),

        # === Operating system error ===
        OSError: lambda e: (
            f"System error ({e.errno})：{e.strerror}" if lang == 'zh'
            else f"System Error ({e.errno}): {e.strerror}"
        ),

        # === Data processing error ===
        KeyError: lambda e: (
            f"A required key is missing when processing the data:{e}" if lang == 'zh'
            else f"{e}"
        ),

        IndexError: lambda e: (
            f"Index out of bounds when processing list or sequence:{e}" if lang == 'zh'
            else f"{e}"
        ),

        LookupError: lambda e: (
            f"Find error, the specified key or index does not exist:{e}" if lang == 'zh'
            else f"{e}"
        ),

        UnicodeDecodeError: lambda e: (
            f"File or data decoding failed, encoding format error:{e.reason}" if lang == 'zh'
            else f" {e.reason}"
        ),

        ValueError: lambda e: (
            f"Invalid value or argument:{e}" if lang == 'zh'
            else f"{e}"
        ),

        # === Program internal error ===
        AttributeError: lambda e: (
            f"Internal program error:{e}" if lang == 'zh'
            else f"{e}"
        ),

        NameError: lambda e: (
            f"Internal program error: undefined variable '{e.name}'" if lang == 'zh' else f"{e}"
        ),

        TypeError: lambda e: (
            f"Internal program error:{e}" if lang == 'zh'
            else f"{e}"
        ),

        RecursionError: lambda e: (
            f"Internal program error: infinite recursion occurred:{e}" if lang == 'zh'
            else f"{e}"
        ),

        ZeroDivisionError: lambda e: (
            f"Arithmetic error: division by zero:{e}" if lang == 'zh'
            else f"{e}"
        ),

        OverflowError: lambda e: (
            f"Arithmetic error: Value exceeds maximum limit:{e}" if lang == 'zh'
            else f"{e}"
        ),

        BrokenPipeError: lambda e: (
            'The connecting pipe is damaged, please check the network connection' if lang == 'zh'
            else "Broken pipe error, check network connection"
        ),
    }

    # Traverse the map and find matching processors
    for exc_types, handler in exception_handlers.items():
        if isinstance(ex, exc_types):
            return handler(ex)

    # === Backup processing logic ===
    error_str = str(ex)
    if any(keyword in error_str.lower() for keyword in [
        'connection', 'connect', 'refused', 'reset', 'timeout', 'retries',
        'connect', 'reject', 'reset', 'timeout', 'Try again', 'host', 'port', 'http', 'tcp','ProxyError'
    ]):
        return _handle_connection_error_detail(ex, lang)

    # Try to extract more specific information from the exception object
    if hasattr(ex, 'error') and ex.error:
        if isinstance(ex.error, dict):
            error_msg = str(ex.error.get('message', ex.error))
        else:
            error_msg = str(ex.error)
        return (
            f"Error details:{error_msg}" if lang == 'zh'
            else f"Error details: {error_msg}"
        )

    if hasattr(ex, 'message') and ex.message:
        return str(ex.message)

    if hasattr(ex, 'detail') and ex.detail:
        if isinstance(ex.detail, dict):
            message = ex.detail.get('message')
            if message:
                return str(message)
            error_info = ex.detail.get('error')
            if error_info:
                if isinstance(error_info, dict):
                    return str(error_info.get('message', error_info))
                return str(error_info)
        return str(ex.detail)

    if hasattr(ex, 'body') and ex.body:
        if isinstance(ex.body, dict):
            message = ex.body.get('message')
            if message:
                return str(message)
            error_info = ex.body.get('error')
            if error_info:
                if isinstance(error_info, dict):
                    return str(error_info.get('message', error_info))
                return str(error_info)
        return str(ex.body)

    #Default error message
    return ''
