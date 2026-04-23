# Define a factory function to return the configured Session
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry


def custom_session_factory():
    sess = requests.Session()
    # Configure retry strategy
    retries = Retry(
        total=3,  # Total number of retries (changed to 3)
        connect=2,  # Number of connection retries
        read=2,  #Read the number of retries
        backoff_factor=2,  # Retry interval (seconds) to avoid instantaneous frequent requests
        status_forcelist=[500, 502, 503, 504]  # Retry when encountering these status codes
    )

    # Attach retry strategy to http and https protocols
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount('http://', adapter)
    sess.mount('https://', adapter)
    return sess
