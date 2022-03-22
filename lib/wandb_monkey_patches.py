import warnings

import requests  # type: ignore
from wandb.sdk.wandb_artifacts import WandbStoragePolicy

WANDB_PORT = 8081
_file_url_original = WandbStoragePolicy._file_url


def _file_url(self, api, entity_name, manifest_entry) -> str:
    """There is a wrong redirection to 8080 port while downloading artifacts from wandb.
    This method is a monkey patch of WandbStoragePolicy to use WANDB_PORT.
    """
    url = _file_url_original(self, api, entity_name, manifest_entry)
    api_key = self._api.api_key

    response = requests.get(url, auth=("api", api_key), stream=True, timeout=5, allow_redirects=False)

    next_url = response.headers.get("Location")  # redirection url
    if next_url and "http://localhost:8080/" in next_url:
        return next_url.replace("http://localhost:8080/", f"http://localhost:{WANDB_PORT}/")
    else:
        warnings.warn(f"WandB monkey patch expected redirection to 8080 port on {url}!")
        return url


def monkey_patch_wandb():
    WandbStoragePolicy._file_url = _file_url  # type: ignore
