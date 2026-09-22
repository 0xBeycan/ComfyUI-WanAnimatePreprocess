"""Model download used by the loader and the SAM3 segmentation on first use."""
import os
import urllib.request

from comfy.utils import ProgressBar
from tqdm import tqdm

from .. import log


def download(url, path):
    """Stream `url` to `path` through a .part file, so an interrupted download never leaves
    a truncated model behind."""
    name = os.path.basename(path)
    log.info(f"downloading {name} from {url}")
    part = path + ".part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-WanAnimatePreprocess"})
        with urllib.request.urlopen(request) as response, open(part, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            comfy_pbar = ProgressBar(total) if total else None
            with tqdm(total=total or None, unit="B", unit_scale=True, desc=name) as pbar:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    out.write(chunk)
                    pbar.update(len(chunk))
                    if comfy_pbar is not None:
                        comfy_pbar.update_absolute(pbar.n)
    except Exception as e:
        raise RuntimeError(f"Could not download {name} from {url} ({e}). Download it by hand to {path}") from e
    os.replace(part, path)
