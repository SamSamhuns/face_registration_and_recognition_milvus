import os
import urllib.request as urllib2


def get_mode_ext(mode):
    return {"image": ".jpg", "video": ".mp4"}[mode]


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def download_url_file(download_url: str, download_path: str) -> None:
    response = urllib2.urlopen(download_url)
    with open(download_path, 'wb') as f:
        f.write(response.read())


async def cache_file_locally(file_cache_path: str, data: bytes) -> None:
    with open(file_cache_path, 'wb') as img_file_ptr:
        img_file_ptr.write(data)
