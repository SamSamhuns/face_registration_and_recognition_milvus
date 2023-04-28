"""
common utils
"""
import os
import urllib.request as urllib2


def get_mode_ext(mode):
    """
    Returns the file extension for the given mode.
    Args:
        mode (str): Either "image" or "video".
    Returns:
        str: The file extension - ".jpg" for image, ".mp4" for video.
    """
    return {"image": ".jpg", "video": ".mp4"}[mode]


def remove_file(path: str) -> None:
    """
    Removes the file at the given path.
    Args:
        path (str): The path of the file to remove.
    """
    if os.path.exists(path):
        os.remove(path)


async def download_url_file(download_url: str, download_path: str) -> None:
    """
    Downloads the file at the given URL to the local path.
    Args:
        download_url (str): The URL of the file to download.
        download_path (str): The local path to save the downloaded file.
    """
    response = urllib2.urlopen(download_url)
    with open(download_path, "wb") as fwtr:
        fwtr.write(response.read())


async def cache_file_locally(file_cache_path: str, data: bytes) -> None:
    """
    Caches the given data to a local file.
    Args:
        file_cache_path (str): The path to save the data.
        data (bytes): The data to cache.
    """
    with open(file_cache_path, "wb") as img_file_ptr:
        img_file_ptr.write(data)
