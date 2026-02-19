"""
common utils
"""

import os

import aiofiles
import httpx


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


async def download_url_file(download_url: str, download_path: str, timeout: int = 60) -> None:
    """
    Downloads the file at the given URL to the local path using requests library.
    This version uses streaming to handle large files and includes basic error handling.
    Args:
        download_url (str): The URL of the file to download.
        download_path (str): The local path to save the downloaded file.
        timeout (int): The timeout in seconds to use for the request.
    """
    # Stream the download to handle large files
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(download_url)
        response.raise_for_status()

        # Write the file out in chunks to avoid using too much memory
        async with aiofiles.open(download_path, "wb") as file:
            async for chunk in response.aiter_bytes(chunk_size=8192):
                await file.write(chunk)


async def cache_file_locally(file_cache_path: str, data: bytes) -> None:
    """
    Caches the given data to a local file.
    Args:
        file_cache_path (str): The path to save the data.
        data (bytes): The data to cache.
    """
    async with aiofiles.open(file_cache_path, "wb") as img_file_ptr:
        await img_file_ptr.write(data)
