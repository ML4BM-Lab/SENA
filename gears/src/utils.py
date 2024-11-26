"""Utilities."""

import os

import requests
from tqdm import tqdm


def download_file(url: str, path: str, skip_if_exists: bool = True) -> None:
    """Download a file with a progress bar.

    The progress bar will display the size in binary units (e.g., KiB for kibibytes,
    MiB for mebibytes, GiB for gibibytes, etc.), which are based on powers of 1024.

    Args:
        url: The URL of the file.
        path: The path where the file will be saved.
        skip_if_exists: If True, skip downloading the file if it already exists.

    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP
            request.
        OSError: If there is an issue with writing the file.
    """
    if skip_if_exists and os.path.exists(path=path):
        print(f"Skipping download because file already exists: {path}")
        return

    print(f"Downloading: {url} -> {path}")
    progress_bar = None
    try:
        with requests.get(url=url, stream=True) as response:
            response.raise_for_status()
            total_size_in_bytes = int(
                response.headers.get(key="content-length", default=0)
            )
            print(f"Total size: {total_size_in_bytes:,} bytes")
            block_size = 1024
            with tqdm(
                total=total_size_in_bytes, unit="iB", unit_scale=True
            ) as progress_bar:
                with open(file=path, mode="wb") as file:
                    for data in response.iter_content(chunk_size=block_size):
                        progress_bar.update(n=len(data))
                        file.write(data)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        raise
    except OSError as e:
        print(f"Error writing file: {e}")
        raise
