from PIL import Image
import requests
from tqdm.auto import tqdm
import uuid
import os
from pathlib import Path
import numpy as np
from duckduckgo_search import DDGS


def collect_images(
    keywords: str, path: str, max_results: int = 30, timeout: tuple = (3, 5)
):
    """Function to collect images using DDGS

    Args:
      keywords (str): keywords for query
      path (str): images path to be saved to
      max_results (int): max number of results. If None, returns results only from the first response. Defaults to None.
      timeout (tuple[float, float]): timeout for request (connect_timeout, read_timeout). Default to (3, 5)

    Returns:
      image_results (dict[str: str]): {'image': image_url,
                                       'url': site_url,
                                       'path': image_path,
                                       'height': image_height,
                                       'width': image_width,
                                       'source': source_of_search}
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    image_results = []
    with DDGS() as ddgs:
        results = list(ddgs.images(keywords=keywords, max_results=max_results))

    for result in tqdm(results):
        try:
            img = Image.open(
                requests.get(result["image"], stream=True, timeout=timeout).raw
            ).convert("RGB")
            if np.array(img).shape[2] == 3:
                image_name = uuid.uuid5(
                    namespace=uuid.NAMESPACE_URL, name=result["image"]
                )
                img.save(f"{path}/{image_name}.jpg")
                image_result = {
                    "image": result["image"],
                    "url": result["url"],
                    "path": f"{path}/{image_name}.jpg",
                    "height": result["height"],
                    "width": result["width"],
                    "source": result["source"],
                }
                image_results.append(image_result)
        except:
            continue

    print(f"[INFO] Downloaded {len(image_results)} images into {path}")
    return image_results


def validate_images(path: str):
    """
    Function to validate images within a path and remove broken images

    Args: path (str): path to validate
    """
    paths = Path(path).rglob("*.jpg")
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            if np.array(img).shape[2] != 3:
                os.remove(path)
                print(f"[INFO] Removed invalid path: {path}")
        except:
            os.remove(path)
            print(f"[INFO] Removed invalid path: {path}")
