import os
import base64
import requests
import cv2
import numpy as np

def upload_image(image: np.ndarray) -> str:
    """
    Uploads the image to an image hosting service and returns the URL.
    If you have a dedicated storage, replace this with your actual upload logic.
    """
    _, buffer = cv2.imencode(".png", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # Log the key and payload size for debugging (for security, avoid printing sensitive info in production)
    print(f"Uploading image with key: {os.getenv('IMGBB_API_KEY')}")
    print(f"Base64 image size: {len(base64_image)} bytes")

    try:
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": os.getenv("IMGBB_API_KEY"), "image": base64_image}
        )

        # Check if response is successful
        if response.status_code == 200:
            return response.json()["data"]["url"]
        else:
            # Log the detailed response if not successful
            print(f"Error Response: {response.text}")
            raise Exception(f"Failed to upload image. Status Code: {response.status_code}")

    except Exception as e:
        print(f"Error during image upload: {e}")
        raise e
