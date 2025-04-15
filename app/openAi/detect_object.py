import openai  
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_object(image_url: str) -> str:
    """
    Uses GPT-4 Turbo with Vision to detect objects in the image that may be obstacles for blind users.
    Returns only the names of objects that could be obstacles.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that identifies potential obstacles for blind users in images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please identify any potential obstacles in the image. Just list the objects that could be a hazard. Just return a single object name"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ],
            },
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content
