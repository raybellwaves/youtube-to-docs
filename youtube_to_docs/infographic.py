import os
from typing import Optional

from google import genai


def generate_infographic(
    image_model: Optional[str], summary_text: str, video_title: str
) -> Optional[bytes]:
    """
    Generates an infographic image using the specified model.
    Returns the image bytes if successful, None otherwise.
    """
    if not image_model:
        return None

    prompt = (
        "Create a visually appealing infographic summarizing the following "
        "video content.\n"
        f"Video Title: {video_title}\n\n"
        f"Summary:\n{summary_text}\n\n"
        "The infographic should be easy to read, professional, and capture "
        "the key points."
    )

    if "gemini" in image_model:
        try:
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                print("Error: GEMINI_API_KEY not found for infographic generation")
                return None

            client = genai.Client(api_key=GEMINI_API_KEY)

            response = client.models.generate_images(
                model=image_model,
                prompt=prompt,
            )

            if response.generated_images:
                return response.generated_images[0].image.image_bytes

            print(f"No image data found in response from {image_model}")
            return None

        except Exception as e:
            print(f"Infographic generation error with {image_model}: {e}")
            return None

    print(f"Image model {image_model} not supported yet.")
    return None
