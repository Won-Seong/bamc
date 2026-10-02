from typing import List, Optional
from io import BytesIO
from PIL import Image

from google import genai
from google.genai import types

def call_vlm(
    client: Optional[genai.Client],
    imgs: List[Image.Image],
    model: str,
    prompt: str,
    temperature: float = 1.0,
    response_format=None
) -> str:
    """Call VLM to get the prompt for the image."""
    if client is None:
        raise ValueError(
            "Gemini client is not initialized. Please ensure GEMINI_API_KEY is set in your .env or environment variables."
        )

    contents = [prompt] + imgs
    config = types.GenerateContentConfig(
        temperature=temperature,
    )

    if response_format is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_format

    max_retries = 10
    base_delay = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                print(f"⚠️ VLM call failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {base_delay} seconds...")
                time.sleep(base_delay)
            else:
                print(f"❌ VLM call failed after {max_retries} attempts. Terminating.")
                raise e

def call_image_generator(
    client: Optional[genai.Client],
    model: str,
    prompt: str
) -> Image.Image:
    """Call image generator to get the image for the prompt."""
    if client is None:
        raise ValueError(
            "Gemini client is not initialized. Please ensure GEMINI_API_KEY is set in your .env or environment variables."
        )
    
    config = types.GenerateImagesConfig(
        number_of_images=1,
        output_mime_type="image/jpeg",
        aspect_ratio="1:1" 
    )

    result = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=config
    )

    image_bytes = result.generated_images[0].image.image_bytes
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    resized_image = img.resize((512, 512))
    return resized_image
