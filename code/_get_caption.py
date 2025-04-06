from openai import OpenAI
from PIL import Image
from _get_model_key import Constants as const
import google.generativeai as genai
import anthropic

def describe_frame_with_gtp(img_base64, 
                            prompt="What's in this image", 
                            model_name="gpt-4o-mini", 
                            img_type="image/jpeg"):

    client = OpenAI(api_key=const.OPEN_AI_KEY)

    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt                       
                    },                    
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_type};base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )
    
    return response.choices[0].message.content

def describe_frame_with_gemini(img_path,
                               prompt="What's in this image", 
                               model_name="gemini-1.5-flash"):
    
    genai.configure(api_key=const.GEMINI_KEY)
    model = genai.GenerativeModel(model_name)
    organ = Image.open(img_path)
    response = model.generate_content([prompt, organ])
    if response.candidates:
        return response.text
    return ""

def describe_frame_with_claude(base64_image, 
                               prompt="What's in this image", 
                               model_name="claude-3-5-sonnet-20241022", 
                               media_type = "image/jpeg"):
    
    client = anthropic.Anthropic(
        api_key= const.CLAUDE_KEY,
    )

    message = client.messages.create(
        model= model_name,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing legal documents."
            },
            {
                "type": "text",
                "text": "Here is the full text of a complex legal agreement: [Insert full text of a 50-page legal agreement here]",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return message.content[0].text