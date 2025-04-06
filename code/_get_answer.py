from openai import OpenAI
from _get_model_key import Constants as const
import anthropic
import google.generativeai as genai

# gpt-4o-mini
# gpt-3.5-turbo

def answer_gpt(prompt, engine="gpt-4o-mini", max_tok=1024):
    client = OpenAI(api_key=const.OPEN_AI_KEY)
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    
    
    completions = client.chat.completions.create(
        model=engine,
        messages=messages,
        max_tokens=max_tok,
        n=1,
        stop=None,
        temperature=0,
    )
    message = completions.choices[0].message.content

    return message

def answer(model_name, prompt, engine="gpt-4o-mini", max_tok=1024):
    
    
    if model_name == "gpt":
        client = OpenAI(api_key=const.OPEN_AI_KEY)
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}]
        
        
        completions = client.chat.completions.create(
            model=engine,
            messages=messages,
            max_tokens=max_tok,
            n=1,
            stop=None,
            temperature=0,
        )
        message = completions.choices[0].message.content
    elif model_name == "claude":
        client = anthropic.Anthropic(
            api_key= const.CLAUDE_KEY,
        )

        messages = client.messages.create(
            model= engine,
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
                    "content": prompt
                }
            ],
        )

        message = messages.content[0].text
    elif model_name == "gemini":
        genai.configure(api_key=const.GEMINI_KEY)
        model = genai.GenerativeModel(model_name=engine)
        response = model.generate_content(prompt)
        message = response.text
    else:
        print("You should choose one model from this list: [gpt, claude, gemini].")
        return None

    return message
