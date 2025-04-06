from dataclasses import dataclass

@dataclass(frozen=True)
class Constants:
    OPEN_AI_KEY: str = ""
    GEMINI_KEY: str = ""
    CLAUDE_KEY: str = ""
