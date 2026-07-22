"""
llm_client.py
-------------
A thin, provider-agnostic wrapper around Anthropic, OpenAI, and Gemini.

Every other module should call:
    client = LLMClient()
    response = client.complete(...)
"""

from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

from src.utils.logger import logger


class LLMClient:
    """
    Provider-independent LLM wrapper.
    """

    def __init__(self, provider: str | None = None):
        self.provider = (provider or LLM_PROVIDER).lower()

        if self.provider == "anthropic":
            import anthropic

            if not ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file."
                )

            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self._model = ANTHROPIC_MODEL

        elif self.provider == "openai":
            import openai

            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )

            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self._model = OPENAI_MODEL

        elif self.provider == "gemini":
            from google import genai

            if not GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Add it to your .env file."
                )

            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._model = GEMINI_MODEL

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

        logger.info(
            f"LLMClient initialized: provider={self.provider}, model={self._model}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def complete(
        self,
        prompt: str,
        system: str = "You are a precise, helpful document-analysis assistant.",
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ) -> str:

        if self.provider == "anthropic":
            return self._complete_anthropic(
                prompt,
                system,
                max_tokens,
                temperature,
            )

        elif self.provider == "openai":
            return self._complete_openai(
                prompt,
                system,
                max_tokens,
                temperature,
            )

        elif self.provider == "gemini":
            return self._complete_gemini(
                prompt,
                system,
                max_tokens,
                temperature,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(
        self,
        messages: list[dict],
        system: str = "You are a precise, helpful document-analysis assistant.",
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ) -> str:

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )

            return response.content[0].text

        elif self.provider == "openai":
            full_messages = [{"role": "system", "content": system}] + messages

            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=full_messages,
            )

            return response.choices[0].message.content

        elif self.provider == "gemini":

            prompt = system + "\n\n"

            for m in messages:
                prompt += f"{m['role']}: {m['content']}\n"

            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )

            return response.text

        raise ValueError(f"Unsupported provider: {self.provider}")

    # ----------------------------------------------------------
    # Provider-specific implementations
    # ----------------------------------------------------------

    def _complete_anthropic(
        self,
        prompt,
        system,
        max_tokens,
        temperature,
    ) -> str:

        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.content[0].text

    def _complete_openai(
        self,
        prompt,
        system,
        max_tokens,
        temperature,
    ) -> str:

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response.choices[0].message.content

    def _complete_gemini(
        self,
        prompt,
        system,
        max_tokens,
        temperature,
    ) -> str:

        response = self._client.models.generate_content(
            model=self._model,
            contents=f"{system}\n\n{prompt}",
        )

        return response.text