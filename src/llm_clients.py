"""
LLM API Client Wrappers for Financial Hallucination Study
Supports: OpenAI GPT-5, Anthropic Claude 4.5, Google Gemini 2.5/3.0
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import streamlit as st


@dataclass
class LLMResponse:
    """Standardized response object from any LLM."""
    model: str
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    error: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def query(self, question: str, context: Optional[str] = None) -> LLMResponse:
        """Send query to LLM and return standardized response."""
        pass
    
    def _build_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build prompt with optional context."""
        if context and context.strip():
            return f"""Use the following context to answer the question. If the answer cannot be determined from the context, say "Cannot determine from provided context."

Context:
{context}

Question: {question}

Answer:"""
        else:
            return f"""Answer the following financial question accurately and concisely. If you're not certain, say so.

Question: {question}

Answer:"""


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-5 client wrapper."""
    
    # Pricing per 1M tokens (GPT-5 series, Nov 2025)
    PRICING = {
        "gpt-5": {"input": 10.00, "output": 30.00},
        "gpt-5-mini": {"input": 2.00, "output": 8.00},
        "gpt-5-nano": {"input": 0.50, "output": 2.00},
    }
    
    def __init__(self, model_name: str = "gpt-5-mini", api_key: str = None):
        super().__init__(model_name, api_key)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def query(self, question: str, context: Optional[str] = None) -> LLMResponse:
        prompt = self._build_prompt(question, context)
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant. Provide accurate, concise answers."},
                    {"role": "user", "content": prompt}
                ],
                #temperature=0.1,
                max_completion_tokens=500
            )
            latency_ms = (time.time() - start_time) * 1000
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            pricing = self.PRICING.get(self.model_name, {"input": 10.0, "output": 30.0})
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
            
            return LLMResponse(
                model=self.model_name,
                answer=response.choices[0].message.content.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost
            )
        except Exception as e:
            return LLMResponse(
                model=self.model_name,
                answer="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0,
                error=str(e)
            )


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude 4.5 client wrapper."""
    
    # Pricing per 1M tokens (Claude 4.5 series, Nov 2025)
    PRICING = {
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    }
    
    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929", api_key: str = None):
        super().__init__(model_name, api_key)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
    
    def query(self, question: str, context: Optional[str] = None) -> LLMResponse:
        prompt = self._build_prompt(question, context)
        
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a financial analyst assistant. Provide accurate, concise answers."
            )
            latency_ms = (time.time() - start_time) * 1000
            
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            pricing = self.PRICING.get(self.model_name, {"input": 3.0, "output": 15.0})
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
            
            return LLMResponse(
                model=self.model_name,
                answer=response.content[0].text.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost
            )
        except Exception as e:
            return LLMResponse(
                model=self.model_name,
                answer="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0,
                error=str(e)
            )


class GeminiClient(BaseLLMClient):
    """Google Gemini 2.5/3.0 client wrapper."""
    
    # Pricing per 1M tokens (Gemini series, Nov 2025)
    PRICING = {
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-3.0-pro-preview": {"input": 2.50, "output": 10.00},
    }
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        super().__init__(model_name, api_key)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.client = genai.GenerativeModel(
            model_name,
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )
    
    def _extract_text(self, response) -> tuple[str, str]:
        """Safely extract text from Gemini response. Returns (text, error)."""
        try:
            if hasattr(response, 'text') and response.text:
                return response.text.strip(), ""
        except ValueError:
            pass
        
        try:
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                return part.text.strip(), ""
                
                reason_map = {1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                reason_name = reason_map.get(finish_reason, str(finish_reason))
                return "", f"No content returned (finish_reason: {reason_name})"
            
            return "", "No candidates in response"
        except Exception as e:
            return "", f"Error extracting response: {str(e)}"
    
    def query(self, question: str, context: Optional[str] = None) -> LLMResponse:
        prompt = self._build_prompt(question, context)
        full_prompt = f"You are a financial analyst assistant. Provide accurate, concise answers.\n\n{prompt}"
        
        start_time = time.time()
        try:
            # Build generation config - disable thinking for 2.5 Pro
            gen_config = {
                "temperature": 0.1,
                "max_output_tokens": 8192,  # Increased significantly
            }
            
            # Disable thinking mode for models that have it enabled by default
            if "2.5-pro" in self.model_name or "3.0" in self.model_name:
                gen_config["thinking_config"] = {"thinking_budget": 0}
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=gen_config
            )
            latency_ms = (time.time() - start_time) * 1000
            
            answer_text, extraction_error = self._extract_text(response)
            
            try:
                input_tokens = self.client.count_tokens(full_prompt).total_tokens
            except:
                input_tokens = len(full_prompt) // 4
            
            output_tokens = 0
            if answer_text:
                try:
                    output_tokens = self.client.count_tokens(answer_text).total_tokens
                except:
                    output_tokens = len(answer_text) // 4
            
            pricing = self.PRICING.get(self.model_name, {"input": 1.25, "output": 5.00})
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
            
            return LLMResponse(
                model=self.model_name,
                answer=answer_text if answer_text else f"[{extraction_error}]",
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                latency_ms=latency_ms,
                cost_usd=cost,
                error=extraction_error if not answer_text else None
            )
        except Exception as e:
            return LLMResponse(
                model=self.model_name,
                answer="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0,
                error=str(e)
            )


# Model registry - Only GPT-5, Claude 4.5, and Gemini 2.5/3.0
AVAILABLE_MODELS = {
    # OpenAI GPT-5 Series
    "GPT-5": {"client": OpenAIClient, "model_id": "gpt-5", "key_name": "OPENAI_API_KEY"},
    "GPT-5 Mini": {"client": OpenAIClient, "model_id": "gpt-5-mini", "key_name": "OPENAI_API_KEY"},
    "GPT-5 Nano": {"client": OpenAIClient, "model_id": "gpt-5-nano", "key_name": "OPENAI_API_KEY"},
    
    # Anthropic Claude 4.5 Series
    "Claude Sonnet 4.5": {"client": AnthropicClient, "model_id": "claude-sonnet-4-5-20250929", "key_name": "ANTHROPIC_API_KEY"},
    "Claude Haiku 4.5": {"client": AnthropicClient, "model_id": "claude-haiku-4-5-20251001", "key_name": "ANTHROPIC_API_KEY"},
    
    # Google Gemini 2.5/3.0 Series
    "Gemini 2.5 Flash": {"client": GeminiClient, "model_id": "gemini-2.5-flash", "key_name": "GOOGLE_API_KEY"},
    "Gemini 2.5 Pro": {"client": GeminiClient, "model_id": "gemini-2.5-pro", "key_name": "GOOGLE_API_KEY"},
    "Gemini 3.0 Pro": {"client": GeminiClient, "model_id": "gemini-3.0-pro-preview", "key_name": "GOOGLE_API_KEY"},
}


def get_client(model_display_name: str) -> BaseLLMClient:
    """Factory function to get initialized LLM client."""
    if model_display_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_display_name}")
    
    config = AVAILABLE_MODELS[model_display_name]
    api_key = st.secrets["api_keys"].get(config["key_name"])
    
    if not api_key:
        raise ValueError(f"API key {config['key_name']} not found in secrets")
    
    return config["client"](model_name=config["model_id"], api_key=api_key)


def check_api_keys() -> dict:
    """Check which API keys are configured."""
    keys_status = {}
    try:
        keys = st.secrets.get("api_keys", {})
        keys_status["OpenAI"] = bool(keys.get("OPENAI_API_KEY"))
        keys_status["Anthropic"] = bool(keys.get("ANTHROPIC_API_KEY"))
        keys_status["Google"] = bool(keys.get("GOOGLE_API_KEY"))
    except Exception:
        keys_status = {"OpenAI": False, "Anthropic": False, "Google": False}
    return keys_status