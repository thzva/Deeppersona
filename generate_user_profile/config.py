"""Configuration module for managing API keys, proxy settings, and OpenAI client configuration.

This module contains the following main features:
- OpenAI API configuration
- Proxy server settings
- OpenAI client initialization
- API call utility functions
- GPT response parsing functions
"""

import os
import json
from openai import OpenAI
from typing import List, Dict, Optional, Any, Union, Tuple

# API Configuration
# Note: In production environment, API keys should be read from environment variables or config files

# GeoNames API Configuration
GEONAMES_USERNAME = "demo"  # Replace with your GeoNames username
GEONAMES_API_BASE = "http://api.geonames.org"

# OpenAI API Configuration
OPENAI_API_KEY = "OPENAI_API_KEY"

# GPT model version to use
GPT_MODEL = "gpt-4.1-mini"

# Proxy Configuration
os.environ['https_proxy'] = ''
os.environ['http_proxy'] = ''

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_completion(messages: List[Dict[str, str]], model: str = GPT_MODEL, temperature: float = 0.2, max_retries: int = 3) -> Optional[str]:
    """Call OpenAI GPT API to get response with retry mechanism

    Args:
        messages (List[Dict[str, str]]): List of messages, each containing role and content
        model (str, optional): GPT model name to use. Defaults to GPT_MODEL from config
        temperature (float, optional): Sampling temperature to control randomness. Defaults to 0.2
        max_retries (int, optional): Maximum number of retries. Defaults to 3

    Returns:
        Optional[str]: Text response from API. Returns None if request fails
    """
    import time
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Incremental wait time: 2s, 4s, 6s
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} retry attempts failed")
                return None


def extract_json_from_markdown(response: str) -> str:
    """Extract JSON content from Markdown code block.

    Args:
        response (str): Response text that may contain Markdown code block

    Returns:
        str: Extracted JSON content or original response
    """
    if response and response.strip().startswith('```') and '```' in response:
        # Extract code block content
        code_content = response.split('```', 2)[1]
        if code_content.startswith('json'):
            code_content = code_content[4:].strip()
        response = code_content.strip()
    return response


def parse_json_response(response: str, default_value: Any = None) -> Any:
    """Parse JSON response and handle possible parsing errors.

    Args:
        response (str): Response text in JSON format
        default_value (Any, optional): Default value to return if parsing fails

    Returns:
        Any: Parsed JSON object or default value
    """
    if not response:
        return default_value
    
    # First try to extract JSON from Markdown
    response = extract_json_from_markdown(response)
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        print(f"Response was: {response[:100]}..." if len(response) > 100 else f"Response was: {response}")
        return default_value


def parse_gpt_response(response: str, expected_fields: List[str] = None, field_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    """Parse GPT response, extract expected fields and apply default values.

    Args:
        response (str): GPT response text
        expected_fields (List[str], optional): List of expected fields
        field_defaults (Dict[str, Any], optional): Dictionary of field default values

    Returns:
        Dict[str, Any]: Dictionary containing all expected fields
    """
    if field_defaults is None:
        field_defaults = {}
    
    # Parse JSON response
    result = parse_json_response(response, {})
    
    # If no expected fields specified, return parsed result directly
    if not expected_fields:
        return result
    
    # Ensure all expected fields are returned
    output = {}
    for field in expected_fields:
        output[field] = result.get(field, field_defaults.get(field))
    
    return output


def parse_nested_json_response(response: str) -> Tuple[Dict[str, Any], bool]:
    """Parse potentially nested JSON response.

    Handles JSON objects that may be nested within Markdown code blocks as JSON string representations.

    Args:
        response (str): Response text that may contain nested JSON

    Returns:
        Tuple[Dict[str, Any], bool]: Parsed JSON object and success flag
    """
    # First extract content from Markdown
    extracted = extract_json_from_markdown(response)
    
    # Try to parse JSON
    try:
        result = json.loads(extracted)
        
        # Check if it's a JSON object represented as a JSON string
        if isinstance(result, dict) and len(result) == 1 and next(iter(result.values())).startswith('{'):
            key = next(iter(result.keys()))
            try:
                nested_json = json.loads(result[key])
                return nested_json, True
            except json.JSONDecodeError:
                pass
        
        return result, True
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        return {}, False
