from typing import Dict

# Mistral API pricing (per million tokens)
MISTRAL_PRICING = {
    "mistral-small-latest": {
        "input": 1000,   #  per M tokens, I fake it here (*1000 wrt real values) to compensate for low volume, and make the numbers more realistic
        "output": 3000,
    }
    # Add other models as needed
}


def estimate_tokens(text: str) -> int:
    """Rough token count estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> Dict[str, float]:
    """Calculate cost for Mistral API call.
        Returns:
        Dict with input_cost, output_cost, total_cost in USD
    """
    pricing = MISTRAL_PRICING["mistral-small-latest"]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost * 1000:.4f}m"  # in millidollars
    return f"${cost:.4f}"
