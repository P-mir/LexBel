
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-cloud",
        action="store_true",
        default=False,
        help="Run tests that require cloud API (OpenAI)"
    )
