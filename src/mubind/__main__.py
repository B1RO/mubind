import sys

from .cli.startup import finish_provider_startup, start_cli_startup

try:
    # Also covers run-local, before importing the CLI or PyTorch.
    start_cli_startup(sys.argv[1:])
    from .cli import app

    app()
finally:
    finish_provider_startup()
