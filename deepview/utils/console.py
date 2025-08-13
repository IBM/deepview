from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.spinner import Spinner
from rich.live import Live
import sys
import os

class ConsoleWrapper:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConsoleWrapper, cls).__new__(cls)
            cls._instance.console = Console()
            cls._instance.live = None
            cls._instance.progress = None
            cls._instance.progress_task = None
        return cls._instance

    def start_progress(self, text="Running DeepView..."):
        if not sys.stdout.isatty() or os.environ.get("IN_POD") == "true":
            self.console.print(text)
            return
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=self.console,
            )
            self.progress.start()
            self.progress_task = self.progress.add_task(text, total=None)

    def stop_progress(self):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.progress_task = None

    def print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)

    def error(self, error):
        self.stop_progress()
        self.console.print(f"[red]{error}[/red]")
        sys.exit(1)