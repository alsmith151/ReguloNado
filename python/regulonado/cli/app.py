from __future__ import annotations

import logging
import sys
from importlib.metadata import PackageNotFoundError, version

import typer
from regulonado.cli.attribute import attribute
from regulonado.cli.config import config, init
from regulonado.cli.dataset import dataset, recompress_dataset
from regulonado.cli.design import design
from regulonado.cli.normalization import normalization_app
from regulonado.cli.pipeline import pipeline as _pipeline
from regulonado.cli.predict import predict
from regulonado.cli.tracks import tracks_app
from regulonado.cli.train import train

app = typer.Typer(no_args_is_help=True)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def _configure_logging(level_name: str) -> None:
    """Install the single stderr handler for every `regulonado` log line.

    Library modules only ever call ``logging.getLogger(__name__)`` and never
    install handlers of their own — this is the one place in the process that
    decides where "regulonado" log records go and at what level. Only the
    "regulonado" logger is touched, never the root logger, so a host
    application (or Hydra, when `training/runner.py` runs as its own
    subprocess and never reaches this callback) keeps control of its own
    logging configuration; "regulonado"'s child loggers still propagate to it.
    """
    level = logging.getLevelName(level_name.upper())
    if not isinstance(level, int):
        raise typer.BadParameter(
            f"Invalid log level {level_name!r}; choose one of "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL",
            param_hint="--log-level",
        )

    package_logger = logging.getLogger("regulonado")
    for existing in list(package_logger.handlers):
        package_logger.removeHandler(existing)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    package_logger.addHandler(handler)
    package_logger.propagate = False
    package_logger.setLevel(level)


def _version_callback(value: bool) -> None:
    if value:
        try:
            package_version = version("regulonado")
        except PackageNotFoundError:
            package_version = "unknown"
        try:
            from regulonado import _rs

            rust_version = _rs.version()
        except (ImportError, AttributeError):
            rust_version = "unknown"
        typer.echo(f"regulonado {package_version} (Rust extension {rust_version})")
        raise typer.Exit()


@app.callback()
def _main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show Python package and Rust extension versions.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Log level for the 'regulonado' logger.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Shorthand for --log-level WARNING.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Shorthand for --log-level DEBUG. Takes priority over --quiet.",
    ),
) -> None:
    """Build datasets, train models, run inference, and manage workflows."""
    effective_level = log_level
    if quiet:
        effective_level = "WARNING"
    if verbose:
        effective_level = "DEBUG"
    _configure_logging(effective_level)


def main() -> None:
    app()


app.command()(config)
app.command()(init)
app.command()(train)
app.command()(recompress_dataset)
app.command()(dataset)
app.command()(predict)
app.command()(design)
app.command()(attribute)
app.command("pipeline")(_pipeline)
app.add_typer(normalization_app, name="normalization")
app.add_typer(tracks_app, name="tracks")


if __name__ == "__main__":
    main()
