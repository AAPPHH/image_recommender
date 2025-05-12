import logging
from pathlib import Path


def setup_logging(log_file: str, log_dir: str = "logs"):
    """
    Initialisiert das Logging und schreibt das Logfile in den Unterordner `log_dir`.
    Erzeugt das Verzeichnis bei Bedarf.
    """
    # Ordner anlegen, falls nicht vorhanden
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    full_path = Path(log_dir) / log_file

    logging.basicConfig(
        level=logging.INFO,
        filename=str(full_path),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
    )


def console_and_log(message: str, level: str = "info"):
    """
    Druckt die Nachricht und loggt sie mit dem angegebenen Level.
    """
    print(message)
    lvl = level.lower()
    if lvl == "info":
        logging.info(message)
    elif lvl == "warning":
        logging.warning(message)
    elif lvl == "error":
        logging.error(message)
    else:
        logging.debug(message)
