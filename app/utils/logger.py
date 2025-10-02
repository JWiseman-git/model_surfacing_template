import logging

def setup_logging(level=logging.INFO):
    """
        Configure application-wide logging.

        Usage:
            - In your entrypoint script (e.g. main.py), call this once:
                setup_logging()

            - In any module where you want to log messages, create a logger:
                logger = logging.getLogger(__name__)

        Arguments:
            level (int): The logging level. Defaults to logging.INFO.
        """

    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
