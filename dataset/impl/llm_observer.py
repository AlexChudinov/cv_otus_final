import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def observer(url: str):
    _logger.info("Notification observer with url %s", url)
