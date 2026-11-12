"""Error reporting that tolerates exceptions raised by device SDK formatters."""


def exception_message(error: BaseException) -> str:
    """Preserve ordinary messages without letting a broken formatter escape."""
    try:
        return str(error)
    except Exception:
        return "{}: exception message is unavailable".format(type(error).__name__)


def exception_description(error: BaseException) -> str:
    """Include the exception type once, even when its SDK formatter is broken."""
    try:
        message = str(error)
    except Exception:
        message = "exception message is unavailable"
    return "{}: {}".format(type(error).__name__, message)
