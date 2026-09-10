"""Error reporting that tolerates exceptions raised by device SDK formatters."""


def exception_message(error: BaseException) -> str:
    """Preserve ordinary messages without letting a broken formatter escape."""
    try:
        return str(error)
    except Exception:
        return "{}: exception message is unavailable".format(type(error).__name__)
