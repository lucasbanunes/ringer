import logging
from typing import Sequence


class CSVFormatter(logging.Formatter):
    """Formats message as a csv file row"""
    def __init__(self,
                 cols: Sequence[str],
                 extra_records: bool,
                 sep: str = ","):
        # self.first_row = True
        # self.cols = list(cols)
        # self.extra_records = extra_records
        # self.sep = sep
        raise NotImplementedError

    def format(record: logging.LogRecord) -> str:
        raise NotImplementedError

    def formatException(exc_info):
        raise NotImplementedError

    def formatStack(stack_info):
        raise NotImplementedError
