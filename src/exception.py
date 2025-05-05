import sys
from src.logger import logging


def error_messgae_detail(error,error_Detail:sys):
    _,_,exc_tb = error_Detail.exc_info()
    error_message = "Error occured in script name [{0}] line number [{1}] error message [{2}]".format(
    str(exc_tb.tb_frame.f_code.co_filename),exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_Detail:sys):
        super().__init__(error_message)
        self.error_message = error_messgae_detail(error_message,error_Detail=error_Detail)

    def __str__(self):
        return self.error_message