import sys
import traceback


def global_exception_hook(exctype, value, tb):
    tb_str = "".join(traceback.format_exception(exctype, value, tb))
    print(f"!!! UNHANDLED EXCEPTION !!!\n{tb_str}")


sys.excepthook = global_exception_hook
