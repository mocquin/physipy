def timeraw_import_units():
    return """from physipy import units"""


def timeraw_import_constants():
    return r"from physipy import constants"


def timeraw_import_math():
    return r"from physipy import math"


def timeraw_import_physipy():
    return r"import physipy"


def timeraw_import_physipy_calculus():
    return r"from physipy import calculus"


def timeraw_import_physipy_dimension():
    return r"from physipy.quantity import dimension"


def timeraw_import_physipy_plot():
    return r"from physipy.quantity import _plot"


def timeraw_import_physipy_quantity():
    return r"from physipy.quantity import quantity"


def timeraw_import_physipy_units():
    return r"from physipy.quantity import units"


def timeraw_import_physipy_utils():
    return r"from physipy.quantity import utils"
