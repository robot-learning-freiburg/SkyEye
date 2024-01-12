import ast
import configparser
import os

_CONVERTERS = {
    "struct": ast.literal_eval
}

# _DEFAULTS_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], "defaults"))
# DEFAULTS = dict()
# if os.path.exists(_DEFAULTS_DIR):
#     for file in os.listdir(_DEFAULTS_DIR):
#         name, ext = os.path.splitext(file)
#         if ext == ".ini":
#             DEFAULTS[name] = os.path.join(_DEFAULTS_DIR, file)
# else:
#     print("Default config not found. It is assumed that all the parameters are in the provided config file.")


def load_config(config_file, defaults_file=None):
    parser = configparser.ConfigParser(allow_no_value=True, converters=_CONVERTERS)
    if defaults_file is not None:
        parser.read([defaults_file, config_file])
    else:
        parser.read([config_file])
    return parser
