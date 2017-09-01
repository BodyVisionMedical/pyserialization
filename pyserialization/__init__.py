from pathlib import Path
from typing import Union

from .simple import Serializable
from .simple import decode
from .simple import encode
from .simple import is_object_serializable
from .simple import to_dict
from .simple import from_dict


def save(obj, path_file: Union[str, Path], allow_implicit_simples=False):
    path_file = Path(path_file)
    path_dir = path_file.parent  # type: Path
    path_dir.mkdir(parents=True, exist_ok=True)
    data = encode(obj, allow_implicit_simples)
    with path_file.open(mode="w") as text_file:
        text_file.write(data)


def load(path_file: Union[str, Path]):
    path_file = Path(path_file)
    with path_file.open(mode="r") as text_file:
        return decode(text_file.read())
