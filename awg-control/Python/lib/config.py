from __future__ import annotations
from math import inf
from pathlib import Path
import toml
from typing import Any, Iterable

class Range:
    def __init__(
        self,
        start: int | float,
        end: int | float,
        incl_start: bool=True,
        incl_end: bool=False
    ):
        self.start = start
        self.end = end
        self.incl_start = incl_start
        self.incl_end = incl_end

    def __contains__(self, X: int | float) -> bool:
        in_start = (X >= self.start) if self.incl_start else (X > self.start)
        in_end = (X <= self.end) if self.incl_end else (X < self.end)
        return in_start and in_end

    def __str__(self) -> str:
        return (
            ("[" if self.incl_start else "(")
            + str(self.start)
            + ", "
            + str(self.end)
            + ("]" if self.incl_end else ")")
        )

class TypedList:
    def __init__(self, types: list, finite: bool=False):
        if len(types) < 1:
            raise ValueError("TypedList: must provide at least one type")
        if not all(isinstance(t, (type, tuple)) for t in types):
            raise ValueError("TypedList: contained values must be types")
        self.types = types
        self.finite = finite

    def verify(self, values: list) -> bool:
        if not isinstance(values, list):
            raise ValueError("TypedList.verify: values must be in a list")
        if self.finite:
            if len(self.types) != len(values):
                raise ValueError("TypedList.verify: incorrect number of values")
            return all(isinstance(v, t) for v, t in zip(values, self.types))
        else:
            return all(type(v) in self.types for v in values)

    def __str__(self) -> str:
        return (
            "typedlist["
            + ", ".join(t.__name__ for t in self.types)
            + f" finite={self.finite}]"
        )

class Config:
    def __init__(self, data: dict[str, ...], verify: dict[str, ...] = None):
        if verify is not None:
            self._data = verify_config(data, verify)
        else:
            self._data = data

    def __getitem__(self, key: str | list[str]) -> ...:
        if isinstance(key, str):
            return dict_access_path(self._data, key.split("."))
        elif isinstance(key, list):
            if not all(isinstance(k, str) for k in key):
                raise KeyError("Config.__getitem__: all keys must be strs")
            return dict_access_path(self._data, key)
        else:
            return KeyError("Config.__getitem__: all keys must be strs")

    def get(self, key: str | list[str], default=None) -> ...:
        if isinstance(key, str):
            return dict_get_path(self._data, key.split("."), default)
        elif isinstance(key, list):
            if not all(isinstance(k, str) for k in key):
                raise KeyError("Config.get: all keys must be strs")
            return dict_Get_path(self._data, key, default)
        else:
            return KeyError("Config.get: all keys must be strs")

    def __getattr__(self, attr: str) -> ...:
        if attr in self._data.keys():
            if isinstance(self._data[attr], dict):
                return Config(self._data[attr], verify=None)
            else:
                return self._data[attr]
        else:
            raise AttributeError(f"invalid attribute {attr}")

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def values(self) -> Iterable[...]:
        return self._data.values()

    def items(self) -> Iterable[tuple[str, ...]]:
        return self._data.items()

    def __eq__(self, other: Config) -> bool:
        return self._data == other._data

    def __neq__(self, other: Config) -> bool:
        return not (self == other)

    def __str__(self) -> str:
        return f"Config({self._data})"

Number = (int, float)
VALUE_TYPES = {bool, int, float, Number, str, list, Any}
VALUE_SETS = {set, Range}

DEF_CONFIG_VER = {
    "awg": {
        "id": {0,1},
        "channel": {
            "index": TypedList([int, int, int, int], finite=True),
            "amplitude": TypedList(
                [
                    int,
                    int,
                    int,
                    int,
                ], 
                finite=True
                ),
        },
        "sequence": {
            "partition": int,
            "segment_chain": TypedList([int], finite=False),
            "nloop": TypedList([int], finite=False),
            "endloop": TypedList([str], finite=False),
        },
    },
    "infile": {
        "paths": TypedList([str], finite=False),
        # "channels": TypedList([int], finite=False),
        "segments": TypedList([int], finite=False)
    },
    "waveform": {
        "enabled": bool,
        "from_infile": bool,
        "params": {
            "f_start": float,
            "df": float,
            "nt": int,
            "sample_rate": float,
            "freq_res": float,
            "amplitude": int,
        },
    },
}

def load_config(infile: Path, verify: dict[str, ...]=None) -> Config:
    """
    Load config options from `infile`. Expected in TOML format.
    """
    return Config(toml.load(infile), verify)

def load_def_config(infile: Path) -> Config:
    """
    Load config options from `infile` and verify default config options.
    Expected in TOML format.
    """
    return Config(toml.load(infile), verify=DEF_CONFIG_VER)

def verify_config(data: dict, verify: dict) -> dict:
    f"""
    Recursively verify the values in `data` (including types) according to the
    specification in `verify`. The keys of `verify` should be mapped to:
        - single type classes
            {VALUE_TYPES}
        - non-list container objects with a `__contains__` method
            {VALUE_SETS}
        - a TypedList
    Keys in `data` that are not contained in `verify` are not checked, but
    KeyError is raised if a key in `verify` is not contained in `data`.
    ValueError is raised for any non-matches.
    """
    if not isinstance(data, dict):
        print(data, verify)
        raise ValueError(
            "verify_config: encountered non-matching top-level structure"
        )
    for type_key, type_value in verify.items():
        if type_key not in data.keys():
            raise KeyError(f"verify_config: data is missing key '{type_key}'")
        if isinstance(type_value, dict):
            verify_config(data[type_key], type_value)
        elif type_value in VALUE_TYPES:
            if type_value == Any:
                continue
            if not isinstance(data[type_key], type_value):
                raise ValueError(
                    f"""
verify_config: invalid type for key '{type_key}'
Expected '{type_value.__name__}' but got '{type(data[type_key]).__name__}'
                    """
                )
        elif type_value in VALUE_SETS or type(type_value) in VALUE_SETS:
            if not data[type_key] in type_value:
                raise ValueError(
                    f"""
verify_config: invalid value encountered for key '{type_key}'
Expected a value in {str(type_value)} but got value {data[type_key]}
                    """
                )
        elif isinstance(type_value, TypedList):
            if not type_value.verify(data[type_key]):
                raise ValueError(
                    f"""
verify_config: invalid types encountered in list for key '{type_key}'
Expected {str(type_value)} but got value {data[type_key]}
                    """
                )
        else:
            raise ValueError("verify_config: encountered invalid specification")
    return data

def dict_access_path(D: dict, path: list[...]):
    """
    Attempts to recursively descent through a dict-like tree structure along a
    path given as a list-like of keys. Raises `KeyError` if any key in the path
    is not found in its appropriate parent node.
    """
    if len(path) <= 0:
        raise ValueError("dict_access_path: empty key path")
    try:
        d = D[path[0]]
    except KeyError:
        raise KeyError(f"invalid key: {path[0]}")
    if len(path) == 1:
        return d
    elif not isinstance(d, dict) or path[0] not in D.keys():
        err = f"""
dict_access_path: descent terminated before the end of the path was reached
Terminating value: {d}
Path remaining: {path}
        """
        raise KeyError(err)
    else:
        return dict_get_path(d, path[1:])

def dict_get_path(D: dict, path: list[...], default=None):
    """
    Attempts to recursively descend through a dict-like tree structure along a
    path given as a list-like of keys. Returns `default` or raises `KeyError` if
    the final or other keys are not found in their appropriate parent nodes,
    respectively.
    """
    assert len(path) > 0
    d = D.get(path[0], default)
    if len(path) == 1:
        return d
    elif not isinstance(d, dict) or path[0] not in D.keys():
        err = f"""
dict_get_path: descent terminated before the end of the path was reached
Terminating value: {d}
Path remaining: {path}
        """
        raise KeyError(err)
    else:
        return dict_get_path(d, path[1:], default)

