from dataclasses import dataclass, field
import inspect
import types
from typing import Any, Annotated, Literal, Optional, TypeAlias, Union, get_args, get_origin


NODE_CLASS_MAPPINGS = {}
NODE_NAME_MAPPINGS = {}


@dataclass
class NumRange:
    min: int | float
    max: int | float
    step: int | float


@dataclass
class Name:
    name: str


@dataclass
class TypeName:
    """Annotation for a wrapper around arbitrary types like 'IMAGE'"""
    name: str


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


IMAGE: TypeAlias = Annotated[Any, TypeName("IMAGE")]
SEGS: TypeAlias = Annotated[Any, TypeName("SEGS")]


def process_annotation(a) -> tuple[str | tuple, Optional[str], bool, bool, dict[str, Any]]:
    settings = {}
    type_name = None
    name = None
    is_list = False
    optional = False

    print(f"Processing annotation for {a!r}")

    if get_origin(a) in (Union, types.UnionType):
        args = get_args(a)
        if len(args) != 2 or args[1] is not type(None):
            raise TypeError(f"We only support unions with NoneType, i.e. Optional")

        a = args[0]

    if get_origin(a) is list:
        is_list = True
        a = get_args(a)[0]

    if get_origin(a) is Annotated:
        for arg in get_args(a)[1:]:
            print(f"arg: {arg!r}")
            match arg:
                case "optional":
                    optional = True
                case "dynamicPrompts":
                    settings["dynamicPrompts"] = True
                case "forceInput":
                    settings["forceInput"] = True
                case NumRange(min, max, step):
                    settings.update({
                        "min": min,
                        "max": max,
                        "step": step,
                    })
                case Name(n):
                    name = n
                case TypeName(n):
                    type_name = n

        a = get_args(a)[0]

    if type_name is None:
        print(f"Checking type {a!r}")
        # TODO validate settings
        if a is int:
            type_name = "INT"
        elif a is bool:
            type_name = "BOOLEAN"
        elif a is float:
            type_name = "FLOAT"
        elif a is str:
            type_name = "STRING"
            # Default dynamicPrompts to False
            if settings.get("multiline", False) and not settings.get("dynamicPrompts", True):
                settings["dynamicPrompts"] = False
        elif a is Any:
            type_name = AnyType("*")
        elif get_origin(a) is Literal:
            type_name = get_args(a)
        else:
            raise ValueError(f"Need to provide TypeName annotation for {a!r}, or you annotated the outer type")

    return type_name, name, optional, is_list, settings


def process_return(a) -> tuple[str, str, bool]:
    t, name, optional, is_list, settings = process_annotation(a)

    # No literal returns
    assert not isinstance(t, tuple)

    if settings:
        raise TypeError(f"Unsupported annotation(s) for return types in {t!r} {name!r}: {settings!r}")

    if optional:
        raise TypeError(f"Optional makes no sense for return types in {t!r} {name!r}")

    if name is None:
        name = t

    return t, name, is_list


def node(category, name=None, input_is_list=None):
    def wrapper(func):
        nonlocal input_is_list, name

        if isinstance(func, type):
            # Support callable classes
            sig = inspect.signature(func.__call__)
            d = func.__dict__.copy()
            d["FUNCTION"] = "__call__"
            bases = func.__bases__ # Keep any existing bases
            skip_self = True
        else:
            def doit(self, *args, **kwargs):
                func(*args, **kwargs)

            sig = inspect.signature(func)
            d = {
                "FUNCTION": "doit",
                "doit": doit,
            }
            bases = ()
            skip_self = False

        arg_is_list = False
        required = {}
        optional = {}
        return_types = []
        return_names = []
        output_is_list = []

        ra = sig.return_annotation
        if get_origin(ra) is not tuple:
            # Turn singletons into a tuple to simplify processing
            ra = tuple[ra]
        elif get_origin(ra) is not tuple:
            raise TypeError("Return type of node must be a type or tuple of types.")

        for rt in get_args(ra):
            return_type, return_name, is_list = process_return(rt)
            return_types.append(return_type)
            return_names.append(return_name)
            output_is_list.append(is_list)

        for arg_name, parameter in list(sig.parameters.items())[1 if skip_self else 0:]:
            print(f"arg_name = {arg_name!r}, parameter = {parameter!r}")
            type_name, n, opt, is_list, settings = process_annotation(parameter.annotation)
            if n:
                raise TypeError(f"Don't use Name annotation on input types: {arg_name!r}")

            if is_list:
                if input_is_list is None:
                    input_is_list = True
                    arg_is_list = True
                elif not input_is_list:
                    raise TypeError(f"All inputs must be is_list or none of them, or use second argument of the decorator: {arg_name!r}")
            elif arg_is_list:
                raise TypeError(f"All inputs must be is_list or none of them, or use second argument of the decorator: {arg_name!r}")

            if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
                settings["default"] = parameter.default

            if isinstance(type_name, tuple):
                if settings:
                    raise TypeError("No settings with list of selections: {n!r}")
                o = type_name
            elif settings:
                o = (type_name, settings)
            else:
                o = (type_name,)

            if opt:
                optional[arg_name] = o
            else:
                required[arg_name] = o

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": required,
                "optional": optional,
            }

        d.update({
            "CATEGORY": category,
            "INPUT_TYPES": INPUT_TYPES,
            "INPUT_IS_LIST": input_is_list,
            "RETURN_TYPES": return_types,
            "RETURN_NAMES": return_names,
            "OUTPUT_IS_LIST": output_is_list,
        })
        new_node = type(func.__name__, bases, d)
        if name is None:
            name = new_node.__name__

        NODE_CLASS_MAPPINGS[name] = new_node
        NODE_NAME_MAPPINGS[new_node.__name__] = name
        return new_node

    return wrapper
