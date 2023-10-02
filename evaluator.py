"""Evaluate Python expressions safely."""

import ast
import inspect
from operator import getitem
import re
import string
from typing import Any, Generator, Optional, Sequence


def safe_vformat(fmt: str, args: Sequence = [], kwargs: dict = {}) -> str:
    return FORMATTER.vformat(fmt, args, kwargs)


def safe_format(fmt: str, *args, **kwargs) -> str:
    return safe_vformat(fmt, args, kwargs)


def safe_getattr(obj: Any, attr: str) -> Any:
    if attr.startswith("_"):
        raise ValueError(f"Not allowed to access attributes starting with underscore.")

    value = getattr(obj, attr)
    if callable(value):
        if inspect.isbuiltin(value):
            if value.__qualname__ not in SAFE_BUILTIN_METHODS:
                raise ValueError(
                    f"Sorry, {value.__qualname__!r} is not on the list of allowed builtin methods."
                )
        else:
            raise ValueError(f"Sorry, {value!r} is not on the list of allowed methods.")

    return value


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": safe_getattr,
    # TODO hasattr
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


SAFE_BUILTIN_METHODS = set(
    [
        "complex.conjugate",
        "dict.iter",
        "dict.get",
        "dict.items",
        "dict.keys",
        "dict.reversed",
        "dict.values",
        "float.as_integer_ratio",
        "float.is_integer",
        "float.hex",
        "float.fromhex",
        "int.bit_length",
        "int.bit_count",
        "int.to_bytes",
        "int.from_bytes",
        "int.as_integer_ratio",
        "list.copy",
        "set.difference",
        "set.intersection",
        "set.isdisjoint",
        "set.issubset",
        "set.issuperset",
        "set.symmetric_difference",
        "set.union",
        "str.capitalize",
        "str.casefold",
        "str.center",
        "str.count",
        "str.encode",
        "str.endswith",
        "str.expandtabs",
        "str.find",
        "str.index",
        "str.isalnum",
        "str.isalpha",
        "str.isascii",
        "str.isdigit",
        "str.isidentifier",
        "str.islower",
        "str.isnumeric",
        "str.isprintable",
        "str.isspace",
        "str.istitle",
        "str.isupper",
        "str.join",
        "str.ljust",
        "str.lower",
        "str.lstrip",
        "str.maketrans",
        "str.partition",
        "str.removeprefix",
        "str.removesuffix",
        "str.replace",
        "str.rfind",
        "str.rindex",
        "str.rjust",
        "str.rpartition",
        "str.rstrip",
        "str.split",
        "str.splitlines",
        "str.startswith",
        "str.strip",
        "str.swapcase",
        "str.title",
        "str.translate",
        "str.upper",
        "str.zfill",
    ]
)


class Safifier:
    """Evaluate an expression in a restricted subset of Python by walking the AST."""

    safe_nodes = (
        ast.Add,
        ast.And,
        ast.arg,
        ast.arguments,
        ast.BinOp,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.BoolOp,
        ast.Call,
        ast.Compare,
        ast.comprehension,
        ast.Constant,
        ast.Dict,
        ast.DictComp,
        ast.Div,
        ast.Eq,
        ast.Expression,
        ast.FloorDiv,
        ast.FormattedValue,
        ast.GeneratorExp,
        ast.Gt,
        ast.GtE,
        ast.IfExp,
        ast.In,
        ast.Invert,
        ast.Is,
        ast.IsNot,
        ast.JoinedStr,
        ast.keyword,
        ast.Lambda,
        ast.List,
        ast.ListComp,
        ast.Load,
        ast.LShift,
        ast.MatMult,
        ast.Mod,
        ast.Mult,
        ast.Not,
        ast.NotEq,
        ast.NotIn,
        ast.Name,
        ast.Or,
        ast.Pass,
        ast.Pow,
        ast.RShift,
        ast.Set,
        ast.SetComp,
        ast.Slice,
        ast.Starred,
        ast.Sub,
        ast.Subscript,
        ast.Tuple,
        ast.UAdd,
        ast.UnaryOp,
        ast.USub,
    )

    def make_field_safe(self, field):
        if field is None or isinstance(field, (int, str)):
            return field

        if isinstance(field, ast.AST):
            return self.make_safe(field)

        if isinstance(field, list):
            return [self.make_safe(node) for node in field]

        raise NotImplementedError(f"Field {field!r} not supported.")

    def make_safe(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Attribute):
            assert isinstance(node.ctx, ast.Load)
            value = self.make_safe(node.value)
            return ast.Call(
                func=ast.Name(id="getattr", ctx=ast.Load()),
                args=[
                    value,
                    ast.Constant(value=node.attr),
                ],
                keywords=[],
            )
        elif isinstance(node, self.safe_nodes):
            fields = [self.make_field_safe(f) for _, f in ast.iter_fields(node)]
            return node.__class__(*fields)
        else:
            raise NotImplementedError(f"Node {node!r} is not supported.")


def safe_compile(expr: str):
    node = ast.parse(expr, mode="eval")
    safifier = Safifier()
    safe_node = safifier.make_safe(node)
    ast.fix_missing_locations(safe_node)
    return compile(safe_node, "<string>", "eval")


def safe_eval(code, local_symbols: dict[str, Any] = {}) -> Any:
    global_symbols = {
        "__builtins__": SAFE_BUILTINS,
    }
    return eval(code, global_symbols, local_symbols)


def evaluate(expr: str, local_symbols: dict[str, Any] = {}) -> Any:
    code = safe_compile(expr)
    return safe_eval(code, local_symbols)


class SafeFormatter(string.Formatter):
    field_pat = re.compile(r"([^\.\]]*)(?:\.(.+)|\[(\d+)\])?")

    def get_field(self, field_name, args, kwargs):
        m = self.field_pat.fullmatch(field_name)
        if m is None:
            raise ValueError(f"Could not parse field name {field_name!r}")

        key, attr, idx = m.groups()
        if key[0].isdigit():
            key = int(key)
        value = self.get_value(key, args, kwargs)
        if attr is not None:
            return safe_getattr(value, attr), key

        if idx is not None:
            return getitem(value, int(idx)), key

        return value, key


FORMATTER = SafeFormatter()


def main():
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("expression")
    args = p.parse_args()
    result = evaluate(args.expression, {})
    print(f"result: {result!r}")


if __name__ == "__main__":
    main()
