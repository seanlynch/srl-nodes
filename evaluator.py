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

    return getattr(obj, attr)


SAFE_BUILTINS = {
    "abs": abs,
    "bytes": bytes,
    "divmod": divmod,
    "format": format,
    "float": float,
    "getattr": safe_getattr,
    "int": int,
    "str": str,
}


class Safifier:
    """Evaluate an expression in a restricted subset of Python by walking the AST."""
    safe_nodes = (
        ast.arg,
        ast.arguments,
        ast.Add,
        ast.BinOp,
        ast.BoolOp,
        ast.Call,
        ast.Compare,
        ast.Constant,
        ast.Dict,
        ast.DictComp,
        ast.Expression,
        ast.FormattedValue,
        ast.GeneratorExp,
        ast.IfExp,
        ast.JoinedStr,
        ast.Lambda,
        ast.List,
        ast.ListComp,
        ast.Load,
        ast.Name,
        ast.Set,
        ast.SetComp,
        ast.Slice,
        ast.Subscript,
        ast.Tuple,
        ast.UnaryOp,
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

        ast.fix_missing_locations(new_node)


def safe_compile(expr: str):
    node = ast.parse(expr, mode="eval")
    safifier = Safifier()
    safe_node = safifier.make_safe(node)
    ast.fix_missing_locations(safe_node)
    return compile(safe_node, "<string>", "eval")


def evaluate(expr: str, local_symbols: dict[str, Any] = {}) -> Any:
    code = safe_compile(expr)
    global_symbols = {
        "__builtins__": SAFE_BUILTINS,
    }
    return eval(code, global_symbols, local_symbols)


class SafeFormatter(string.Formatter):
    field_pat = re.compile(r"([^\.\]]*)\.(.+)|\[(\d+)\]")
    def get_field(self, field_name, args, kwargs):
        m = self.field_pat.fullmatch(field_name)
        if m is None:
            raise ValueError(f"Could not parse field name {field_name!r}")

        key, attr, idx = m.groups()
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
