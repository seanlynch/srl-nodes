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


GLOBAL_SYMBOLS = {
    "abs": abs,
    "bytes": bytes,
    "divmod": divmod,
    "format": format,
    "float": float,
    "int": int,
    "safe_format": safe_format,
    "safe_vformat": safe_vformat,
    "str": str,
}


SAFE_FUNCTIONS = set(GLOBAL_SYMBOLS.values())


class EvalCtx:
    """Context for the evaluator."""

    def __init__(self, symbols: dict[str, Any], parent: Optional["EvalCtx"] = None):
        self.symbols = symbols
        self.parent = parent

    def lookup_symbol(self, name: str):
        try:
            return self.symbols[name]
        except KeyError:
            if self.parent is None:
                raise NameError(f"name {name!r} is not defined")
            return self.parent.lookup_symbol(name)

    def push_symbols(self, symbols: dict[str, Any]):
        return self.__class__(symbols, self)


GLOBAL_CTX = EvalCtx(GLOBAL_SYMBOLS)


class Lambda:
    def __init__(self, args: ast.arguments, body: ast.AST, evaluator: "Evaluator", ctx: EvalCtx):
        num_non_defaults = len(args.posonlyargs) + len(args.args) - len(args.defaults)
        defaults = {}
        for a, d in zip((args.posonlyargs + args.args)[num_non_defaults:], args.defaults):
            defaults[a.arg] = evaluator.evaluate(d, ctx)

        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            if d is not None:
                defaults[a.arg] = evaluator.evaluate(d, ctx)

        parameters = []
        for a in args.posonlyargs:
            p = inspect.Parameter(
                a.arg,
                inspect.Parameter.POSITIONAL_ONLY,
                default=defaults.get(a.arg, inspect.Parameter.empty),
            )
            parameters.append(p)

        for a in args.args:
            p = inspect.Parameter(
                a.arg,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults.get(a.arg, inspect.Parameter.empty),
            )
            parameters.append(p)

        if args.vararg is not None:
            parameters.append(inspect.Parameter(
                args.vararg.arg,
                inspect.Parameter.VAR_POSITIONAL,
            ))

        for a in args.kwonlyargs:
            p = inspect.Parameter(
                a.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=defaults.get(a.arg, inspect.Parameter.empty),
            )
            parameters.append(p)

        if args.kwarg is not None:
            parameters.append(inspect.Parameter(
                args.kwarg.arg,
                inspect.Parameter.VAR_KEYWORD,
            ))

        self.signature = inspect.Signature(parameters)
        self.body = body
        self.evaluator = evaluator
        self.ctx = ctx
        self.defaults = [evaluator.evaluate(d, ctx) for d in args.defaults]

    def __call__(self, *args, **kwargs):
        ba = self.signature.bind(*args, **kwargs)
        ba.apply_defaults()
        ctx = self.ctx.push_symbols(ba.arguments)
        return self.evaluator.evaluate(self.body, ctx)


class Evaluator:
    """Evaluate an expression in a restricted subset of Python by walking the AST."""

    def comprehend(self, generators: list[ast.comprehension], ctx: EvalCtx) -> Generator[EvalCtx, None, None]:
        comp = generators[0]
        assert not comp.is_async
        it = self.evaluate(comp.iter, ctx)
        t = comp.target
        for i in it:
            symbols = {}
            if isinstance(t, ast.Name):
                symbols[t.id] = i
            elif isinstance(t, ast.Tuple):
                for elt, value in zip(t.elts, i):
                    assert isinstance(elt, ast.Name)
                    symbols[elt.id] = value

            new_ctx = ctx.push_symbols(symbols)
            for ifexp in comp.ifs:
                if not self.evaluate(ifexp, new_ctx):
                    break
            else:
                if len(generators) > 1:
                    for j in self.comprehend(generators[1:], new_ctx):
                        yield j
                else:
                    yield new_ctx

    def evaluate(self, node: ast.AST, ctx: EvalCtx) -> Any:
        meth = getattr(self, f"eval_{node.__class__.__name__}", None)
        if meth is None:
            raise NotImplementedError(f"Not implemented: {node!r}")

        return meth(node, ctx)

    def eval_Attribute(self, node: ast.Attribute, ctx: EvalCtx):
        assert isinstance(node.ctx, ast.Load)
        obj = self.evaluate(node.value, ctx)
        return safe_getattr(obj, node.attr)

    def eval_BinOp(self, node, ctx: EvalCtx) -> Any:
        left = self.evaluate(node.left, ctx)
        right = self.evaluate(node.right, ctx)
        match type(node.op):
            case ast.Add:
                return left + right
            case ast.BitAnd:
                return left & right
            case ast.BitOr:
                return left | right
            case ast.BitXor:
                return left ^ right
            case ast.Div:
                return left / right
            case ast.FloorDiv:
                return left // right
            case ast.In:
                return left in right
            case ast.Is:
                return left is right
            case ast.IsNot:
                return left is not right
            case ast.LShift:
                return left << right
            case ast.Mod:
                return left % right
            case ast.NotIn:
                return left not in right
            case ast.Pow:
                return left ** right
            case _:
                raise NotImplementedError(f"BinOp {node.op} is not supported.")

    def eval_BoolOp(self, node: ast.BoolOp, ctx: EvalCtx) -> bool:
        match type(node.op):
            case ast.And:
                for value in node.values:
                    if not self.evaluate(value, ctx):
                        # Short circuit evaluation means we stop as soon
                        # as we get a false value.
                        return False
                return True
            case ast.Or:
                for value in node.values:
                    if self.evaluate(value, ctx):
                        # Stop as soon as we get a true value.
                        return True
                return False
            case _:
                raise NotImplementedError(f"Unimplemented boolean operation {node.op!r}")

    def eval_Call(self, node: ast.Call, ctx: EvalCtx) -> Any:
        func = self.evaluate(node.func, ctx)
        # Only allow calling of Lambdas and functions that are known to be safe
        # TODO need to support methods as well
        if not isinstance(func, Lambda) and func not in SAFE_FUNCTIONS:
            raise ValueError(f"Not allowed to call {func!r}")

        args = [self.evaluate(arg, ctx) for arg in node.args]
        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = self.evaluate(kw.value, ctx)

        return func(*args, **kwargs)

    def eval_Compare(self, node: ast.Compare, ctx: EvalCtx) -> bool:
        left = self.evaluate(node.left, ctx)
        for op, c in zip(node.ops, node.comparators):
            right = self.evaluate(c, ctx)
            match type(op):
                case ast.Eq:
                    res = left == right
                case ast.Gt:
                    res = left > right
                case ast.GtE:
                    res = left >= right
                case ast.Lt:
                    res = left < right
                case ast.LtE:
                    res = left <= right
                case ast.NotEq:
                    res = left != right
                case _:
                    raise NotImplementedError(f"Compare op {op!r} is not supported.")
            if not res:
                return False

            left = right
        return True

    def eval_Constant(self, node: ast.Constant, ctx: EvalCtx) -> Any:
        return node.value

    def eval_Dict(self, node: ast.Dict, ctx: EvalCtx) -> dict:
        r = {}
        for k, v in zip(node.keys, node.values):
            value = self.evaluate(v, ctx)
            if k is None:
                r.update(value)
            else:
                key = self.evaluate(k, ctx)
                r[key] = value

        return r

    def eval_DictComp(self, node: ast.DictComp, ctx: EvalCtx) -> dict:
        return {
            self.evaluate(node.key, c): self.evaluate(node.value, c)
            for c in self.comprehend(node.generators, ctx)
        }

    def eval_Expression(self, node: ast.Expression, ctx: EvalCtx) -> Any:
        return self.evaluate(node.body, ctx)

    def eval_FormattedValue(self, node: ast.FormattedValue, ctx: EvalCtx) -> str:
        value = self.evaluate(node.value, ctx)
        if node.conversion != -1:
            value = FORMATTER.convert_field(value, chr(node.conversion))

        format_spec = "" if node.format_spec is None else self.evaluate(node.format_spec, ctx)
        return FORMATTER.format_field(value, format_spec)

    def eval_GeneratorExp(self, node: ast.GeneratorExp, ctx: EvalCtx):
        return (self.evaluate(node.elt, c) for c in self.comprehend(node.generators, ctx))

    def eval_IfExp(self, node: ast.IfExp, ctx: EvalCtx) -> Any:
        test = self.evaluate(node.test, ctx)
        if test:
            return self.evaluate(node.body, ctx)
        else:
            return self.evaluate(node.orelse, ctx)

    def eval_JoinedStr(self, node: ast.JoinedStr, ctx: EvalCtx) -> str:
        return "".join(self.evaluate(v, ctx) for v in node.values)

    def eval_Lambda(self, node: ast.Lambda, ctx: EvalCtx) -> Lambda:
        return Lambda(node.args, node.body, self, ctx)

    def eval_List(self, node: ast.List, ctx: EvalCtx) -> list:
        assert isinstance(node.ctx, ast.Load)
        return [self.evaluate(elt, ctx) for elt in node.elts]

    def eval_ListComp(self, node: ast.ListComp, ctx: EvalCtx) -> list:
        return [self.evaluate(node.elt, c) for c in self.comprehend(node.generators, ctx)]

    def eval_Name(self, node: ast.Name, ctx: EvalCtx) -> Any:
        assert isinstance(node.ctx, ast.Load)
        return ctx.lookup_symbol(node.id)

    def eval_Set(self, node: ast.Set, ctx: EvalCtx) -> set:
        return set(self.evaluate(elt, ctx) for elt in node.elts)

    def eval_SetComp(self, node: ast.SetComp, ctx: EvalCtx) -> set:
        return set(self.evaluate(node.elt, c) for c in self.comprehend(node.generators, ctx))

    def eval_Slice(self, node: ast.Slice, ctx: EvalCtx) -> slice:
        lower = None if node.lower is None else self.evaluate(node.lower, ctx)
        upper = None if node.upper is None else self.evaluate(node.upper, ctx)
        step = None if node.step is None else self.evaluate(node.step, ctx)
        return slice(lower, upper, step)

    def eval_Subscript(self, node: ast.Subscript, ctx: EvalCtx) -> slice:
        value = self.evaluate(node.value, ctx)
        s = self.evaluate(node.slice, ctx)
        # TODO see if this is actually safe
        return getitem(value, s)

    def eval_Tuple(self, node: ast.Tuple, ctx: EvalCtx) -> tuple:
        assert isinstance(node.ctx, ast.Load)
        return tuple(self.evaluate(elt, ctx) for elt in node.elts)

    def eval_UnaryOp(self, node: ast.UnaryOp, ctx: EvalCtx) -> Any:
        v = self.evaluate(node, ctx)
        match type(node.op):
            case ast.Invert:
                return ~v
            case ast.Not:
                return not v
            case ast.UAdd:
                return +v
            case ast.USub:
                return -v
            case _:
                raise NotImplementedError(f"Unary operation {node.op} is not supported.")


def evaluate(expr: str, symbols: dict[str, Any] = {}) -> Any:
    node = ast.parse(expr, mode="eval")
    assert isinstance(node, ast.Expression)
    evaluator = Evaluator()
    return evaluator.evaluate(node, EvalCtx(symbols, GLOBAL_CTX))


def safe_getattr(obj: Any, attr: str) -> Any:
    if attr.startswith("_"):
        raise ValueError(f"Not allowed to access attributes starting with underscore.")

    return getattr(obj, attr)


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
