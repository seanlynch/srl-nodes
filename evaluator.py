"""Evaluate Python expressions safely."""

import ast
from operator import getitem
from typing import Any


class Visitor(ast.NodeVisitor):
    # String format conversion numbers
    CONV = {
        -1: "",
        115: "!s",
        114: "!r",
        97: "!a",
    }

    def __init__(self, symbols: dict[str, Any]):
        self.indent = 0
        self.stack = []
        self.symbols = symbols
        super().__init__()

    def comprehend(self, generators: list[ast.comprehension]):
        """Set the symbol table based on the comprehensions and then yield."""
        comp = generators[0]
        assert not comp.is_async
        it = self.evaluate(comp.iter)
        t = comp.target
        for i in it:
            symbols = {}
            if isinstance(t, ast.Name):
                symbols[t.id] = i
            elif isinstance(t, ast.Tuple):
                for elt, value in zip(t.elts, i):
                    assert isinstance(elt, ast.Name)
                    symbols[elt.id] = value

            old_symbols = self.push_symbols(symbols)
            for ifexp in comp.ifs:
                if not self.evaluate(ifexp, symbols):
                    break
            else:
                if len(generators) > 1:
                    for j in self.comprehend(generators[1:]):
                        yield j
                else:
                    yield None

            self.symbols = old_symbols

    def evaluate(self, node):
        self.visit(node)
        return self.stack.pop()

    def eval_elts(self, elts):
        for elt in elts:
            self.visit(elt)
            yield self.stack.pop()

    def generate(self, node):
        """Implement list and set comprehensions and generator expressions."""
        for i in self.comprehend(node.generators):
            yield self.evaluate(node.elt)

    def make_lambda(self, args: ast.arguments, body: ast.AST):
        assert not args.posonlyargs
        assert not args.kwonlyargs
        defaults = [self.evaluate(d) for d in args.defaults]
        def func(*a, **kw):
            symbols = {arg.arg: v for arg, v in zip(args.args, a)}
            rest = args.args[len(a):]
            for i, arg in enumerate(rest, len(defaults)-len(rest)):
                if arg.arg in kw:
                    symbols[arg.arg] = kw[arg.arg]
                    del kw[arg.arg]
                elif i < 0:
                    symbols[arg.arg] = defaults[i]
                else:
                    raise ArugumentError(f"Failed to provide argument for {arg.arg}")

            old_symbols = self.push_symbols(symbols)
            print(ast.dump(body))
            self.visit(body)
            self.symbols = old_symbols
            return self.stack.pop()

        return func

    def push_symbols(self, symbols: dict[str, Any]) -> dict[str, Any]:
        print(f"push_symbols: {symbols!r}")
        old_symbols = self.symbols
        self.symbols = {**old_symbols, **symbols}
        return old_symbols

    def visit_Add(self, node):
        self.stack[-2] += self.stack[-1]

    def visit_Attribute(self, node):
        assert isinstance(self.ctx, ast.Load)

        # Need to be very careful with this one
        raise NotImplementedError("Attribute lookup is not yet implemented.")

    def visit_BinOp(self, node):
        # Visit in reverse polish order
        self.visit(node.left)
        self.visit(node.right)
        self.visit(node.op)
        # Pop the second operand off the stack so the ops don't have to
        self.stack.pop()

    def visit_BitAnd(self, node):
        self.stack[-2] &= self.stack[-1]

    def visit_BitOr(self, node):
        self.stack[-2] |= self.stack[-1]

    def visit_BitXor(self, node):
        self.stack[-2] ^= self.stack[-1]

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            for value in node.values:
                self.visit(value)
                if not self.stack[-1]:
                    # Short circuit evaluation means we stop as soon
                    # as we get a false value.
                    break

                self.stack.pop()
            else:
                self.stack.append(True)
        elif isinstance(node.op, ast.Or):
            for value in node.values:
                self.visit(value)
                if self.stack[-1]:
                    # Stop as soon as we get a true value.
                    break

                self.stack.pop()
            else:
                self.stack.append(False)
        else:
            raise NotImplementedError(f"Unimplemented boolean operation {node.op!r}")

    def visit_Call(self, node):
        func = self.evaluate(node.func)
        args = list(self.eval_elts(node.args))
        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = self.evaluate(kw.value)

        self.stack.append(func(*args, **kwargs))

    def visit_Compare(self, node):
        self.visit(node.left)
        for op, c in zip(node.ops, node.comparators):
            self.visit(c)
            self.visit(op)
            if not self.stack.pop(-2):
                self.stack[-1] = False
                break
        else:
            self.stack[-1] = True

    def visit_Constant(self, node):
        self.stack.append(node.value)

    def visit_Dict(self, node):
        r = {}
        for k, v in zip(node.keys, node.values):
            self.visit(v)
            value = self.stack.pop()
            if k is None:
                r.update(value)
            else:
                self.visit(k)
                r[self.stack.pop()] = value

        self.stack.append(r)

    def visit_DictComp(self, node):
        r = {self.evaluate(node.key): self.evaluate(node.value) for i in self.comprehend(node.generators)}
        self.stack.append(r)

    def visit_Div(self, node):
        self.stack[-2] /= self.stack[-1]

    def visit_Eq(self, node):
        self.stack[-2] = self.stack[-2] == self.stack[-1]

    def visit_Expression(self, node):
        assert len(self.stack) == 0
        self.visit(node.body)
        print(self.stack)
        assert len(self.stack) == 1

    def visit_FloorDiv(self, node):
        self.stack[-2] //= self.stack[-1]

    def visit_FormattedValue(self, node):
        self.visit(node.value)
        value = self.stack.pop()
        conv = self.CONV[node.conversion]
        if node.format_spec is None:
            format_spec = ""
        else:
            self.visit(node.format_spec)
            format_spec = f":{self.stack[-1]}"

        # Cheat and use str.format() rather than trying to reimplement formatting
        self.stack[-1] = f"{{{conv}{format_spec}}}".format(value)

    def visit_GeneratorExp(self, node):
        self.stack.append(self.generate(node))

    def visit_GtE(self, node):
        self.stack[-2] = self.stack[-2] >= self.stack[-1]

    def visit_Gt(self, node):
        self.stack[-2] = self.stack[-2] > self.stack[-1]

    def visit_IfExp(self, node):
        test = self.evaluate(self.test)
        if test:
            self.visit(self.body)
        else:
            self.visit(self.orelse)

    def visit_In(self, node):
        self.stack[-2] = self.stack[-2] in self.stack[-1]

    def visit_Invert(self, node):
        self.stack[-1] = ~self.stack[-1]

    def visit_Is(self, node):
        self.stack[-2] = self.stack[-2] is self.stack[-1]

    def visit_IsNot(self, node):
        self.stack[-2] = self.stack[-2] is not self.stack[-1]

    def visit_JoinedStr(self, node):
        r = []
        for value in node.values:
            self.visit(value)
            r.append(self.stack.pop())

        self.stack.append("".join(r))

    def visit_Lambda(self, node):
        self.stack.append(self.make_lambda(node.args, node.body))

    def visit_List(self, node):
        assert isinstance(node.ctx, ast.Load)
        self.stack.append(list(self.eval_elts(node.elts)))

    def visit_ListComp(self, node):
        self.stack.append(list(self.generate(node)))

    def visit_LShift(self, node):
        self.stack[-2] <<= self.stack[-1]

    def visit_Lt(self, node):
        self.stack[-2] = self.stack[-2] < self.stack[-1]

    def visit_LtE(self, node):
        self.stack[-2] = self.stack[-2] <= self.stack[-1]

    def visit_Mod(self, node):
        self.stack[-2] %= self.stack[-1]

    def visit_Mult(self, node):
        self.stack[-2] *= self.stack[-1]

    def visit_Name(self, node):
        assert isinstance(node.ctx, ast.Load)
        self.stack.append(self.symbols[node.id])

    def visit_Not(self, node):
        self.stack[-1] = not self.stack[-1]

    def visit_NotEq(self, node):
        self.stack[-2] = self.stack[-2] != self.stack[-1]

    def visit_NotIn(self, node):
        self.stack[-2] = self.stack[-2] not in self.stack[-1]

    def visit_Pow(self, node):
        self.stack[-2] **= self.stack[-1]

    def visit_RShift(self, node):
        self.stack[-2] >>= self.stack[-1]

    def visit_Set(self, node):
        self.stack.append(set(self.eval_elts(node.elts)))

    def visit_SetComp(self, node):
        self.stack.append(set(self.generate(node)))

    def visit_Slice(self, node):
        lower = self.evaluate(node.lower)
        upper = self.evaluate(node.upper)
        step = None if node.step is None else self.evaluate(node.step)
        self.stack.append(slice(lower, upper, step))

    def visit_Subscript(self, node):
        value = self.evaluate(node.value)
        s = self.evaluate(node.slice)
        self.stack.append(getitem(value, s))

    def visit_Tuple(self, node):
        assert isinstance(node.ctx, ast.Load)
        self.stack.append(tuple(self.eval_elts(node.elts)))

    def visit_UAdd(self, node):
        self.stack[-1] = +self.stack[-1]

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        self.visit(node.op)

    def visit_USub(self, node):
        self.stack[-1] = -self.stack[-1]

    def generic_visit(self, node):
        raise NotImplementedError(f"Not implemented: {node!r}")


def evaluate(expr: str, variables: dict[str, Any]) -> Any:
    node = ast.parse(expr, mode="eval")
    visitor = Visitor({})
    return visitor.evaluate(node)


def main():
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("expression")
    args = p.parse_args()
    result = evaluate(args.expression, {})
    print(f"result: {result!r}")


if __name__ == "__main__":
    main()
