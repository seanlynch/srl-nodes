import unittest
from typing import Annotated, Any, Literal, Optional

from node_decorator import NODE_CLASS_MAPPINGS, NODE_NAME_MAPPINGS, Name, NumRange, node, IMAGE, SEGS


@node("utils", "SRL Conditional Interrupt")
def SrlConditionalInterrupt(interrupt: Annotated[bool, "forceInput"], inp: Any) -> Annotated[Any, Name("output")]:
    ...


@node("utils", "SRL Format String")
def SrlFormatString(
        format: str = "first input via str(): {}, second input via repr(): {!r}, third input by index: {2}",
        in0: Optional[Any] = None,
        in1: Optional[Any] = None,
        in2: Optional[Any] = None,
        in3: Optional[Any] = None,
        in4: Optional[Any] = None,
) -> str:
    ...


@node("utils", "SRL Num Expr")
class SrlNumExpr:
    def __call__(
            self,
            expr: Annotated[str, "multiline"],
            i0: Optional[Annotated[int, "forceInput"]],
            i1: Optional[Annotated[int, "forceInput"]],
            i2: Optional[Annotated[int, "forceInput"]],
            f0: Optional[Annotated[float, "forceInput"]],
            f1: Optional[Annotated[float, "forceInput"]],
            f2: Optional[Annotated[float, "forceInput"]]
    ) -> tuple[bool, int, float]:
        ...


@node("utils", "SRL Filter Image List")
def SrlFilterImageList(images: list[IMAGE], keep: list[Annotated[bool, "forceInput"]]) -> tuple[list[Annotated[IMAGE, Name("t_images")]], list[Annotated[IMAGE, Name("f_images")]]]:
    ...


@node("utils", "SRL Count SEGS")
def SrlCountSegs(segs: SEGS, ignore_none: bool = False) -> int:
    ...


@node("images", "SRL Scale Image")
class SrlScaleImage:
    def __call__(
        self,
        image: IMAGE,
        what: Literal["width", "height", "both", "factor", "longest", "shortest", "mebipixels"],
        how: Literal["up", "down", "both"],
        width: Annotated[int, NumRange(1, 4096, 1)],
        height: Annotated[int, NumRange(1, 4096, 1)],
        factor: Annotated[float, NumRange(0.001, 100.0, 0.001)],
        longest_shortest: Annotated[int, NumRange(1, 4096, 1)],
        mebipixels: Annotated[float, NumRange(0.01, 20, 0.01)],
) -> tuple[IMAGE, Annotated[int, Name("width")], Annotated[int, Name("height")], Annotated[float, Name("width_factor")], Annotated[float, Name("height_factor")], Annotated[bool, Name("did_scale")], Annotated[bool, Name("did_scale_up")]]:
        ...


def main():
    print(f"NODE_CLASS_MAPPINGS = {NODE_CLASS_MAPPINGS!r}")
    print(f"NODE_NAME_MAPPINGS = {NODE_NAME_MAPPINGS!r}")

    for name, cls in NODE_CLASS_MAPPINGS.items():
        print(name)
        print(f"  {cls.INPUT_TYPES()!r}")


if __name__ == "__main__":
    main()

