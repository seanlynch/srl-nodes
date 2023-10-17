import itertools
from typing import Annotated, Any, Literal, Optional

import comfy.utils
import nodes

from . import evaluator
from .node_decorator import NODE_CLASS_MAPPINGS, NODE_NAME_MAPPINGS, Name, NumRange, node, IMAGE, SEGS


@node("utils", "SRL Conditional Interrupt")
def SrlConditionalInterrupt(interrupt: Annotated[bool, "forceInput"], inp: Any) -> Annotated[Any, Name("output")]:
    """Interrupt processing if the boolean input is true. Pass through the other input."""

    if interrupt:
        nodes.interrupt_processing()

    return inp


@node("utils", "SRL Format String")
def SrlFormatString(
        format: str = "first input via str(): {}, second input via repr(): {!r}, third input by index: {2}",
        in0: Optional[Any] = None,
        in1: Optional[Any] = None,
        in2: Optional[Any] = None,
        in3: Optional[Any] = None,
        in4: Optional[Any] = None,
) -> str:
    return evaluator.safe_format(format, in0, in1, in2, in3, in4)


@node("utils", "SRL Num Expr")
class SrlNumExpr:
    """Evaluate a numerical expression safely."""

    def __init__(self):
        self.last_expr = None
        self.last_code = None

    def __call__(
            self,
            expr: Annotated[str, "multiline"],
            i0: Optional[Annotated[int, "forceInput"]],
            i1: Optional[Annotated[int, "forceInput"]],
            i2: Optional[Annotated[int, "forceInput"]],
            f0: Optional[Annotated[float, "forceInput"]],
            f1: Optional[Annotated[float, "forceInput"]],
            f2: Optional[Annotated[float, "forceInput"]],
    ) -> tuple[bool, int, float]:
        if expr == self.last_expr:
            code = self.last_code
        else:
            code = self.last_code = evaluator.safe_compile(expr)

        res = evaluator.safe_eval(code, {
            "i0": i0,
            "i1": i1,
            "i2": i2,
            "f0": f0,
            "f1": f1,
            "f2": f2,
        })
        return (bool(res), int(res), float(res))


@node("utils", "SRL Filter Image List")
def SrlFilterImageList(images: list[IMAGE], keep: list[Annotated[bool, "forceInput"]]) -> tuple[list[Annotated[IMAGE, Name("t_images")]], list[Annotated[IMAGE, Name("f_images")]]]:
    """Filter an image list based on a list of bools"""

    im1, im2 = itertools.tee(zip(images, keep))
    return (
        [im for im, k in im1 if k],
        [im for im, k in im2 if not k],
    )


@node("utils", "SRL Count SEGS")
def SrlCountSegs(segs: SEGS, ignore_none: bool = False) -> int:
    """Count the number of segs."""
    if segs is None:
        if ignore_none:
            raise TypeError("segs is None, expected SEGS")
        else:
            return 0
    else:
        return len(segs[1])


@node("images", "SRL Scale Image")
class SrlScaleImage:
    """Scale an image under specified conditions."""

    def scale_dim(self, how, idim, odim, width, height) -> tuple[int, int, float, bool, bool]:
        scaled = scaled_up = False
        factor = 1.0

        if how in ("up", "both") and idim < odim:
            scaled = scaled_up = True
            factor = odim / idim
        elif how in ("down", "both") and idim > odim:
            scaled = True
            factor = odim / idim

        if scaled:
            width *= factor
            height *= factor

        return round(width), round(height), factor, scaled, scaled_up

    def __call__(
            self,
            image: IMAGE,
            what: Literal["width", "height", "both", "factor", "longest", "shortest", "mebipixels"],
            how: Literal["up", "down", "both"],
            width: Annotated[int, NumRange(1, nodes.MAX_RESOLUTION, 1)],
            height: Annotated[int, NumRange(1, nodes.MAX_RESOLUTION, 1)],
            factor: Annotated[float, NumRange(0.001, 100.0, 0.001)],
            longest_shortest: Annotated[int, NumRange(1, nodes.MAX_RESOLUTION, 1)],
            mebipixels: Annotated[float, NumRange(0.01, 20, 0.01)],
    ) -> tuple[IMAGE, Annotated[int, Name("width")], Annotated[int, Name("height")], Annotated[float, Name("width_factor")], Annotated[float, Name("height_factor")], Annotated[bool, Name("did_scale")], Annotated[bool, Name("did_scale_up")]]:
        samples = image.movedim(-1, 1)
        w, h = samples.shape[2:4]
        scaled = scaled_up = False
        new_width = w
        new_height = h
        width_factor = height_factor = 1.0

        match what:
            case "factor":
                new_width = w * factor
                new_height = h * factor
                scaled = factor != 1.0
                scaled_up = factor > 1.0
                width_factor = height_factor = factor
            case "both":
                new_width = width
                new_height = height
                width_factor = new_width / w
                height_factor = new_height / h
                scaled = new_width != width or new_height != height
                scaled_up = new_width > width
            case _:
                match what:
                    case "width":
                        idim = w
                        odim = width
                    case "height":
                        idim = h
                        odim = height
                    case "longest":
                        idim = max(width, height)
                        odim = longest_shortest
                    case "shortest":
                        idim = min(width, height)
                        odim = longest_shortest
                    case "mebipixels":
                        idim = width * height / 1048576
                        odim = mebipixels * 1048576
                    case _:
                        raise ValueError("'what' must be width, height, longest, shortest, or mebipixels")

                new_width, new_height, ofactor, scaled, scaled_up = self.scale_dim(how, idim, odim, width, height)
                width_factor = height_factor = ofactor

        if scaled:
            samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")

        return (samples.movedim(1, -1), new_width, new_height, width_factor, height_factor, scaled, scaled_up)
