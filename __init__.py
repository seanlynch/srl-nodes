import itertools

import comfy.utils
import nodes

from . import evaluator


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class SrlConditionalInterrupt:
    """Interrupt processing if the boolean input is true. Pass through the other input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interrupt": ("BOOLEAN", {"forceInput": True}),
                "inp": (any_typ,),
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, interrupt, inp):
        if interrupt:
            nodes.interrupt_processing()

        return (inp,)


class SrlFormatString:
    """Use Python f-string syntax to generate a string using the inputs as the arguments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "first input via str(): {}, second input via repr(): {!r}, third input by index: {2}",
                    },
                ),
            },
            "optional": {
                "in0": (any_typ,),
                "in1": (any_typ,),
                "in2": (any_typ,),
                "in3": (any_typ,),
                "in4": (any_typ,),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, format, **kwargs):
        return (evaluator.safe_vformat(format, list(kwargs.values()), {}),)


class SrlNumExpr:
    """Evaluate a numerical expression safely."""

    def __init__(self):
        self.last_expr = None
        self.last_code = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expr": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "i0": ("INT", {"forceInput": True}),
                "i1": ("INT", {"forceInput": True}),
                "i2": ("INT", {"forceInput": True}),
                "f0": ("FLOAT", {"forceInput": True}),
                "f1": ("FLOAT", {"forceInput": True}),
                "f2": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "FLOAT")
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, expr, **kwargs):
        if expr == self.last_expr:
            code = self.last_code
        else:
            code = self.last_code = evaluator.safe_compile(expr)

        res = evaluator.safe_eval(code, kwargs)
        return (bool(res), int(res), float(res))


class SrlFilterImageList:
    """Filter an image list based on a list of bools"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "keep": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("t_images", "f_images")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, images, keep):
        im1, im2 = itertools.tee(zip(images, keep))
        return (
            [im for im, k in im1 if k],
            [im for im, k in im2 if not k],
        )


class SrlCountSegs:
    """Count the number of segs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "ignore_none": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, segs, ignore_none):
        if segs is None:
            if ignore_none:
                raise TypeError("segs is None, expected SEGS")
            else:
                return 0
        else:
            return (len(segs[1]),)


class SrlScaleImage:
    """Scale an image under specified conditions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "what": ("width", "height", "both", "factor", "longest", "shortest", "mebipixels"),
                "how": ("up", "down", "both"),
                "width": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "factor": ("FLOAT",),
                "longest_shortest": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "mebipixels": ("FLOAT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("image", "width", "height", "width_factor", "height_factor", "did_scale", "did_scale_up")
    FUNCTION = "doit"
    CATEGORY = "images"

    def scale_dim(self, how, idim, odim, width, height):
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

        return width, height, factor, scaled, scaled_up

    def doit(self, image, what, how, width, height, factor, longest_shortest, mebipixels, ratio):
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

                new_width, new_height, ofactor, scaled, scaled_up = self.scale_dim(how, idim, odim, width, height)
                width_factor = height_factor = ofactor

        if scaled:
            samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")

        return (samples.movedim(1, -1), new_width, new_height, width_factor, height_factor, scaled, scaled_up)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SRL Conditional Interrrupt": SrlConditionalInterrupt,
    "SRL Format String": SrlFormatString,
    "SRL Filter Image List": SrlFilterImageList,
    "SRL Num Expr": SrlNumExpr,
    "SRL Count SEGS": SrlCountSegs,
    "SRL Scale Image": SrlScaleImage,
}


# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SrlConditionalInterrupt": "SRL Conditional Interrupt",
    "SrlFormatString": "SRL Format String",
    "SrlFilterImageList": "SRL Filter Image List",
    "SrlNumExpr": "SRL Num Expr",
    "SrlCountSegs": "SRL Count SEGS",
    "SrlScaleImage": "SRL Scale Image",
}
