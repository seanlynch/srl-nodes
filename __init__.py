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
                "format": ("STRING", {
                    "multiline": False,
                    "default": "first input via str(): {}, second input via repr(): {!r}, third input by index: {2}, fifth input by name: {in4}",
                }),
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
        # Allow referencing arguments both by name and index.
        return (evaluator.safe_vformat(format, list(kwargs.values()), kwargs),)


class SrlNumExpr:
    """Evaluate a numerical expression safely."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expr": ("STRING", {"multiline": True}),
            },
            "optional": {
                "b0": ("BOOLEAN", {"forceInput": True}),
                "b1": ("BOOLEAN", {"forceInput": True}),
                "b2": ("BOOLEAN", {"forceInput": True}),
                "i0": ("INT", {"forceInput": True}),
                "i1": ("INT", {"forceInput": True}),
                "i2": ("INT", {"forceInput": True}),
                "f0": ("FLOAT", {"forceInput": True}),
                "f1": ("FLOAT", {"forceInput": True}),
                "f2": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("BOOL", "INT", "FLOAT")

    def doit(self, expr, **kwargs):
        pass


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

    def doit(self, segs, ignore_none):
        if segs is None:
            if ignore_none:
                raise TypeError("segs is None, expected SEGS")
            else:
                return 0
        else:
            return len(segs[1])


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SRL Conditional Interrrupt": SrlConditionalInterrupt,
    "SRL Format String": SrlFormatString,
    "SRL Filter Image List": SrlFilterImageList,
    "SRL Count SEGS": SrlCountSegs,
}


# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SrlConditionalInterrupt": "SRL Conditional Interrupt",
    "SrlFormatString": "SRL Format String",
    "SrlFilterImageList": "SRL Filter Image List",
    "SrlCountSegs": "SRL Count SEGS",
}
