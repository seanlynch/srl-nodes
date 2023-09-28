import inspect
import textwrap

import nodes


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
        return (format.format(*kwargs.values(), **kwargs),)


class SrlEval:
    """Evaluate any Python code as a function with the given inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parameters": ("STRING", {"multiline": False, "default": "a, b=None, c=\"foo\", *rest"}),
                "code": ("STRING", {"multiline": True, "default": "code goes here\nreturn a + b"}),
            },
            "optional": {
                "arg0": (any_typ,),
                "arg1": (any_typ,),
                "arg2": (any_typ,),
                "arg3": (any_typ,),
                "arg4": (any_typ,),
            }
        }

    RETURN_TYPES = (any_typ,)
    FUNCTION = "doit"
    CATEGORY = "utils"

    def doit(self, parameters, code, **kw):
        # Indent the code for the main body of the function
        func_code = textwrap.indent(code, "    ")
        source = f"def func({parameters}):\n{func_code}"

        # The provided code can mutate globals or really do anything, but ComfyUI isn't secure to begin with.
        loc = {}
        exec(source, globals(), loc)
        func = loc["func"]

        argspec = inspect.getfullargspec(func)
        # We don't allow variable keyword arguments or keyword only arguments, but we do allow varargs
        assert argspec.varkw is None
        assert not argspec.kwonlyargs

        input_names = list(self.INPUT_TYPES()["optional"].keys())
        parameter_names = argspec.args

        # Convert the list of defaults into a dictionary to make it easier to use
        default_list = argspec.defaults if argspec.defaults is not None else []
        defaults = {parameter_name: default for parameter_name, default in zip(parameter_names[-len(default_list):], default_list)}

        # We handle substituting default values ourselves in order to support *args
        args = [kw[input_name] if input_name in kw else defaults[parameter_name] for parameter_name, input_name in zip(parameter_names, input_names)]

        # Support *args
        if argspec.varargs is not None:
            unnamed_inputs = input_names[len(argspec.args):]
            # I considered requiring the remaining inputs to be contiguous, but I don't think it's helpful.
            args += [kw[input_name] for input_name in unnamed_inputs if input_name in kw]

        ret = func(*args)
        return (ret,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SRL Conditional Interrrupt": SrlConditionalInterrupt,
    "SRL Format String": SrlFormatString,
    "SRL Eval": SrlEval,
}


# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SrlConditionalInterrupt": "SRL Conditional Interrupt",
    "SrlFormatString": "SRL Format String",
    "SrlEval": "SRL Eval",
}
