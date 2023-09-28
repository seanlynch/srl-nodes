# SRL's nodes

This is a collection of nodes I find useful.

# Nodes

## SRL Conditional Interrupt

![Screenshot of SrlConditionalInterrupt](screenshots/SrlConditionalInterrupt.png)

Interrupts the currently running prompt if the "interrupt" input is
true. Also passes through an item of any type for sequencing purposes.

## SRL Format String

![Screenshot of SrlFormatString](screenshots/SrlFormatString.png)

Format a string using Python's string format syntax. You can use both
positional and named argument syntax.

## SRL Eval

![Screenshot of SrlEval](screenshots/SrlEval.png)

Evaluate arbitrary Python code as a function. Code is evaluated as:

```
def func({parameters}):
    code\_line1
    code\_line2
    ...
```

Right now this uses the globals dictionary from the module, so be
careful about anything that might mutate globals, such as importing
modules. I don't have any intention to try to make it secure because
ComfyUI itself isn't remotely secure, but I will probably try to
improve isolation at some point in the future to prevent unintended
problems. PRs gladly accepted if you have need for this.

# License

All code in this repository is copyright Sean Richard Lynch and any
other srl-nodes authors and released under the AGPLv3 unless otherwise
specified. Please see [LICENSE.txt](LICENSE.txt) for the full text of
the license.
