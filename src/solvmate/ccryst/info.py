

DEBUG = 0
INFO = 1
WARNING = 2


LOG_LEVEL = 1
def log(*args, **kwargs):
    """
    Simple logging interface.
    """
    if "level" in kwargs:
        level = kwargs["level"]
        del kwargs["level"]
    else:
        level = DEBUG
    if "indent" in kwargs:
        indent = kwargs["indent"]
        del kwargs["indent"]
    else:
        indent = 0
    if "elt" in kwargs:
        elt = kwargs["elt"]
        del kwargs["elt"]
    else:
        elt = 0
    if level >= LOG_LEVEL:
        if elt == "header":
            print("-"*80)
        print("\t"*indent,end="",)
        print(*args,**kwargs)
        if elt == "header":
            print("-"*80)
    

