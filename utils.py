"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: 

Utility/Auxiliar code

Holds utility code that can be reused by several modules.

"""


import sys
from itertools import chain
from collections import deque

def dynamically_init_class(module_name, **kwargs):
    """Dynamically initializes a python object based
    on the given class name that resides inside module
    specified by the `module_name`.
    
    The `class` name must be specified as an additional argument,
    this argument will be caught under kwargs variable.
    
    The reason for not directly specifying the class as argument is 
    because `class` is a reserved keyword in python, which may be
    confusing if it is seen as an argument of a function. 
    Additionally, this way the function integrates nicely with the
    `.get_kwargs()` method from the `Param` object.

    Parameters
    ----------
    module_name : str
        the name of the module where the class resides
    kwargs : Dict[str, object]
        python dictionary that holds the variables and their values
        that are used as arguments during the class initialization.
        Note that the variable `class` must be here and that it will
        not be passed as an initialization argument since it is removed
        from this dict.
    
    Returns
        ----------
        object
            python instance
    """

    class_name = kwargs.pop("class")
    return getattr(sys.modules[module_name], class_name)(**kwargs)


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
