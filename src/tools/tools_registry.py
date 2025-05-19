import inspect
from autogen_core.tools import FunctionTool

_TOOL_REGISTRY = {}


def tool(name=None, description=None):
    """
    Decorator to mark a function as a tool.

    Args:
        name: Optional name for the tool (defaults to function name)
        description: Optional description of the tool
    """

    def decorator(func):
        # Determine tool name and description
        tool_name = name if name is not None else func.__name__
        tool_desc = description if description is not None else (func.__doc__ or "")

        # Store metadata on the function itself
        func._tool_name = tool_name
        func._tool_description = tool_desc

        # Register the function in the global registry
        _TOOL_REGISTRY[tool_name] = func

        return func

    # Handle case where decorator is used without parentheses
    if callable(name):
        func, name = name, None
        return decorator(func)

    return decorator


def collect_tools(function_tool_class, modules=None):
    """
    Create FunctionTool objects for all registered tools.

    Args:
        function_tool_class: The FunctionTool class from your framework
        modules: Optional list of modules to scan for additional tools

    Returns:
        Dict mapping tool names to FunctionTool objects
    """
    # If modules are provided, scan them for decorated functions
    if modules is not None:
        if not isinstance(modules, list):
            modules = [modules]

        for module in modules:
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, '_tool_name'):
                    _TOOL_REGISTRY[obj._tool_name] = obj

    # Create FunctionTool objects for each registered function
    tool_dict = {}
    for name, func in _TOOL_REGISTRY.items():
        tool_dict[name] = function_tool_class(
            func=func,
            name=name,
            description=func._tool_description,
            strict=True
        )

    return tool_dict