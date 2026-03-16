# Auto-import all loader modules so their @register_loader decorators run.
import pkgutil
import importlib

for _, modname, _ in pkgutil.iter_modules(__path__):
    if modname != "registry":
        importlib.import_module(f"{__name__}.{modname}")
