from mmengine.registry import HOOKS
from mmengine.visualization import WandbHook

# Register WandbHook explicitly
@HOOKS.register_module()
class CustomWandbHook(WandbHook):
    pass