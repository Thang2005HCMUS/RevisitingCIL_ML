from .base import TuningStrategy
from .linear_probe import FreezeStrategy
from .scale_shift import SSFStrategy
from .adapter import AdapterStrategy
from .prompt import PromptStrategy  # Tương tự cho VPT
from .bn import BNTuningStrategy
from .finetune import FinetuneStrategy
def get_tuning_strategy(convnet_type: str, model_name: str = "") -> TuningStrategy:
    """
    Hàm Factory để trả về TuningStrategy phù hợp dựa trên tên convnet và model_name.
    """
    convnet_type = convnet_type.lower()
    model_name = model_name.lower()
    
    if "ssf" in convnet_type:
        return SSFStrategy()
    elif "adapter" in convnet_type:
        return AdapterStrategy()
    elif "vpt" in convnet_type:
        return PromptStrategy()
    elif "bn" in convnet_type or "bn" in model_name:
        return BNTuningStrategy()
    elif "finetune" in convnet_type or "finetune" in model_name:
        return FinetuneStrategy()
    else:
        # Mặc định là SimpleCIL (đóng băng hoàn toàn)
        return FreezeStrategy()