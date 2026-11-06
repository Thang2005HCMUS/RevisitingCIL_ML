from .base import TuningStrategy
from .linear_probe import FreezeStrategy
from .scale_shift import SSFStrategy
from .adapter import AdapterStrategy
# from .prompt import PromptStrategy  # Tương tự cho VPT

def get_tuning_strategy(convnet_type: str) -> TuningStrategy:
    """
    Hàm Factory để trả về TuningStrategy phù hợp dựa trên tên convnet.
   [cite: 23]
    """
    convnet_type = convnet_type.lower()
    
    if "ssf" in convnet_type:
        return SSFStrategy()
    elif "adapter" in convnet_type:
        return AdapterStrategy()
    elif "vpt" in convnet_type:
        return PromptStrategy() 
        pass
    else:
        # Mặc định là SimpleCIL (đóng băng hoàn toàn)[cite: 3, 23]
        return FreezeStrategy()