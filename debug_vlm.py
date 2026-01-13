
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except ImportError as e:
    print(f"transformers import failed: {e}")

try:
    import peft
    print(f"peft: {peft.__version__}")
except ImportError as e:
    print(f"peft import failed: {e}")

try:
    import qwen_vl_utils
    print("qwen_vl_utils imported successfully")
except ImportError as e:
    print(f"qwen_vl_utils import failed: {e}")

try:
    from Webcane3.fine_tuned_vlm import FineTunedVLM, QWEN_VLM_AVAILABLE
    print(f"QWEN_VLM_AVAILABLE: {QWEN_VLM_AVAILABLE}")
    
    if QWEN_VLM_AVAILABLE:
        vlm = FineTunedVLM()
        print(f"VLM Adapter Path: {vlm.adapter_path}")
        print(f"VLM Available check: {vlm.is_available()}")
except Exception as e:
    print(f"Error importing/init FineTunedVLM: {e}")
