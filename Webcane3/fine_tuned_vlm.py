"""
Fine-tuned Qwen2-VL model for UI element clicking.
Used as fallback when NVIDIA/Gemini vision agents fail.

Based on the training/inference approach from checkkkkk.ipynb:
- Uses Qwen2-VL-2B-Instruct base model
- Loads PEFT adapter from checkpoint-1800
- Outputs: click(point='<point>X Y</point>') where X,Y are 0-1000 normalized coordinates
"""

import os
import re
import torch
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO

# Lazy imports to avoid loading heavy dependencies unless needed
QWEN_VLM_AVAILABLE = False
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    QWEN_VLM_AVAILABLE = True
except ImportError:
    pass


class FineTunedVLM:
    """
    Wrapper for the fine-tuned Qwen2-VL model.
    
    Lazy-loaded to avoid VRAM usage until actually needed.
    This is used as a fallback when NVIDIA and Gemini vision agents fail.
    """
    
    # Base model to use
    BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    
    def __init__(self, adapter_path: str = None):
        """
        Initialize the fine-tuned VLM wrapper.
        
        Args:
            adapter_path: Path to the PEFT adapter checkpoint.
                         If None, uses default checkpoint-1800 location.
        """
        if adapter_path is None:
            adapter_path = os.path.join(
                os.path.dirname(__file__), 
                "archive"
            )
        
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.loaded = False
        self._load_error = None
        
        # Check if adapter config exists
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            self._load_error = f"Adapter config not found at {config_path}"
            print(f"[Fine-tuned VLM] Warning: {self._load_error}")
    
    def is_available(self) -> bool:
        """Check if the VLM can be used."""
        return QWEN_VLM_AVAILABLE and self._load_error is None
    
    def load(self) -> bool:
        """
        Load model and processor (lazy loading).
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if self.loaded:
            return True
        
        if not QWEN_VLM_AVAILABLE:
            print("[Fine-tuned VLM] Required packages not available")
            print("[Fine-tuned VLM] Install: pip install transformers peft qwen-vl-utils bitsandbytes accelerate")
            return False
        
        if self._load_error:
            print(f"[Fine-tuned VLM] Cannot load: {self._load_error}")
            return False
        
        try:
            print(f"[Fine-tuned VLM] Loading model from {self.adapter_path}...")
            
            # 4-bit quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            # Load base model
            print("[Fine-tuned VLM] Loading base model (this may take a moment)...")
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            # Load PEFT adapter
            print(f"[Fine-tuned VLM] Loading adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.BASE_MODEL_ID,
                min_pixels=256 * 28 * 28,
                max_pixels=1024 * 28 * 28
            )
            
            self.loaded = True
            print("[Fine-tuned VLM] Model loaded successfully")
            return True
            
        except Exception as e:
            self._load_error = str(e)
            print(f"[Fine-tuned VLM] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_click(
        self, 
        image_bytes: bytes, 
        instruction: str,
        image_size: Tuple[int, int] = None
    ) -> Tuple[int, int]:
        """
        Predict click coordinates for given instruction.
        
        Args:
            image_bytes: Screenshot as PNG bytes
            instruction: Click instruction (e.g., "Click Email address")
            image_size: Optional (width, height) of original image for coordinate conversion
        
        Returns:
            (x, y) coordinates in screen pixels, or (-1, -1) if failed
        """
        if not self.load():
            return (-1, -1)
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            width, height = image.size
            
            if image_size:
                width, height = image_size
            
            # Construct prompt in same format as training
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,  # Greedy decoding for strict output
                    temperature=0.0
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"[Fine-tuned VLM] Raw output: {output_text}")
            
            # Parse output
            norm_x, norm_y = self.parse_output(output_text)
            
            if norm_x < 0 or norm_y < 0:
                print("[Fine-tuned VLM] Failed to parse coordinates")
                return (-1, -1)
            
            # Convert to screen coordinates (model outputs 0-1000 normalized scale)
            screen_x = int((norm_x / 1000) * width)
            screen_y = int((norm_y / 1000) * height)
            
            print(f"[Fine-tuned VLM] Normalized: ({norm_x}, {norm_y}) -> Screen: ({screen_x}, {screen_y})")
            
            return (screen_x, screen_y)
            
        except Exception as e:
            print(f"[Fine-tuned VLM] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return (-1, -1)
    
    def parse_output(self, output: str) -> Tuple[int, int]:
        """
        Parse model output format: click(point='<point>X Y</point>')
        
        The model outputs coordinates in 0-1000 normalized scale.
        
        Args:
            output: Raw model output string
        
        Returns:
            (x, y) normalized coordinates or (-1, -1) if parsing fails
        """
        try:
            # Pattern: <point>X Y</point>
            match = re.search(r'<point>(\d+)\s+(\d+)</point>', output)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                return (x, y)
            
            # Fallback: try to find any two numbers in sequence
            numbers = re.findall(r'\d+', output)
            if len(numbers) >= 2:
                x = int(numbers[0])
                y = int(numbers[1])
                # Sanity check - should be in 0-1000 range
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    return (x, y)
            
            return (-1, -1)
            
        except Exception as e:
            print(f"[Fine-tuned VLM] Parse error: {e}")
            return (-1, -1)
    
    def unload(self):
        """Unload the model to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Fine-tuned VLM] Model unloaded")


# Singleton instance for lazy loading
_vlm_instance: Optional[FineTunedVLM] = None


def get_fine_tuned_vlm(adapter_path: str = None) -> Optional[FineTunedVLM]:
    """
    Get the singleton fine-tuned VLM instance.
    
    Args:
        adapter_path: Optional custom adapter path
    
    Returns:
        FineTunedVLM instance or None if not available
    """
    global _vlm_instance
    
    if not QWEN_VLM_AVAILABLE:
        return None
    
    if _vlm_instance is None:
        _vlm_instance = FineTunedVLM(adapter_path)
    
    return _vlm_instance


def predict_click_coordinates(
    image_bytes: bytes, 
    instruction: str,
    image_size: Tuple[int, int] = None
) -> Tuple[int, int]:
    """
    Convenience function to predict click coordinates.
    
    Args:
        image_bytes: Screenshot as PNG bytes
        instruction: Click instruction (e.g., "Click Email address")
        image_size: Optional (width, height) for coordinate conversion
    
    Returns:
        (x, y) screen coordinates or (-1, -1) if failed
    """
    vlm = get_fine_tuned_vlm()
    if vlm is None:
        return (-1, -1)
    
    return vlm.predict_click(image_bytes, instruction, image_size)
