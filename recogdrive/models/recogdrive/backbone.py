from typing import List, Optional, Tuple, Union
import types
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from .utils.conversation import get_conv_template

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


def hidden_size_from_vlm_config(config):
    for nested_name in ("text_config", "llm_config"):
        nested = getattr(config, nested_name, None)
        hidden_size = getattr(nested, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    hidden_size = getattr(config, "hidden_size", None)
    return int(hidden_size) if hidden_size is not None else None


def resolve_qwen_model_class(checkpoint_path: str, config: object = None):
    """Choose Qwen3-VL vs Qwen2.5-VL from config, not the folder name."""
    loaded = config
    if loaded is None:
        try:
            loaded = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        except Exception:
            loaded = None
    blob = " ".join(
        [
            str(checkpoint_path),
            str(getattr(loaded, "model_type", "") or "") if loaded is not None else "",
            " ".join(getattr(loaded, "architectures", None) or []) if loaded is not None else "",
        ]
    ).lower()
    if "qwen3" in blob:
        if Qwen3VLForConditionalGeneration is not None:
            return Qwen3VLForConditionalGeneration
        if AutoModelForMultimodalLM is None:
            raise ImportError(
                "Qwen3-VL is unavailable. Install a transformers build with Qwen3VL "
                "or copy UniDriveVLA qwenvl3/transformers_replace."
            )
        return AutoModelForMultimodalLM
    if Qwen2_5_VLForConditionalGeneration is None:
        raise ImportError(
            "Qwen2.5-VL is unavailable. Use a Qwen3-VL / UniDriveVLA checkpoint."
        )
    return Qwen2_5_VLForConditionalGeneration


class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=False,
                device_map=self.device
            ).eval()
            self.enable_sdpa()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self.configure_internvl()
            self.num_image_token = 256

        elif self.model_type == 'qwen':
            qwen_model_cls = resolve_qwen_model_class(checkpoint_path)
            self.model = qwen_model_cls.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            ).eval()
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def enable_sdpa(self):
        # official InternVL flash-attn path: 1487ms vs SDPA 715ms on 3090, same tokens
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None and hasattr(language_model, "set_attn_implementation"):
            language_model.set_attn_implementation("sdpa")

        def sdpa_attn(module, x):
            bsz, seqlen, width = x.shape
            qkv = module.qkv(x).reshape(bsz, seqlen, 3, module.num_heads, width // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            if module.qk_normalization:
                heads, dim = query.shape[1], query.shape[-1]
                query = module.q_norm(query.transpose(1, 2).flatten(-2, -1)).view(bsz, seqlen, heads, dim).transpose(1, 2)
                key = module.k_norm(key.transpose(1, 2).flatten(-2, -1)).view(bsz, seqlen, heads, dim).transpose(1, 2)
            context = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=module.scale)
            context = context.transpose(1, 2).reshape(bsz, seqlen, width)
            return module.proj_drop(module.proj(context))

        for module in self.model.modules():
            if module.__class__.__name__ == "InternAttention":
                module._naive_attn = types.MethodType(sdpa_attn, module)
        print("Backbone attention: sdpa")

    def configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")

    def forward(self, pixel_values: Union[torch.Tensor, List[str]], questions: List[str], num_patches_list: Optional[List[int]] = None):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        if self.model_type == "qwen":
            return self.forward_qwen(pixel_values, questions)
        
        model_dtype = next(self.model.parameters()).dtype

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, padding='max_length', max_length=2800)
        longest = max(map(len, model_inputs['input_ids']))
        if longest > 2800:
            raise ValueError(f"VLM prompt has {longest} tokens > max_length 2800 "
                             f"(images tiled into up to {max(num_patches_list)} patches)")
        device = torch.device(self.device)
        pin = device.type == "cuda"
        input_ids = torch.tensor(model_inputs['input_ids'], pin_memory=pin).to(device, non_blocking=True)
        attention_mask = torch.tensor(model_inputs['attention_mask'], pin_memory=pin).to(device, non_blocking=True)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

        return self.model(
                pixel_values=pixel_values.to(model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
        )

    def forward_qwen(self, image_paths: Union[torch.Tensor, List[str]], questions: List[str]):
        if process_vision_info is None:
            raise ImportError("qwen_vl_utils is required for Qwen-VL preprocessing.")
        if not isinstance(image_paths, list):
            raise TypeError("Qwen-VL backbone expects image_paths as a list of file paths.")

        messages_batch = []
        for image_path, question in zip(image_paths, questions):
            messages_batch.append([
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question.replace("<image>\n", "").replace("<image>", "")},
                    ],
                },
            ])

        texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        image_inputs = []
        video_inputs = []
        for messages in messages_batch:
            image_input, video_input = process_vision_info(messages)
            image_inputs.extend(image_input or [])
            video_inputs.extend(video_input or [])

        model_inputs = self.tokenizer(
            text=texts,
            images=image_inputs,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    
