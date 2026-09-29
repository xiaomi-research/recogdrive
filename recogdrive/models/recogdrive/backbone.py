from typing import List, Optional, Tuple, Union
import types
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
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

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Qwen2 / Qwen3 LLMs
FLEX_OWN_COMPILE = "flex_attention_own_compile"

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
                 device: str = "cuda",
                 max_length: int = 2800):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
            max_length (int): Padded prompt length; multi-view prompts need more than the single-view 2800.
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device
        self.max_length = max_length

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
            self.set_attention()
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

    def set_attention(self):
        # Vision: SDPA (the official InternVL flash-attn path: 1487ms vs SDPA 715ms on 3090, same tokens).
        # Language: flex attention. The left-padded prompts need a causal + padding mask, which puts SDPA on a kernel
        # whose backward costs 8x its forward on a 3090; flex skips the masked blocks, with the same outputs up to
        # bf16 rounding (padding rows identical). It runs outside the compiled decoder blocks, compiled by
        # transformers itself: inside a block graph torch 2.6 finds no flex backward kernel that fits a 3090.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        language_attention = None
        language_model = getattr(self.model, "language_model", None)
        if language_model is not None and hasattr(language_model, "set_attn_implementation"):
            try:
                from transformers import AttentionInterface, AttentionMaskInterface
                from transformers.integrations.flex_attention import flex_attention_forward
                from transformers.masking_utils import flex_attention_mask

                AttentionInterface.register(FLEX_OWN_COMPILE, torch.compiler.disable(flex_attention_forward))
                AttentionMaskInterface.register(FLEX_OWN_COMPILE, flex_attention_mask)
                candidates = (FLEX_OWN_COMPILE, "sdpa")
            except ImportError:
                candidates = ("sdpa",)
            for language_attention in candidates:
                try:
                    language_model.set_attn_implementation(language_attention)
                    break
                except (ImportError, ValueError):
                    continue

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
        print(f"Backbone attention: vision sdpa, language {language_attention}")

    def add_lora(self, rank: int, alpha: float, dropout: float, targets: Optional[List[str]] = None) -> None:
        """LoRA adapters (peft) on the VLM's linear layers named in `targets`, by default the language model's
        attention and MLP projections (the vision towers name theirs differently). Base weights stay as loaded."""
        from peft import LoraConfig, inject_adapter_in_model

        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets or LORA_TARGETS))
        inject_adapter_in_model(config, self.model)

    def configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")

    def forward(self, pixel_values: Union[torch.Tensor, List[str]], questions: List[str],
                num_patches_list: Optional[List[int]] = None) -> torch.Tensor:
        """The language model's final hidden states (B, tokens, hidden), the same values as `hidden_states[-1]` of
        the chat model's forward. Only the decoder runs: its LM head would compute logits over every position that
        nothing uses."""
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        if self.model_type == "qwen":
            return self.forward_qwen(pixel_values, questions)
        
        model_dtype = next(self.model.parameters()).dtype

        # one entry of num_patches_list per <image> placeholder, in prompt order
        counts = iter(num_patches_list)
        queries = []
        for question in questions:
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for _ in range(query.count('<image>')):
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * next(counts) + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, padding='max_length', max_length=self.max_length)
        longest = max(map(len, model_inputs['input_ids']))
        if longest > self.max_length:
            raise ValueError(f"VLM prompt has {longest} tokens > max_length {self.max_length} "
                             f"(images tiled into up to {max(num_patches_list)} patches)")
        device = torch.device(self.device)
        pin = device.type == "cuda"
        input_ids = torch.tensor(model_inputs['input_ids'], pin_memory=pin).to(device, non_blocking=True)
        attention_mask = torch.tensor(model_inputs['attention_mask'], pin_memory=pin).to(device, non_blocking=True)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # InternVLChatModel.forward without the LM head: image features replace the <IMG_CONTEXT> embeddings in
        # prompt order (masked_scatter instead of its boolean-index assignment, which syncs with the host)
        language_model = self.model.language_model
        embeds = language_model.get_input_embeddings()(input_ids)
        vit_embeds = self.model.extract_feature(pixel_values.to(model_dtype))
        selected = (input_ids == self.img_context_token_id).unsqueeze(-1)
        embeds = embeds.masked_scatter(selected, vit_embeds.reshape(-1, embeds.shape[-1]).to(embeds.dtype))
        return language_model.get_decoder()(
            inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=False,
        ).last_hidden_state

    def forward_qwen(self, image_paths: Union[torch.Tensor, List[str]], questions: List[str]):
        if process_vision_info is None:
            raise ImportError("qwen_vl_utils is required for Qwen-VL preprocessing.")
        if not isinstance(image_paths, list):
            raise TypeError("Qwen-VL backbone expects image_paths as a list of file paths.")

        messages_batch = []
        for image_path, question in zip(image_paths, questions):
            paths = [image_path] if isinstance(image_path, str) else list(image_path)
            messages_batch.append([
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": path} for path in paths),
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

        # The generation model's own hidden_states[-1]: for Qwen3-VL it is the last decoder layer before the final
        # norm, unlike self.model.model's. logits_to_keep=1 leaves the LM head one position instead of every token.
        return self.model(**model_inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states[-1]
    
