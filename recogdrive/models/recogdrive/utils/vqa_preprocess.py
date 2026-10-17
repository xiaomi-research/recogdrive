"""CPU preprocessing for VQA batches handled by the shared ReCogDrive VLM."""

import torch

from recogdrive.models.recogdrive.backbone import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    process_vision_info,
)
from recogdrive.models.recogdrive.utils.internvl_preprocess import load_image

VQA_SYSTEM_PROMPT = "You are an autonomous driving assistant. Answer the user's question using the image evidence."


def answer_only_tensors(prompt_ids, full_ids, max_length):
    """Keep a sample unpadded; only assistant answer tokens receive LM loss."""
    prompt_ids, full_ids = list(prompt_ids), list(full_ids)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("VQA prompt tokenization is not a prefix of the prompt-plus-answer tokenization")
    answer_ids = full_ids[len(prompt_ids):]
    if not answer_ids:
        raise ValueError("VQA answer must contain at least one token")
    if len(full_ids) > max_length:
        raise ValueError(f"VQA prompt and answer need {len(full_ids)} tokens, exceeding vqa.max_length={max_length}")
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(([-100] * len(prompt_ids)) + answer_ids, dtype=torch.long),
        "answer_length": len(answer_ids),
    }


def _token_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=True, padding=False, truncation=False)["input_ids"]


def _prepare_internvl(tokenizer, image_paths, question, answer, max_length, num_image_token, short_sides):
    image_tensors = [load_image(path, short_side=short_sides[0 if index == 0 else 1])
                     for index, path in enumerate(image_paths)]
    patch_counts = [int(image.shape[0]) for image in image_tensors]
    image_prefix = "".join(
        f"Image {index + 1}: {IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * (num_image_token * count)}{IMG_END_TOKEN}\n"
        for index, count in enumerate(patch_counts)
    )
    user = f"{image_prefix}Question: {question}"
    prefix = (f"<|im_start|>system\n{VQA_SYSTEM_PROMPT}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    prompt_ids = _token_ids(tokenizer, prefix)
    full_ids = _token_ids(tokenizer, prefix + answer + "<|im_end|>")
    tensors = answer_only_tensors(prompt_ids, full_ids, max_length)
    features = {key: value for key, value in tensors.items() if key != "answer_length"}
    features.update(pixel_values=torch.cat(image_tensors, dim=0), num_patches=torch.tensor(patch_counts),
                    answer_lengths=tensors["answer_length"])
    return features


def _prepare_qwen(processor, image_paths, question, answer, max_length):
    if process_vision_info is None:
        raise ImportError("qwen_vl_utils is required to prepare Qwen-VL VQA samples")
    messages = [
        {"role": "system", "content": VQA_SYSTEM_PROMPT},
        {"role": "user", "content": [
            *({"type": "image", "image": path} for path in image_paths),
            {"type": "text", "text": question},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_messages = messages + [{"role": "assistant", "content": answer}]
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    common = {"images": image_inputs, "videos": video_inputs or None, "padding": False, "return_tensors": "pt"}
    prompt_ids = processor(text=[prompt_text], **common)["input_ids"][0].tolist()
    processed = processor(text=[full_text], **common)
    full_ids = processed["input_ids"][0].tolist()
    tensors = answer_only_tensors(prompt_ids, full_ids, max_length)
    features = {key: value for key, value in tensors.items() if key != "answer_length"}
    features.update({key: value for key, value in processed.items()
                     if key not in ("input_ids", "attention_mask") and torch.is_tensor(value)})
    features["answer_lengths"] = tensors["answer_length"]
    return features


def prepare_vqa_features(features, tokenizer, model_type, max_length, num_image_token=256, short_sides=(None, None)):
    """Turn one raw VQA record into CPU tensors; used through the agent's worker transform."""
    question, answer = features["question"], features["answer"]
    image_paths = features["image_paths"]
    if str(model_type).lower() == "internvl":
        processed = _prepare_internvl(tokenizer, image_paths, question, answer, max_length, num_image_token,
                                      short_sides)
    elif str(model_type).lower() == "qwen":
        processed = _prepare_qwen(tokenizer, image_paths, question, answer, max_length)
    else:
        raise ValueError(f"Unsupported VQA backbone {model_type!r}; expected 'internvl' or 'qwen'")
    result = {key: value for key, value in features.items() if key not in ("question", "answer", "image_paths")}
    result.update(processed)
    result["task"] = "vqa"
    return result
