import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_TRANSFORM_CACHE = {}


def build_transform(input_size):
    cached = _TRANSFORM_CACHE.get(input_size)
    if cached is not None:
        return cached
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    _TRANSFORM_CACHE[input_size] = transform
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, short_side=None, augment=None, num_patches=None):
    """InternVL tiles of an image; `short_side` rescales it first (e.g. 960 / 480 for multi-view),
    `augment` is a PIL -> PIL training augmentation. Optional `num_patches` includes the global thumbnail."""
    image = (image_file if isinstance(image_file, Image.Image) else Image.open(image_file)).convert('RGB')
    if short_side:
        scale = short_side / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
    if augment is not None:
        image = augment(image)
    transform = build_transform(input_size=input_size)
    if num_patches is None:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        if type(num_patches) is not int or num_patches < 1 or num_patches == 2:
            raise ValueError("num_patches must be 1 or an integer >= 3 (crops plus a global thumbnail)")
        crops = 1 if num_patches == 1 else num_patches - 1
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, min_num=crops, max_num=crops)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
