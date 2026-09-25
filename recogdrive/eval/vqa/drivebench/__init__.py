"""DriveBench (adapted from https://github.com/drive-bench/toolkit, Apache-2.0, see LICENSE here).

Clean or corrupted nuScenes camera images. Scores: MCQ accuracy, BLEU / ROUGE-L / CIDEr on open questions
(needs the `language-evaluation` package), and with --gpt a GPT-3.5 grade (reads OPENAI_API_KEY).
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from recogdrive.eval.vqa.drivebench import gpt_prompts

HERE = Path(__file__).parent
SYSTEM_PROMPT = (HERE / "system_prompt.txt").read_text(encoding="utf-8")
MAX_NEW_TOKENS, MIN_NEW_TOKENS, BATCH_SIZE, SESSION_LEN = 4096, 50, 4, 30000
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
VIEWS = {camera: camera[len("CAM_"):].replace("_", " ") + " VIEW" for camera in CAMERAS}
ALL_CAMERAS = ("You are provided with up to six camera images in the sequence "
               "[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT].")
BLANK_SIZE = 448
BUCKETS = {  # question_type -> [(tag, question kind, GPT grading prompt)], first tag present wins
    "perception": [(2, "VQA", gpt_prompts.PERCEPTION_VQA_PROMPT), (0, "MCQ", gpt_prompts.PERCEPTION_MCQ_PROMPT)],
    "prediction": [(3, "VQA", gpt_prompts.PREDICTION_VQA_PROMPT)],
    "planning": [(1, "VQA", gpt_prompts.PLANNING_VQA_PROMPT)],
    "behavior": [(0, "MCQ", gpt_prompts.BEHAVIOR_MCQ_PROMPT)],
}
SAFE_ACTIONS_QUESTION = "In this scenario, what are safe actions to take for the ego vehicle?"


def read_items(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def camera_prompt(image_paths):
    """System prompt naming only the cameras present, and one view placeholder per camera."""
    found = set()
    for path in image_paths:
        match = re.search(r"samples/([^/]+)/", path)
        if match is None or match.group(1) not in CAMERAS:
            raise ValueError(f"cannot tell the camera of image path {path!r}")
        found.add(match.group(1))
    ordered = [camera for camera in CAMERAS if camera in found]
    if not ordered:
        raise ValueError("no camera images")
    listed = ", ".join(ordered)
    sentence = (f"You are provided with a single camera image: [{listed}]." if len(ordered) == 1
                else f"You are provided with {len(ordered)} camera images in the sequence [{listed}].")
    placeholders = "".join(f"<{VIEWS[camera]}>:\n<image>\n" for camera in ordered)
    return SYSTEM_PROMPT.replace(ALL_CAMERAS, sentence), placeholders


def load(args):
    blank = None
    if args.corruption == "NoImage":
        import numpy as np
        from PIL import Image

        blank = Image.fromarray(np.zeros((BLANK_SIZE, BLANK_SIZE, 3), dtype=np.uint8))
    samples = []
    for index, item in enumerate(read_items(args.data)):
        paths = [os.path.join(args.image_root, p) for p in item["image_path"].values() if isinstance(p, str)]
        system, placeholders = camera_prompt(paths)
        if blank is not None:
            images = [blank] * len(paths)
        elif args.corruption:
            images = [p.replace("nuscenes/samples", f"corruption/{args.corruption}") for p in paths]
        else:
            images = paths
        samples.append({"id": str(index), "prompt": placeholders + system + item["question"], "images": images})
    return samples


def choice(answer) -> str:
    """The A-D option an answer picks, or '' when it names none."""
    text = str(answer or "").strip().lower()
    for letter in "ABCD":
        if re.search(rf"\b({letter.lower()}|option {letter.lower()})\b", text):
            return letter
    return ""


def bucket(item):
    """(task, question kind, GPT prompt) of an item, or None for the unscored prediction MCQs."""
    qtype = item["question_type"]
    for tag, kind, template in BUCKETS.get(qtype, []):
        if tag in item["tag"]:
            return qtype, kind, template
    if qtype == "prediction" and 0 in item["tag"]:
        return None
    raise ValueError(f"unhandled DriveBench question type {qtype!r} with tag {item['tag']}")


def language_metrics(items):
    try:
        import language_evaluation
    except ImportError:
        return "skipped: pip install language-evaluation"
    evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
    return dict(evaluator.run_evaluation([str(i["pred"] or "") for i in items], [i["answer"] for i in items]))


def visual_description(descriptions, item):
    """Visual description of the object a question refers to, e.g. <c1,CAM_FRONT,767.5,513.3>."""
    question = item["question"]
    if question == SAFE_ACTIONS_QUESTION:
        return "No visual description needed for this question."
    match = re.search(r"<(.*)>", question)
    if match is None:
        return None
    prefix = ",".join(match.group(1).split(",")[:2])
    objects = descriptions[item["scene_token"]]["key_frames"][item["frame_token"]]["key_object_infos"]
    key = next((k for k in objects if k.startswith(f"<{prefix}")), None)
    if key is None:
        raise ValueError(f"no described object for {prefix} in {item['scene_token']}/{item['frame_token']}")
    return objects[key]["Visual_description"].lower()


class GptJudge:
    """Grades answers 0-100 with GPT-3.5; every reply is appended to a log, which also resumes a run."""

    URL = "https://api.openai.com/v1/chat/completions"
    MODEL = "gpt-3.5-turbo"
    INSTRUCTION = "You are an evaluator who rates answers based on their closeness to the correct answer."
    ATTEMPTS = 5

    def __init__(self, log_path: Path):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("--gpt needs OPENAI_API_KEY in the environment")
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.done = {}
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.done[self.key(entry)] = entry["gpt_score"]
        self.descriptions = json.loads((HERE / "visual_description.json").read_text(encoding="utf-8"))

    @staticmethod
    def key(item) -> str:
        return f"{item['scene_token']}_{item['frame_token']}_{item['question']}"

    def request(self, prompt: str) -> str:
        import requests

        payload = {"model": self.MODEL, "temperature": 0.0, "max_tokens": 512,
                   "messages": [{"role": "user", "content": f"{self.INSTRUCTION}\n{prompt}"}]}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.ATTEMPTS):
            try:
                response = requests.post(self.URL, headers=headers, json=payload, timeout=120)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                error = str(exc)
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"GPT grading failed after {self.ATTEMPTS} attempts: {error}")

    def grade(self, item, template: str) -> int:
        reply = self.done.get(self.key(item))
        if reply is None:
            fields = {"GT": item["answer"], "PRED": item["pred"], "QUESTION": item["question"]}
            description = visual_description(self.descriptions, item)
            if description:
                fields["DESC"] = description
            reply = self.request(template.format(**fields)).strip()
            with self.lock, self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**item, "desc": description, "gpt_score": reply}, ensure_ascii=False) + "\n")
        match = re.search(r"Total Score:\s*(\d{1,3})\b", reply)
        if match is None:
            raise ValueError(f"no 'Total Score' in the GPT reply: {reply[:200]}")
        return int(match.group(1))


def finish(args, samples, answers):
    name = args.corruption or "clean"
    items = [dict(item, pred=answers.get(str(index))) for index, item in enumerate(read_items(args.data))]
    (args.out / f"{name}.json").write_text(json.dumps(items, indent=4, ensure_ascii=False), encoding="utf-8")
    judge = GptJudge(args.out / "gpt_eval_logs" / f"{name}_eval_log.json") if args.gpt else None
    groups = {}
    for item in items:
        found = bucket(item)
        if found is not None:
            task, kind, template = found
            groups.setdefault((task, kind, template), []).append(item)
    scores = {}
    for (task, kind, template), group in groups.items():
        entry = scores.setdefault(task, {}).setdefault(kind, {})
        if kind == "MCQ":
            entry["accuracy"] = sum(choice(i["answer"]) == choice(i["pred"]) for i in group) / len(group)
        else:
            entry["language_metrics"] = language_metrics(group)
        if judge is not None:
            with ThreadPoolExecutor(32) as pool:
                grades = list(pool.map(lambda i: judge.grade(i, template), group))
            entry["gpt_score"] = sum(grades) / len(grades)
    return scores
