"""DriveLM-nuScenes: answers every key-frame question from the six cameras and writes output.json plus the
challenge submission.json (fill in the team fields). Scores come from the DriveLM evaluation server."""

import json
import os

from recogdrive.models.recogdrive.backbone import system_message as SYSTEM_PROMPT  # noqa: F401  (read by the CLI)

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]
VIEWS = {camera: camera[len("CAM_"):].replace("_", " ") + " VIEW" for camera in CAMERAS}
HEADER = "The following images are captured simultaneously from different cameras mounted on the same ego vehicle:\n"
MAX_NEW_TOKENS, MIN_NEW_TOKENS, BATCH_SIZE, SESSION_LEN = 4096, 50, 8, 30000
TEAM = {"method": "method", "team": "team", "authors": ["authors"], "email": "email",
        "institution": "institution", "country": "country"}


def prompt(question: str) -> str:
    return HEADER + "\n".join(f"<{VIEWS[camera]}>:\n<image>" for camera in CAMERAS) + f"\n{question}"


def load(args):
    with open(args.data, encoding="utf-8") as f:
        scenes = json.load(f)
    samples = []
    for scene_id, scene in scenes.items():
        for frame_id, frame in scene.get("key_frames", {}).items():
            paths = frame.get("image_paths", {})
            images = [os.path.join(args.image_root, paths[camera]) for camera in CAMERAS if camera in paths]
            qa_pairs = [qa for section in frame.get("QA", {}).values() for qa in section]
            for index, qa in enumerate(qa_pairs):
                samples.append({"id": f"{scene_id}_{frame_id}_{index}", "prompt": prompt(qa["Q"]), "images": images})
    return samples


def finish(args, samples, answers):
    results = sorted(
        ({"id": s["id"], "question": s["prompt"], "answer": answers[s["id"]]}
         for s in samples if answers.get(s["id"]) is not None),
        key=lambda row: row["id"],
    )
    (args.out / "output.json").write_text(json.dumps(results, indent=4), encoding="utf-8")
    submission = args.out / "submission.json"
    submission.write_text(json.dumps({**TEAM, "results": results}, indent=4), encoding="utf-8")
    return {"answered": len(results), "questions": len(samples), "submission": str(submission)}
