import os
import re
import json
import argparse

from tqdm import tqdm
from multiprocessing import Pool

from evaluate.utils import preprocess_answer
from evaluate.request import GPTEvaluation
from evaluate.prompts import (PERCEPTION_MCQ_PROMPT, PERCEPTION_VQA_PROMPT,
                              PREDICTION_MCQ_PROMPT, PREDICTION_VQA_PROMPT, 
                              PLANNING_VQA_PROMPT, BEHAVIOR_MCQ_PROMPT)


class EvaluationSuit:
    def __init__(self, 
                 api_key: str, 
                 log_file: str, 
                 desc_file: str,
                 eval_gpt: bool = True,
                 temperature: float = 0.0,
                 max_tokens: int = 3000):
        """Initialize the Evaluation Suite.
        Args:
            api_key (str): Your OpenAI API key.
            log_file (str): Path to the file where all GPT responses will be logged.
            desc_file (str): Path to the file containing visual descriptions.
            eval_gpt (bool): Whether to evaluate GPT score.
            temperature (float): GPT temperature.
            max_tokens (int): GPT max tokens.
        """
        
        self.eval_gpt = eval_gpt
        self.language_eval = None
        if eval_gpt:
            if not api_key:
                raise ValueError("Set OPENAI_API_KEY or pass --key with --eval-gpt.")
            with open(desc_file, encoding="utf-8") as stream:
                self.original_data = json.load(stream)
            self.gpt_score_eval = GPTEvaluation(api_key, temperature=temperature,
                                                max_tokens=max_tokens, log_file=log_file)

        # Task - Question_type - Metric: List[data]
        self.results = {
            "perception": {
                "MCQ": {"gpt": [], "accuracy": []},
                "VQA": {"gpt": [], "language": []}
            },
            "prediction": {
                "VQA": {"gpt": [], "language": []}
            },
            "planning": {
                "VQA": {"gpt": [], "language": []}
            },
            "behavior": {
                "MCQ": {"gpt": [], "accuracy": []}}
        }

    def extract_visual_description(self, scene_token, frame_token, question):
        """Extracts the visual description of the referred object for better evaluation.
        """
        
        # TODO: workaround for a specific planning question
        if question == 'In this scenario, what are safe actions to take for the ego vehicle?':
            return "No visual description needed for this question."
        
        key_object = self.original_data[scene_token]["key_frames"][frame_token]["key_object_infos"]
        
        # research identifier, e.g. <c1,CAM_FRONT,767.5,513.3>
        object_id_match = re.search(r'<(.*)>', question)
        if object_id_match:
            object_id = object_id_match.group(1)
            # e.g., <c1, CAM_FRONT>
            object_id_prefix = ','.join(object_id.split(',')[:2])
        else:
            return None
        
        matching_object_key = next((key for key in key_object if key.startswith(f"<{object_id_prefix}")), None)
        
        if matching_object_key:
            desc = key_object[matching_object_key]['Visual_description'].lower()
            return desc
        else:
            raise ValueError(f"No matching object found for prefix: {object_id_prefix}")

    def forward(self, data_item):
        """Routes the QA pair into the correct bucket based on question_type and tag.
        """
        
        question = data_item["question"]
        question_type = data_item["question_type"]
        tag = data_item["tag"]
        scene_token = data_item["scene_token"]
        frame_token = data_item["frame_token"]
        
        # if question refer to an object, extract its visual description
        if self.eval_gpt:
            data_item['desc'] = self.extract_visual_description(scene_token, frame_token, question)
        
        if question_type == "perception":
            # Perception - VQA
            if 2 in tag:
                self.results["perception"]['VQA']["gpt"].append((data_item, PERCEPTION_VQA_PROMPT))
                self.results["perception"]['VQA']["language"].append(data_item)
            # Perception - MCQ   
            elif 0 in tag:
                self.results["perception"]['MCQ']["gpt"].append((data_item, PERCEPTION_MCQ_PROMPT))
                self.results["perception"]['MCQ']["accuracy"].append(data_item)
            else:
                raise NotImplementedError(f"Tag {tag} not handled for perception.")
            
        elif question_type == "prediction":
            # Prediction - VQA
            if 3 in tag:
                self.results["prediction"]['VQA']["gpt"].append((data_item, PREDICTION_VQA_PROMPT))
                self.results["prediction"]['VQA']["language"].append(data_item)
            # Prediction - MCQ
            elif 0 in tag:
                # TODO: not used for now
                pass
            else:
                raise NotImplementedError(f"Tag {tag} not handled for prediction.")
            
        elif question_type == "planning":
            # Planning - VQA
            if 1 in tag:
                self.results["planning"]['VQA']["gpt"].append((data_item, PLANNING_VQA_PROMPT))
                self.results["planning"]['VQA']["language"].append(data_item)
            else:
                raise NotImplementedError(f"Tag {tag} not handled for planning.")
            
        elif question_type == "behavior":
            # Behavior - MCQ
            if 0 in tag:
                self.results["behavior"]['MCQ']["accuracy"].append(data_item)
                self.results["behavior"]['MCQ']["gpt"].append((data_item, BEHAVIOR_MCQ_PROMPT))
            else:
                raise NotImplementedError(f"Tag {tag} not handled for behavior.")
            
        else:
            raise NotImplementedError(f"Task category {question_type} not implemented.")

    def eval_gpt_score(self, data):
        """Evaluate GPT Score.
        """
        with Pool(32) as p:
            scores = p.map(self.gpt_score_eval.forward, data)
        avg_score = sum(map(float, scores)) / len(scores)
        return avg_score

    def eval_accuracy(self, data):
        """Evaluate Accuracy.
        """
        scores = []
        for data_item in data:
            answer = preprocess_answer(data_item["answer"])
            GT = preprocess_answer(data_item["pred"])
            scores.append(answer == GT)
        accuracy = sum(scores) / len(scores)
        return accuracy

    def eval_language(self, data):
        """Evaluate Language Metric (e.g. BLEU, ROUGE_L, CIDEr).
        """
        pred_list = [data_item['pred'] for data_item in data]
        GT_list = [data_item['answer'] for data_item in data]
        
        if self.language_eval is None:
            import language_evaluation

            self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        results_gen = self.language_eval.run_evaluation(pred_list, GT_list)
        results_gen_dict = {f"{k}": v for k, v in results_gen.items()}
        return results_gen_dict

    def evaluation(self):
        """Run evaluation for all tasks and return a dictionary of scores.
        """
        print("Evaluation start!")
        scores = {
            "perception": {"MCQ": {}, "VQA": {}},
            "prediction": {"VQA": {}},
            "planning": {"VQA": {}},
            "behavior": {"MCQ": {}}
        }
        evaluation_mapping = {
            "perception": {
            "MCQ": [
                ("gpt", self.eval_gpt_score, "gpt_score"),
                ("accuracy", self.eval_accuracy, "accuracy")
            ],
            "VQA": [
                ("gpt", self.eval_gpt_score, "gpt_score"),
                ("language", self.eval_language, "language_metrics")
            ]
            },
            "prediction": {
            "VQA": [
                ("gpt", self.eval_gpt_score, "gpt_score"),
                ("language", self.eval_language, "language_metrics")
            ]
            },
            "planning": {
            "VQA": [
                ("gpt", self.eval_gpt_score, "gpt_score"),
                ("language", self.eval_language, "language_metrics")
            ]
            },
            "behavior": {
            "MCQ": [
                ("gpt", self.eval_gpt_score, "gpt_score"),
                ("accuracy", self.eval_accuracy, "accuracy")
            ]
            }
        }

        for task, qtypes in evaluation_mapping.items():
            for qtype, metrics in qtypes.items():
                for key, func, target in metrics:
                    if key == "gpt" and not self.eval_gpt:
                        continue
                    # Evaluate only if there is data for the metric.
                    if self.results[task][qtype].get(key):
                        scores[task][qtype][target] = func(self.results[task][qtype][key])
                
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Evaluation for Autonomous Driving')
    parser.add_argument('path', type=str, help='Path to prediction file')
    # GPT evaluation arguments
    parser.add_argument('--key', type=str, default='', help='OpenAI API key')
    parser.add_argument('--eval-gpt', action='store_true', help='Evaluate GPT score')
    parser.add_argument('--temperature', '-t', type=float, default=0.0, help='GPT temperature')
    parser.add_argument('--max-tokens', '-m', type=int, default=3000, help='GPT max tokens')
    args = parser.parse_args()

    # Log path for GPT evaluation
    corruption = os.path.basename(args.path).split(".")[0]
    log_dir = os.path.join(os.path.dirname(args.path), "gpt_eval_logs")
    log_path = os.path.join(log_dir, f"{corruption}_eval_log.json")

    # if key is not provided, read from environment variable
    if not args.key:
        args.key = os.getenv('OPENAI_API_KEY')

    with open(args.path, 'r') as f:
        data = json.load(f)

    desc_file = 'data/visual_description.json'
    evaluation = EvaluationSuit(args.key, 
                                log_path, 
                                desc_file, 
                                eval_gpt=args.eval_gpt,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens)

    for data_item in tqdm(data):
        evaluation.forward(data_item)

    final_scores = evaluation.evaluation()
    print("Final Scores:")
    print(json.dumps(final_scores, indent=4))