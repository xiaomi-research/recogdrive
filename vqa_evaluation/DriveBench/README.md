# DriveBench reproduction

Prepare the official DriveBench images and annotations and generate `data/drivebench-test-final.json` from `drivebench-test.json` with the provided preprocessing script. The annotation paths must resolve from this directory. The corruption settings read the corresponding dataset image variants. [DriveBench data preparation](https://github.com/worldbench/DriveBench/blob/main/docs/DATA_PREPAER.md)

```bash
cd vqa_evaluation/DriveBench
python tools/preprocess.py data/drivebench-test.json data/drivebench-test-final.json
MODEL=/path/to/llava-1.5-7b bash script/llava1.5-7b.sh 1
MODEL=/path/to/internvl2-8b bash script/internvl.sh 1
PYTHONPATH=. python evaluate/eval.py res/llava-1.5-7b/clean.json
```

The launchers run clean, text-only and corrupted conditions and save a separate JSON for each. For open-ended language metrics, install the COCO evaluation dependencies, including Java. [Language evaluator setup](https://github.com/bckim92/language-evaluation#installation-and-usage)

The optional GPT judge accepts `--eval-gpt` with `OPENAI_API_KEY` or `--key`. The DriveLM GPT-Score and the DriveBench task scores are reported separately. In ReCogDrive Table 5, the DriveBench scores use the clean columns in Appendix Table 7 and average the four task scores. The original baseline checkpoint revisions and prediction JSONs are not part of this repository. [ReCogDrive paper](https://arxiv.org/html/2506.08052v2) [DriveBench baseline results](https://github.com/worldbench/DriveBench#benchmark-study)
