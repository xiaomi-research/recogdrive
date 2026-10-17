# Closed-loop simulator evaluation

`evaluator.name` accepts `neuroncap`, `hugsim`, and `alpasim`. Each uses the existing `SubprocessEvaluator` lifecycle: the training process snapshots the checkpoint, the evaluator runs in a child, and `poll()` / `finish()` collect the parsed metrics. If `evaluator.gpus` is unset, the run is deferred until training releases its GPUs.

Every evaluator needs an explicit simulator argv, one result artifact, and a map from ReCogDrive metric names to official result fields. Commands are YAML argument lists, never shell strings. The runner recognizes `{output_dir}`, `{checkpoint}`, `{protocol}`, `{policy_host}`, and `{policy_port}`. For HUGSIM it also recognizes `{hugsim_base_config}` and `{hugsim_launcher}`. Environment mappings are passed to their child process without writing their values into the runner config; use the process environment for secrets.

The full selected `agent` config is serialized as `config/agent/trained_agent.yaml` for both the prestarted services and the HUGSIM-owned client. Optional `policy_overrides` entries use explicit `agent.*=value` paths, for example `agent.trajectory_sampling.time_horizon=5`; unknown paths fail. Evaluation flags such as `grpo`, hidden-state caching, and backbone training are then disabled where those keys exist, while the configured agent target and model-specific options stay intact.

You can evaluate a saved checkpoint without starting training. Use the same Python environment and `PYTHONPATH` used to run ReCogDrive; the output path must be fresh (or empty):

```bash
python scripts/verify/closedloop_eval.py \
  --training-config /path/to/experiment/.hydra/config.yaml \
  --checkpoint /path/to/experiment/last.ckpt \
  --protocol alpasim \
  --output-dir /path/to/new/closedloop-eval
```

The standalone command snapshots the checkpoint and its `_vlm` export, runs the configured protocol through the same subprocess evaluator, and writes `standalone_result.json` plus the normal logs under the output directory. It uses the protocol-specific evaluator section in the saved Hydra config; `evaluator.on_end` does not suppress this explicit run.

```yaml
evaluator:
  name: alpasim
  every_n_epochs: 0
  on_end: true
  gpus: "0"
  alpasim:
    policy_host: 127.0.0.1
    policy_port: 6789
    readiness: tcp
    policy_startup_timeout_s: 300
    simulator_timeout_s: 7200
    simulator_cwd: /workspace/alpasim
    simulator_command:
      - uv
      - run
      - alpasim_wizard
      - deploy=local
      - topology=1gpu
      - driver=manual
      - driver_source=external_static
      - wizard.log_dir={output_dir}/alpasim
      - 'wizard.external_services.driver=["127.0.0.1:{policy_port}"]'
    result_file: alpasim/aggregate/metrics_results.txt
    metrics:
      collision_at_fault: collision_at_fault
      collision_rear: collision_rear
      offroad: offroad
      duration_frac_20s: duration_frac_20s
```

The AlpaSim runner starts ReCogDrive’s gRPC `EgodriverService` first, waits for its TCP port, then starts the official wizard. The wizard's output directory contains `aggregate/metrics_results.txt`; the parser reads each configured metric's official `mean=` value. It does not derive or replace AlpaSim scores from rollout files. See the [official AlpaSim tutorial](https://github.com/NVlabs/alpasim/blob/main/docs/TUTORIAL.md) and [operations guide](https://github.com/NVlabs/alpasim/blob/main/docs/OPERATIONS.md) for scene and deployment setup.

For HUGSIM, supply a copyable official base YAML and run one scenario per evaluator job. The evaluator makes a run-local copy, changes only the base YAML's top-level `output_dir` and `ltf_path`, then lets HUGSIM start the ReCogDrive FIFO client after creating its episode pipes.

```yaml
evaluator:
  name: hugsim
  on_end: true
  gpus: "0"
  hugsim:
    simulator_base_config: /workspace/HUGSIM/configs/sim/nuscenes_base.yaml
    simulator_cwd: /workspace/HUGSIM
    simulator_timeout_s: 7200
    simulator_command:
      - python
      - closed_loop.py
      - --scenario_path
      - /workspace/HUGSIM/configs/benchmark/nuscenes/my_scenario.yaml
      - --base_path
      - "{hugsim_base_config}"
      - --camera_path
      - /workspace/HUGSIM/configs/sim/nuscenes_camera.yaml
      - --kinematic_path
      - /workspace/HUGSIM/configs/sim/kinematic.yaml
      - --ad
      - ltf
      - --ad_cuda
      - "0"
    result_file: hugsim_ltf/*/eval.json
    metrics:
      hdscore: hdscore
      route_completion: rc
      pdms: pdms
```

The evaluator forwards `CHECKPOINT`, the checkpoint's `VLM_PATH`, the serialized agent config, the selected Python executable, and the NAVSIM devkit path to HUGSIM's ReCogDrive client. Supply `NAVSIM_EXP_ROOT`, `OPENSCENE_DATA_ROOT`, map variables, `PYTHONPATH`, and HUGSIM dataset/asset settings through the parent environment or `simulator_env`. The `simulator_cwd` keeps HUGSIM's relative scene and asset paths rooted at its checkout. Its official `closed_loop.py` writes `eval.json` for one `<scene>_<mode>` episode, including `hdscore`, `rc`, and `pdms`; this integration reports that episode's values. It does not claim the full multi-scenario HUGSIM benchmark. See the [official HUGSIM launch interface](https://github.com/hyzhou404/HUGSIM/blob/main/README.md) and [score writer](https://github.com/hyzhou404/HUGSIM/blob/main/closed_loop.py).

NeuroNCAP uses ReCogDrive's HTTP service (`GET /alive`) and starts the configured official runner only after that endpoint responds. For a single scenario/run, point `result_file` at the official run's `metrics.json`; the default mapping reads its numeric `ncap_score`. The raw `any_collide@0.0s` field is a boolean episode indicator, so it is not reported as an aggregate collision rate. For the full benchmark, configure `simulator_command` to invoke the official scenario runner and its official aggregator; the aggregator prints an `Overall` table that can be captured by setting `result_file: simulator.log` and mapping `ncap_score` and numeric `CR@0.0s`. A site-specific wrapper may set the official runner's paths and feed `{policy_port}` to its model endpoint. Set `simulator_timeout_s` to cover the full run (the reference single-machine script documents runs that can take over a day). See the [official run instructions](https://github.com/atonderski/neuro-ncap/blob/main/docs/how-to-run.md), [evaluator](https://github.com/atonderski/neuro-ncap/blob/main/neuro_ncap/components/evaluator.py), [aggregator](https://github.com/atonderski/neuro-ncap/blob/main/scripts/aggregate_results.py), and [scoring protocol](https://github.com/atonderski/neuro-ncap/blob/main/docs/evaluation-protocol-overview.md).

`result_file` may be relative to the evaluator's run directory or an explicit absolute path. It must identify exactly one file and must be updated by the current run; glob patterns that match multiple per-episode files fail rather than silently averaging them. JSON results support exact top-level keys and dotted nested keys. Text results support AlpaSim's `metric: mean=value` rows and NeuroNCAP's printed `Overall` table. Every mapped value must be numeric and finite. Missing files, stale files, nonzero simulator exits, policy exits, and timeouts are reported as evaluator errors.

The runner waits up to `policy_startup_timeout_s` (default 300 s) for the policy and up to `simulator_timeout_s` (default 3600 s) for the simulator. On completion, failure, timeout, or interruption it terminates only process groups that it started. TCP readiness confirms that the AlpaSim service port accepts connections; it does not replace an official simulator rollout or score.

CPU tests use fake processes to exercise readiness, failure, timeout, cleanup, result parsing, and collection. They do not prove official simulator compatibility or produce benchmark scores; those require the official simulator, assets, and configured command.

## Full scene plans and official aggregation

The same evaluator now accepts an ordered `simulator_commands` list instead of the single `simulator_command`. Each command in a multi-command plan must have its own `episode_result_files` path, and an explicit `aggregate_command` must be supplied. The runner executes all scenes, checks that each distinct artifact was created or updated, then starts the official aggregator. Missing/stale/duplicate scene results fail before aggregation. It does not calculate an arithmetic episode mean on behalf of a benchmark.

```yaml
evaluator:
  hugsim:
    simulator_command: []
    simulator_base_config: /workspace/HUGSIM/configs/sim/nuscenes_base.yaml
    simulator_cwd: /workspace/HUGSIM
    simulator_commands:
      - [python, closed_loop.py, --scenario_path, /assets/scenario-a.yaml,
         --base_path, "{hugsim_base_config}", --camera_path, /assets/camera.yaml,
         --kinematic_path, /assets/kinematic.yaml, --ad, ltf, --ad_cuda, "0"]
      - [python, closed_loop.py, --scenario_path, /assets/scenario-b.yaml,
         --base_path, "{hugsim_base_config}", --camera_path, /assets/camera.yaml,
         --kinematic_path, /assets/kinematic.yaml, --ad, ltf, --ad_cuda, "0"]
    episode_result_files:
      - hugsim_ltf/<scene-a>_<mode>/eval.json
      - hugsim_ltf/<scene-b>_<mode>/eval.json
    aggregate_command: [python, /path/to/benchmark-approved-aggregate.py,
                        "{output_dir}/hugsim_ltf", "{output_dir}/official_score.json"]
    aggregate_cwd: /workspace/HUGSIM
    aggregate_timeout_s: 7200
    result_file: official_score.json
    metrics: {score: score}
```

This is a configuration template: replace scene names/modes, asset paths, and the aggregator with the exact program and grading rules of the selected benchmark release. The template does not claim that a script named `benchmark-approved-aggregate.py` is shipped by HUGSIM. If the chosen release supplies only per-episode scores, report per-episode results until its full-benchmark grading rule/program is available; do not label an invented mean as official HD score.

Commands may also use `{episode_index}` and `{episode_output_dir}` for distinct outputs. `aggregate_cwd` defaults to `simulator_cwd`; the simulator timeout applies to each command, and `aggregate_timeout_s` applies separately to scoring. For NeuroNCAP, a full official scenario-runner command can be followed by `[python, scripts/aggregate_results.py, /path/to/this-run-results]` with `result_file: aggregate.log` and mapped `ncap_score`/`CR@0.0s`. Keep the official aggregator's completeness checks enabled. The official [aggregation program](https://github.com/atonderski/neuro-ncap/blob/main/scripts/aggregate_results.py) checks scenario/run files before printing its `Overall` table.

Each run records scene and aggregation exits in `closedloop_run_status.json`, alongside `simulator_0000.log`, subsequent scene logs, and `aggregate.log`. A scoring failure, timeout, invalid metric, or cleanup failure makes the run fail; standalone output and training's error report retain the concrete stage error plus child exit and log path. OS SIGTERM cleanup is checked with a real training process and evaluation child in CPU tests, as well as helper-level checks.
