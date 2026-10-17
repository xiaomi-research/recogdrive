"""Run an official closed-loop simulator beside a ReCogDrive policy service.

The evaluator owns only the processes it starts. NeuroNCAP uses its HTTP
``/alive`` endpoint; AlpaSim uses TCP readiness because its service does not
implement gRPC health. HUGSIM starts its FIFO client itself for every episode,
so this evaluator starts only the configured simulator command for HUGSIM.

Commands are argument arrays (never shell strings). A simulator command may
use ``{output_dir}``, ``{checkpoint}``, ``{protocol}``, ``{policy_host}``, and
``{policy_port}`` placeholders. Its configured result artifact and metric
mapping must identify one official output; this module does not aggregate
per-rollout scores.
"""

import argparse
import glob
import json
import logging
import math
import os
from pathlib import Path, PurePath
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional
from urllib.error import URLError
from urllib.request import urlopen

from omegaconf import OmegaConf

from recogdrive.eval.registry import LAUNCH_ENV, SubprocessEvaluator, build_evaluators, register, snapshot
from recogdrive.infer import vlm_export_path

logger = logging.getLogger(__name__)

DEFAULT_PORTS = {"neuroncap": 9000, "alpasim": 6789}
DEFAULT_METRICS = {
    "neuroncap": {"ncap_score": "ncap_score"},  # per-run collision flags are booleans, not aggregate rates
    "hugsim": {"hdscore": "hdscore", "route_completion": "rc", "pdms": "pdms"},
    "alpasim": {
        "collision_at_fault": "collision_at_fault",
        "collision_rear": "collision_rear",
        "offroad": "offroad",
        "duration_frac_20s": "duration_frac_20s",
    },
}
ENV_PREFIX = "RECOGDRIVE_CLOSEDLOOP_"
PLACEHOLDER = re.compile(
    r"\{(output_dir|checkpoint|protocol|policy_host|policy_port|hugsim_base_config|hugsim_launcher|episode_index|episode_output_dir)\}"
)


class ClosedLoopEvaluator(SubprocessEvaluator):
    """SubprocessEvaluator adapter for the official external simulator runners."""

    protocol = ""

    def __init__(self, cfg, section, output_dir: Path):
        super().__init__(cfg, section, output_dir)
        if self.protocol not in DEFAULT_METRICS:
            raise ValueError(f"unsupported closed-loop protocol: {self.protocol!r}")
        self.policy_owned_by_simulator = bool(
            OmegaConf.select(section, "policy_owned_by_simulator", default=self.protocol == "hugsim")
        )
        if self.protocol == "hugsim" and not self.policy_owned_by_simulator:
            raise ValueError("HUGSIM must start its FIFO policy client from closed_loop.py")

    def _env_values(self, key: str) -> Dict[str, str]:
        values = _to_container(OmegaConf.select(self.section, key))
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError(f"evaluator.{self.protocol}.{key} must be a mapping")
        out = {}
        for name, value in values.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"invalid environment variable name in {key}: {name!r}")
            if value is None or isinstance(value, (dict, list, tuple)):
                raise ValueError(f"{key}.{name} must be a scalar value")
            out[name] = str(value)
        return out

    def child_env(self) -> Dict[str, str]:
        env = super().child_env()
        for kind in ("simulator_env", "policy_env"):
            for name, value in self._env_values(kind).items():
                env[f"{ENV_PREFIX}{kind.upper()}_{name}"] = value
        return env

    @staticmethod
    def _devkit_root() -> Path:
        import navsim

        return Path(navsim.__file__).resolve().parents[1]

    def _agent_config(self, ckpt: Path, out_dir: Path, vlm_path: str) -> Path:
        agent = OmegaConf.to_container(self.cfg.agent, resolve=True)
        if not isinstance(agent, dict):
            raise ValueError("cfg.agent must be a mapping for closed-loop inference")
        merged = OmegaConf.create(agent)
        overrides = OmegaConf.select(self.section, "policy_overrides") or []
        for override in overrides:
            if not isinstance(override, str) or "=" not in override:
                raise ValueError("policy_overrides entries must be dotted agent.* assignments")
            key, value = override.split("=", 1)
            if not key.startswith("agent."):
                raise ValueError(f"policy override must start with 'agent.': {key!r}")
            path = key[len("agent."):]
            node = agent
            parts = path.split(".")
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(f"unknown policy override path: {key!r}")
                node = node[part]
            if not isinstance(node, dict) or parts[-1] not in node:
                raise ValueError(f"unknown policy override path: {key!r}")
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([f"{path}={value}"]))
        agent = OmegaConf.to_container(merged, resolve=True)
        # Preserve the selected model adapter and its complete training config. Only change inference state.
        for key, value in {
            "checkpoint_path": str(ckpt.resolve()), "grpo": False, "cache_mode": False,
            "cache_hidden_state": False, "train_backbone": False, "vlm_path": vlm_path,
        }.items():
            if key in agent:
                agent[key] = value
        config_dir = out_dir / "config"
        agent_dir = config_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(agent), agent_dir / "trained_agent.yaml")
        return config_dir

    def _policy(self, ckpt: Path, out_dir: Path, host: str, port: int, config_dir: Path) -> List[str]:
        devkit = self._devkit_root()
        entry = devkit / "navsim" / "planning" / "script" / "run_recogdrive_infer.py"
        if not entry.is_file():
            raise FileNotFoundError(f"NAVSIM inference entry not found: {entry}")

        searchpath = f"[pkg://navsim.planning.script.config.common,{config_dir.resolve().as_uri()}]"
        hydra_searchpath = "hydra.searchpath=" + searchpath
        def quoted(value) -> str:
            return json.dumps(str(value))

        return [
            sys.executable, str(entry), hydra_searchpath, "agent=trained_agent", "train_test_split=mini",
            "infer.checkpoint=" + quoted(ckpt.resolve()),
            "infer.serve=" + self.protocol,
            "infer.port=" + str(port),
            "infer.host=" + quoted(host),
            "output_dir=" + quoted((out_dir / "policy").resolve()),
        ]

    def command(self, ckpt: Path, out_dir: Path, tag: str) -> List[str]:
        raw_commands = _simulation_commands({
            "simulator_command": _to_container(OmegaConf.select(self.section, "simulator_command")),
            "simulator_commands": _to_container(OmegaConf.select(self.section, "simulator_commands")),
        })
        aggregate = _to_container(OmegaConf.select(self.section, "aggregate_command"))
        aggregate = _argv({"aggregate_command": aggregate}, "aggregate_command") if aggregate else None
        episode_files = _to_container(OmegaConf.select(self.section, "episode_result_files")) or []
        _validate_plan(raw_commands, aggregate, episode_files)
        result_file = OmegaConf.select(self.section, "result_file")
        result_is_default = not isinstance(result_file, str) or not result_file.strip()
        if result_is_default:
            defaults = {"neuroncap": "metrics.json", "hugsim": "eval.json",
                        "alpasim": "aggregate/metrics_results.txt"}
            result_file = defaults[self.protocol]
        configured_metrics = _to_container(OmegaConf.select(self.section, "metrics"))
        metrics = DEFAULT_METRICS[self.protocol] if configured_metrics in (None, {}) else configured_metrics
        if not isinstance(metrics, dict) or not metrics or any(
                not isinstance(k, str) or not isinstance(v, str) or not v for k, v in metrics.items()):
            raise ValueError(f"evaluator.{self.protocol}.metrics must map metric names to official field names")

        host = str(OmegaConf.select(self.section, "policy_host") or "127.0.0.1")
        configured_port = OmegaConf.select(self.section, "policy_port")
        port = int(DEFAULT_PORTS.get(self.protocol, 0) if configured_port is None else configured_port)
        if not self.policy_owned_by_simulator and port <= 0:
            raise ValueError("policy_port must be a positive TCP port")
        hugsim_base = None
        if self.protocol == "hugsim" and OmegaConf.select(self.section, "simulator_base_config"):
            source = Path(str(OmegaConf.select(self.section, "simulator_base_config"))).expanduser().resolve()
            base = OmegaConf.load(source)
            if not isinstance(OmegaConf.select(base, "output_dir"), str):
                raise ValueError(f"HUGSIM base config has no string 'output_dir' field: {source}")
            hugsim_base = out_dir / "config" / "hugsim_base.yaml"
            hugsim_base.parent.mkdir(parents=True, exist_ok=True)
            base.output_dir = str((out_dir / "hugsim_").resolve())
            repo_root = Path(__file__).resolve().parents[2]
            base.ltf_path = str(repo_root / "scripts" / "infer" / "run_recogdrive_hugsim.sh")
            OmegaConf.save(base, hugsim_base)
            if result_is_default:
                result_file = "hugsim_ltf/*/eval.json"
        values = {
            "output_dir": str(out_dir.resolve()), "checkpoint": str(ckpt.resolve()),
            "protocol": self.protocol, "policy_host": host, "policy_port": str(port),
            "hugsim_base_config": str(hugsim_base.resolve()) if hugsim_base else "",
            "hugsim_launcher": str(Path(__file__).resolve().parents[2] / "scripts" / "infer" /
                                    "run_recogdrive_hugsim.sh"),
        }
        simulator_commands = []
        expanded_episode_files = []
        for index, raw_command in enumerate(raw_commands):
            episode_values = {**values, "episode_index": str(index),
                              "episode_output_dir": str((out_dir / "episodes" / f"{index:04d}").resolve())}
            simulator_commands.append([_expand(arg, episode_values) for arg in raw_command])
            if episode_files:
                expanded_episode_files.append(_expand(episode_files[index], episode_values))
        aggregate_command = [_expand(arg, values) for arg in aggregate] if aggregate else None
        if expanded_episode_files and len(set(expanded_episode_files)) != len(expanded_episode_files):
            raise ValueError("episode_result_files must identify distinct episode artifacts")
        vlm_path = str(vlm_export_path(ckpt) or OmegaConf.select(self.cfg, "agent.vlm_path") or "")
        config_dir = self._agent_config(ckpt, out_dir, vlm_path)
        devkit_root = self._devkit_root()
        policy_command = None if self.policy_owned_by_simulator else self._policy(
            ckpt, out_dir, host, port, config_dir)
        simulator_cwd = OmegaConf.select(self.section, "simulator_cwd")
        config = {
            "protocol": self.protocol,
            "policy_owned_by_simulator": self.policy_owned_by_simulator,
            "policy_command": policy_command,
            "agent_config_dir": str(config_dir.resolve()),
            "navsim_devkit_root": str(devkit_root),
            "policy_python": sys.executable,
            "policy_host": host,
            "policy_port": port,
            "readiness": str(OmegaConf.select(self.section, "readiness") or
                             ("http" if self.protocol == "neuroncap" else "tcp")),
            "policy_startup_timeout_s": _timeout(
                OmegaConf.select(self.section, "policy_startup_timeout_s"), 300, "policy_startup_timeout_s"),
            "simulator_timeout_s": _timeout(
                OmegaConf.select(self.section, "simulator_timeout_s"), 3600, "simulator_timeout_s"),
            "simulator_command": simulator_commands[0] if len(simulator_commands) == 1 else None,
            "simulator_commands": simulator_commands if len(simulator_commands) > 1 else None,
            "episode_result_files": expanded_episode_files,
            "aggregate_command": aggregate_command,
            "aggregate_cwd": _expand(str(OmegaConf.select(self.section, "aggregate_cwd")), values)
                if OmegaConf.select(self.section, "aggregate_cwd") else None,
            "aggregate_timeout_s": _timeout(OmegaConf.select(self.section, "aggregate_timeout_s"),
                                               3600, "aggregate_timeout_s"),
            "simulator_cwd": _expand(str(simulator_cwd), values) if simulator_cwd else None,
            "result_file": result_file,
            "metrics": metrics,
            "simulator_env": list(self._env_values("simulator_env")),
            "policy_env": list(self._env_values("policy_env")),
            "checkpoint": str(ckpt.resolve()),
            "vlm_path": vlm_path,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        config_path = out_dir / "closedloop_runner.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        try:
            config_path.chmod(0o600)
        except OSError:
            pass
        runner = Path(__file__).resolve().parents[2] / "scripts" / "verify" / "closedloop_eval.py"
        return [sys.executable, str(runner), str(config_path)]

    def read_result(self, out_dir: Path) -> Dict[str, float]:
        result = json.loads((out_dir / "recogdrive_result.json").read_text(encoding="utf-8"))
        metrics = result.get("metrics") if isinstance(result, dict) else None
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("closed-loop runner produced no metrics")
        return _finite_metrics(metrics)

    def failure_result(self, out_dir: Path, returncode: int) -> Dict:
        try:
            status = json.loads((out_dir / "closedloop_run_status.json").read_text(encoding="utf-8"))
            error = status.get("error") if isinstance(status, dict) else None
            if not error and isinstance(status, dict) and status.get("cleanup_errors"):
                error = f"owned-process cleanup failed: {status['cleanup_errors']}"
            if isinstance(error, str) and error:
                return {"error": f"{error}; child exit {returncode}, see {out_dir / 'eval.log'}"}
        except (OSError, ValueError, TypeError):
            pass
        return super().failure_result(out_dir, returncode)


@register("neuroncap")
class NeuroNCAPEvaluator(ClosedLoopEvaluator):
    protocol = "neuroncap"


@register("hugsim")
class HUGSIMEvaluator(ClosedLoopEvaluator):
    protocol = "hugsim"


@register("alpasim")
class AlpaSimEvaluator(ClosedLoopEvaluator):
    protocol = "alpasim"


def _expand(value: str, values: Mapping[str, str]) -> str:
    def replace(match):
        name = match.group(1)
        if name not in values:
            raise ValueError(f"unknown closed-loop command placeholder {{{name}}}")
        return values[name]
    return PLACEHOLDER.sub(replace, value)


def _to_container(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _timeout(value, default: float, name: str) -> float:
    value = default if value is None else value
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and greater than zero")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and greater than zero") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number


def _clean_env(kind: str, keys: List[str]) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in LAUNCH_ENV and not k.startswith("TORCHELASTIC") and not k.startswith(ENV_PREFIX)}
    for name in keys:
        key = f"{ENV_PREFIX}{kind.upper()}_{name}"
        if key not in os.environ:
            raise ValueError(f"missing evaluator environment value for {name}")
        env[name] = os.environ[key]
    return env


def _probe_ready(host: str, port: int, readiness: str) -> bool:
    address = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    if readiness == "tcp":
        try:
            with socket.create_connection((address, port), timeout=0.5):
                return True
        except OSError:
            return False
    if readiness == "http":
        try:
            with urlopen(f"http://{address}:{port}/alive", timeout=0.5) as response:
                return response.status == 200
        except (OSError, URLError, TimeoutError):
            return False
    raise ValueError(f"readiness must be 'http' or 'tcp', got {readiness!r}")


def _require_port_free(host: str, port: int) -> None:
    """Refuse a known occupied service port instead of accepting another process's readiness."""
    if port <= 0 or port > 65535:
        raise ValueError("policy_port must be a positive TCP port")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"policy service port {host}:{port} is already in use; refusing to use an unowned listener"
            ) from exc


def _wait_ready(proc: subprocess.Popen, host: str, port: int, readiness: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f"policy service exited before readiness (exit {code}); see policy.log")
        if _probe_ready(host, port, readiness):
            return
        time.sleep(0.1)
    raise TimeoutError(f"policy service did not become ready at {host}:{port} within {timeout_s:g}s")


def _stop_owned(proc: Optional[subprocess.Popen], grace_s: float = 3.0) -> None:
    if proc is None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            proc.poll()  # reap the direct child before checking for its remaining owned descendants
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _argv(config: dict, key: str) -> List[str]:
    command = config.get(key)
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError(f"{key} must be a non-empty argv list; shell strings are not accepted")
    return command


def _simulation_commands(config: dict) -> List[List[str]]:
    many = config.get("simulator_commands")
    if many:
        if config.get("simulator_command"):
            raise ValueError("set simulator_command or simulator_commands, not both")
        if not isinstance(many, list):
            raise ValueError("simulator_commands must be a list of argv lists")
        return [_argv({"command": command}, "command") for command in many]
    return [_argv(config, "simulator_command")]


def _validate_plan(commands, aggregate, episode_files) -> None:
    if not isinstance(episode_files, list) or any(not isinstance(path, str) or not path for path in episode_files):
        raise ValueError("episode_result_files must be a list of result paths")
    if episode_files and len(episode_files) != len(commands):
        raise ValueError("episode_result_files must provide one artifact per simulator command")
    if len(commands) > 1 and (not aggregate or len(episode_files) != len(commands)):
        raise ValueError("multi-scenario evaluation requires episode_result_files and an official aggregate_command")


def _start(command: List[str], env: dict, log_path: Path, cwd=None) -> subprocess.Popen:
    kwargs = {"env": env, "cwd": cwd, "stdout": None, "stderr": subprocess.STDOUT}
    kwargs["start_new_session"] = os.name == "posix"
    with log_path.open("w", encoding="utf-8") as log:
        kwargs["stdout"] = log
        return subprocess.Popen(command, **kwargs)


def _wait_simulator(simulator: subprocess.Popen, policy: Optional[subprocess.Popen], timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        code = simulator.poll()
        if code is not None:
            return code
        if policy is not None and policy.poll() is not None:
            _stop_owned(simulator)
            raise RuntimeError("policy service exited while the simulator was running; see policy.log")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_owned(simulator)
            raise TimeoutError(f"simulator exceeded its {timeout_s:g}s timeout; see simulator.log")
        time.sleep(min(0.1, remaining))


def _result_paths(result_file: str, out_dir: Path) -> List[Path]:
    pattern = Path(result_file)
    if not pattern.is_absolute() and ".." in PurePath(result_file).parts:
        raise ValueError("relative result_file cannot contain '..'")
    search = str(pattern if pattern.is_absolute() else out_dir / pattern)
    paths = [Path(path) for path in glob.glob(search, recursive=True) if Path(path).is_file()]
    root = out_dir.resolve()
    for path in paths:
        if not Path(result_file).is_absolute() and not path.resolve().is_relative_to(root):
            raise ValueError(f"result_file escaped the evaluation directory: {path}")
    if len(paths) != 1:
        raise ValueError(f"result_file {result_file!r} matched {len(paths)} files; select one official aggregate artifact")
    return paths


def _fingerprint(paths: List[Path]) -> Dict[str, tuple]:
    return {str(path.resolve()): (path.stat().st_mtime_ns, path.stat().st_size) for path in paths}


def _json_value(document, dotted: str):
    if isinstance(document, dict) and dotted in document:
        return document[dotted]
    value = document
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"official metric field {dotted!r} is missing")
        value = value[part]
    return value


def _text_values(text: str) -> Dict[str, float]:
    values = {}
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for line in text.splitlines():
        match = re.match(rf"\s*([^:|]+):\s*mean\s*=\s*({number})", line)
        if match:
            values[match.group(1).strip()] = float(match.group(2))
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in text.splitlines() if line.strip().startswith("|")]
    for index, row in enumerate(rows[:-1]):
        if row and row[0] == "Overall" and rows[index + 1] and rows[index + 1][0] == "model":
            values.update({key: float(value) for key, value in zip(row[1:], rows[index + 1][1:])})
    return values


def _official_values(path: Path, metrics: dict) -> Dict[str, float]:
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        raw = {name: _json_value(document, field) for name, field in metrics.items()}
    else:
        raw_text = _text_values(path.read_text(encoding="utf-8"))
        raw = {name: raw_text[field] if field in raw_text else _missing(field) for name, field in metrics.items()}
    return _finite_metrics(raw)


def _missing(field: str):
    raise KeyError(f"official metric field {field!r} is missing from the result artifact")


def _finite_metrics(metrics: dict) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} must be a finite numeric value, got {value!r}")
        out[str(key)] = float(value)
    if not out:
        raise ValueError("official result contains no mapped metrics")
    return out


def run(config: dict, out_dir: Path) -> Dict[str, float]:
    """Run all configured scenes and an optional official aggregator, without inventing scores."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = config.get("protocol")
    simulator_commands = _simulation_commands(config)
    aggregate_command = _argv(config, "aggregate_command") if config.get("aggregate_command") else None
    episode_files = config.get("episode_result_files") or []
    _validate_plan(simulator_commands, aggregate_command, episode_files)
    metrics = config.get("metrics")
    if protocol not in DEFAULT_METRICS or not isinstance(metrics, dict) or not metrics:
        raise ValueError("runner config needs a known protocol and non-empty official metric mapping")
    policy_timeout = _timeout(config.get("policy_startup_timeout_s"), 300, "policy_startup_timeout_s")
    simulator_timeout = _timeout(config.get("simulator_timeout_s"), 3600, "simulator_timeout_s")
    aggregate_timeout = _timeout(config.get("aggregate_timeout_s"), 3600, "aggregate_timeout_s")
    before_paths = _result_paths(config["result_file"], out_dir) if glob.glob(
        str(Path(config["result_file"]) if Path(config["result_file"]).is_absolute()
            else out_dir / config["result_file"]), recursive=True) else []
    before = _fingerprint(before_paths)
    policy = simulator = aggregator = None
    seen_episode_paths = set()
    status = {"protocol": protocol, "success": False, "stages": []}
    try:
        if not config.get("policy_owned_by_simulator", False):
            policy_command = _argv(config, "policy_command")
            _require_port_free(config["policy_host"], int(config["policy_port"]))
            policy_env = _clean_env("POLICY_ENV", config.get("policy_env", []))
            if protocol == "alpasim":
                policy_env["ALPASIM_DRIVER_HOST"] = str(config["policy_host"])
                policy_env["ALPASIM_DRIVER_PORT"] = str(config["policy_port"])
            policy = _start(policy_command, policy_env, out_dir / "policy.log")
            _wait_ready(policy, config["policy_host"], int(config["policy_port"]),
                        config["readiness"], policy_timeout)

        sim_env = _clean_env("SIMULATOR_ENV", config.get("simulator_env", []))
        if protocol == "hugsim" and config.get("policy_owned_by_simulator", False):
            sim_env["CHECKPOINT"] = str(config["checkpoint"])
            sim_env["VLM_PATH"] = str(config.get("vlm_path", ""))
            sim_env["AGENT_CONFIG_DIR"] = str(config["agent_config_dir"])
            sim_env["RECOGDRIVE_PYTHON"] = str(config["policy_python"])
            sim_env["NAVSIM_DEVKIT_ROOT"] = str(config["navsim_devkit_root"])
        for index, simulator_command in enumerate(simulator_commands):
            log_name = "simulator.log" if len(simulator_commands) == 1 else f"simulator_{index:04d}.log"
            previous_episode = {}
            if episode_files:
                pattern = episode_files[index]
                previous_paths = _result_paths(pattern, out_dir) if glob.glob(
                    str(Path(pattern) if Path(pattern).is_absolute() else out_dir / pattern), recursive=True) else []
                previous_episode = _fingerprint(previous_paths)
            stage = {"kind": "simulator", "index": index, "log": log_name, "returncode": None}
            status["stages"].append(stage)
            simulator = _start(simulator_command, sim_env, out_dir / log_name, config.get("simulator_cwd"))
            stage["returncode"] = _wait_simulator(simulator, policy, simulator_timeout)
            if stage["returncode"] != 0:
                raise RuntimeError(f"simulator {index} exited with status {stage['returncode']}; see {log_name}")
            if episode_files:
                paths = _result_paths(episode_files[index], out_dir)
                if _fingerprint(paths) == previous_episode:
                    raise ValueError(f"episode {index} result artifact was not updated by this run: {paths[0]}")
                resolved = paths[0].resolve()
                if resolved in seen_episode_paths:
                    raise ValueError(f"episode artifacts must be distinct: {resolved}")
                seen_episode_paths.add(resolved)
                stage["result_file"] = str(paths[0])
            _stop_owned(simulator)
            simulator = None

        if aggregate_command:
            stage = {"kind": "aggregate", "log": "aggregate.log", "returncode": None}
            status["stages"].append(stage)
            aggregator = _start(aggregate_command, sim_env, out_dir / "aggregate.log",
                                config.get("aggregate_cwd") or config.get("simulator_cwd"))
            stage["returncode"] = _wait_simulator(aggregator, None, aggregate_timeout)
            if stage["returncode"] != 0:
                raise RuntimeError(f"official aggregator exited with status {stage['returncode']}; see aggregate.log")

        paths = _result_paths(config["result_file"], out_dir)
        if _fingerprint(paths) == before:
            raise ValueError(f"result artifact was not updated by this run: {paths[0]}")
        result = _official_values(paths[0], metrics)
        payload = {"protocol": protocol, "result_file": str(paths[0]), "metrics": result}
        (out_dir / "recogdrive_result.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        status["success"] = True
        return result
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_errors = []
        for owned in (aggregator, simulator, policy):
            try:
                _stop_owned(owned)
            except Exception as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if cleanup_errors:
            status["success"] = False
            status["cleanup_errors"] = cleanup_errors
        (out_dir / "closedloop_run_status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False), encoding="utf-8")
        if cleanup_errors and "error" not in status:
            raise RuntimeError(f"owned-process cleanup failed: {cleanup_errors}")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a ReCogDrive closed-loop simulator evaluation")
    parser.add_argument("config", nargs="?", type=Path,
                        help="internal runner config (normally generated by a training evaluator)")
    parser.add_argument("--training-config", type=Path,
                        help="saved full Hydra config, for evaluating a checkpoint without retraining")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint to evaluate in standalone mode")
    parser.add_argument("--protocol", choices=sorted(DEFAULT_METRICS), help="closed-loop protocol")
    parser.add_argument("--output-dir", type=Path, help="fresh output directory for standalone mode")
    args = parser.parse_args()
    standalone = (args.training_config, args.checkpoint, args.protocol, args.output_dir)
    if any(value is not None for value in standalone):
        if args.config or any(value is None for value in standalone):
            parser.error("standalone mode requires --training-config, --checkpoint, --protocol, and --output-dir only")
        return _standalone(args.training_config, args.checkpoint, args.protocol, args.output_dir)
    if args.config is None:
        parser.error("provide the internal config or all standalone options")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        run(config, args.config.parent)
    except BaseException:
        logger.exception("closed-loop evaluation failed")
        raise
    return 0


def _standalone(training_config: Path, checkpoint: Path, protocol: str, output_dir: Path) -> int:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"standalone output directory must be empty: {output_dir}")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    cfg = OmegaConf.load(training_config)
    OmegaConf.set_struct(cfg, False)
    if OmegaConf.select(cfg, "evaluator") is None:
        raise ValueError("training config has no evaluator section")
    cfg.evaluator.name = protocol
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "training_config.yaml")
    frozen_checkpoint = snapshot(checkpoint, output_dir / "checkpoint")
    evaluators = build_evaluators(cfg, output_dir)
    if len(evaluators) != 1:
        raise RuntimeError(f"expected one {protocol} evaluator, built {len(evaluators)}")
    evaluator = evaluators[0]
    try:
        evaluator.submit(frozen_checkpoint, "standalone")
        results = evaluator.collect(block=True)
    except BaseException:
        evaluator.cancel()
        raise
    if not results:
        raise RuntimeError("closed-loop evaluator completed without a result")
    _, metrics = results[0]
    payload = {"protocol": protocol, "checkpoint": str(checkpoint), "metrics": metrics}
    result_path = output_dir / "standalone_result.json"
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 1 if isinstance(metrics, dict) and "error" in metrics else 0


def _handle_term(signum, _frame):
    raise InterruptedError(f"closed-loop runner interrupted by signal {signum}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_term)
    try:
        raise SystemExit(_main())
    except KeyboardInterrupt:
        raise SystemExit(130)
