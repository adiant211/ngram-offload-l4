#!/usr/bin/env python3
"""exp_train launcher: one TOML file drives the trainer.

  python exp.py check                       # GPU, packages, parity checks (~5 min, needs the GPU)
  python exp.py prepare                     # tokenize FineWeb into data/ (sizes in [prepare])
  python exp.py train                       # train; resumes from the newest checkpoint after a crash
  python exp.py train --dry-run             # print the trainer command and exit
  python exp.py stop                        # graceful stop: finish the update, checkpoint, exit

  -c FILE                 use another config (default: exp_config.toml)
  --set SECTION.KEY=VAL   override one setting, repeatable (e.g. --set batch.micro_tokens=10240)

Every run lives in runs/<run.name>/: the config it was started with, train.log, metrics.jsonl
(one line per update), results.jsonl (one summary per process) and ckpt/. Re-running
`python exp.py train` with the same run.name continues that run.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"

# Allowed keys per section (typos are errors, not silently ignored settings).
TRAINER_SECTIONS = {
    "model": {"preset", "ngram_params", "layer_drop", "layer_drop_dist", "layer_drop_schedule",
              "layer_drop_final", "layer_drop_end", "depth_attn", "delta_block", "delta_blocks", "delta_variant"},
    "batch": {"micro_tokens", "grad_accum", "seq_len"},
    "schedule": {"steps", "warmup_steps", "decay_frac", "min_lr_frac", "stop_at", "max_minutes"},
    "optimizer": {"optimizer", "dion_lr", "dion_fraction", "dion_mu", "dion_weight_decay", "scalar_lr",
                  "adamw_lr", "grad_clip", "dion_triton", "dion_compile"},
    "performance": {"compile", "compile_mode", "loss_chunks", "fast_ce", "fla_nosync", "residual_dtype",
                    "update_threads", "gpu_mem_cap_gib"},
    "eval": {"eval_every", "eval_tokens", "eval_exit_at"},
    "checkpoint": {"ckpt_minutes", "ckpt_every", "keep", "resume", "max_skips"},
}
OTHER_SECTIONS = {
    "run": {"name", "seed", "runs_dir", "cache_dir", "max_restarts"},
    "data": {"train", "val", "tokenizer"},
    "prepare": {"train_tokens", "val_tokens"},
}
BOOL_OPTIONAL = {"fast_ce", "fla_nosync", "dion_triton", "dion_compile"}   # --x / --no-x
STORE_TRUE = {"delta_block"}                                                # --x when true
ZERO_MEANS_DEFAULT = {"update_threads", "gpu_mem_cap_gib"}                  # 0: let the trainer decide
SHAPE_KEYS = {"d_model", "n_layers", "full_attn_every", "n_heads", "n_kv_heads", "head_dim", "gdn_heads",
              "gdn_head_dim", "ffn_dim", "rope_theta", "ngram.dim", "ngram.inject_layers", "ngram.orders",
              "ngram.heads_per_order", "ngram.lr", "ngram.beta2", "ngram.canonical_tokens",
              "ngram.stochastic_round"}

# Measured on one A100-SXM4-80GB, 700m preset, ~786k-token updates, current stack (see RESULTS.md).
MEASURED = {10240: (26379, 30.6), 16384: (27504, 42.7), 20480: (27502, 51.2), 24576: (28145, 58.1)}


def split_key(key):
    """'model.shape."ngram.dim"' -> ['model', 'shape', 'ngram.dim'] (quoted parts may contain dots)."""
    parts, current, quoted = [], "", False
    for ch in key:
        if ch == '"':
            quoted = not quoted
        elif ch == "." and not quoted:
            parts.append(current)
            current = ""
        else:
            current += ch
    return parts + [current]


def load_config(path, sets):
    config = tomllib.loads(Path(path).read_text())
    for item in sets:
        key, value = item.split("=", 1)
        *sections, name = split_key(key)
        try:
            value = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            pass                                    # bare strings: --set run.name=small
        node = config
        for part in sections:
            node = node.setdefault(part, {})
        node[name] = value
    model = config.get("model", {})
    if model.get("layer_drop_end") == "decay":     # reach layer_drop_final when the LR decay starts
        model["layer_drop_end"] = round(1 - config.get("schedule", {}).get("decay_frac", 0.2), 6)
    allowed = {**TRAINER_SECTIONS, **OTHER_SECTIONS}
    for section, values in config.items():
        if section == "model" and "shape" in values:
            for key in values["shape"]:
                if key not in SHAPE_KEYS:
                    raise SystemExit(f"unknown setting model.shape.{key} (known: {sorted(SHAPE_KEYS)})")
        if section not in allowed:
            raise SystemExit(f"unknown section [{section}] (known: {sorted(allowed)})")
        for key in values:
            if key not in allowed[section] and not (section == "model" and key == "shape"):
                raise SystemExit(f"unknown setting {section}.{key} (known: {sorted(allowed[section])})")
    return config


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else HERE / path


def run_dir(config):
    return path_of(config["run"].get("runs_dir", "runs")) / config["run"]["name"]


def trainer_args(config, attempt=0):
    args = []
    for section, keys in TRAINER_SECTIONS.items():
        for key, value in config.get(section, {}).items():
            if key == "shape":
                continue
            flag = "--" + key.replace("_", "-")
            if key in BOOL_OPTIONAL:
                args.append(flag if value else "--no-" + key.replace("_", "-"))
            elif key in STORE_TRUE:
                args += [flag] if value else []
            elif key in ZERO_MEANS_DEFAULT and not value:
                continue
            else:
                args += [flag, str(value)]
    for key, value in config.get("model", {}).get("shape", {}).items():
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        args += ["--model-override", f"{key}={value}"]
    data = config["data"]
    val = path_of(data["val"]) if data.get("val") else None
    rd = run_dir(config)
    args += ["--data", str(path_of(data["train"])), "--eval-data", str(val) if val and val.exists() else "",
             "--tokenizer", str(path_of(data["tokenizer"])), "--seed", str(config["run"].get("seed", 1234)),
             "--ckpt-dir", str(rd / "ckpt"), "--metrics", str(rd / "metrics.jsonl"),
             "--output", str(rd / "results.jsonl"), "--run-name", f"{config['run']['name']}-a{attempt}"]
    return args


def summary(config):
    b, s, m = config["batch"], config["schedule"], config["model"]
    update = b["micro_tokens"] * b["grad_accum"]
    lines = [f"run {config['run']['name']}: {m['preset']} preset, {m['ngram_params'] / 1e9:.2f}B n-gram table",
             f"update {b['micro_tokens']:,} x {b['grad_accum']} = {update:,} tokens; schedule {s['steps']:,} updates "
             f"= {update * s['steps'] / 1e9:.2f}B tokens, warmup {s['warmup_steps']} updates"]
    if m["preset"] == "700m" and not m.get("shape") and b["micro_tokens"] in MEASURED:
        tps, gib = MEASURED[b["micro_tokens"]]
        if m.get("layer_drop"):
            tps = int(tps * (1 + 0.43 * m["layer_drop"]))   # ~+8.6% at p_max 0.2, constant schedule
        hours = update * s["steps"] / tps / 3600
        lines.append(f"A100-80GB reference: ~{tps:,} tokens/s, ~{gib} GiB GPU memory, "
                     f"full schedule ~{hours:.0f} h ({hours / 24:.1f} days)")
    if m.get("delta_block"):
        lines.append("delta_block on: measured -14% throughput in this stack (see SETTINGS.md)")
    return "\n".join(lines)


def newest_checkpoint(rd):
    done = sorted(d.name for d in (rd / "ckpt").glob("step_*") if (d / "COMPLETE").exists())
    return done[-1] if done else None


def train(config, args):
    rd = run_dir(config)
    cmd = [sys.executable, "-u", "trainer.py"] + trainer_args(config)
    print(summary(config), flush=True)
    if args.dry_run:
        print("COMMAND (cwd code/):", " ".join(cmd))
        return 0
    if not path_of(config["data"]["train"]).exists():
        raise SystemExit(f"missing {config['data']['train']}: run `python exp.py prepare` first")
    rd.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (rd / f"config-{stamp}.toml").write_text(args.config_text)
    cache = path_of(config["run"].get("cache_dir", "compile-cache"))
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
    (rd / "supervisor.pid").write_text(str(os.getpid()))
    state = dict(child=None, stop=False)

    def forward(signum, frame):
        state["stop"] = True
        if state["child"] is not None and state["child"].poll() is None:
            state["child"].send_signal(signal.SIGTERM)   # trainer: finish update, checkpoint, exit 0
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    log = open(rd / "train.log", "a")
    code, stalled = 1, 0
    for attempt in range(config["run"].get("max_restarts", 20) + 1):
        before = newest_checkpoint(rd)
        message = f"SUPERVISOR start {attempt} {time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(message, flush=True)
        log.write(message + "\n")
        cmd = [sys.executable, "-u", "trainer.py"] + trainer_args(config, attempt)
        state["child"] = subprocess.Popen(cmd, cwd=CODE, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True)
        for line in state["child"].stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = state["child"].wait()
        if code == 0 or state["stop"]:
            break
        # A failure that repeats without any new checkpoint would loop forever: stop and report it.
        stalled = stalled + 1 if newest_checkpoint(rd) == before else 0
        if stalled >= 3:
            message = "SUPERVISOR 3 failures in a row without a new checkpoint: giving up, see train.log"
            print(message, flush=True)
            log.write(message + chr(10))
            break
        message = f"SUPERVISOR trainer exited with {code}; resuming from the last checkpoint in 15 s"
        print(message, flush=True)
        log.write(message + "\n")
        time.sleep(15)
    (rd / "supervisor.pid").unlink(missing_ok=True)
    print(f"SUPERVISOR done (exit {code})", flush=True)
    return code


def prepare(config):
    data, sizes = config["data"], config.get("prepare", {})
    train, val = path_of(data["train"]), path_of(data["val"])
    assert train.parent == val.parent and train.name.endswith("-train.bin") and val.name.endswith("-val.bin"), \
        "data.train / data.val must be <dir>/<prefix>-train.bin and <dir>/<prefix>-val.bin"
    prefix = train.name[:-len("-train.bin")]
    train.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(CODE / "prepare_fineweb.py"), "--tokenizer", str(path_of(data["tokenizer"])),
           "--train-tokens", str(int(sizes.get("train_tokens", 1_000_000_000))),
           "--val-tokens", str(int(sizes.get("val_tokens", 2_097_153))), "--prefix", prefix]
    print("PREPARE", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=train.parent)


def check():
    steps = [
        ("GPU", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]),
        ("packages", [sys.executable, "-c", "import torch, fla, dion, tokenizers; "
                      "print('torch', torch.__version__, 'cuda', torch.version.cuda, '| fla', fla.__version__, "
                      "'| gpu', torch.cuda.get_device_name(0))"]),
        ("n-gram RAM path parity", [sys.executable, "-u", "pinned_ngram.py"]),
        ("FLA sync removal parity", [sys.executable, "-u", "fla_nosync.py"]),
        ("fast cross-entropy parity", [sys.executable, "-u", "fast_flce.py"]),
        ("Delta Block parity (FP32)", [sys.executable, "-u", "delta_block.py"]),
    ]
    failed = []
    for name, cmd in steps:
        print(f"== {name}", flush=True)
        if subprocess.call(cmd, cwd=CODE) != 0:
            failed.append(name)
    print("CHECK", "passed" if not failed else f"FAILED: {failed}", flush=True)
    return 1 if failed else 0


def stop(config):
    pid_file = run_dir(config) / "supervisor.pid"
    if not pid_file.exists():
        raise SystemExit(f"no running supervisor for {config['run']['name']} ({pid_file} not found)")
    os.kill(int(pid_file.read_text()), signal.SIGTERM)
    print("sent SIGTERM: the trainer finishes its current update, writes a checkpoint and exits")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "prepare", "check", "stop"))
    p.add_argument("-c", "--config", default=str(HERE / "exp_config.toml"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    config = load_config(args.config, args.set)
    args.config_text = Path(args.config).read_text() + "".join(f"\n# --set {s}" for s in args.set)
    if args.command == "train":
        return train(config, args)
    if args.command == "prepare":
        return prepare(config)
    if args.command == "check":
        return check()
    return stop(config)


if __name__ == "__main__":
    sys.exit(main())
