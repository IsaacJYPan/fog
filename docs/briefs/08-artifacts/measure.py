"""Brief 08 timing shims. No model, loss or trainer source is changed.

Run `python docs/briefs/08-artifacts/measure.py ppo|replay` from the repository.
Set FOG08_OUTPUT and FOG08_MANIFEST. PPO reuses brief 06's timing hooks and
command, returning the trainer's normal halt signal after two saved updates.
"""
from __future__ import annotations

import contextlib
import ctypes
import csv
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("FOG08_OUTPUT", "runs/update-speed/evidence"))
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, value):
    (OUT / name).write_text(json.dumps({"label": "smoke", **value}, indent=2) + "\n")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Parts(contextlib.AbstractContextManager):
    """Synchronized, non-overlapping forward blocks and one native optimizer step."""

    def __init__(self, agent, optimizer, name):
        self.agent, self.optimizer, self.name = agent, optimizer, name
        self.seconds = {}
        self.handles = []
        self.finished = False
        self.started = None
        self.rows = None
        self.calls = {}

    def clock(self):
        sync()
        return time.perf_counter()

    def __enter__(self):
        def before_agent(_module, args):
            if not self.finished:
                self.seconds.clear()
                self.calls.clear()
                self.started = self.clock()
                key = "actions" if "actions" in args[0] else "action"
                self.rows = len(args[0][key])

        def after_agent(_module, _args, _output):
            if not self.finished:
                self.forward_end = self.clock()
                self.forward_seconds = self.forward_end - self.started
                self.seconds["forward_other"] = self.forward_seconds - sum(self.seconds.values())

        self.handles.extend([self.agent.register_forward_pre_hook(before_agent),
                             self.agent.register_forward_hook(after_agent)])
        for attribute, category in (("backbone", "shared_trunk"),
                                    ("shared_latent", "shared_trunk"),
                                    ("belief", "opponent_belief"),
                                    ("self_belief", "self_belief"),
                                    ("policy_value", "policy_value_heads")):
            module = getattr(self.agent, attribute, None)
            if module is None:
                continue
            starts = []

            def before(_m, _a, starts=starts):
                if not self.finished:
                    starts.append(self.clock())

            def after(_m, _a, _o, starts=starts, category=category):
                if not self.finished:
                    self.seconds[category] = self.seconds.get(category, 0) + self.clock() - starts.pop()
                    self.calls[category] = self.calls.get(category, 0) + 1

            self.handles.extend([module.register_forward_pre_hook(before),
                                 module.register_forward_hook(after)])
        self.original_backward = torch.autograd.backward
        self.original_step = self.optimizer.step

        def backward(*args, **kwargs):
            if self.finished:
                return self.original_backward(*args, **kwargs)
            start = self.clock()
            self.seconds["loss_assembly"] = start - self.forward_end
            result = self.original_backward(*args, **kwargs)
            self.backward_end = self.clock()
            self.seconds["backward"] = self.backward_end - start
            return result

        def step(*args, **kwargs):
            if self.finished:
                return self.original_step(*args, **kwargs)
            start = self.clock()
            self.seconds["gradient_clip_and_metrics"] = start - self.backward_end
            result = self.original_step(*args, **kwargs)
            end = self.clock()
            self.seconds["optimizer_step"] = end - start
            self.total = end - self.started
            self.finished = True
            return result

        torch.autograd.backward = backward
        self.optimizer.step = step
        return self

    def __exit__(self, *exc):
        torch.autograd.backward = self.original_backward
        self.optimizer.step = self.original_step
        for handle in self.handles:
            handle.remove()
        if self.finished:
            dump(self.name, {"rows": self.rows, "seconds": self.seconds,
                            "shares": {k: v / self.total for k, v in self.seconds.items()},
                            "total_seconds": self.total, "forward_seconds": self.forward_seconds,
                            "calls": self.calls, "partition_seconds": sum(self.seconds.values()),
                            "scope": "one native minibatch; synchronized before and after every part; shared_trunk includes both perspectives' backbone and shared_latent; forward_other contains projections, context assembly and timing-hook overhead"})
        return False


def profiler_dump(command):
    """Permit only this profiler child to inspect its own benchmark parent.

    Linux's default ptrace rule prevents a child from reading its parent. Set
    PR_SET_PTRACER on this process to that child's PID, then revoke it. No host
    policy or permissions on any other process are changed.
    """
    read_fd, write_fd = os.pipe()
    gate = "import os,sys; os.read(int(sys.argv[1]),1); os.execv(sys.argv[2],sys.argv[2:])"
    child = subprocess.Popen([sys.executable, "-c", gate, str(read_fd), *command],
                             pass_fds=(read_fd,), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    os.close(read_fd)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                          ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    try:
        if libc.prctl(0x59616d61, child.pid, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PTRACER profiler-child grant failed")
        os.write(write_fd, b"1")
        stdout, stderr = child.communicate(timeout=15)
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
    finally:
        os.close(write_fd)
        if child.poll() is None:
            child.kill()
            child.communicate()
        if libc.prctl(0x59616d61, 0, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PTRACER revoke failed")


def stacks(stop):
    for index in range(3):
        if stop.wait(0.1):
            return
        command = [sys.executable.replace("python", "py-spy"), "dump", "--pid", str(os.getpid())]
        try:
            result = profiler_dump(command)
            (OUT / f"update-stack-{index + 1}.txt").write_text(result.stdout + result.stderr)
            dump(f"update-stack-{index + 1}.json", {"command": command, "exit_code": result.returncode,
                                                   "update": 2})
        except (OSError, subprocess.TimeoutExpired) as error:
            dump(f"update-stack-{index + 1}.json", {"error": str(error), "update": 2})


def ppo():
    from fog.training import ppo as trainer
    legacy = runpy.run_path(str(ROOT / "docs/briefs/06-artifacts/training_timing.py"))
    legacy["install"](OUT / "training-timing.jsonl")
    original_update, original_phase = trainer.ppo_update, trainer._maybe_chunk_update
    updates = 0

    def update(agent, optimizer, batch, *args, **kwargs):
        nonlocal updates
        updates += 1
        if updates != 2:
            return original_update(agent, optimizer, batch, *args, **kwargs)
        stop = threading.Event()
        thread = threading.Thread(target=stacks, args=(stop,), daemon=True)
        thread.start()
        try:
            with Parts(agent, optimizer, "ppo-parts.json"):
                return original_update(agent, optimizer, batch, *args, **kwargs)
        finally:
            stop.set()
            thread.join(timeout=16)

    def phase(setup, *args, **kwargs):
        result = original_phase(setup, *args, **kwargs)
        if result == "updated" and updates == 2:
            from fog.training.training_state import save_training_state
            save_training_state(setup, setup.ckpt_dir / "ppo_short.pt")
            sync()
            dump("ppo-stop.json", {"updates": updates, "decisions": setup.decisions_seen,
                                   "wall_seconds_through_checkpoint": time.time() - float(os.environ["FOG08_STARTED"]),
                                   "checkpoint": str(setup.ckpt_dir / "ppo_short.state.pt"),
                                   "reason": "brief requires exactly two complete updates"})
            return "halt"
        return result

    trainer.ppo_update, trainer._maybe_chunk_update = update, phase
    sys.argv = ["fog.train", "--config", "configs/collect_gpu_check.yaml", "--trainer", "ppo",
                "--set", "log.dir=runs/collect_gpu", "league.snapshot_every_decisions=8000",
                f"env.rollout_workers={os.environ['FOG08_WORKERS']}"]
    runpy.run_module("fog.train", run_name="__main__")


def replay():
    from fog.config import load_config_with_args
    from fog.models.agent import FogAgent
    from fog.training.bc import bc_loss, _select_trainable_parameters
    from fog.training.dataset import ReplayDataset
    config = load_config_with_args(Path("configs/replay_blueprint_cpu.yaml"),
                                   ["device=cuda", "bc.batch_size=1024"])
    dump("replay-config.json", {"config": config})
    seed = int(config["seed"])
    torch.manual_seed(seed)
    agent = FogAgent.from_tables(config).cuda().train()
    dataset = ReplayDataset(Path(os.environ["FOG08_MANIFEST"]))
    bc = config["bc"]
    opt = torch.optim.AdamW(_select_trainable_parameters(agent, "full"), lr=float(bc["lr"]))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=math.ceil(len(dataset) / 1024) * int(bc["epochs"]), eta_min=float(bc["lr_final"]))
    scaler = torch.amp.GradScaler("cuda")
    rows = steps = epoch = skipped = 0
    samples = []
    checkpoint = Path("runs/update-speed/replay.state.pt")
    elapsed_before = 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        agent.load_state_dict(state["agent"])
        opt.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        steps, rows = state["steps"], state["rows"]
        elapsed_before = state["seconds"]
        samples = state["samples"]
        skipped = state["skipped"]
        torch.random.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    batches_per_epoch = len(dataset) // 1024
    epoch, offset = divmod(steps, batches_per_epoch)
    sync()
    started = time.perf_counter()
    while steps < 200:
        for index, batch in enumerate(dataset.batches(1024, shuffle=True, seed=seed + epoch, device="cuda")):
            if index < offset:
                continue
            if len(batch["action"]) != 1024:
                continue
            opt.zero_grad(set_to_none=True)
            # The CUDA branch in bc.py uses autocast and GradScaler even when the
            # recipe's top-level mixed_precision is false. Reproduce that branch.
            with (Parts(agent, opt, "replay-parts.json") if steps == 1 else contextlib.nullcontext()):
                with torch.amp.autocast("cuda"):
                    total, metrics = bc_loss(agent, batch, config["loss"])
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
            if scaler.get_scale() < scale:
                skipped += 1
                continue
            schedule.step()
            rows += len(batch["action"])
            steps += 1
            if steps <= 3 or steps % 20 == 0:
                sync()
                samples.append({"step": steps, "rows": rows, "seconds": elapsed_before + time.perf_counter() - started,
                                "loss": float(total.detach())})
            if steps % 20 == 0:
                sync()
                torch.save({"agent": agent.state_dict(), "optimizer": opt.state_dict(),
                            "schedule": schedule.state_dict(), "scaler": scaler.state_dict(),
                            "steps": steps, "rows": rows, "skipped": skipped, "samples": samples,
                            "seconds": elapsed_before + time.perf_counter() - started,
                            "cpu_rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                           checkpoint)
            if steps == 200:
                break
        epoch += 1
        offset = 0
    sync()
    seconds = elapsed_before + time.perf_counter() - started
    dump("replay-summary.json", {"steps": steps, "rows": rows, "seconds": seconds,
                                 "seconds_per_step": seconds / steps, "rows_per_second": rows / seconds,
                                 "skipped_nonfinite_steps": skipped, "samples": samples,
                                 "dataset_rows": len(dataset), "manifest": os.environ["FOG08_MANIFEST"],
                                 "random_weights": True, "batch_size": 1024, "warmup_steps": 0,
                                 "scope": "200 successful native CUDA BC updates including batch building and the synchronized second-step profile; loading model and dataset excluded; each epoch's partial final batch skipped"})
    with (OUT / "replay-metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=samples[0])
        writer.writeheader()
        writer.writerows(samples)


if __name__ == "__main__":
    torch.set_num_threads(1)
    {"ppo": ppo, "replay": replay}[sys.argv[1]]()
