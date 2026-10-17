"""Single Raspberry Pi application: Astra wake word -> command -> action.

The default mode continuously listens for the separate Astra ONNX wake
model. After a wake it captures one command, classifies it with the AI231
CRNN-BiGRU ONNX model, and
dispatches the Pi GPIO/simulated action. ``--no-wake`` keeps the same app but
uses reliable SPACE-to-record command mode instead.

No target-hardware latency claim is made here; timings printed by this app are
measurements from the Pi on which it is run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.audio.capture import (  # noqa: E402
    default_input_device,
    list_input_devices,
    open_input_stream,
    record_push_to_talk,
    record_vad,
)
from src.audio.preprocessing import normalize_peak, resample, rms, trim_or_pad  # noqa: E402
from src.config import load_commands_config, repo_root  # noqa: E402
from src.inference import OnnxInference  # noqa: E402
from src.wake_word import WakeWordInference  # noqa: E402

from .actions import cleanup, execute  # noqa: E402
from .assistant_services import command_chime, event, microphone_level, wake_chime  # noqa: E402

WAKE_LABEL = "ASTRA"
WAKE_LABELS = repo_root() / "configs" / "astra_labels.yaml"


def _load_command_mapping(path: Path) -> dict[str, dict[str, str]]:
    """Map the model's 93 variation labels to benchmark intent/slot fields."""
    if not path.is_file():
        raise FileNotFoundError(f"command mapping CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["class"]: row for row in csv.DictReader(fh)}


def _command_details(label: str, mapping: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    """Return (intent, slot, display text) for an AI231 classifier output."""
    if label in {"UNKNOWN", "SILENCE", "OUT_OF_SCOPE"}:
        return "OUT_OF_SCOPE", "", "unsupported or uncertain command"
    if label.endswith("__OTHER_SLOT"):
        intent = label.removesuffix("__OTHER_SLOT")
        return intent, "", f"{intent} with an unsupported value"
    row = mapping.get(label)
    if row is None:
        return "OUT_OF_SCOPE", "", f"unmapped label: {label}"
    intent, slot = row["intent"], row["slot_value"]
    return intent, slot, f"{intent}" + (f" | {slot}" if slot else "")


class BenchmarkLogger:
    """Append class-benchmark-compatible JSONL records for each decision."""

    def __init__(self, log_dir: Path, run_id: str, mapping: dict[str, dict[str, str]]) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"{run_id}_{stamp}.log"
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self.mapping = mapping

    def command(self, label: str, infer_ms: float, audio_ms: float) -> None:
        intent, slot, _ = _command_details(label, self.mapping)
        record = {
            "intent": intent,
            "slot": slot,
            "infer_ms": round(float(infer_ms), 1),
            "audio_ms": round(float(audio_ms), 1),
            "model_label": label,
        }
        print(json.dumps(record), file=self._file, flush=True)

    def wake(self, score: float, infer_ms: float) -> None:
        print(json.dumps({"event": "wake", "wake_label": WAKE_LABEL,
                          "wake_score": round(float(score), 4),
                          "wake_infer_ms": round(float(infer_ms), 1)}),
              file=self._file, flush=True)

    def close(self) -> None:
        self._file.close()


class AdbPhoneStream:
    """A sounddevice-like stream backed by an Android ADB microphone relay.

    The companion Android app serves signed 16-bit, mono, 16 kHz PCM on a TCP
    port on the phone. ``adb forward`` exposes that port locally on the Pi.
    This intentionally bypasses desktop PipeWire/PulseAudio virtual devices,
    which are not dependable inputs for a headless PortAudio application.
    """

    def __init__(self, port: int, sample_rate: int, source_sample_rate: int = 16_000) -> None:
        self.port = port
        self.sample_rate = sample_rate
        self.source_sample_rate = source_sample_rate
        if source_sample_rate % sample_rate:
            raise ValueError("ADB phone sample rate must be an integer multiple of the model sample rate")
        self._decimation = source_sample_rate // sample_rate
        self._socket: socket.socket | None = None
        self._pending = bytearray()

    def __enter__(self):
        self._connect(timeout_sec=12.0)
        return self

    def _connect(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                self._socket = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                # ADB forwarding can deliver audio in uneven bursts. Keep the
                # connected microphone socket blocking instead of treating a
                # quiet/late packet as a disconnection after five seconds.
                self._socket.settimeout(None)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(
            f"could not reach Android microphone relay on 127.0.0.1:{self.port}. "
            "Open the companion app on the unlocked phone and start its microphone relay."
        ) from last_error

    def __exit__(self, *_exc) -> None:
        self._close()

    def _close(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._socket = None

    def read(self, frames: int):
        if self._socket is None:
            raise RuntimeError("ADB phone stream is not connected")
        source_frames = frames * self._decimation
        byte_count = source_frames * 2  # PCM s16le, mono
        while len(self._pending) < byte_count:
            try:
                packet = self._socket.recv(max(4096, byte_count - len(self._pending)))
            except OSError:
                packet = b""
            if not packet:
                # Android may briefly tear down the foreground service socket
                # when its UI/activity changes.  The relay server accepts a
                # new client, so reconnect rather than killing the assistant.
                self._close()
                self._pending.clear()
                self._connect(timeout_sec=12.0)
                continue
            self._pending.extend(packet)
        raw = bytes(self._pending[:byte_count])
        del self._pending[:byte_count]
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if self._decimation > 1:
            audio = audio[::self._decimation]
        return audio.reshape(-1, 1), False


class NativeRateResamplingStream:
    """Capture at the microphone's supported rate and expose 16 kHz reads.

    The USB microphone on the deployment Pi only accepts 44.1 kHz.  Models
    and VAD stay at 16 kHz; this adapter resamples every requested chunk so
    their timing and feature representation remain correct.
    """

    def __init__(self, device: int | None, source_rate: int, target_rate: int) -> None:
        self.device = device
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._stream = None

    def __enter__(self):
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.source_rate, channels=1, device=self.device, dtype="float32"
        )
        self._stream.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.__exit__(*exc)
        self._stream = None

    def read(self, frames: int):
        if self._stream is None:
            raise RuntimeError("microphone stream is not open")
        source_frames = max(1, round(frames * self.source_rate / self.target_rate))
        raw, overflow = self._stream.read(source_frames)
        audio = resample(raw[:, 0].astype(np.float32), self.source_rate, self.target_rate)
        # The VAD and wake loop require exactly the requested 16 kHz frame
        # size even when a rounded source-rate read lands one sample short.
        audio = trim_or_pad(audio, frames).astype(np.float32)
        return audio.reshape(-1, 1), overflow


def _prepare_adb_phone(port: int) -> None:
    """Verify one authorized Android device and forward its microphone port."""
    try:
        devices = subprocess.run(["adb", "devices"], check=True, text=True,
                                 capture_output=True, timeout=10).stdout.splitlines()[1:]
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError("ADB is unavailable. Install it with: sudo apt install -y adb") from exc
    if not any(line.rstrip().endswith("\tdevice") for line in devices):
        raise RuntimeError("No authorized Android phone found. Connect USB, unlock the phone, and allow USB debugging.")
    try:
        subprocess.run(["adb", "forward", f"tcp:{port}", f"tcp:{port}"], check=True, timeout=10)
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"Could not forward Android microphone port {port} over ADB") from exc


def _wait_key(prompt: str) -> str:
    print(prompt, end="", flush=True)
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        return input().strip()[:1]


def _record_command(stream, engine, mode: str):
    fc = engine.feature_cfg
    if mode == "vad":
        from src.audio.vad import VadConfig

        audio = record_vad(stream, VadConfig(sample_rate=fc.sample_rate))
    else:
        audio = record_push_to_talk(stream, fc.sample_rate, max_duration_sec=3.0)
    if audio is None or len(audio) == 0:
        return None
    return trim_or_pad(normalize_peak(audio, fc.normalize_target), int(fc.sample_rate * fc.max_duration_sec))


def _drain_after_wake(stream, sample_rate: int, duration_sec: float) -> None:
    """Discard the end of the wake phrase before command VAD begins."""
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        stream.read(max(1, int(sample_rate * min(0.05, deadline - time.monotonic()))))


def _dispatch_command(stream, engine, mode: str, mapping: dict[str, dict[str, str]], wake_at: float | None = None,
                      benchmark: BenchmarkLogger | None = None,
                      post_wake_delay_sec: float = 0.0) -> None:
    if wake_at is not None and post_wake_delay_sec:
        print(f"  [COMMAND] Wake heard. Waiting {post_wake_delay_sec:.2f} s for the command...")
        _drain_after_wake(stream, engine.feature_cfg.sample_rate, post_wake_delay_sec)
    print("  [COMMAND] Listening for command...")
    event("Listening for command")
    audio = _record_command(stream, engine, mode)
    if audio is None:
        print("  (no command speech detected)")
        return
    pred = engine.predict(audio)
    intent, slot, display = _command_details(pred.label, mapping)
    audio_ms = len(audio) / engine.feature_cfg.sample_rate * 1000.0
    if benchmark is not None:
        benchmark.command(pred.label, pred.latency_ms, audio_ms)
    print(f"  Command:    {display}")
    print(f"  Class:      {pred.label}")
    print(f"  Confidence: {100 * pred.confidence:.1f}%")
    print(f"  [LATENCY] command inference: {pred.latency_ms:.1f} ms (this Pi)")
    event("Command recognized", command=display, intent=intent, slot=slot,
          confidence=round(100 * pred.confidence, 1))
    action = execute(intent)
    command_chime(intent != "OUT_OF_SCOPE")
    print(f"  Action:     {action}")
    # This is the authoritative UI event: canonical intent + slot + action.
    event("Action completed", command=display, intent=intent, slot=slot,
          action=action, state_revision=time.time_ns())
    if wake_at is not None:
        elapsed = (time.perf_counter() - wake_at) * 1000
        print(f"  [LATENCY] wake -> command result: {elapsed:.1f} ms (includes user speech capture; this Pi)")


def _status(chunk: np.ndarray, state: str = "[ACTIVE]", message: str = 'Say "Astra"') -> None:
    level = min(12, int(rms(chunk) * 500))
    meter = "#" * level + "-" * (12 - level)
    print(f"\r\033[2K  {state} LISTENING FOR WAKE WORD | {message} | mic [{meter}]", end="", flush=True)


def _wake_loop(stream, wake, command, args, mapping: dict[str, dict[str, str]], benchmark: BenchmarkLogger | None = None) -> None:
    fc = command.feature_cfg
    window, hop = int(fc.sample_rate * fc.max_duration_sec), int(fc.sample_rate * 0.5)
    ring, positives, cooldown_until = np.empty(0, dtype=np.float32), 0, 0.0
    print("[rpi] [ACTIVE] Astra wake listener started. Ctrl+C stops.")
    while True:
        chunk, _overflow = stream.read(hop)
        mono = chunk[:, 0].astype(np.float32)
        microphone_level(min(1.0, rms(mono) * 18))
        ring = np.concatenate((ring, mono))[-window:]
        if time.perf_counter() < cooldown_until:
            positives = 0
            _status(mono, "[COOLDOWN]", "waiting after command")
            continue
        if cooldown_until:
            cooldown_until = 0.0
            event("Waiting for wake word")
        if ring.size < window:
            _status(mono)
            continue
        if rms(ring) < args.wake_silence_rms:
            positives = 0
            _status(mono, "[SILENCE GATE]", "quiet input ignored")
            continue
        _status(mono)
        pred = wake.predict(ring)
        score = pred.raw_probs.get(WAKE_LABEL, 0.0)
        if args.debug_wake:
            print(f"\n  [DEBUG] rms={rms(ring):.5f} wake_score={score:.3f} threshold={wake.threshold:.3f}")
        positives = positives + 1 if wake.is_wake_prediction(pred) else 0
        if positives < args.wake_confirmations:
            continue
        woke_at = time.perf_counter()
        print(f"\n  [DETECTED] Astra ({100 * score:.1f}%)")
        event("Astra detected", wake_score=round(100 * score, 1))
        wake_chime()
        print(f"  [LATENCY] wake inference: {pred.latency_ms:.1f} ms (this Pi)")
        if benchmark is not None:
            benchmark.wake(score, pred.latency_ms)
        _dispatch_command(stream, command, args.mode, mapping, woke_at, benchmark, args.wake_command_delay_sec)
        ring, positives = np.empty(0, dtype=np.float32), 0
        cooldown_until = time.perf_counter() + args.wake_cooldown_sec


def _no_wake_loop(stream, command, mode: str, mapping: dict[str, dict[str, str]], benchmark: BenchmarkLogger | None = None) -> None:
    print("[rpi] No-wake mode. Press SPACE to record a command; Q quits.")
    while True:
        key = _wait_key("  > ")
        if key.lower() == "q":
            return
        if key in (" ", "\x20"):
            _dispatch_command(stream, command, mode, mapping, benchmark=benchmark)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Raspberry Pi Astra + AI231 command voice controller")
    p.add_argument("--command-model", default=str(repo_root() / "models" / "ai231_command_hpo_nosampler.onnx"))
    p.add_argument("--command-labels", default=str(repo_root() / "configs" / "ai231_command_variations.yaml"))
    p.add_argument("--command-mapping", default=str(repo_root() / "models" / "benchmark_mapping.csv"))
    p.add_argument("--command-threshold", type=float, default=0.0,
                   help="0 keeps the Mac-tested argmax behavior; raise after OOS threshold tuning")
    p.add_argument("--wake-model", default=str(repo_root() / "models" / "astra_crnn_bigru_verifier.onnx"))
    p.add_argument("--no-wake", action="store_true", help="use SPACE-to-record command mode in this same app")
    p.add_argument("--mode", choices=["vad", "push"], default="push",
                   help="command capture mode; push records the same fixed 3-second window used by the Mac controller")
    p.add_argument("--device", type=int)
    p.add_argument("--capture-sample-rate", type=int, default=None,
                   help="native microphone rate; defaults to the selected device's advertised rate")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--phone-adb", action="store_true",
                   help="use the experimental Android microphone relay over USB/ADB")
    p.add_argument("--phone-adb-port", type=int, default=47100,
                   help="local/Android TCP port used by --phone-adb (default: 47100)")
    p.add_argument("--wake-threshold", type=float, default=0.60)
    p.add_argument("--wake-confirmations", type=int, default=1)
    p.add_argument("--wake-silence-rms", type=float, default=0.005)
    p.add_argument("--wake-cooldown-sec", type=float, default=2.0)
    p.add_argument("--wake-command-delay-sec", type=float, default=0.35,
                   help="seconds to wait after Astra before the fixed command capture starts")
    p.add_argument("--debug-wake", action="store_true")
    p.add_argument("--benchmark-log-dir", type=Path, default=Path.home() / "vcm_benchmark",
                   help="directory for class-benchmark JSONL output")
    p.add_argument("--benchmark-run-id", default="me2",
                   help="identifier used in the benchmark log filename")
    p.add_argument("--no-benchmark-log", action="store_true",
                   help="do not create class-benchmark compatible logs")
    args = p.parse_args(argv)
    if args.list_devices:
        for d in list_input_devices():
            print(f"[{d['index']}] {d['name']} ({d['channels']} ch, {d['default_sr']} Hz)")
        return 0
    if (args.wake_confirmations < 1 or args.wake_silence_rms < 0
            or args.wake_cooldown_sec < 0 or args.wake_command_delay_sec < 0):
        p.error("wake confirmations must be >= 1; RMS, cooldown, and command delay must be non-negative")
    if not 1 <= args.phone_adb_port <= 65535:
        p.error("phone ADB port must be between 1 and 65535")
    command_path = Path(args.command_model)
    if not command_path.exists():
        p.error(f"command ONNX model not found: {command_path}")
    wake_path = Path(args.wake_model)
    if not args.no_wake and not wake_path.exists():
        p.error(f"wake ONNX model not found: {wake_path}")

    command_labels_path = Path(args.command_labels)
    if not command_labels_path.exists():
        p.error(f"command labels not found: {command_labels_path}")
    mapping_path = Path(args.command_mapping)
    if not mapping_path.exists():
        p.error(f"command mapping not found: {mapping_path}")
    labels, _ = load_commands_config(command_labels_path)
    command = OnnxInference(command_path, label_map=labels)
    command.rejection_threshold = args.command_threshold
    mapping = _load_command_mapping(mapping_path)
    wake = None if args.no_wake else WakeWordInference(wake_path, WAKE_LABELS, WAKE_LABEL, args.wake_threshold)
    if args.phone_adb:
        _prepare_adb_phone(args.phone_adb_port)
        device_description = f"Android ADB microphone relay (USB; port {args.phone_adb_port})"
    else:
        device = args.device if args.device is not None else default_input_device()
        if args.capture_sample_rate is not None:
            capture_rate = args.capture_sample_rate
        else:
            import sounddevice as sd
            capture_rate = round(float(sd.query_devices(device, "input")["default_samplerate"]))
        device_description = f"device {device}; capture {capture_rate} Hz -> model {command.feature_cfg.sample_rate} Hz"
    print("=" * 60)
    print(" ME2 Raspberry Pi Voice Controller")
    print(f" command model : {command_path.name}")
    print(f" wake model    : {'disabled (no-wake mode)' if wake is None else wake_path.name}")
    print(f" microphone    : {device_description}; command mode: {args.mode}")
    benchmark = None if args.no_benchmark_log else BenchmarkLogger(
        args.benchmark_log_dir.expanduser(), args.benchmark_run_id, mapping
    )
    if benchmark is not None:
        print(f" benchmark log: {benchmark.path}")
    print("=" * 60)
    try:
        stream_context = (
            AdbPhoneStream(args.phone_adb_port, command.feature_cfg.sample_rate)
            if args.phone_adb
            else NativeRateResamplingStream(device, capture_rate, command.feature_cfg.sample_rate)
        )
        with stream_context as stream:
            event("Waiting for wake word", microphone=f"Registered: {device_description}")
            if wake is None:
                _no_wake_loop(stream, command, args.mode, mapping, benchmark)
            else:
                _wake_loop(stream, wake, command, args, mapping, benchmark)
    except KeyboardInterrupt:
        print("\n[rpi] Stopped.")
    finally:
        if benchmark is not None:
            benchmark.close()
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
