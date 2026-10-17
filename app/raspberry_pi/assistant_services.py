"""Local music, weather, spoken feedback, and dashboard state for Astra."""
from __future__ import annotations
import json, os, random, socket, subprocess, sys, tempfile, threading, time, urllib.parse, urllib.request, wave
import numpy as np
from pathlib import Path

STATE = Path.home()/".local/state/me2-astra.json"
MUSIC = Path.home()/"Music"
_lock = threading.Lock(); _proc = None; _tracks=[]; _index=0; _volume=55
MPV_SOCKET = Path.home()/".cache/me2-astra-mpv.sock"
PIPER_MODEL = Path.home()/".local/share/piper-voices/en_US-lessac-low.onnx"
_speech_lock = threading.Lock()

def _mpv(command):
    """Send a non-blocking JSON IPC command to the current mpv process."""
    if not MPV_SOCKET.exists(): return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(.3); client.connect(str(MPV_SOCKET))
            client.sendall((json.dumps({"command": command}) + "\n").encode())
        return True
    except OSError: return False

def wake_chime():
    """Play a brief local acknowledgement without delaying command capture."""
    path = Path.home()/".cache/me2-astra-wake-v2.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rate=16000; tone=np.concatenate([
            0.15*np.sin(2*np.pi*523*np.arange(int(rate*.08))/rate),
            np.zeros(int(rate*.035)),
            0.15*np.sin(2*np.pi*659*np.arange(int(rate*.12))/rate),
        ])
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(rate)
            out.writeframes((np.clip(tone, -1, 1)*32767).astype("<i2").tobytes())
    subprocess.Popen(["aplay", "-D", "plughw:CARD=Headphones,DEV=0", str(path)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def command_chime(success: bool):
    """Short completion (up) or timeout/rejection (down) cue."""
    path = Path.home()/(".cache/me2-astra-success-v2.wav" if success else ".cache/me2-astra-expired-v2.wav")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rate=16000; freqs=(784, 988, 1319) if success else (420, 280)
        length=.065 if success else .09
        tone=np.concatenate([.22*np.sin(2*np.pi*f*np.arange(int(rate*length))/rate) for f in freqs])
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(rate)
            out.writeframes((np.clip(tone,-1,1)*32767).astype("<i2").tobytes())
    subprocess.Popen(["aplay", "-D", "plughw:CARD=Headphones,DEV=0", str(path)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _state(**values):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    # A previous process can be interrupted while writing this lightweight
    # dashboard state file. Treat an empty/corrupt file as fresh state rather
    # than preventing the microphone listener from starting.
    try:
        old = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    except (OSError, json.JSONDecodeError):
        old = {}
    old.update(values); old["updated_at"] = time.strftime("%H:%M:%S")
    STATE.write_text(json.dumps(old), encoding="utf-8")

def event(status: str, **values): _state(status=status, **values)
def microphone_level(level: float): _state(mic_level=round(float(level), 4))

def _scan():
    global _tracks
    _tracks = sorted(p for p in MUSIC.rglob("*") if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"})
    return _tracks

def _speak(text: str):
    """TTS intentionally disabled: instant sound cues keep Astra responsive."""
    return

def quick_speak(text: str):
    """Natural offline Piper response; runs asynchronously from recognition."""
    if not PIPER_MODEL.exists() or not _speech_lock.acquire(blocking=False): return
    def worker():
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out: path = out.name
            piper = Path(sys.executable).with_name("piper")
            subprocess.run([str(piper), "--model", str(PIPER_MODEL), "--output_file", path],
                           input=text, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["aplay", "-D", "plughw:CARD=Headphones,DEV=0", path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception: pass
        finally:
            if path: Path(path).unlink(missing_ok=True)
            _speech_lock.release()
    threading.Thread(target=worker, daemon=True).start()

def shutil_which(name):
    from shutil import which
    return which(name)

def _play(index=None):
    global _proc, _index
    tracks = _tracks or _scan()
    if not tracks: return "No music files found in Music"
    if index is not None: _index = index % len(tracks)
    if _proc and _proc.poll() is None: _proc.terminate()
    MPV_SOCKET.unlink(missing_ok=True)
    _proc = subprocess.Popen(["mpv", "--no-video", "--audio-device=alsa/plughw:CARD=Headphones,DEV=0", f"--volume={_volume}", f"--input-ipc-server={MPV_SOCKET}", str(tracks[_index])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    title=tracks[_index].stem; event("Playing music", song=title, music="playing", volume=_volume); _speak(f"Playing {title}"); return f"Playing {title}"

def music(command):
    global _index, _volume, _proc
    with _lock:
        if command == "PLAY_MUSIC":
            if _proc and _proc.poll() is None: _proc.send_signal(18); event("Playing music", music="playing"); return "Music resumed"
            return _play()
        if command in {"MEDIA_PAUSE", "PAUSE"} and _proc and _proc.poll() is None:
            _proc.send_signal(19); event("Music paused", music="paused"); _speak("Music paused"); return "Music paused"
        if command in {"MEDIA_STOP", "STOP"}:
            if _proc and _proc.poll() is None: _proc.terminate()
            event("Music stopped", music="stopped"); _speak("Music stopped"); return "Music stopped"
        if command in {"MEDIA_NEXT", "NEXT"}: _index += 1; return _play(_index)
        if command in {"VOLUME_UP", "VOLUME_DOWN"}:
            _volume=max(0,min(100,_volume + (10 if command == "VOLUME_UP" else -10)))
            _mpv(["set_property", "volume", _volume])
            event("Volume changed", music="playing" if _proc and _proc.poll() is None else "stopped", volume=_volume)
            _speak(f"Volume {_volume} percent")
            return f"Volume {_volume}%"
    return "Music action unavailable"

def weather():
    key=os.environ.get("ME2_OPENWEATHER_API_KEY", "")
    if not key: return "Weather key is not configured"
    query=urllib.parse.urlencode({"lat":"14.6760","lon":"121.0437","units":"metric","appid":key})
    try:
        data=json.load(urllib.request.urlopen("https://api.openweathermap.org/data/2.5/weather?"+query, timeout=8))
        text=f"Quezon City: {round(data['main']['temp'])} degrees Celsius, {data['weather'][0]['description']}"
        event("Weather updated", weather=text); quick_speak(text); return text
    except Exception: return "Weather service is unavailable"
