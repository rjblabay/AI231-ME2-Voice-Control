# Raspberry Pi 4 benchmark

Measurements were collected with the committed ONNX models on a Raspberry Pi
4 Model B. The USB microphone captured at 44.1 kHz and was resampled to the
16 kHz mono model input. The launcher is `./run_astra_headless_mac.sh`, which
starts the listener on the Pi through SSH.

Nineteen interactive wake/command trials yielded:

| Measure | Mean | p95 | Real-time factor (p95 / 3 s window) |
| --- | ---: | ---: | ---: |
| Astra wake inference | 97.7 ms | 116.5 ms | 0.039× |
| Command inference | 161.6 ms | 176.0 ms | 0.059× |
| Wake-to-command result | 3,972.8 ms | 3,993.6 ms | N/A |

The wake-to-command figure includes the configured 0.75 s delay, microphone
capture, and user speech; it is not pure inference latency. RTF uses the
fixed 3-second input window.

To reproduce an interactive run after deploying to the Pi:

```bash
./run_astra_headless_mac.sh
```

The script prints each wake and command inference latency and writes the Pi
benchmark log path at startup.
