"""Raspberry Pi action handlers.

Maps recognized commands to real device/GPIO actions. This module is ONLY
imported on the Raspberry Pi (see app/raspberry_pi/inference.py). It does the
RPi.GPIO import lazily inside functions so that importing this module on a Mac
does not fail.

To wire up real hardware, edit the ``GPIO_PIN`` constants and the bodies of
the ``_gpio_*`` helpers. Until then, every action falls back to printing a
simulated result so the pipeline works end-to-end without hardware.
"""
from __future__ import annotations

import sys
import datetime as dt
from .assistant_services import event, music, quick_speak, weather

# --- Hardware wiring (edit for your Pi) -------------------------------------
GPIO_LIGHT = 17
GPIO_RELAY = 27
HAS_GPIO = False
_gpio = None


def _ensure_gpio():
    """Import and set up RPi.GPIO once. Returns False if unavailable."""
    global HAS_GPIO, _gpio
    if HAS_GPIO:
        return True
    try:
        import RPi.GPIO as GPIO  # noqa: deferred - Pi only

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GPIO_LIGHT, GPIO.OUT)
        GPIO.setup(GPIO_RELAY, GPIO.OUT)
        _gpio = GPIO
        HAS_GPIO = True
        return True
    except Exception as exc:  # not on a Pi, or GPIO not installed
        print(f"[rpi] GPIO unavailable ({exc}); using simulated actions.", file=sys.stderr)
        return False


def light_on() -> str:
    if _ensure_gpio():
        _gpio.output(GPIO_LIGHT, _gpio.HIGH)
        return "LIGHTS ON (GPIO)"
    return "LIGHTS ON (simulated)"


def light_off() -> str:
    if _ensure_gpio():
        _gpio.output(GPIO_LIGHT, _gpio.LOW)
        return "LIGHTS OFF (GPIO)"
    return "LIGHTS OFF (simulated)"


def relay_on() -> str:
    if _ensure_gpio():
        _gpio.output(GPIO_RELAY, _gpio.HIGH)
        return "RELAY ON (GPIO)"
    return "RELAY ON (simulated)"


def relay_off() -> str:
    if _ensure_gpio():
        _gpio.output(GPIO_RELAY, _gpio.LOW)
        return "RELAY OFF (GPIO)"
    return "RELAY OFF (simulated)"


# --- Command dispatch --------------------------------------------------------
def execute(command: str) -> str:
    """Run the device action for a command. Returns a status string."""
    if command == "LIGHT_ON":
        return light_on()
    if command == "LIGHT_OFF":
        return light_off()
    if command in ("PLAY_MUSIC", "MEDIA_PAUSE", "PAUSE", "MEDIA_STOP", "STOP", "MEDIA_NEXT", "NEXT", "VOLUME_UP", "VOLUME_DOWN"):
        return music(command)
    if command == "WEATHER":
        return weather()
    if command == "TIME":
        now = dt.datetime.now().strftime("%-I:%M %p")
        text = f"The time is {now}"
        quick_speak(text)
        event("Time updated", time=now)
        return text
    if command in ("TEMP_UP", "TEMP_DOWN", "TEMPERATURE"):
        return f"TEMP {command} (simulated)"
    if command in ("WEATHER", "TIME", "BRIGHTNESS", "COLOR", "TIMER", "ALARM", "CREATE_REMINDER", "LIST_REMINDERS", "CALL", "MESSAGE"):
        return f"{command} (simulated)"
    if command in ("UNKNOWN", "SILENCE", "OUT_OF_SCOPE"):
        return "(no action)"
    return f"(no action for {command})"


def cleanup() -> None:
    if HAS_GPIO:
        _gpio.cleanup()
