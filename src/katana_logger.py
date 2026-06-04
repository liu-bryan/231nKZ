"""Katana ZERO input logger.

Logs keyboard and mouse-button events to a timestamped JSONL file. F9 toggles
logging and is meant to be bound to the same OBS start/stop-recording hotkey, so
the `session_start` event aligns with video frame 0.

What this logs (and what it deliberately does NOT):
  - keydown / keyup            -> movement, jump, dodge, interact, slow-mo
  - mousedown / mouseup        -> attack (left button); CLICK TIMING ONLY
  - NO continuous mouse movement, NO cursor coordinates.

Why no mouse coordinates: aim direction is recovered from the video instead.
The in-game crosshair is a labeled YOLO class, so the policy reads aim as
(cursor - player) in normalized video-frame space. That is robust to the game
running in a movable window (absolute desktop coordinates from pynput could not
be aligned to a window-only OBS capture anyway). The logger therefore only needs
to tell us *when* an attack happened, not where the cursor was.
"""

import os
import json
from datetime import datetime, timezone

from pynput import keyboard, mouse

HOTKEY = keyboard.Key.f9  # MUST match the OBS start/stop-recording hotkey.

os.makedirs("logs", exist_ok=True)

# Regenerated on every arming press so each run gets its own timestamped file.
LOG_FILE = None
armed = False


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write(event):
    if not armed or LOG_FILE is None:
        return
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def on_press(key):
    global armed, LOG_FILE

    if key == HOTKEY:
        if armed:
            # Emit session_stop BEFORE disarming, otherwise write() drops it.
            write({"type": "session_stop", "t": ts()})
            armed = False
            print("[STOP] Logging")
        else:
            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            LOG_FILE = f"logs/run_{run_id}.jsonl"
            armed = True
            print(f"[START] Logging -> {LOG_FILE}")
            write({"type": "session_start", "t": ts()})
        return

    if not armed:
        return

    try:
        k = key.char
    except AttributeError:
        k = str(key)
    write({"t": ts(), "type": "keydown", "key": k})


def on_release(key):
    if not armed:
        return
    try:
        k = key.char
    except AttributeError:
        k = str(key)
    write({"t": ts(), "type": "keyup", "key": k})


def on_click(x, y, button, pressed):
    # Coordinates are intentionally discarded; only the click event + timing is
    # recorded. Aim comes from the YOLO-detected crosshair in the video.
    if not armed:
        return
    write({
        "t": ts(),
        "type": "mousedown" if pressed else "mouseup",
        "button": str(button),
    })


print("Katana ZERO input logger ready.")
print(f"Press {HOTKEY} to START/STOP logging (must match OBS).")
print("A new log file will be created in ./logs/ each time logging starts.")

keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# No on_move handler: continuous mouse movement is not logged.
mouse_listener = mouse.Listener(on_click=on_click)

keyboard_listener.start()
mouse_listener.start()

keyboard_listener.join()
mouse_listener.join()
