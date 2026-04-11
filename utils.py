import sys
import os
import subprocess


def notification_sound() -> None:
    plat = sys.platform

    # --- Windows ---
    if plat.startswith("win"):
        try:
            import winsound  # built-in on Windows

            # System sound (less likely to be blocked than Beep in some setups)
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            return
        except Exception:
            pass

    # --- macOS ---
    if plat == "darwin":
        # Prefer built-in binaries (present on most macOS installs)
        for cmd in (
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            ["osascript", "-e", "beep 1"],
        ):
            try:
                subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                pass

    # --- Linux / other Unix ---
    # There is no single built-in audio command across all distros.
    # Try common ones if available.
    for cmd in (
        ["canberra-gtk-play", "-i", "complete"],
        ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
        ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
    ):
        try:
            subprocess.run(
                cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return
        except Exception:
            pass

    # Fallback: terminal bell (best-effort)
    print("\a", end="", flush=True)
