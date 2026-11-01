import time
from pathlib import Path

FLEET = "F1"
HOLDERS_TOKEN = "m" * 40


def block(directory, holders_url="http://127.0.0.1:9/holders"):
    directory = Path(directory)
    return {"fleet": FLEET, "holders_url": holders_url, "holders_token_file": str(directory / "holders-token"),
            "state_file": str(directory / "working-state.json"), "hub_directive": str(directory / "hub-home" / "sparkdata" / "working-state" / "directive")}


def write_token(directory):
    path = Path(directory) / "holders-token"
    path.write_text(HOLDERS_TOKEN + "\n")
    path.chmod(0o600)


def unheld(swapper):
    swapper.working.accept({"fleet": FLEET, "now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hours": []}, time.time())
