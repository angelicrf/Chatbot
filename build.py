#!/usr/bin/env python3
from __future__ import annotations

import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS_FILE = ROOT / "requirements.txt"


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.check_call(command, env=env)


def create_virtual_env() -> None:
    if VENV_DIR.exists():
        print("Virtual environment already exists at .venv")
        return

    print("Creating virtual environment at .venv...")
    venv.create(VENV_DIR, with_pip=True)
    print("Virtual environment created.")


def get_venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_requirements() -> None:
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Could not find requirements file: {REQUIREMENTS_FILE}")

    python_exe = get_venv_python()
    if not python_exe.exists():
        raise FileNotFoundError(f"Virtual environment Python not found at {python_exe}")

    print("Upgrading pip in the virtual environment...")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    print(f"Installing dependencies from {REQUIREMENTS_FILE}...")
    run_command([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    print("Dependencies installed.")


def print_next_steps() -> None:
    print("\nSetup complete.")
    if platform.system() == "Windows":
        print("Activate the environment with:")
        print("  .\\.venv\\Scripts\\Activate.ps1")
        print("or")
        print("  .\\.venv\\Scripts\\activate.bat")
    else:
        print("Activate the environment with:")
        print("  source .venv/bin/activate")

    print("\nThen run:")
    print("  python run_chatbot.py")
    print("  python telemetry_analysis.py")
    print("  python telemetry_visualization.py")


def main() -> None:
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or newer is required.")

    create_virtual_env()
    install_requirements()
    print_next_steps()


if __name__ == "__main__":
    main()
