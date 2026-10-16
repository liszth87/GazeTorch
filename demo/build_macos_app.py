"""Build a local launcher bundle for the current checkout and Python environment.

Run with the demo's virtualenv Python. The bundle is local, not redistributable.
"""
import argparse
from pathlib import Path
import plistlib
import shlex
import shutil
import sys


def build(destination):
    if sys.platform != 'darwin':
        raise RuntimeError('This launcher is for macOS')
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} already exists; choose a new output path')
    executable = destination / 'Contents/MacOS'
    executable.mkdir(parents=True)
    shutil.copy2(Path(sys.executable).resolve(), executable / 'GazeTorchPython')
    sites = [path for path in sys.path if 'site-packages' in path]
    script = '\n'.join([
        '#!/bin/zsh',
        f'cd {shlex.quote(str(root))} || exit 1',
        f'export PYTHONHOME={shlex.quote(sys.base_prefix)}',
        f'export PYTHONPATH={shlex.quote(":".join(sites + [str(root)]))}',
        f'exec {shlex.quote(str(executable / "GazeTorchPython"))} {shlex.quote(str(root / "demo/app.py"))}',
        '',
    ])
    launcher = executable / 'GazeTorch'
    launcher.write_text(script)
    launcher.chmod(0o755)
    info = {
        'CFBundleName': 'GazeTorch Demo',
        'CFBundleDisplayName': 'GazeTorch Demo',
        'CFBundleIdentifier': 'org.gazetorch.webcam-demo',
        'CFBundleExecutable': 'GazeTorch',
        'CFBundlePackageType': 'APPL',
        'CFBundleVersion': '1',
        'CFBundleShortVersionString': '0.1',
        'NSCameraUsageDescription': 'Use the webcam locally to calibrate and display a gaze dot.',
        'NSHighResolutionCapable': True,
    }
    with (destination / 'Contents/Info.plist').open('wb') as stream:
        plistlib.dump(info, stream)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='GazeTorch Demo.app')
    args = parser.parse_args()
    print(build(args.output))
