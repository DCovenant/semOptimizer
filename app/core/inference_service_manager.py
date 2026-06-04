"""Launch & supervise the out-of-process YOLO inference service.

The service runs in its own interpreter (.venv-infer, Python 3.12 + ROCm torch)
because it can't share the CARLA-pinned 3.7 venv. This manager finds that
interpreter, starts inference/service.py as a subprocess, and tears it down on
app exit. The service stays warm across Run-YOLO toggles so we only pay the
~13s GPU kernel-JIT warmup once per app session.
"""
import os
import subprocess
import sys

# repo root = two levels up from app/core/
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SOCKET_PATH = "/tmp/sem_inference.sock"
_SERVICE_PY = os.path.join(_ROOT, "inference", "service.py")
_VENV_PYTHON = os.path.join(_ROOT, ".venv-infer", "bin", "python")


class InferenceServiceManager:
    def __init__(self, model="models/yolov8n.pt", device="auto",
                 socket_path=SOCKET_PATH):
        self._model = model
        self._device = device
        self._socket_path = socket_path
        self._proc = None

    @property
    def socket_path(self):
        return self._socket_path

    def is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _kill_stale(self):
        """Kill any service bound to our socket from a prior session/crash.

        Services are kept warm across Run-YOLO toggles, but if the app crashed or
        was killed (no clean closeEvent) the old subprocess lingers — and because
        each new service unlinks+rebinds the socket, a client can race into a
        stale one running OLD code ("unknown message type"). Clearing them here
        guarantees a single, current-code service per socket.
        """
        try:
            subprocess.run(["pkill", "-f", "%s --socket %s"
                            % (_SERVICE_PY, self._socket_path)], timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass

    def ensure_running(self):
        """Start the service if not already up. Returns (ok, message)."""
        if self.is_running():
            return True, "Inference service already running."
        if not os.path.exists(_VENV_PYTHON):
            return False, ("Service venv missing (%s). Create it with: "
                           "python3.12 -m venv .venv-infer && "
                           ".venv-infer/bin/pip install torch torchvision "
                           "--index-url https://download.pytorch.org/whl/rocm6.4 "
                           "&& .venv-infer/bin/pip install ultralytics lapx"
                           % _VENV_PYTHON)
        self._kill_stale()   # clear orphans from a prior session before starting
        try:
            self._proc = subprocess.Popen(
                [_VENV_PYTHON, _SERVICE_PY,
                 "--socket", self._socket_path,
                 "--model", self._model,
                 "--device", self._device],
                cwd=_ROOT,                       # so relative model paths resolve
                stdout=sys.stdout, stderr=sys.stderr,
            )
        except OSError as e:
            self._proc = None
            return False, "Failed to launch inference service: %s" % e
        return True, "Started inference service (device=%s)." % self._device

    def stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
