"""Out-of-process YOLO inference service + its wire protocol.

Only `protocol` is import-safe from the 3.7 app; `service` pulls in torch and
must run under .venv-infer (Python 3.12 + ROCm). Keep this __init__ empty so
importing inference.protocol never drags torch in.
"""
