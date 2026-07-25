# Security

This project generates synthetic token inputs and does not require external model files.

Do not run untrusted ONNX models or serialized PyTorch objects in the benchmark environment. Model formats and custom operators can consume unexpected resources, and unsafe deserialization can execute code.

Report security issues privately to the repository owner rather than opening a public issue.
