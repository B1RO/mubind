# Inference Services

Runs `Complexa` and `ESMFold2` inside isolated `uv` virtual environments and keeps them warm on the GPU over Unix sockets.

```text
services/
├── provider.py          Starts and monitors warm Complexa and ESMFold2 workers (mubind provider)
├── complexa.py          Builds Complexa configs and runs standalone or persistent scaffold jobs
├── complexa_service.py  Unix-socket client for a running Complexa worker
├── folding_service.py   Unix-socket lifecycle and client for ESMFold2
├── complexa_worker.py   Isolated Complexa worker process (no mubind imports)
├── esmfold2_worker.py   Isolated ESMFold2 worker process (no mubind imports)
└── worker_server.py     Shared Unix-socket RPC server and parent watchdog
```
