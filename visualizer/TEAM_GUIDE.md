# Team setup

1. Pull the repository default branch (`master`).
2. Install Python 3.10+ and FFmpeg.
3. Run `./visualizer/run.sh` and open http://127.0.0.1:8765.
4. No sample rollouts are bundled. Import a `rollouts.npz` from an evaluation, or submit a CPU evaluation from the UI (needs the D3IL runtime and `requirements.txt`).
5. Use the shared timeline and playback-speed controls; export the current layout as MP4/GIF.

See [README.md](README.md) for metric definitions, import format, export limits, optional simulator execution, and asset licenses.
