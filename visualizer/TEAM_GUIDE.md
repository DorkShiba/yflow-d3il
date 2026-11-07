# Team setup

1. Pull the repository default branch (`master`).
2. Install Python 3.10+ and FFmpeg.
3. Run `./visualizer/run.sh` and open http://127.0.0.1:8765.
4. Select Pushing or Inserting. The three verified sample rollouts load without a checkpoint, simulator installation, or GPU.
5. Use the shared timeline and playback-speed controls; export the current layout as MP4/GIF.

For full evaluation, install the existing D3IL runtime and `requirements.txt`; follow `fm_baseline/LIPO.md`. The six visual examples are selected first episodes, not a substitute for the aggregate experiment reports. Existing local stores keep their previous dashboard. To view the newly bundled defaults, choose a fresh `VISUALIZER_STORE` directory without deleting existing results.

See [README.md](README.md) for metric definitions, import format, export limits, optional simulator execution, and asset licenses.
