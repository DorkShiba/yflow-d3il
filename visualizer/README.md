# D3IL rollout viewer

Run `./visualizer/run.sh`, then open http://127.0.0.1:8765. Python 3.10+ is required; the script creates `.venv-visualizer` and installs the web dependencies. Install FFmpeg separately for video export. Use `PORT=8766` to change ports or `VISUALIZER_STORE=/path/to/store` for a separate dashboard/job store.

No sample rollouts are bundled (the 2026-11-04 Baseline/Blending/LiPo examples were removed with the blend LiPo definition). The dashboard is empty in a fresh store; submit a CPU evaluation from the UI or import a `rollouts.npz`.

## Playback and smoothness

- Close-up view focuses on the hand and blocks; **확대 / 전체** toggles the camera. Drag or scroll to adjust it.
- Traces retain the most recent second. Playback, seeking and speed are synchronized across panels; shorter rollouts hold their final state.
- Each panel displays achieved hand **XY jerk**, its trailing one-second RMS, and episode RMS. Jerk uses unfiltered third backward finite differences at the recorded control interval (35 ms for these runs). Initial samples without enough history remain unavailable.
- The second graph shows command velocity change at chunk boundaries. Dashed lines mark new chunks. Graphs share axes across methods and show a trailing two-second window. Playback speed never rescales the physical metric.
- Inserting includes the 17 fixed walls from the pinned D3IL task definition. Robot meshes, block display height, and table appearance are visual representations, not a complete MuJoCo scene replica.

## Import and export

Import an existing `rollouts.npz` and specify its actual step interval. Supported arrays include post-step hand/desired positions, block XY/yaw, actions, plan starts, initial hand XY, optional joint/XYZ block/goal records. Without joint records, the arm stays at a reference pose; no IK reconstruction is invented. Unknown imports receive no guessed goal metadata.

Select up to four panels, a time interval, speed, and MP4/GIF. 3D exports include the close-up and smoothness plots: MP4 is 640×720 per panel at 30 fps (maximum 60 output seconds), GIF 480×540 at 15 fps (maximum 15 output seconds). Keep the browser open during capture. Frames are capped at 256 MiB and removed after successful encoding. Two-dimensional exports use 640×480 or 480×360 per panel. MP4 uses H.264/yuv420p/BT.709/faststart.

## Optional checkpoint execution

Use a separate D3IL-compatible runtime, with the repository dependencies (`requirements.txt`) and Pillow installed. Specify that Python executable, D3IL and data directories in the UI (the model code is bundled in `fm_baseline/models`). The viewer runs the baseline evaluator on CPU with paired noise and joint recording; it does not train a model. Pushing supports fresh/fixed replan noise; Inserting supports fresh noise only. MuJoCo frame capture is optional and requires a working graphics backend. WebGL playback and video export use recorded states and do not require D3IL locally.

Only run trusted checkpoint/context files. Server access is restricted to localhost and same-origin requests. Inputs, source hashes, job manifests and results remain in the selected local store. Restarted in-progress jobs are marked interrupted, not automatically rerun.

## Tests and assets

```sh
.venv-visualizer/bin/python -m pip install httpx
.venv-visualizer/bin/python -m unittest discover -s tests -p 'test_visualizer*.py' -v
```

Tests cover input validation, metrics, an empty fresh store, origin checks and video encoding. They do not establish real-robot performance. Panda visual meshes derive from DeepMind MuJoCo Menagerie (Apache-2.0); Three.js and its loaders use MIT. Licenses, pinned provenance and model source files are included under `static/panda` and `static/vendor`.
