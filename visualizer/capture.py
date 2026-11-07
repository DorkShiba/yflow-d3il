"""Optional read-only frame capture; imported only by visualizer-enabled evaluation."""
from pathlib import Path
from PIL import Image

class Capture:
    def __init__(self, env, root, camera='oblique'):
        import mujoco
        self.mj = mujoco
        self.env = env
        self.root = Path(root)
        self.renderer = mujoco.Renderer(env.scene.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [.5, 0, .0]
        self.camera.distance = 1.5
        self.camera.azimuth = 90
        self.camera.elevation = -90 if camera == 'top' else -45

    def save(self, ci, ti, step):
        self.renderer.update_scene(self.env.scene.data, camera=self.camera)
        folder = self.root / f'c{ci:03d}_t{ti}'
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.renderer.render()).save(folder / f'{step:06d}.jpg', quality=90)

    def close(self):
        self.renderer.close()
