"""Transform only the three block positions, using the simulator's TCP frame."""

import numpy as np

BLOCK_POSITIONS = (slice(8, 11), slice(13, 16), slice(18, 21))


class StackingFrame:
    def __init__(self, env):
        import mujoco

        robot = env.robot
        robot.receiveState()
        self.mujoco = mujoco
        self.model = env.scene.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = env.scene.data.qpos
        self.indices = np.asarray(robot.joint_indices)
        self.body_id = self.model.body(robot.add_id2model_key("tcp")).id
        if self.indices.shape != (7,):
            raise ValueError("Expected seven arm joints")
        np.testing.assert_allclose(
            self.position(robot.current_j_pos), robot.current_c_pos_global,
            rtol=0, atol=1e-8,
        )

    def position(self, joints):
        self.data.qpos[self.indices] = joints
        self.mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.body_id].copy()

    def relative(self, observation):
        result = np.asarray(observation, dtype=np.float32).copy()
        tcp = self.position(result[:7])
        for xyz in BLOCK_POSITIONS:
            result[xyz] -= tcp
        return result
