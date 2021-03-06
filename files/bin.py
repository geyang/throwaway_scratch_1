import os
from gym.utils import EzPickle
from . import fetch_env
import numpy as np


class BinEnv(fetch_env.FetchEnv, EzPickle):
    def __init__(self, action, reward_type="sparse", **kwargs):
        from ml_logger import logger
        logger.upload_file(__file__)

        self.action = action
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'bin:joint': [1.25, 0.53, 0.4, 0, 0., 0., 0.],
            # use same location to correct set the target height
            'object0:joint': [1.25, 0.53, 0.6, 0, 0., 0., 0.],
            # 'object0:joint': [1.25, 0.95 if "place" in action else 0.53, 1, 1, 0., 0., 0.],
        }
        _kwargs = dict(
            obj_keys=("bin", "object0"),
            obs_keys=("object0",),
            goal_key="object0",
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=self.initial_qpos,
            reward_type=reward_type
        )
        _kwargs.update(kwargs)
        fetch_env.FetchEnv.__init__(self, "bin.xml", **_kwargs)
        EzPickle.__init__(self)

    def _reset_sim(self):
        """
        :return: True, Read by the reset function to know this is ready.
        """
        if self.action == "pick":
            bin_pos = self._reset_body("bin")
            self._reset_body("object0", bin_pos[:3])
        elif self.action == "place":
            # we keep the object at its original position
            self._reset_body("object0")
            self._reset_body("bin")
        elif self.action == "place-fix-block":
            # we keep the object at its original position
            self._reset_body("object0", self.initial_qpos['object0:joint'])
            self._reset_body("bin")
        elif self.action == "fix-bin":
            self._reset_body("object0")
            self._reset_body("bin", self.initial_qpos['bin:joint'])
        else:
            for obj_key in self.obj_keys:
                self._reset_body(obj_key)
        self.sim.forward()
        return True

    def _step_callback(self):
        super()._step_callback()
        if not self.action:
            return
        if "place" in self.action:
            # goal setting
            self.goal = self.sim.data.get_site_xpos("bin").copy()
            self.goal[2] = self.initial_heights['object0']
        # todo: change to default behavior after stabilization
        if "fix-bin" in self.action:
            # todo: fix the location of the bin
            original_pos = self.initial_qpos['bin:joint']
            original_pos[2] = self.initial_heights['bin']
            self._reset_body("bin", original_pos)

    def _sample_goal(self):
        if self.action == "pick":
            xpos = bin_xpos = self.sim.data.get_site_xpos("bin").copy()
            while np.linalg.norm(xpos - bin_xpos) < 0.1:
                xpos = super()._sample_goal()
            return xpos
        elif self.action == "place":
            bin_xpos = self.sim.data.get_site_xpos("bin").copy()
            bin_xpos[2] = self.initial_heights['object0']
            return bin_xpos

        return super()._sample_goal()
