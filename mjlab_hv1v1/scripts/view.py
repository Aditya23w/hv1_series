#!/usr/bin/env python3
"""Open the HV1V1 training model (not the viewer model) in the MuJoCo viewer.

This is the model mjlab actually trains against: real actuator gains, foot
pads instead of a foot hull, and the solved initial pose.
"""
import mujoco.viewer

from mjlab.entity.entity import Entity
from mjlab_hv1v1.hv1v1_constants import get_hv1v1_robot_cfg

if __name__ == "__main__":
  mujoco.viewer.launch(Entity(get_hv1v1_robot_cfg()).spec.compile())
