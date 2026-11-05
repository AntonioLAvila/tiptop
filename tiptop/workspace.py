from dataclasses import replace
from functools import cache

from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from cutamp.envs.utils import unit_quat
from cutamp.utils.rerun_utils import log_curobo_mesh_to_rerun

from tiptop.config import tiptop_cfg
from tiptop.utils import get_robot_rerun

tensor_args = TensorDeviceType()

X_OFFSET = 0.324  # table rear edge is this far behind the robot base
Z_TABLE_OFFSET = 0.0047625  # table top is this far below the robot base

TABLE_DIMS = (1.2319, 0.7747, 0.02)
TABLE_FRONT_X = TABLE_DIMS[0] - X_OFFSET
TABLE_REAR_X = -X_OFFSET

COLUMN_DIMS = (0.0762, 0.0381, 1.2)
COLUMN_FRONT_X = 0.60325 + X_OFFSET
COLUMN_REAR_X = TABLE_REAR_X + (TABLE_FRONT_X - COLUMN_FRONT_X)
COLUMN_HALF_Y = 0.34925


def _column(name: str, front: bool, left: bool, color: list[int]) -> Cuboid:
    """Column at the given corner of the table."""
    dx, dy, dz = COLUMN_DIMS
    pose = [
        COLUMN_FRONT_X - dx / 2 if front else COLUMN_REAR_X + dx / 2,
        (1 if left else -1) * (COLUMN_HALF_Y - dy / 2),
        dz / 2 - Z_TABLE_OFFSET,
        *unit_quat,
    ]
    return Cuboid(name, dims=list(COLUMN_DIMS), pose=pose, color=color)


def fr3_workspace() -> tuple[Cuboid, ...]:
    tx, _, tz = TABLE_DIMS
    table = Cuboid(
        "table",
        dims=list(TABLE_DIMS),
        pose=[tx / 2 - X_OFFSET, 0.0, -tz / 2 - Z_TABLE_OFFSET, *unit_quat],
        color=[2, 48, 117],
    )
    ceiling = Cuboid(
        "ceiling",
        dims=list(TABLE_DIMS),
        pose=[tx / 2 - X_OFFSET, 0.0, -tz / 2 - Z_TABLE_OFFSET + 1.1176, *unit_quat],
        color=[2, 48, 117],
    )
    wall = Cuboid(
        'wall',
        dims=[0.01, 5, 5],
        pose=[-0.3048 - X_OFFSET, 0, 0, *unit_quat],
        color=[2, 48, 117]
    )
    FR_column = _column("FR_column", front=True, left=False, color=[255, 0, 255])
    FL_column = _column("FL_column", front=True, left=True, color=[255, 0, 0])
    RR_column = _column("RR_column", front=False, left=False, color=[0, 87, 217])
    RL_column = _column("RL_column", front=False, left=True, color=[0, 87, 217])
    return (table, ceiling, FR_column, FL_column, RL_column, RR_column, wall)


def ur5_workspace() -> tuple[Cuboid, ...]:
    pedestal_table = Cuboid(
        "pedestal_table",
        dims=[0.48, 0.87, 0.02],
        pose=[-0.05, 0, -0.01, *unit_quat],
        color=[222, 184, 135],
    )
    # ceiling is more for avoiding crazy large motions
    ceiling = Cuboid("ceiling", dims=[1.5, 2.0, 0.01], pose=[0.5, 0.0, 1.0, *unit_quat], color=[255, 255, 255])

    # humans work to the left and right of the UR5...
    wall_left = Cuboid("wall_left", dims=[1.5, 0.1, 1.0], pose=[0.0, 0.8, 0.2, *unit_quat], color=[225, 225, 225])
    wall_right = Cuboid("wall_right", dims=[1.5, 0.1, 1.0], pose=[0.0, -0.8, 0.2, *unit_quat], color=[225, 225, 225])
    obstacles = (pedestal_table, ceiling, wall_left, wall_right)
    return obstacles


@cache
def workspace_cuboids() -> tuple[Cuboid, ...]:
    """Return workspace cuboids for the configured robot, with names prefixed by 'workspace_' to avoid collisions with objects discovered by actual perception."""
    cfg = tiptop_cfg()
    if cfg.robot.type == "fr3_robotiq":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "panda":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "fr3":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "panda_robotiq":
        cuboids = fr3_workspace()
    elif cfg.robot.type == "ur5":
        cuboids = ur5_workspace()
    else:
        raise ValueError(f"Unknown robot type: {cfg.robot.type}")
    return tuple(replace(c, name=f"workspace_{c.name}") for c in cuboids)


if __name__ == "__main__":
    import rerun as rr

    rr.init("robot_workspace", spawn=True)
    get_robot_rerun()
    for obj in workspace_cuboids():
        log_curobo_mesh_to_rerun(f"world/{obj.name}", obj.get_mesh(), static_transform=True)
