import time
from typing import Annotated

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import typer
from toppra.constraint import JointAccelerationConstraint, JointVelocityConstraint
from x2robot import Robot, connect
from x2robot.sdk import (
    GripperPosition,
    HeadPose,
    JointPositions,
    LiftPosition,
    ManipulatorControlMode,
    ManipulatorControlModeParam,
    RobotModeParam,
    RobotWorkMode,
)


def move_arm_joints_toppra(arm, target_positions: list, v_max=1.0, a_max=3):
    lower_limits = np.array([-2.792, 0.0, -3.14, -1.57, -1.4, -1.745])
    upper_limits = np.array([2.792, 3.44, 0.0, 1.57, 1.4, 1.745])

    current_state = arm.get_joint_states()
    q_start = np.array(current_state.position)
    q_end = np.array(target_positions)

    if np.any(q_end < lower_limits) or np.any(q_end > upper_limits):
        print("Error: Target position out of limits!")
        return
    q_start = np.clip(q_start, lower_limits, upper_limits)

    num_joints = len(q_start)

    waypoints = np.stack([q_start, q_end])
    path = ta.SplineInterpolator([0, 1], waypoints)

    pc_vel = JointVelocityConstraint([v_max] * num_joints)
    pc_acc = JointAccelerationConstraint([a_max] * num_joints)

    instance = algo.TOPPRA([pc_vel, pc_acc], path)
    traj = instance.compute_trajectory(0, 0)

    if traj is None:
        print("TOPP-RA planning failed.")
        return

    duration = traj.duration
    dt = 0.002
    ts = np.arange(0, duration, dt)

    for t in ts:
        q_t = traj(t)
        q_t_safe = np.clip(q_t, lower_limits, upper_limits)

        joint_cmd = JointPositions()
        joint_cmd.positions = q_t_safe.tolist()
        arm.set_joint_positions(joint_cmd)
        time.sleep(dt)

    final_cmd = JointPositions()
    final_cmd.positions = q_end.tolist()
    arm.set_joint_positions(final_cmd)
    print("  Movement Finished.")


def move_arms_to_zero(robot: Robot):
    zero_positions = [0.0] * 6
    for arm_name, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        print(f"  Setting {arm_name} arm to zero ...")
        move_arm_joints_toppra(arm, zero_positions)
        time.sleep(0.5)


def move_grippers_to_zero(robot: Robot):
    for gripper_name in ["left", "right"]:
        gripper = getattr(robot, f"{gripper_name}_gripper")
        print(f"  Setting {gripper_name} gripper to zero ...")
        gripper.set_position(GripperPosition(position=0.0))
        time.sleep(0.5)


def main(
    server: Annotated[
        str, typer.Option(help="robot server address")
    ] = "localhost:50051",
    head_yaw: Annotated[
        float, typer.Option(help="head yaw angle (rad), range [-0.06, 0.90]")
    ] = 0.0,
    head_pitch: Annotated[
        float, typer.Option(help="head pitch angle (rad), range [-1.20, 1.20]")
    ] = 0.3,
    lift_position: Annotated[float, typer.Option(help="lift position")] = 0.1,
    skip_arms: Annotated[bool, typer.Option(help="skip arm zeroing")] = False,
    skip_grippers: Annotated[bool, typer.Option(help="skip gripper zeroing")] = False,
):
    print(f"Connecting to Quanta X1 at {server} ...")
    robot = connect(f"x2://{server}")

    robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK))
    robot.robot_control.set_manipulator_control_mode(
        ManipulatorControlModeParam(
            mode=ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS
        )
    )
    time.sleep(0.5)

    # ---- Head ----
    print(f"\n[Head] yaw={head_yaw}, pitch={head_pitch}")
    robot.head.set_pose(HeadPose(pitch=head_pitch, yaw=head_yaw))
    time.sleep(1.0)

    # ---- Lift ----
    print(f"[Lift] {lift_position}")
    robot.lift.set_lift_position(LiftPosition(position=lift_position))
    time.sleep(1.0)

    # ---- Arms ----
    if not skip_arms:
        print("\n[Arms] resetting to zero ...")
        move_arms_to_zero(robot)
    else:
        print("\n[Arms] skipped")

    # ---- Grippers ----
    if not skip_grippers:
        print("\n[Grippers] resetting to zero ...")
        move_grippers_to_zero(robot)
    else:
        print("\n[Grippers] skipped")

    print("\nDone.")


if __name__ == "__main__":
    typer.run(main)
