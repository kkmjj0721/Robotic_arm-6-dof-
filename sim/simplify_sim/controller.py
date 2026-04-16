from __future__ import annotations

import math
import glfw
import mujoco
import numpy as np

from collections import Counter
from dataclasses import dataclass
from config import Config
from kinematics import ModelRefs, Observation, Scenario, TargetState, angles_to_direction, observe, set_configuration, set_pivot_world


@dataclass
class UserCommand:
    """
        输入命令，一帧内累积得到的用户命令
    """
    pitch_steps: int = 0                # 俯仰
    yaw_steps: int = 0                  # 偏航
    insert_steps: int = 0               # 插入与拔出
    reset: bool = False                 # 重新生成固定点
    quit: bool = False                  # 退出


def command_from_counter(counter: Counter[str]) -> UserCommand:
    """
        把按键计数器转换成更有语义的 `UserCommand`
    """
    return UserCommand(
        pitch_steps=counter["pitch_up"] - counter["pitch_down"],                    # 俯仰步数等于 上抬次数 - 下压次数 
        yaw_steps=counter["yaw_left"] - counter["yaw_right"],                       # 偏航步数等于 左转次数 - 右转次数  
        insert_steps=counter["insert_deeper"] - counter["insert_shallower"],        # 插入步数等于 插入次数 - 回退次数 
        reset=bool(counter["reset"]),                                               # 只要这一帧里收到过 `reset` 键，就把它当成真
        quit=bool(counter["quit"]),                                                 # 只要这一帧里收到过 `quit` 键，就把它当成真
    )


def apply_key_to_counter(counter: Counter[str], key: int) -> None:
    """
        把 GLFW 的原始按键码映射到计数器中的语义键。
    """
    # 如果按下 `I`，表示俯仰上抬一次。
    if key == glfw.KEY_I:
        # 记录一次“俯仰上抬”事件。
        counter["pitch_up"] += 1
    # 如果按下 `K`，表示俯仰下压一次。
    elif key == glfw.KEY_K:
        # 记录一次“俯仰下压”事件。
        counter["pitch_down"] += 1
    # 如果按下 `J`，表示偏航左转一次。
    elif key == glfw.KEY_J:
        # 记录一次“偏航左转”事件。
        counter["yaw_left"] += 1
    # 如果按下 `L`，表示偏航右转一次。
    elif key == glfw.KEY_L:
        # 记录一次“偏航右转”事件。
        counter["yaw_right"] += 1
    # 如果按下 `U`，表示插入更深一次。
    elif key == glfw.KEY_U:
        # 记录一次“向深处插入”事件。
        counter["insert_deeper"] += 1
    # 如果按下 `O`，表示回退一次。
    elif key == glfw.KEY_O:
        # 记录一次“向外回退”事件。
        counter["insert_shallower"] += 1
    # 如果按下 `N`，表示请求重置场景。
    elif key == glfw.KEY_N:
        # 记录一次“重置”事件。
        counter["reset"] += 1
    # 如果按下 `X`，表示请求退出程序。
    elif key == glfw.KEY_X:
        # 记录一次“退出”事件。
        counter["quit"] += 1


def target_direction(target: TargetState) -> np.ndarray:
    """
        根据目标偏航角和俯仰角计算目标方向单位向量
        target：期望姿态
    """
    # 使用运动学里的讲我们现在的姿态转换为单位向量
    return angles_to_direction(target.yaw_ref, target.pitch_ref)


def update_target(target: TargetState, cmd: UserCommand, scenario: Scenario, cfg: Config) -> None:
    """
        根据一帧用户命令更新目标状态,更新的是“目标”，不是直接更新关节值
        target：期望姿态
        cmd：控制命令
        scenario：初始值（关节角，固定点位置等）
        cfg：配置参数
    """
    # 更新偏航
    if cmd.yaw_steps:
        # 把离散步数换成角度增量，并累加到偏航目标上
        target.yaw_ref += math.radians(cfg.yaw_step_deg * cmd.yaw_steps)
    # 更新俯仰
    if cmd.pitch_steps:
        pitch_limit = math.radians(cfg.pitch_limit_deg)
        target.pitch_ref = float(np.clip(target.pitch_ref + math.radians(cfg.pitch_step_deg * cmd.pitch_steps), -pitch_limit, pitch_limit))
    # 更新插入深度
    if cmd.insert_steps:
        target.insertion_ref = float(np.clip(target.insertion_ref + cfg.insert_step * cmd.insert_steps, scenario.s_bounds[0], scenario.s_bounds[1]))


def joint_center_bias(q_active: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """
        q_active：
        ranges：
        主任务是让杆的近端和末端去追目标位置；“如果不影响主任务太多，关节尽量别贴着上下限，往区间中间站一点
        让末端几何误差变小，让 dq 尽量接近 center_bias；让控制器在追踪任务的同时，尽量别把关节用到边界上
    """
    # 计算每个关节上下限的中点（比如关节限制在正负pi之间，那么这里就是0）
    mid = 0.5 * (ranges[:, 0] + ranges[:, 1])
    # 计算每个关节上下限区间的一半宽度（从中点向两边最多移动多少）
    half = 0.5 * (ranges[:, 1] - ranges[:, 0])
    # 
    return -0.25 * (q_active - mid) / np.maximum(half, 1e-9)


def solve_control_step(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs, scenario: Scenario, obs: Observation, target: TargetState, cfg: Config) -> np.ndarray:
    """
        控制器
        model：mujoco模型
        data：模型数据
        refs：
        scenario：
        obs：
        target：
        cfg：配置参数
        使用最小二乘法实现解析解逆解
    """
    # 先把目标角度转换成目标方向单位向量
    desired_dir = target_direction(target)
    # 已知固定点和目标插入量时，杆近端点应该位于固定点反方向退回 insertion_ref 的位置。
    desired_prox = scenario.pivot_world - target.insertion_ref * desired_dir
    # 已知近端点、杆长和方向时，杆末端点的位置也就确定了
    desired_tip = desired_prox + obs.rod_length * desired_dir

    def residual(candidate: Observation) -> np.ndarray:
        """
            残差函数
            残差不是关节空间误差，而是“当前杆近端/末端位置”和“目标近端/末端位置”的差
            如果当前姿态和目标姿态一致，即残差为0
        """
        # 把近端误差和末端误差拼成一个 6 维向量
        return np.concatenate([cfg.prox_weight * (candidate.rod_prox_world - desired_prox), cfg.tip_weight * (candidate.rod_tip_world - desired_tip)])

    # 计算当前姿态对应的残差
    base_residual = residual(obs)
    # 读取所有关节自由度，用于构建雅可比矩阵
    ndof = refs.active_ranges.shape[0]
    # 定义一个雅可比矩阵（6 x ndof）
    jac = np.zeros((base_residual.size, ndof), dtype=float)

    # 设置有限差分步长
    eps = 1e-5
    # 对每个关节分别做一次微小扰动，数值估计残差对关节的导数。
    for i in range(ndof):
        # 从当前关节配置出发复制一份试探解。
        q_trial = obs.q_active.copy()
        # 只扰动第 i 个关节
        q_trial[i] += eps
        # 把扰动后的关节值裁剪到限位范围内
        # 否则极端情况下差分点可能跑到物理无效区间
        q_trial = np.clip(q_trial, refs.active_ranges[:, 0], refs.active_ranges[:, 1])
        # 每次差分前重新写回固定点位置
        # 这是为了防止前一轮试探状态残留影响当前几何
        set_pivot_world(model, refs, scenario.pivot_world)
        # 把试探关节配置写进 MuJoCo，并刷新前向运动学
        set_configuration(model, data, refs, q_trial)
        # 用有限差分公式估计残差对第 i 个关节的偏导数
        jac[:, i] = (residual(observe(model, data, refs)) - base_residual) / eps

    # 差分结束后，把固定点恢复成当前场景的固定值
    set_pivot_world(model, refs, scenario.pivot_world)
    # 差分结束后，把关节配置恢复到当前真实观测对应的姿态
    set_configuration(model, data, refs, obs.q_active)
    # 计算关节回中偏置
    # 它会作为一个次目标被拼进增广系统里
    center_bias = joint_center_bias(obs.q_active, refs.active_ranges)

    # 构造增广矩阵 A
    # 上半部分是主任务雅可比，下半部分是回中正则项
    a = np.vstack([jac, cfg.center_weight * np.eye(ndof)])
    # 构造增广右端项 b
    # 主任务希望残差变小，所以是 -base_residual；正则项希望往区间中心移动，所以加上 center_bias
    b = np.concatenate([-base_residual, cfg.center_weight * center_bias])
    # 根据当前最小奇异值自适应增加阻尼
    # 越接近奇异，阻尼越大，解就越稳，但动作也会更保守
    damping = cfg.base_damping + cfg.damping_gain / max(obs.sigma_min, 1e-5)
    # 求解阻尼最小二乘方程
    # 形式上是 (A^T A + λ^2 I)dq = A^T b
    dq = np.linalg.solve(a.T @ a + (damping**2) * np.eye(ndof), a.T @ b)

    # 计算这一步关节增量的整体大小
    dq_norm = float(np.linalg.norm(dq))
    # 如果一步走得太大，就按比例缩小
    # 这相当于给数值逆解再加一层步长保护。
    if dq_norm > cfg.max_dq_norm:
        # 把增量整体缩放到允许的最大范数
        dq *= cfg.max_dq_norm / dq_norm

    # 返回“当前关节 + 本步增量”得到的新关节配置，并再次裁剪到关节限位内
    return np.clip(obs.q_active + dq, refs.active_ranges[:, 0], refs.active_ranges[:, 1])
