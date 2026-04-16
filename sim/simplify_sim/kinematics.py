from __future__ import annotations

import math
import mujoco
import numpy as np

from dataclasses import dataclass
from config import ACTIVE_JOINTS, PIVOT_BODY, PIVOT_SITE, ROD_BODY, ROD_GEOM, ROD_TIP_SITE


@dataclass
class ModelRefs:
    """
        保存 MuJoCo 模型里各类命名对象对应的索引引用
    """
    active_qpos: np.ndarray                 # 主动关节在 qpos 向量中的位置索引
    active_dof: np.ndarray                  # 主动关节在 dof 向量中的位置索引
    active_ranges: np.ndarray               # 主动关节的上下限范围
    rod_body: int                           # 受控杆件 body 的 MuJoCo ID
    rod_geom: int                           # 受控杆件 geom 的 MuJoCo ID    
    rod_tip_site: int                       # 受控杆件末端点 site 的 MuJoCo ID
    pivot_body: int                         # 固定点参考 body 的 MuJoCo ID
    pivot_site: int                         # 固定点可视化 site 的 MuJoCo ID

@dataclass
class Scenario:
    """
        保存一次仿真回合里固定不变的场景信息
    """
    q_active: np.ndarray                        # 本回合初始的5个主动变量值
    pivot_world: np.ndarray                     # 本回合固定点在世界坐标系中的位置
    s_bounds: tuple[float, float]               # 插入量允许变化的范围

@dataclass
class TargetState:
    """
        保存当前期望的三维控制目标
    """
    yaw_ref: float              # 期望的偏航角
    pitch_ref: float            # 期望的俯仰角
    insertion_ref: float        # 期望的插入深度    

@dataclass
class Observation:
    """
        保存当前的观测信息(包含了此时此刻机械臂的一切几何信息)
    """
    q_active: np.ndarray                    # 当前主动关节的角度    
    rod_prox_world: np.ndarray              # 受控杆件近端点的世界坐标
    rod_tip_world: np.ndarray               # 受控杆件末端点的世界坐标
    rod_dir: np.ndarray                     # 受控杆件的方向向量（可以表示杆件的朝向，姿态）
    rod_length: float                       # 受控杆件的长度
    pivot_world: np.ndarray                 # 固定点的世界坐标
    pivot_error_norm: float                 # 固定点到杆轴线的最短偏差长度（误差多少）
    insertion: float                        # 固定点沿杆方向相对近端点的位置坐标
    sigma_min: float                        # 当前 tip 位置雅可比的最小奇异值
    rod_yaw: float                          # 当前杆方向对应的 yaw 角
    rod_pitch: float                        # 当前杆方向对应的 pitch 角


def normalize(vec: np.ndarray) -> np.ndarray:
    """
        将任意非零向量归一化为单位向量
        vec：需要转换的向量
    """
    norm = np.linalg.norm(vec)

    # 如果向量几乎为零，则抛出异常避免数值错误
    if norm < 1e-12:
        raise ValueError("无法对零向量进行归一化")
    
    return vec / norm 


def angles_to_direction(yaw: float, pitch: float) -> np.ndarray:
    """
        根据 yaw 和 pitch 角生成对应的三维方向向量
        yaw：偏航角（弧度）
        pitch：俯仰角（弧度）
        当 pitch=0 时，方向在水平面上，当 pitch>0 时，z 分量变大，水平分量长度等于 cos(pitch)，再由 yaw 分配到 x/y
    """ 
    # 中间值，避免下面重复计算
    cos_pitch = math.cos(pitch)
    # 按球坐标到笛卡尔坐标的关系返回单位方向向量。
    return np.array(
        # X 分量由 cos(pitch) * cos(yaw) 决定。
        [cos_pitch * math.cos(yaw), cos_pitch * math.sin(yaw), math.sin(pitch)],
        dtype=float,
    )

def direction_to_angles(direction: np.ndarray) -> tuple[float, float]:
    """
        根据三维方向向量反解 yaw 和 pitch 角
        direction：方向向量
    """
    # 先把输入方向向量归一化，防止比例影响角度计算。
    direction = normalize(direction)
    # 用 atan2 计算平面内的 yaw 角。
    yaw = math.atan2(direction[1], direction[0])
    # 用 atan2 计算相对于水平面的 pitch 角。
    pitch = math.atan2(direction[2], math.hypot(direction[0], direction[1]))
    # 返回解出的 yaw 和 pitch。
    return yaw, pitch


def build_refs(model: mujoco.MjModel) -> ModelRefs:
    """
        从模型里构建所有名字到索引的运行时引用
        model：mujoco模型
    """
    # 将主动关节名称解析成关节 ID 数组（按照先后定义顺序排列的）
    active_joint_ids = np.array(
        # 逐个名称在 MuJoCo 模型里查找对应的 joint ID。
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ACTIVE_JOINTS],
        # 明确数组元素类型为整数。
        dtype=int,
    )

    # 返回一个完整的引用结构体
    return ModelRefs(
        # 关节角度
        active_qpos=np.array([model.jnt_qposadr[jid] for jid in active_joint_ids], dtype=int),
        # 关节自由度
        active_dof=np.array([model.jnt_dofadr[jid] for jid in active_joint_ids], dtype=int),
        # 读取每个主动关节的活动范围
        active_ranges=np.array([model.jnt_range[jid].copy() for jid in active_joint_ids], dtype=float),
        # 查找受控杆件 body 的 ID
        rod_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROD_BODY),
        # 查找受控杆件 geom 的 ID
        rod_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ROD_GEOM),
        # 查找受控杆件远端 tip site 的 ID
        rod_tip_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ROD_TIP_SITE),
        # 查找固定点 site 的 ID
        pivot_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PIVOT_SITE),
        # 查找固定点参考 body 的 ID
        pivot_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PIVOT_BODY),
    )

def set_pivot_world(model: mujoco.MjModel, refs: ModelRefs, pivot_world: np.ndarray) -> None:
    """
        将固定点参考 body 直接移动到随机给定的世界坐标
        model：mujoco模型
        refs：
        pivot_world：生成的固定的世界坐标
    """
    # 直接覆盖 pivot_ref body 的三维坐标
    model.body_pos[refs.pivot_body] = pivot_world


def set_configuration(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs, q_active: np.ndarray,) -> None:
    """
        将一组主动关节值写入 MuJoCo 数据并前向计算
        model：mujoco模型
        data：模型数据
        refs：
        q_active：设置关节角度（逆解出来的）
    """
    # 清零所有关节的状态
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    # 仅把主动关节位置写入对应的 qpos
    data.qpos[refs.active_qpos] = q_active

    # 运行 MuJoCo 前向运动学以刷新所有派生状态
    mujoco.mj_forward(model, data)


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs, site_id: int, ) -> np.ndarray:
    """
        计算某个 site 对主动关节的平移雅可比矩阵
        model：mujoco模型
        data：模型数据
        refs：
        site_id：site的索引
        雅可比矩阵包含：位置雅可比矩阵（平移）、旋转雅可比矩阵；其中位置雅可比矩阵，管物体在 X、Y、Z 三个方向上移动有多快；把“电机的旋转速度”翻译成“末端的直线移动速度”
        我们只取位置雅可比矩阵（即前向运动学对关节求导的部分）；因为我们为了绕 tcp 来做移动，但我们是通过远端点以及近端点（或tcp点）来确定怎么移动的，即控制末端点的位置
        来实现移动；又因为两点确定一条直线，所以只用关注末端点的位置即可；所以只需要雅可比矩阵的上半部分；
    """
    # 创建完整系统自由度维度下的平移雅可比矩阵
    jacp = np.zeros((3, model.nv), dtype=float)
    # 调用 MuJoCo 计算指定 site 的平移雅可比
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    # 仅返回主动关节对应的列，供控制器和奇异性评估使用
    return jacp[:, refs.active_dof].copy()


def rod_sigma_min(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs) -> float:
    """
        计算当前 tip 位置雅可比矩阵的最小奇异值，用于衡量奇异性
        model：mujoco模型
        data：模型数据
        refs：
        最小奇异值即最难移动的方向；当最小奇异值趋近于 0 时，意味着机械臂在某个特定方向上完全丧失了运动能力
    """
    # 对杆末端位置雅可比做奇异值分解并只取奇异值数组
    singular_values = np.linalg.svd(site_jacobian(model, data, refs, refs.rod_tip_site), compute_uv=False)

    return float(singular_values[-1])


def modeled_link_axis(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs, ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
        从 MuJoCo 当前状态里重建受控杆件的轴线和端点位置
        model：mujoco模型
        data：模型数据
        refs：
        因为我们夹持一个工作杆；夹持点并不是真实的原点；我们需要重构出这个原点
    """
    # 读取受控杆 body 原点在世界坐标系下的位置
    body_world = data.xpos[refs.rod_body].copy()
    # 读取受控杆 body 在世界坐标系下的旋转矩阵
    rot = data.xmat[refs.rod_body].reshape(3, 3)
    # 约定杆轴沿 body 局部 `-Y` 方向（单位方向）
    axis = -rot[:, 1]
    # 读取杆末端 site 的世界坐标
    tip_world = data.site_xpos[refs.rod_tip_site].copy()
    # 读取杆 geom 中心在 body 局部坐标系下的位置
    center_local = model.geom_pos[refs.rod_geom].copy()
    # 读取杆 geom 的半长度
    half_length = float(model.geom_size[refs.rod_geom][1])
    # 先把局部几何中心偏移旋转到世界系，再投影到杆轴方向上
    center_along_axis = float(np.dot(rot @ center_local, axis))
    # 从几何中心沿负轴方向退回半个长度，就得到杆近端点
    prox_world = body_world + (center_along_axis - half_length) * axis
    # 半长度乘 2 得到整根杆长度
    rod_length = 2.0 * half_length
    # 返回杆近端、杆末端、杆长和单位方向
    return prox_world, tip_world, rod_length, axis


def observe(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs) -> Observation:
    """
        定义观测汇总函数，把当前 MuJoCo 状态转换成控制器可直接使用的几何量
        model：mujoco模型
        data：模型数据
        refs：
    """
    # 先重建杆的近端点、末端点、长度和方向
    rod_prox_world, rod_tip_world, rod_length, rod_dir = modeled_link_axis(model, data, refs)
    # 读取固定点 site 的世界坐标
    pivot_world = data.site_xpos[refs.pivot_site].copy()
    # 计算从杆近端指向固定点的向量
    delta = pivot_world - rod_prox_world
    # 先算 delta 在杆方向上的投影，再从 delta 里减掉它，剩下的就是固定点到杆轴线的最短连线向量
    pivot_error = delta - rod_dir * np.dot(delta, rod_dir)
    # 把杆方向转换成界面友好的偏航角和俯仰角
    rod_yaw, rod_pitch = direction_to_angles(rod_dir)

    return Observation(
        q_active=data.qpos[refs.active_qpos].copy(),
        rod_prox_world=rod_prox_world,
        rod_tip_world=rod_tip_world,
        rod_dir=rod_dir,
        rod_length=rod_length,
        pivot_world=pivot_world,
        pivot_error_norm=float(np.linalg.norm(pivot_error)),
        insertion=float(np.dot(delta, rod_dir)),
        sigma_min=rod_sigma_min(model, data, refs),
        rod_yaw=rod_yaw,
        rod_pitch=rod_pitch,
    )


def make_initial_target(obs: Observation) -> TargetState:
    """
        初始目标生成函数，让重置后的目标和当前姿态完全对齐
        obs：观测类
        用当前观测直接生成一份初始目标；这样重置场景后不会一开始就出现很大的目标误差
    """
    # 先把当前姿态视为正确起点，后续再响应用户按键偏移目标
    return TargetState(yaw_ref=obs.rod_yaw, pitch_ref=obs.rod_pitch, insertion_ref=obs.insertion)
