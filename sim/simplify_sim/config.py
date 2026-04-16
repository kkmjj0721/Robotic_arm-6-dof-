from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# 运行时允许主动求解的 5 个关节名称（因为这里仿真中无法做到随着移动而自由跟随，这里做成也需要解算的方式）
ACTIVE_JOINTS = ("q1", "q2", "q3", "tool_yaw", "tool_pitch")
# 被当作 RCM 受控杆件的 body 名称
ROD_BODY = "tool_modeled_link"
# 被当作 RCM 受控杆件的几何体名称
ROD_GEOM = "tool_modeled_link_geom"
# 声明受控杆件远端 tip site 的名称（杆件末端点）
ROD_TIP_SITE = "tool_modeled_link_tip"
# 固定点可视化 site 的名称
PIVOT_SITE = "pivot_site"
# 固定点参考 body 的名称
PIVOT_BODY = "pivot_ref"
# 记录当前配置文件所在目录的上上上级目录（即项目根目录），方便后续构建模型路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# 加载的 MuJoCo 模型路径
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "Simplify_model" /"rcm_final_end_modeled_link_position.xml"


@dataclass
class Config:
    model_path: Path = DEFAULT_MODEL_PATH.resolve()                    # 模型路径
    seed: int = 42                                                     # 随机种子(可自行配置修改，这里固定为 42 以保证结果可复现) 

    # 初始场景采样约束
    arm_margin: float = 0.18            # 在初始化随机姿态时，为了防止机械臂初始就卡在关节极限上，设置了一定的退让余量
    wrist_margin: float = 0.12
    z_floor: float = 0.05               # 高度下限；防止随机生成的固定点或工具杆直接插到地板以下
    sample_attempts: int = 400          # 因为很多随机姿态可能不符合要求（例如卡入奇异区、碰地等），所以最多允许尝试随机采样 400 次

    # 奇异性与安全阈值
    min_sigma: float = 0.05             # 雅可比矩阵的最小奇异值；在初始采样时，如果姿态的奇异值小于 0.05，说明即将进入死锁（奇异点），系统会丢弃这个采样
    hard_sigma: float = 0.025           # 如果某一步算出的新姿态导致奇异值跌破 0.025，控制器会拒绝移动，避免机械臂失控

    # 交互控制步长与限制（想要更加细致可以适当减少）
    pitch_step_deg: float = 1.0         # 单次 pitch 指令对应的目标角度步长，单位是度。
    yaw_step_deg: float = 1.0           # 单次 yaw 指令对应的目标角度步长，单位是度。
    pitch_limit_deg: float = 85.0       # pitch 目标允许的最大绝对角度，单位是度。
    insert_step: float = 0.004          # 单次插入指令对应的目标插入量步长，单位是米。

    # 逆运动学求解器（IK）权重参数
    prox_weight: float = 10.0           # 近端权重  
    tip_weight: float = 10.0            # 远端权重
    center_weight: float = 0.12

    # 求解器阻尼与速度限制
    base_damping: float = 0.015
    damping_gain: float = 0.005
    max_dq_norm: float = 0.08

    # 固定点误差容忍度（由于这是数值求解，杆子不可能 100% 完美穿过支点。如果偏差大于1厘米，或者一瞬间偏差突然增大了4毫米，系统会认为该计算步不安全并拒绝执行）
    max_pivot_error: float = 0.01               # 定义允许的最大固定点误差，单位是米
    max_pivot_error_jump: float = 0.004         # 定义允许的固定点误差跃迁上限，单位是米

