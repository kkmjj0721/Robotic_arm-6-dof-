from __future__ import annotations

import math
import glfw
import mujoco
import numpy as np

from collections import Counter
from config import Config
from controller import apply_key_to_counter, command_from_counter, solve_control_step, target_direction, update_target
from kinematics import ModelRefs, Observation, Scenario, TargetState, build_refs, make_initial_target, observe, set_configuration, set_pivot_world



def target_pose_error(obs: Observation, target: TargetState, scenario: Scenario) -> float:
    """
        计算当前姿态和目标姿态之间的几何误差
        obs：当前观测（状态）
        target：目标姿态
        scenario：仿真开始的初始值
    """
    # 先根据目标角度得到期望方向
    desired_dir = target_direction(target)
    # 再根据固定点和目标插入量反推出期望杆近端位置
    desired_prox = scenario.pivot_world - target.insertion_ref * desired_dir
    # 再根据杆长和方向反推出期望杆末端位置
    desired_tip = desired_prox + obs.rod_length * desired_dir
    # 取近端误差和末端误差中更大的那个（当整根杆都接近期望几何时，才会认为误差足够小）
    return max(float(np.linalg.norm(obs.rod_prox_world - desired_prox)), float(np.linalg.norm(obs.rod_tip_world - desired_tip)))


def sample_scenario(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs, cfg: Config, rng: np.random.Generator) -> Scenario:
    """
        随机采样一个可行场景（我们每次产生的随机点不一定都可以做到俯仰偏航等操作的，比如说在工作空间边界、奇异点附近）
        model：mujoco模型
        data：模型数据
        refs：
        cfg：配置参数
        rng：

        “可行”，指的是：关节不贴边、姿态不奇异、杆不穿地、固定点真正在杆轴线上
    """
    # 为 5 个主动关节设置采样边界裕量。
    # 前 3 个大臂关节用 `arm_margin`，后 2 个末端关节用 `wrist_margin`。
    margins = np.array([cfg.arm_margin, cfg.arm_margin, cfg.arm_margin, cfg.wrist_margin, cfg.wrist_margin], dtype=float)
    # 在最大尝试次数内不断重试，直到采到一个满足全部条件的场景。
    for _ in range(cfg.sample_attempts):
        # 计算每个关节区间的总宽度。
        widths = refs.active_ranges[:, 1] - refs.active_ranges[:, 0]
        # 计算带裕量后的采样下界。
        low = refs.active_ranges[:, 0] + widths * margins
        # 计算带裕量后的采样上界。
        high = refs.active_ranges[:, 1] - widths * margins
        # 在安全区间内均匀采样一组主动关节。
        q_active = rng.uniform(low, high)
        # 把采样到的关节值写入模型。
        set_configuration(model, data, refs, q_active)
        # 读取这组关节下对应的几何观测。
        obs = observe(model, data, refs)
        # 如果最小奇异值太小，说明姿态太接近奇异，直接放弃。
        if obs.sigma_min < cfg.min_sigma:
            # 进入下一轮采样。
            continue
        # 如果杆近端或杆末端太低，说明姿态太靠近地面，直接放弃。
        if obs.rod_prox_world[2] < cfg.z_floor or obs.rod_tip_world[2] < cfg.z_floor:
            # 进入下一轮采样。
            continue
        # 给固定点采样留出离两端的安全边界。
        # 这样固定点不会贴着杆端，后续插入/回退也才有余量。
        s_margin = min(0.05, 0.18 * obs.rod_length)
        # 合法插入量最小值。
        s_min = s_margin
        # 合法插入量最大值。
        s_max = obs.rod_length - s_margin
        # 如果这根杆太短，以至于留边后没有合法区间，就放弃。
        if s_max <= s_min:
            # 进入下一轮采样。
            continue
        # 在合法区间内随机采样一个插入量。
        insertion_ref = float(rng.uniform(s_min, s_max))
        # 用“杆近端 + 插入量 * 杆方向”构造一个真正落在当前杆轴上的固定点。
        pivot_world = obs.rod_prox_world + insertion_ref * obs.rod_dir
        # 如果固定点本身低于地板，也放弃。
        if pivot_world[2] < cfg.z_floor:
            # 进入下一轮采样。
            continue
        # 把固定点位置写回模型。
        set_pivot_world(model, refs, pivot_world)
        # 再做一次前向计算，确保 site 世界坐标已经同步到新位置。
        mujoco.mj_forward(model, data)
        # 理论上固定点是按当前杆轴构造出来的，所以 RCM 误差应接近 0。
        # 如果这里不够接近，说明数值或几何状态存在异常，也直接放弃。
        if observe(model, data, refs).pivot_error_norm > 1e-8:
            # 进入下一轮采样。
            continue
        # 把成功采样的结果打包返回。
        return Scenario(q_active=q_active.copy(), pivot_world=pivot_world.copy(), s_bounds=(s_min, s_max))
    # 如果所有尝试都失败，就说明当前配置下难以找到合法场景。
    raise RuntimeError("failed to sample a feasible scenario")


def reset_runtime(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs, cfg: Config, rng: np.random.Generator) -> tuple[Scenario, TargetState, Observation]:
    """
        生成一个新的运行时状态
        model：mujoco模型
        data：模型数据
        refs：
        cfg：
        rng：
    """
    # 先随机采样一个合法场景。
    scenario = sample_scenario(model, data, refs, cfg, rng)
    # 把场景里的固定点写回模型。
    set_pivot_world(model, refs, scenario.pivot_world)
    # 把场景里的主动关节配置写回模型。
    set_configuration(model, data, refs, scenario.q_active)
    # 读取当前完整观测。
    obs = observe(model, data, refs)
    # 用当前观测初始化目标，这样刚重置时不会立刻产生控制误差。
    return scenario, make_initial_target(obs), obs


def maybe_accept_step(model: mujoco.MjModel, data: mujoco.MjData, refs: ModelRefs, scenario: Scenario, current_obs: Observation, q_next: np.ndarray, cfg: Config) -> Observation:
    """
        判断一个候选关节更新是否值得接受
    """
    # 先确保模型中的固定点位置和当前场景一致。
    set_pivot_world(model, refs, scenario.pivot_world)
    # 把候选关节配置写入模型。
    set_configuration(model, data, refs, q_next)
    # 读取候选姿态对应的几何观测。
    candidate = observe(model, data, refs)
    # 计算这一步相对于当前姿态造成的 RCM 误差变化量。
    error_jump = candidate.pivot_error_norm - current_obs.pivot_error_norm
    # 如果候选姿态本身的 RCM 误差太大，或者一步让误差突然恶化太多，就拒绝它。
    if candidate.pivot_error_norm > cfg.max_pivot_error or error_jump > cfg.max_pivot_error_jump:
        # 拒绝前先把固定点恢复为当前场景值。
        set_pivot_world(model, refs, scenario.pivot_world)
        # 再把关节配置恢复到当前观测对应的状态。
        set_configuration(model, data, refs, current_obs.q_active)
        # 返回当前观测，表示“这一步没走成”。
        return current_obs
    # 如果候选姿态把系统带到过于接近奇异的位置，也拒绝。
    if candidate.sigma_min < cfg.hard_sigma:
        # 同样先恢复固定点位置。
        set_pivot_world(model, refs, scenario.pivot_world)
        # 再恢复关节配置。
        set_configuration(model, data, refs, current_obs.q_active)
        # 返回当前观测，表示候选步未被接受。
        return current_obs
    # 通过所有检查，就接受这个候选姿态。
    return candidate


def overlay_text(obs: Observation, target: TargetState, scenario: Scenario) -> tuple[str, str]:
    """
        生成叠加到 MuJoCo 画面上的文字
    """
    # 构造左侧标签列。
    left = "\n".join(["RCM Rod", "RCM Error", "Insertion", "Pitch", "Yaw", "sigma_min"])
    # 构造右侧数值列。
    # 这里把内部单位和显示单位做了适当转换，比如 RCM 误差显示成毫米更直观。
    right = "\n".join(["tool_modeled_link", f"{obs.pivot_error_norm * 1000.0:7.3f} mm", f"{obs.insertion:.4f} / {target.insertion_ref:.4f} m", f"{math.degrees(obs.rod_pitch):6.2f} / {math.degrees(target.pitch_ref):6.2f} deg", f"{math.degrees(obs.rod_yaw):6.2f} / {math.degrees(target.yaw_ref):6.2f} deg", f"{obs.sigma_min:.4f}"])
    # 在左侧标题后补充固定点坐标、目标方向和控制说明的小节标题。
    left += "\n\nPivot XYZ\nTarget dir\nControls"
    # 右侧先补两个换行，用来和左侧的小节标题对齐。
    right += "\n\n"
    # 追加固定点坐标字符串。
    right += np.array2string(scenario.pivot_world, precision=3, suppress_small=True)
    # 换行后准备追加目标方向向量。
    right += "\n"
    # 追加目标方向单位向量。
    right += np.array2string(target_direction(target), precision=3, suppress_small=True)
    # 追加按键帮助文本。
    right += "\nI/K pitch  J/L yaw  U/O insert  N reset  X quit"
    # 返回左右两列字符串。
    return left, right

def run_viewer(cfg: Config) -> None:
    """
        运行整个可视化仿真主循环
    """
    # 从 XML 文件路径加载 MuJoCo 模型。
    model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
    # 为该模型创建一份运行时数据。
    data = mujoco.MjData(model)
    # 解析模型中所有后续高频访问的关键索引。
    refs = build_refs(model)
    # 创建随机数生成器。
    # 把种子放在配置里，是为了场景重现和调参时更稳定。
    rng = np.random.default_rng(cfg.seed)

    # 初始化 GLFW。
    # 如果这一步失败，说明图形环境不可用，后面就没必要继续。
    if not glfw.init():
        # 直接抛出异常终止程序。
        raise RuntimeError("failed to initialize GLFW")
    # 创建窗口。
    window = glfw.create_window(1400, 1000, "Kinematic RCM - tool_modeled_link", None, None)
    # 如果窗口创建失败，则先清理 GLFW 再报错。
    if not window:
        # 释放 GLFW 资源。
        glfw.terminate()
        # 抛出异常提示窗口创建失败。
        raise RuntimeError("failed to create GLFW window")
    # 把这个窗口对应的 OpenGL 上下文设为当前上下文。
    glfw.make_context_current(window)
    # 打开垂直同步。
    # 这样渲染交换缓冲时会跟显示器刷新率同步，画面更稳定。
    glfw.swap_interval(1)

    # 创建一个一帧内累计按键事件的计数器。
    pending = Counter()

    # 定义键盘回调函数。
    # 回调只负责收集输入，不在这里直接改目标或算逆解，这样主循环结构更清楚。
    def key_callback(window_handle, key, scancode, action, mods) -> None:
        # 这些参数当前实现不需要，所以显式丢弃。
        del window_handle, scancode, mods
        # 只处理“按下”和“按住重复”两类事件。
        if action in (glfw.PRESS, glfw.REPEAT):
            # 把本次按键写进计数器。
            apply_key_to_counter(pending, key)

    # 把键盘回调注册到窗口上。
    glfw.set_key_callback(window, key_callback)

    # 创建 MuJoCo 相机对象。
    cam = mujoco.MjvCamera()
    # 创建 MuJoCo 渲染选项对象。
    opt = mujoco.MjvOption()
    # 创建 MuJoCo 场景缓存对象。
    scn = mujoco.MjvScene(model, maxgeom=10000)
    # 创建 MuJoCo 渲染上下文。
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # 初始化一个新的场景、目标和观测。
    scenario, target, obs = reset_runtime(model, data, refs, cfg, rng)
    # 把相机注视点放在固定点附近。
    cam.lookat[:] = scenario.pivot_world
    # 设定相机与目标的距离。
    cam.distance = 1.2
    # 设定相机俯仰角。
    cam.elevation = -22.0
    # 设定相机水平绕转角。
    cam.azimuth = 130.0

    # 在终端打印一条启动完成消息。
    print("Viewer ready.", flush=True)
    # 当窗口仍处于打开状态时，持续执行主循环。
    while not glfw.window_should_close(window):
        # 先处理窗口和输入事件。
        glfw.poll_events()
        # 把累计按键事件转换成一帧命令。
        cmd = command_from_counter(pending)
        # 本帧命令已经读走，计数器清空，等待下一帧重新累计。
        pending.clear()

        # 如果收到退出命令，就结束主循环。
        if cmd.quit:
            # 退出循环，后面会统一销毁窗口和终止 GLFW。
            break
        # 如果收到重置命令，就重新生成场景。
        if cmd.reset:
            # 重新采样场景、目标和观测。
            scenario, target, obs = reset_runtime(model, data, refs, cfg, rng)
            # 相机继续跟随新的固定点。
            cam.lookat[:] = scenario.pivot_world
        # 如果没有重置，就只更新目标。
        else:
            # 把本帧用户命令作用到目标 yaw / pitch / insertion 上。
            update_target(target, cmd, scenario, cfg)
        # 只有当当前姿态和目标姿态之间确实有可见误差时，才去求一步逆解。
        if target_pose_error(obs, target, scenario) > 1e-5:
            # 根据当前观测和目标计算一份候选关节配置。
            q_next = solve_control_step(model, data, refs, scenario, obs, target, cfg)
            # 再通过安全过滤器决定是否接受这一步。
            obs = maybe_accept_step(model, data, refs, scenario, obs, q_next, cfg)

        # 读取当前帧缓冲区尺寸。
        width, height = glfw.get_framebuffer_size(window)
        # 如果窗口被最小化或尺寸暂时非法，就跳过这一帧渲染。
        if width <= 0 or height <= 0:
            # 直接进入下一帧循环。
            continue
        # 根据当前窗口尺寸构造 MuJoCo 视口。
        viewport = mujoco.MjrRect(0, 0, width, height)
        # 用当前 `model + data + camera` 组合更新可渲染场景。
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        # 把场景真正画到屏幕对应的视口里。
        mujoco.mjr_render(viewport, scn, con)
        # 生成屏幕叠加文字。
        left, right = overlay_text(obs, target, scenario)
        # 把叠加文字画到窗口左上角。
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, left, right, con)
        # 交换前后缓冲区，让这一帧真正显示出来。
        glfw.swap_buffers(window)

    # 主循环退出后销毁窗口。
    glfw.destroy_window(window)
    # 主循环退出后终止 GLFW。
    glfw.terminate()