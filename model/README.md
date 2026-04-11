# Final Modeling Files

- `rcm_final.urdf`: visualization-oriented URDF with a 2R universal joint, a fixed sleeve, and a sliding inner rod.
- `rcm_final_position.xml`: MuJoCo MJCF using position actuators.
- `rcm_final_force.xml`: MuJoCo MJCF using motor actuators for direct force/torque input.

Notes:

- Topology is `3R + tool_yaw + tool_pitch + tool_slide`.
- `q1` rotates about `Z`.
- `q2` and `q3` rotate about local `X`.
- `tool_slide` translates along the tool rod axis, which follows local `-Y`.
- The tool-side universal joint is explicit and angle-readable:
  - `tool_yaw`: yaw about the local `Z` axis.
  - `tool_pitch`: pitch about the local `-X` axis.
- The tool-local frame follows the stage-one `Ralign` mapping:
  - local `X -> -Y(link3)`
  - local `Y -> -Z(link3)`
  - local `Z ->  X(link3)`
- The sleeve hangs from the universal joint. Only the inner rod slides.

## Basic Parameters

The stage-one document defines the active 3-DOF arm symbolically as:

- `X = sin(q1) * (L2 sin(q2) + L3 sin(q2 + q3))`
- `Y = -cos(q1) * (L2 sin(q2) + L3 sin(q2 + q3))`
- `Z = L1 + L2 cos(q2) + L3 cos(q2 + q3)`

The current final model uses the following parameter mapping.

### Active 3-DOF Arm

- `L1 = 0.12 m`
  - Meaning: offset from the base yaw joint `q1` to the shoulder joint `q2` along base `+Z`.
- `L2 = 0.12 m`
  - Meaning: upper-arm length from `q2` to `q3`.
- `L3 = 0.12 m`
  - Meaning: forearm length from `q3` to the tool local mount / passive universal-joint center.

### Tool-Side Geometry

- Universal-joint center to sleeve front end: `0.18 m`
- Inner rod visible length: `0.34 m`
- Inner rod radius: `0.006 m`
- Sleeve radius: `0.0095 m`
- Tool slide range: `0.00 m` to `0.15 m`

### Other Current Modeling Dimensions

- Base pedestal height: `0.12 m`
- Pedestal radius: `0.014 m`
- Upper-arm radius: `0.018 m`
- Forearm radius: `0.012 m`
- Universal-joint shell radius: `0.016 m`

## Important Note

The stage-one PDF provides the kinematic symbols and coordinate conventions, but it does not provide fixed numeric values for `L1/L2/L3`.
The numeric values above are the current assumed dimensions used by the files in this folder, chosen to match the existing final model consistently.

## 基本参数

下面这组参数是按 `11/阶段一：建模与分析.pdf` 里的符号定义，映射到当前“最终版建模”文件中的实际数值。

### 阶段一符号与当前模型映射

| 阶段一符号 | 当前模型含义 | 当前取值 |
| --- | --- | --- |
| `L1` | 基座中心到第二关节 `q2` 的竖直偏置 | `0.12 m` |
| `L2` | 第二关节 `q2` 到第三关节 `q3` 的连杆长度 | `0.12 m` |
| `L3` | 第三关节 `q3` 到末端万向节中心 `Pend` 的连杆长度 | `0.12 m` |

### 末端与工具参数

| 参数 | 含义 | 当前取值 |
| --- | --- | --- |
| `Pend` | 末端万向节中心，对应 `tool_local_mount` / `tool_local_body` 安装点 | 位于 `forearm_link` 末端之后 `0.12 m` |
| 套筒长度 | 工具固定外套筒可视长度 | `0.18 m` |
| 内杆长度 | 工具滑动内杆可视长度 | `0.34 m` |
| `tool_slide` 行程 | 工具轴向伸缩范围 | `0.00 ~ 0.15 m` |
| 套筒半径 | 外套筒半径 | `0.0095 m` |
| 内杆半径 | 滑动内杆半径 | `0.006 m` |
| 球壳半径 | 万向节外观球壳半径 | `0.016 m` |

### 当前关节范围

| 关节 | 当前范围 |
| --- | --- |
| `q1` | `[-pi, pi]` |
| `q2` | `[-1.91986218, 1.91986218] rad` |
| `q3` | `[-2.44346095, 2.44346095] rad` |
| `tool_yaw` | `[-pi, pi]` |
| `tool_pitch` | `[-1.57079633, 1.57079633] rad` |
| `tool_slide` | `[0.00, 0.15] m` |

### 说明

- 阶段一文档只给出了 `L1/L2/L3` 的符号定义，没有给出唯一数值；上表数值是当前最终模型采用的建模值。
- 当前模型中的工具端使用 `2R` 万向节加滑动内杆，因此阶段一里的工具距离参数 `L` 并没有被建成一个固定常数，而是由工具几何长度和 `tool_slide` 共同体现。
- 当前三自由度机械臂主链长度按阶段一对应关系可写为：`L1 = 0.12 m`，`L2 = 0.12 m`，`L3 = 0.12 m`。
