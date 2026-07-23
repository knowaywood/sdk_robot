# X2Robot Client SDK（Desktop）

本目录提供了一个 **X2Robot 模型服务** 的 **Desktop** 示例客户端，用于演示 SDK 推理流程与交互方式。
**该 sample 仅支持 Desktop 机型**

---

## 📦 环境要求

- 需在与
  <https://github.com/X-Square-Robot/sdk_robot#virtual-environment-setup>
  的 **同一 Python 虚拟环境** 中完成安装
- Linux
- 可访问模型服务的网络环境

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r samples/desktop/requirements.txt
```

### 2. 启动 SDK 客户端

```bash
bash samples/desktop/scripts/start_sdk_desktop.sh
```

本分支推荐的显式启动方式：

```bash
bash samples/desktop/scripts/start_sdk_desktop.sh \
  "192.168.1.120" \
  32195 \
  "Pick up the cucumber from the table with the right hand, hand it over to the left hand, and place it in the box with the left hand." \
  "end_pose" \
  0 \
  "192.168.31.251:50051" \
  30 \
  50 \
  4
```

最后三个参数分别是模型动作频率、SDK目标下发频率和 chunk
重叠区（以原始模型步数计）。第5个参数只为兼容旧命令而保留，异步版本不再使用整数插值倍率。

### 🧠 推理流程说明

- **整体流程（简述）**：
  - 连接 **模型服务**（WebSocket），并接收服务端 `metadata`
  - 连接 **机器人 SDK 服务**（`x2://<ROBOT_SDK_URL>`），设置机器人为 SDK 工作模式与控制模式
  - 主线程循环采集观测并同步等待 WebSocket 模型响应
  - 独立执行线程在等待推理期间继续按目标频率消费当前 action buffer
  - 响应返回后，根据本次请求期间实际已提交的动作数删除过期前缀，再原子切换到新 buffer
  - 原始30 Hz模型轨迹按时间重采样到50 Hz；位姿旋转使用 Slerp，夹爪使用零阶保持
- **一次推理循环对应的代码逻辑**：
  - `_collect_sensor_data()`：从机器人读取状态/相机图像，组装为模型输入
  - `_capture_inference_context()`：记录请求编号、时间和累计动作序号
  - `predict_sync()`：通过 WebSocket 发送输入（msgpack 序列化）并等待模型返回
  - `_execute_actions()`：裁掉过期动作、拼接短重叠区并更新执行 buffer
  - `_execution_loop()`：独立于模型推理，以目标频率调用 SDK
- **关于“按回车逐步执行”**：
  - 当前 Desktop sample 虽然启动脚本传了 `--debug-step`，但 `DesktopClient` 里并没有实现按回车暂停的逻辑；因此默认会连续运行（需要逐步调试请参考 `Quanta_X1` sample 的 `debug_step` 行为，或自行在 DesktopClient 中加入断点/交互暂停）。

该实现属于第一阶段的延迟感知异步 chunk 调度，不是论文中在 flow
去噪过程内执行 VJP/inpainting 的完整 RTC。

### ⚙️ 默认配置说明

脚本内置以下默认参数：

| 参数名 | 说明 | 默认值 |
|  ----  | ----  | ----  |
|`MODEL_ADDRESS` | 模型服务IP地址 |  `39.101.65.229` |
|`MODEL_PORT` | 模型服务端口 | `1175` |
|`INSTRUCTION` | 初始发送给模型的指令 | `Pick up the green cup and place it on the tray` |
|`CONTROL_MODE` | 机器人控制模式 | `end_pose` |
|`INTERPOLATE_MULTIPLIER` | 旧命令兼容参数，异步调度器忽略 |`0` |
|`ROBOT_SDK_URL` | 机器人SDK地址 | `192.168.10.1:50051` |
|`MODEL_ACTION_HZ` | 原始模型 action 时间频率 | `30` |
|`EXEC_HZ` | SDK 动作目标下发频率 | `50` |
|`OVERLAP_MODEL_STEPS` | 新旧 chunk 短重叠区 | `4` |

### 📌 注意事项

- 启动前请确认模型服务已正常运行并可访问。
- 首次推理前没有旧动作可执行，机器人会保持初始目标位姿；收到首个 chunk 后开始异步执行。
- 如果日志出现 `Action buffer underrun`，表示模型响应速度已慢于剩余 buffer，需要继续降低推理延迟或增加预测 horizon。
- `Installed action chunk` 日志会输出端到端延迟、已提交/跳过步数、重叠步数和新 buffer 长度。
- 该示例主要用于 `Desktop` 类型推理流程验证与 SDK 集成测试。
- 推理日志与模型返回结果将直接输出在终端中。

### 📄 说明

本示例仅用于 SDK 测试、功能验证及集成参考。
