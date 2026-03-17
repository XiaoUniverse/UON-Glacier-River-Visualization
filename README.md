# Sea Ice & Ocean Wave 3D Flexural-Gravity Wave Simulation

**海冰与海洋波浪三维交互模拟**

这是一个基于 **Python + PyQt5 + Matplotlib** 的交互式可视化工具，用于模拟海洋表面重力波（ocean surface gravity waves）对浮动冰架（ice shelf / ice sheet）的弯曲-重力波（flexural-gravity waves）激励过程。

主要展示冰架前端在来波作用下的弹性弯曲变形，以及冰前开放水域的波浪传播特征。

GitHub 仓库：[(https://github.com/XiaoUniverse/UON-Glacier-River-Visualization)](https://github.com/XiaoUniverse/UON-Glacier-River-Visualization)

---

## 功能特点

- 实时调整物理参数，立即重新计算色散关系与反射系数
- 支持暂停/播放、时间重置、动画速度调节
- 鼠标滚轮缩放 3D 视角
- 冰体与水体采用不同渐变着色，便于区分位移幅度
- 显示实时最大位移量与模拟时间（分钟）

---

## 运行效果预览

![程序主界面截图]\(resources/main_window.png)

> 示例参数：冰厚 \~200 m，总水深 600 m，冰架长度 10 km

---

## 安装依赖

```bash
pip install numpy scipy matplotlib PyQt5
```

或使用 conda：

```bash
conda install numpy scipy matplotlib pyqt
```

> 注意：部分旧版 matplotlib 可能需要额外安装 `PyQt5-sip`

---

## 运行方式

```bash
python main.py
```

---

## 参数说明与调节效果

| 参数                          | 物理意义          | 典型范围          | 调节效果                                     |
| --------------------------- | ------------- | ------------- | ---------------------------------------- |
| **Total Water Depth H**     | 总水深（冰底到海底）    | 100–2000 m    | H 增大 → 波速加快，波长变长，冰架弯曲波长拉长，衰减变慢           |
| **Sub-ice Water Depth Hs**  | 冰下水层厚度（冰底到海底） | 50–1500 m     | Hs 减小 → 冰下约束增强，冰架弯曲刚性增强，短波反射率升高，近场振幅变大   |
| **Ice Thickness h**         | 冰架厚度          | 10–500 m      | h 增大 → 弯曲刚度 D ∝ h³，冰架几乎不弯曲，仅整体抬升，长波透过率升高 |
| **Young's Modulus E**       | 冰的杨氏模量        | 1e8 – 1e11 Pa | E 增大 → 刚性增强，冰架变形幅度急剧减小                   |
| **Ice Shelf Length L**      | 冰架沿传播方向长度     | 1000–50000 m  | L 变短 → 边界反射干扰严重，驻波明显；L 变长 → 接近半无限冰架，衰减平滑 |
| **Row Stride / Col Stride** | 3D 表面网格抽样步长   | 1–20          | 增大 stride → 性能提升，但表面细节丢失；建议 1–4          |
| **Interval (ms)**           | 动画每帧间隔        | 1–200 ms      | 减小 → 动画更流畅；增大 → 省资源，适合慢速观察               |
| **Time Step (s)**           | 模拟时间步长 dt     | 0.001–0.5 s   | 增大 dt → 动画推进更快，但可能数值不稳定；建议 0.005–0.05    |

---

## 快速调节建议

| 想看到的现象        | 推荐调节方向                 | 预期视觉效果                   |
| ------------- | ---------------------- | ------------------------ |
| 冰架几乎不弯曲，仅长波抬升 | h ↑↑ 或 E ↑↑ 或 Hs ↓     | 冰面接近水平，缓慢整体起伏            |
| 强短波弯曲（高频涟漪）   | h ↓ 或 E ↓，Hs 较大        | 冰面前端出现高频褶边，快速衰减          |
| 很长的弯曲波长       | H ↑↑ 或 h ↓             | 冰面波浪与水面波浪波长接近，贯穿整个冰架     |
| 明显驻波 / 反射干涉   | L 设短（3–8 km），k 范围包含共振点 | 冰架表面出现节点与腹点，渐变水体 + 半透明冰体 |

> 建议保持默认参数，减小 stride 至 1，增大 Interval 可获得最佳观赏效果（但最耗性能）

---

## 已知限制与改进方向

- 当前仅考虑一维传播（忽略横向三维效应）
- 入射波为单频高斯包络，尚未实现真实海浪谱（如 JONSWAP）
- 未包含冰内粘弹性、破裂判据、海冰漂移等
- Matplotlib 3D 渲染在高分辨率网格下性能较差，可考虑迁移至 **PyVista / vedo**

