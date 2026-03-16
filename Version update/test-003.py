import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QDoubleSpinBox,
                             QPushButton, QLabel, QGroupBox, QFrame)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import LightSource  # 引入光源计算
import matplotlib

# 配置中文字体，防止乱码
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',
    'SimHei',
    'Noto Sans CJK SC'
]
matplotlib.rcParams['axes.unicode_minus'] = False


# ==============================
# 原始物理计算（保持不变）
# ==============================
def reflection(L, beta, gamma, ratio, k):
    M = np.zeros((7, 7), dtype=complex)
    coeffs = [beta, 0, 0, 0, 1 - gamma * k ** 2, 0, ratio * k ** 2]
    our_roots = np.roots(coeffs)

    idx = np.argsort(np.real(our_roots))
    our_roots = our_roots[idx]

    for count1 in range(3):
        r = our_roots[count1]
        M[0, count1], M[1, count1], M[2, count1] = r, r ** 2, r ** 3
        M[3, count1], M[4, count1] = r ** 4 * np.exp(r * L), r ** 5 * np.exp(r * L)
        M[5, count1], M[6, count1] = np.exp(r * L), r * np.exp(r * L)

    for count1 in range(3, 6):
        r = our_roots[count1]
        M[0, count1] = r * np.exp(-r * L)
        M[1, count1] = r ** 2 * np.exp(-r * L)
        M[2, count1] = r ** 3 * np.exp(-r * L)
        M[3, count1] = r ** 4
        M[4, count1] = r ** 5
        M[5, count1] = 1
        M[6, count1] = r

    M[5, 6] = -1
    M[6, 6] = -1j * k * ratio

    C = np.zeros((7,), dtype=complex)
    C[5] = 1
    C[6] = -1j * k * ratio

    LHS = np.linalg.solve(M, C)
    return LHS[:6], LHS[6], M, our_roots


# ==============================
# 绘制带侧面封闭的实体体积块 (剖面视觉核心)
# ==============================
def plot_volume(ax, x, y, Z_top, Z_bot, top_cmap, side_color, ls):
    X, Y = np.meshgrid(x, y)

    # 标量兼容，确保底部是一个二维数组
    if np.isscalar(Z_bot):
        Z_bot = np.full_like(Z_top, Z_bot)

    # 1. 顶面：带有光影明暗效果的起伏曲面
    rgb_top = ls.shade(Z_top, cmap=cm.get_cmap(top_cmap), blend_mode='soft')
    ax.plot_surface(X, Y, Z_top, facecolors=rgb_top, rstride=1, cstride=3, antialiased=False, shade=False)

    # 2. 侧面：纯色+默认多边形阴影，构建厚重剖面感
    # 前剖面 (Y最小)
    X_front = np.vstack((X[0, :], X[0, :]))
    Y_front = np.vstack((Y[0, :], Y[0, :]))
    Z_front = np.vstack((Z_bot[0, :], Z_top[0, :]))
    ax.plot_surface(X_front, Y_front, Z_front, color=side_color, shade=True)

    # 后剖面 (Y最大)
    X_back = np.vstack((X[-1, :], X[-1, :]))
    Y_back = np.vstack((Y[-1, :], Y[-1, :]))
    Z_back = np.vstack((Z_bot[-1, :], Z_top[-1, :]))
    ax.plot_surface(X_back, Y_back, Z_back, color=side_color, shade=True)

    # 左剖面 (X最小)
    X_left = np.vstack((X[:, 0], X[:, 0]))
    Y_left = np.vstack((Y[:, 0], Y[:, 0]))
    Z_left = np.vstack((Z_bot[:, 0], Z_top[:, 0]))
    ax.plot_surface(X_left, Y_left, Z_left, color=side_color, shade=True)

    # 右剖面 (X最大)
    X_right = np.vstack((X[:, -1], X[:, -1]))
    Y_right = np.vstack((Y[:, -1], Y[:, -1]))
    Z_right = np.vstack((Z_bot[:, -1], Z_top[:, -1]))
    ax.plot_surface(X_right, Y_right, Z_right, color=side_color, shade=True)


# ==============================
# 主程序
# ==============================
class SimulationApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("海冰-海洋波动三维光影模拟系统")
        self.setGeometry(100, 100, 1500, 850)

        # 默认参数
        self.H = 500
        self.H_s = 300
        self.h = 200
        self.E = 1.1e9
        self.L_sheet = 10000
        self.dt = 0.05
        self.current_t = 0
        self.is_running = False

        # 可视化参数
        self.amp_scale = 5
        self.vert_exag = 4

        # 光源系统：方位角135度，仰角45度，模拟真实阳光投射
        self.light_source = LightSource(azdeg=135, altdeg=45)

        self.init_ui()
        self.recompute_physics()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ==========================
    # UI界面
    # ==========================
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        sidebar = QVBoxLayout()
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(320)
        sidebar_frame.setLayout(sidebar)
        layout.addWidget(sidebar_frame)

        # --------物理参数--------
        phys_group = QGroupBox("物理参数")
        phys_layout = QFormLayout()
        self.spin_H = self.create_spin(100, 2000, self.H)
        self.spin_Hs = self.create_spin(50, 1500, self.H_s)
        self.spin_h = self.create_spin(10, 500, self.h)
        self.spin_E = self.create_spin(1e8, 1e11, self.E, decimals=0)
        self.spin_L = self.create_spin(1000, 50000, self.L_sheet)

        phys_layout.addRow("总水深 H (m)", self.spin_H)
        phys_layout.addRow("冰下水深 Hs (m)", self.spin_Hs)
        phys_layout.addRow("冰厚 h (m)", self.spin_h)
        phys_layout.addRow("杨氏模量 E (Pa)", self.spin_E)
        phys_layout.addRow("冰架长度 L (m)", self.spin_L)
        phys_group.setLayout(phys_layout)
        sidebar.addWidget(phys_group)

        # --------可视化--------
        vis_group = QGroupBox("可视化控制")
        vis_layout = QFormLayout()
        self.spin_amp = self.create_spin(1, 100, self.amp_scale)
        self.spin_vert = self.create_spin(1, 20, self.vert_exag)
        vis_layout.addRow("振幅放大倍数", self.spin_amp)
        vis_layout.addRow("垂直夸张倍数", self.spin_vert)
        vis_group.setLayout(vis_layout)
        sidebar.addWidget(vis_group)

        # --------动画--------
        ctrl_group = QGroupBox("动画控制")
        ctrl_layout = QVBoxLayout()
        self.spin_dt = self.create_spin(0.001, 0.5, self.dt, decimals=3)
        ctrl_layout.addWidget(QLabel("时间步长"))
        ctrl_layout.addWidget(self.spin_dt)

        self.btn_play = QPushButton("播放 / 暂停")
        self.btn_play.clicked.connect(self.toggle_animation)
        self.btn_reset = QPushButton("时间重置")
        self.btn_reset.clicked.connect(self.reset_animation)

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_reset)
        ctrl_group.setLayout(ctrl_layout)
        sidebar.addWidget(ctrl_group)

        self.info_label = QLabel()
        sidebar.addWidget(self.info_label)
        sidebar.addStretch()

        # --------Matplotlib--------
        self.fig = Figure(figsize=(10, 8), facecolor="#101010")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, 1)

    def create_spin(self, min_v, max_v, init_v, decimals=2):
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(init_v)
        sb.setDecimals(decimals)
        sb.valueChanged.connect(self.recompute_physics)
        return sb

    # ==========================
    # 物理计算
    # ==========================
    def recompute_physics(self):
        H = self.spin_H.value()
        H_s = self.spin_Hs.value()
        h = self.spin_h.value()
        E = self.spin_E.value()
        L_sheet = self.spin_L.value()
        nu = 0.33
        rho_i, rho_w, g = 922.5, 1025, 9.8066

        D = E * h ** 3 / (12 * (1 - nu ** 2))
        beta = D / (H ** 4 * rho_w * g)
        gamma = (rho_i * h * H_s) / (H ** 2 * rho_w)
        L1 = L_sheet / H

        self.T0 = np.sqrt(H / g)
        self.L_scale = (L_sheet / L1) / 1000
        self.k_values = np.linspace(0.01, 50, 400)

        self.x_water = np.linspace(0, L1, 300)
        self.x_shelf = np.linspace(-L1, 0, 300)

        dw_list, ds_list = [], []

        for k in self.k_values:
            vc, R, _, roots = reflection(L1, beta, gamma, H / H_s, k)

            dw_list.append(
                np.exp(-1j * k * self.x_water) +
                R * np.exp(1j * k * self.x_water)
            )

            s_m = np.zeros_like(self.x_shelf, dtype=complex)
            for c in range(3):
                s_m -= (roots[c] ** 2 / k ** 2) * vc[c] * np.exp(roots[c] * (self.x_shelf + L1))
                s_m -= (roots[c + 3] ** 2 / k ** 2) * vc[c + 3] * np.exp(roots[c + 3] * self.x_shelf)

            ds_list.append((H_s / H) * s_m)

        self.dw_store = np.array(dw_list)
        self.ds_store = np.array(ds_list)
        dx = self.x_water[1] - self.x_water[0]

        self.f_hat_k = (1 / (2 * np.pi)) * dx * (
                self.dw_store @ np.exp(-0.5 * (self.x_water - 6) ** 2)
        )
        self.y_range = np.linspace(-1, 1, 10)

    # ==========================
    # 动画控制
    # ==========================
    def toggle_animation(self):
        self.is_running = not self.is_running

    def reset_animation(self):
        self.current_t = 0

    # ==========================
    # 更新画面 (核心渲染循环)
    # ==========================
    def update_frame(self):
        # --- 新增：在清空画布前，保存当前的鼠标视角状态 ---
        current_elev = self.ax.elev
        current_azim = self.ax.azim

        # 如果是刚初始化，可能获取不到，给个默认的漂亮剖面视角
        if current_elev is None:
            current_elev = 15
            current_azim = -60

        if self.is_running:
            self.current_t += self.spin_dt.value()

        self.ax.clear()
        self.ax.set_facecolor("#101010")

        time_factor = np.exp(-1j * self.k_values * self.current_t) * self.f_hat_k
        dk = self.k_values[1] - self.k_values[0]

        disp_w = dk * np.real(self.dw_store.T @ time_factor)
        disp_s = dk * np.real(self.ds_store.T @ time_factor)

        amp = self.spin_amp.value()
        vert = self.spin_vert.value()

        disp_w = disp_w * amp * vert
        disp_s = disp_s * amp * vert

        max_disp = max(np.max(np.abs(disp_w)), np.max(np.abs(disp_s)))
        if max_disp == 0:
            max_disp = 1.0

            # 创建拓扑高度网格
        Z_top_water = np.tile(disp_w, (len(self.y_range), 1))
        Z_top_shelf = np.tile(disp_s, (len(self.y_range), 1))

        # ---- 定义层级深度 (剖面视觉调整) ----
        # 设定底部深度，使海水有个深邃的视觉剖面
        z_base = -max(2.0, max_disp * 3.5)
        # 冰架的可视化厚度 (成比例显示冰块厚感)
        ice_vis_thick = max(0.5, max_disp * 0.8)

        # 海水的侧面颜色 (深海幽蓝)
        deep_water_color = "#051C2C"
        # 冰块的侧面颜色 (冷光浅蓝)
        ice_side_color = "#99CDE1"

        # 1. 绘制冰架实体 (冰层厚度)
        plot_volume(
            ax=self.ax, x=self.x_shelf * self.L_scale, y=self.y_range,
            Z_top=Z_top_shelf,
            Z_bot=Z_top_shelf - ice_vis_thick,
            top_cmap="bone", side_color=ice_side_color, ls=self.light_source
        )

        # 2. 绘制冰架下方的海水
        # (贴合冰底延伸到深海，使得冰面被抬起，下方依然是水)
        plot_volume(
            ax=self.ax, x=self.x_shelf * self.L_scale, y=self.y_range,
            Z_top=Z_top_shelf - ice_vis_thick,
            Z_bot=z_base,
            top_cmap="GnBu_r", side_color=deep_water_color, ls=self.light_source
        )

        # 3. 绘制敞水区海面与海水
        plot_volume(
            ax=self.ax, x=self.x_water * self.L_scale, y=self.y_range,
            Z_top=Z_top_water,
            Z_bot=z_base,
            top_cmap="GnBu_r", side_color=deep_water_color, ls=self.light_source
        )

        # 视界与轴线调整
        self.ax.set_zlim(z_base, max_disp * 2.5)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-2, 2)

        # 压扁 y 轴，拉长 x 轴，以突出波浪横切面效果
        self.ax.set_box_aspect((4, 1.2, 1.5))

        # --- 修改：恢复清空画布前的鼠标视角 ---
        self.ax.view_init(elev=current_elev, azim=current_azim)

        t_min = self.current_t * self.T0 / 60
        self.ax.set_title(f"模拟时间: {t_min:.2f} 分钟", color="white")
        self.ax.set_xlabel("距离 (km)", color="white")
        self.ax.set_ylabel("横向", color="white")
        self.ax.set_zlabel("位移/深度", color="white")

        # 坐标轴颜色适配暗黑主题
        self.ax.tick_params(colors="white")
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        self.info_label.setText(
            f"最大水面位移: {np.max(np.abs(disp_w)):.3f}\n"
            f"最大冰架位移: {np.max(np.abs(disp_s)):.3f}"
        )

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulationApp()
    window.show()
    sys.exit(app.exec_())