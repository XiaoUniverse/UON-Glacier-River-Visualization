import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QDoubleSpinBox,
                             QPushButton, QLabel, QGroupBox, QFrame)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm


# --- 原始物理逻辑函数 (保持不变) ---

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
        M[0, count1], M[1, count1], M[2, count1] = r * np.exp(-r * L), r ** 2 * np.exp(-r * L), r ** 3 * np.exp(-r * L)
        M[3, count1], M[4, count1], M[5, count1], M[6, count1] = r ** 4, r ** 5, 1, r

    M[5, 6], M[6, 6] = -1, -1j * k * ratio
    C = np.zeros((7,), dtype=complex)
    C[5] = 1;
    C[6] = -1j * k * ratio
    LHS = np.linalg.solve(M, C)
    return LHS[:6], LHS[6], M, our_roots


def plot_solid_block(ax, x, y, disp, thickness, cmap_name, alpha_base, vmin=-1.0, vmax=1.0):
    X, Y = np.meshgrid(x, y)
    Z_top = np.tile(disp, (len(y), 1))
    Z_bot = Z_top - thickness
    norm = cm.colors.Normalize(vmin, vmax)
    color_map = cm.get_cmap(cmap_name)

    if cmap_name == 'Greys':
        color_indices = 0.02 + 0.08 * norm(Z_top)
        face_colors = color_map(color_indices)
    else:
        face_colors = color_map(norm(Z_top))

    ax.plot_surface(X, Y, Z_top, facecolors=face_colors, alpha=alpha_base, shade=False, antialiased=True)
    ax.plot_surface(X, Y, Z_bot, color=face_colors[0, 0, 0:3] * 0.4, alpha=alpha_base * 0.5, shade=True)

    for y_idx in [0, -1]:
        Xs = np.vstack([x, x])
        Ys = np.full_like(Xs, y[y_idx])
        Zs = np.vstack([disp, disp - thickness])
        mid_color = face_colors[len(y) // 2, len(x) // 2, 0:3]
        ax.plot_surface(Xs, Ys, Zs, color=mid_color, alpha=alpha_base * 0.8)


# --- PyQt 界面类 ---

class SimulationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冰架波动力学 3D 仿真控制台")
        self.setGeometry(100, 100, 1400, 800)

        # 初始物理参数
        self.H = 500.0
        self.H_s = 300.0
        self.h = 200.0
        self.nu = 0.33
        self.E = 1.1e9
        self.L_sheet = 10000.0
        self.dt = 0.05
        self.current_t = 0.0
        self.is_running = False

        self.init_ui()
        self.recompute_physics()

        # 设置动画定时器 (QTimer)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 约 33 FPS

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- 左侧控制面板 ---
        sidebar = QVBoxLayout()
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(300)
        sidebar_frame.setLayout(sidebar)
        layout.addWidget(sidebar_frame)

        # 物理参数组
        phys_group = QGroupBox("物理参数")
        phys_layout = QFormLayout()

        self.spin_H = self.create_double_spin(100, 2000, self.H, "水深 H")
        self.spin_Hs = self.create_double_spin(50, 1500, self.H_s, "冰下水深 Hs")
        self.spin_h = self.create_double_spin(10, 1000, self.h, "冰厚 h")
        self.spin_E = self.create_double_spin(1e8, 1e11, self.E, "模量 E", decimals=0)
        self.spin_L = self.create_double_spin(1000, 50000, self.L_sheet, "冰架长度 L")

        phys_layout.addRow("总水深 (H):", self.spin_H)
        phys_layout.addRow("冰下深度 (Hs):", self.spin_Hs)
        phys_layout.addRow("冰厚度 (h):", self.spin_h)
        phys_layout.addRow("Young's E:", self.spin_E)
        phys_layout.addRow("冰架长度:", self.spin_L)
        phys_group.setLayout(phys_layout)
        sidebar.addWidget(phys_group)

        # 动画控制组
        ctrl_group = QGroupBox("动画控制")
        ctrl_layout = QVBoxLayout()

        self.spin_dt = self.create_double_spin(0.001, 0.5, self.dt, "步长 dt", decimals=3)
        ctrl_layout.addWidget(QLabel("时间步长 (dt):"))
        ctrl_layout.addWidget(self.spin_dt)

        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_animation)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self.toggle_animation)
        self.btn_reset = QPushButton("重置时间")
        self.btn_reset.clicked.connect(self.reset_animation)

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(self.btn_play)
        btn_hbox.addWidget(self.btn_reset)
        ctrl_layout.addLayout(btn_hbox)
        ctrl_group.setLayout(ctrl_layout)
        sidebar.addWidget(ctrl_group)

        sidebar.addStretch()

        # --- 右侧 Matplotlib 画布 ---
        self.fig = Figure(figsize=(10, 8), facecolor='#1A1A1A')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, 1)

    def create_double_spin(self, min_v, max_v, init_v, name, decimals=2):
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(init_v)
        sb.setDecimals(decimals)
        if decimals == 0: sb.setSingleStep(1e8)
        sb.valueChanged.connect(self.recompute_physics)
        return sb

    def recompute_physics(self):
        """核心计算逻辑：当参数改变时触发"""
        # 获取界面参数
        H = self.spin_H.value()
        H_s = self.spin_Hs.value()
        h = self.spin_h.value()
        E = self.spin_E.value()
        L_sheet = self.spin_L.value()
        nu = 0.33
        rho_i, rho_w, g = 922.5, 1025, 9.8066

        # 派生参数
        D = E * h ** 3 / (12 * (1 - nu ** 2))
        beta = D / (H ** 4 * rho_w * g)
        gamma = (rho_i * h * H_s) / (H ** 2 * rho_w)
        L1 = L_sheet / H
        self.T0 = np.sqrt(H / g)
        self.L_scale = (L_sheet / L1) / 1000

        # 计算模态存储
        self.k_values = np.linspace(0.01, 50, 400)
        self.x_water = np.linspace(0, L1, 300)
        self.x_shelf = np.linspace(-L1, 0, 300)

        dw_list, ds_list, ds4_list = [], [], []

        for k in self.k_values:
            vc, R, _, roots = reflection(L1, beta, gamma, H / H_s, k)
            dw_list.append(np.exp(-1j * k * self.x_water) + R * np.exp(1j * k * self.x_water))

            s_m = np.zeros_like(self.x_shelf, dtype=complex)
            s_m4 = np.zeros_like(self.x_shelf, dtype=complex)
            for c in range(3):
                s_m -= (roots[c] ** 2 / k ** 2) * vc[c] * np.exp(roots[c] * (self.x_shelf + L1)) + \
                       (roots[c + 3] ** 2 / k ** 2) * vc[c + 3] * np.exp(roots[c + 3] * self.x_shelf)
                s_m4 -= (roots[c] ** 6 / k ** 2) * vc[c] * np.exp(roots[c] * (self.x_shelf + L1)) + \
                        (roots[c + 3] ** 6 / k ** 2) * vc[c + 3] * np.exp(roots[c + 3] * self.x_shelf)

            ds_list.append((H_s / H) * s_m)
            ds4_list.append((H_s / H) * s_m4)

        self.dw_store = np.array(dw_list)
        self.ds_store = np.array(ds_list)
        self.ds4_store = np.array(ds4_list)

        # 初始包络
        dx = self.x_water[1] - self.x_water[0]
        self.f_hat_k = (1 / (2 * np.pi)) * dx * (
                self.dw_store @ np.exp(-0.5 * (self.x_water - 6) ** 2) +
                self.ds_store @ np.exp(-0.5 * (self.x_shelf + 6) ** 2) +
                beta * (self.ds4_store @ np.exp(-0.5 * (self.x_shelf + 6) ** 2))
        )
        self.y_range = np.linspace(-0.8, 0.8, 8)

        if not self.is_running:
            self.update_frame()

    def toggle_animation(self):
        self.is_running = not self.is_running
        self.btn_play.setText("停止" if self.is_running else "播放")

    def reset_animation(self):
        self.current_t = 0.0
        self.update_frame()

    def update_frame(self):
        if self.is_running:
            self.current_t += self.spin_dt.value()

        # 记录当前视角
        curr_elev = self.ax.elev if self.ax.elev is not None else 25
        curr_azim = self.ax.azim if self.ax.azim is not None else -55

        self.ax.clear()
        self.ax.set_facecolor('#1A1A1A')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.grid(False)

        # 时间计算
        time_factor = np.exp(-1j * self.k_values * self.current_t) * self.f_hat_k
        dk = self.k_values[1] - self.k_values[0]
        disp_w = dk * np.real(self.dw_store.T @ time_factor)
        disp_s = dk * np.real(self.ds_store.T @ time_factor)

        # 渲染
        self.ax.set_box_aspect((4, 1.5, 1))

        # 1. 冰川 (白色)
        plot_solid_block(self.ax, self.x_shelf * self.L_scale, self.y_range, disp_s,
                         thickness=0.6, cmap_name='Greys', alpha_base=1.0, vmin=-2, vmax=2)

        # 2. 海水 (纯蓝)
        plot_solid_block(self.ax, self.x_water * self.L_scale, self.y_range, disp_w,
                         thickness=0.8, cmap_name='Blues_r', alpha_base=0.8, vmin=-1.5, vmax=1.5)

        # 样式应用
        t_min = self.current_t * self.T0 / 60
        self.ax.set_title(f"Simulation: {t_min:.2f} min", fontsize=12, color='white')
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.set_xlabel('Distance (km)', color='white')
        self.ax.set_zlabel('Displacement', color='white')

        # 限制范围
        limit_x = self.x_water[-1] * self.L_scale
        self.ax.set_xlim(-limit_x, limit_x)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-2, 2)

        # 还原视角以便支持拖拽
        self.ax.view_init(elev=curr_elev, azim=curr_azim)

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulationApp()
    window.show()
    sys.exit(app.exec_())