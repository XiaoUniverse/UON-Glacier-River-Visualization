"""
Compared with the previous version, the rendering speed has been improved.
"""
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QDoubleSpinBox,
                             QPushButton, QLabel, QGroupBox, QFrame)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# 配置中文字体，防止乱码
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False


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


def plot_volume(ax, x, y, Z_top, Z_bot, top_cmap, side_color, ls,
                use_vertical_gradient=False, top_color=None, bottom_color=None):
    X, Y = np.meshgrid(x, y)
    if np.isscalar(Z_bot):
        Z_bot = np.full_like(Z_top, Z_bot)

    # 顶面
    rgb_top = ls.shade(Z_top, cmap=cm.get_cmap(top_cmap), blend_mode='soft')
    ax.plot_surface(X, Y, Z_top, facecolors=rgb_top,
                    rstride=2, cstride=5,
                    antialiased=False, shade=False)

    z_min = Z_bot.min()
    z_max = Z_top.max()
    z_range = z_max - z_min + 1e-10

    if use_vertical_gradient and top_color is not None and bottom_color is not None:
        grad_cmap = LinearSegmentedColormap.from_list("water_gradient", [bottom_color, top_color], N=256)
    else:
        grad_cmap = None

    def plot_side(X_side, Y_side, Z_side, default_color):
        if use_vertical_gradient and grad_cmap is not None:
            n_z = 4
            z_bot = Z_side[0, :]
            z_top = Z_side[1, :]
            frac = np.linspace(0, 1, n_z)[:, np.newaxis]
            Z_fine = z_bot * (1 - frac) + z_top * frac

            z_norm = (Z_fine - z_min) / z_range
            z_norm = np.power(z_norm, 1.5)
            facecolors = grad_cmap(z_norm)

            X_fine = np.tile(X_side[0, :], (n_z, 1))
            Y_fine = np.tile(Y_side[0, :], (n_z, 1))

            ax.plot_surface(X_fine, Y_fine, Z_fine, facecolors=facecolors,
                            rstride=1, cstride=4,
                            antialiased=False, alpha=0.92, shade=True)
        else:
            ax.plot_surface(X_side, Y_side, Z_side, color=default_color,
                            rstride=2, cstride=3,
                            antialiased=False, shade=True)

    # 前后左右四个侧面
    for idx, (X_slice, Y_slice) in enumerate([
        ((X[0,:], X[0,:]), (Y[0,:], Y[0,:])),           # 前
        ((X[-1,:], X[-1,:]), (Y[-1,:], Y[-1,:])),       # 后
        ((X[:,0], X[:,0]), (Y[:,0], Y[:,0])),           # 左
        ((X[:,-1], X[:,-1]), (Y[:,-1], Y[:,-1]))        # 右
    ]):
        X_side = np.vstack(X_slice)
        Y_side = np.vstack(Y_slice)
        if idx < 2:  # 前后
            Z_side = np.vstack((Z_bot[idx%len(Z_bot),:], Z_top[idx%len(Z_top),:]))
        else:        # 左右
            Z_side = np.vstack((Z_bot[:,idx%len(Z_bot[0])], Z_top[:,idx%len(Z_top[0])]))
        plot_side(X_side, Y_side, Z_side, side_color)


class SimulationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("海冰-海洋波动三维光影模拟系统（白色背景优化版）")
        self.setGeometry(100, 100, 1500, 850)

        self.H = 600
        self.H_s = 300
        self.h = 200
        self.E = 1.1e9
        self.L_sheet = 10000
        self.dt = 0.05
        self.current_t = 0
        self.is_running = False

        self.amp_scale = 5
        self.vert_exag = 4

        self.light_source = LightSource(azdeg=135, altdeg=45)

        # 缓存变量
        self.cached_disp_w = None
        self.cached_disp_s = None
        self.last_amp = None
        self.last_vert = None

        self.init_ui()
        self.recompute_physics()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        sidebar = QVBoxLayout()
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(320)
        sidebar_frame.setLayout(sidebar)
        layout.addWidget(sidebar_frame)

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

        vis_group = QGroupBox("可视化控制")
        vis_layout = QFormLayout()
        self.spin_amp = self.create_spin(1, 100, self.amp_scale)
        self.spin_vert = self.create_spin(1, 20, self.vert_exag)
        vis_layout.addRow("振幅放大倍数", self.spin_amp)
        vis_layout.addRow("垂直夸张倍数", self.spin_vert)
        vis_group.setLayout(vis_layout)
        sidebar.addWidget(vis_group)

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

        # Matplotlib 画布 - 白色背景
        self.fig = Figure(figsize=(10, 8), facecolor="white")
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

        self.k_values = np.linspace(0.01, 50, 140)
        self.x_water = np.linspace(0, L1, 110)
        self.x_shelf = np.linspace(-L1, 0, 110)
        self.y_range = np.linspace(-1, 1, 6)

        dw_list, ds_list = [], []
        for k in self.k_values:
            vc, R, _, roots = reflection(L1, beta, gamma, H / H_s, k)
            dw_list.append(np.exp(-1j * k * self.x_water) + R * np.exp(1j * k * self.x_water))
            s_m = np.zeros_like(self.x_shelf, dtype=complex)
            for c in range(3):
                s_m -= (roots[c] ** 2 / k ** 2) * vc[c] * np.exp(roots[c] * (self.x_shelf + L1))
                s_m -= (roots[c + 3] ** 2 / k ** 2) * vc[c + 3] * np.exp(roots[c + 3] * self.x_shelf)
            ds_list.append((H_s / H) * s_m)

        self.dw_store = np.array(dw_list)
        self.ds_store = np.array(ds_list)
        dx = self.x_water[1] - self.x_water[0]
        self.f_hat_k = (1 / (2 * np.pi)) * dx * (self.dw_store @ np.exp(-0.5 * (self.x_water - 6) ** 2))

    def toggle_animation(self):
        self.is_running = not self.is_running

    def reset_animation(self):
        self.current_t = 0

    def update_frame(self):
        current_elev = getattr(self.ax, 'elev', 15)
        current_azim = getattr(self.ax, 'azim', -60)

        amp = self.spin_amp.value()
        vert = self.spin_vert.value()

        force_recompute = self.is_running or \
                          self.last_amp != amp or \
                          self.last_vert != vert or \
                          self.cached_disp_w is None

        if force_recompute:
            if self.is_running:
                self.current_t += self.spin_dt.value()

            time_factor = np.exp(-1j * self.k_values * self.current_t) * self.f_hat_k
            dk = self.k_values[1] - self.k_values[0]
            disp_w_raw = dk * np.real(self.dw_store.T @ time_factor)
            disp_s_raw = dk * np.real(self.ds_store.T @ time_factor)

            self.cached_disp_w = disp_w_raw * amp * vert
            self.cached_disp_s = disp_s_raw * amp * vert
            self.last_amp = amp
            self.last_vert = vert

        disp_w = self.cached_disp_w
        disp_s = self.cached_disp_s

        max_disp = max(np.max(np.abs(disp_w)), np.max(np.abs(disp_s))) or 1.0

        self.ax.clear()
        self.ax.set_facecolor("white")

        z_base = -max(2.0, max_disp * 3.5)
        ice_vis_thick = max(0.5, max_disp * 0.8)

        Z_top_water = np.tile(disp_w, (len(self.y_range), 1))
        Z_top_shelf = np.tile(disp_s, (len(self.y_range), 1))

        plot_volume(self.ax, self.x_shelf * self.L_scale, self.y_range,
                    Z_top_shelf, Z_top_shelf - ice_vis_thick,
                    "bone", "#99CDE1", self.light_source, False)

        plot_volume(self.ax, self.x_shelf * self.L_scale, self.y_range,
                    Z_top_shelf - ice_vis_thick, z_base,
                    "GnBu_r", "#051C2C", self.light_source, False)

        plot_volume(self.ax, self.x_water * self.L_scale, self.y_range,
                    Z_top_water, z_base,
                    "GnBu_r", None, self.light_source, True,
                    "#4C8BF5", "#001020")

        self.ax.set_zlim(z_base, max_disp * 2.5)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-2, 2)
        self.ax.set_box_aspect((4, 1.2, 1.5))
        self.ax.view_init(elev=current_elev, azim=current_azim)

        t_min = self.current_t * self.T0 / 60
        self.ax.set_title(f"模拟时间: {t_min:.2f} 分钟", color="black")
        self.ax.set_xlabel("距离 (km)", color="black")
        self.ax.set_ylabel("横向", color="black")
        self.ax.set_zlabel("位移/深度", color="black")
        self.ax.tick_params(colors="black")

        self.ax.xaxis.pane.fill = True
        self.ax.yaxis.pane.fill = True
        self.ax.zaxis.pane.fill = True
        self.ax.xaxis.pane.set_facecolor("white")
        self.ax.yaxis.pane.set_facecolor("white")
        self.ax.zaxis.pane.set_facecolor("white")

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