"""
This version is rendered using the GPU. It has lower latency and better visual quality, but the functionality is not fully complete.
"""
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QDoubleSpinBox,
                             QPushButton, QLabel, QGroupBox, QFrame)
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLSurfacePlotItem, GLGridItem
from pyqtgraph import Vector


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
# GPU 渲染函数（已适配动态比例尺 + 固定视觉盒子）
# ==============================
def plot_volume_gl(glview, x, y, Z_top, Z_bot, top_color, side_color,
                   use_vertical_gradient=False,
                   gradient_top_color=None, gradient_bottom_color=None):
    """
    x, y, Z_top, Z_bot 已经是按 (4 : 1.2 : 1.5) 视觉比例缩放后的坐标
    """
    # 1. 顶面
    top_surf = GLSurfacePlotItem(
        x=x, y=y, z=Z_top,
        color=top_color,
        shader='shaded',
        glOptions='opaque'
    )
    glview.addItem(top_surf)

    # 2. 侧面
    def add_long_side(fixed_y, Z_bot_row, Z_top_row, is_gradient):
        if is_gradient and gradient_top_color is not None and gradient_bottom_color is not None:
            n_vert = 8
            frac = np.linspace(0, 1, n_vert)
            Z_levels = Z_bot_row[:, np.newaxis] * (1 - frac) + Z_top_row[:, np.newaxis] * frac
            y_param = np.full(n_vert, fixed_y)
            colors = np.zeros((len(x), n_vert, 4))
            for c in range(4):
                colors[:, :, c] = (gradient_bottom_color[c] * (1 - frac) +
                                   gradient_top_color[c] * frac)
            side = GLSurfacePlotItem(
                x=x, y=y_param, z=Z_levels,
                colors=colors,
                shader='shaded',
                glOptions='opaque'
            )
        else:
            y_param = np.array([fixed_y, fixed_y])
            Z_levels = np.column_stack((Z_bot_row, Z_top_row))
            side = GLSurfacePlotItem(
                x=x, y=y_param, z=Z_levels,
                color=side_color,
                shader='shaded',
                glOptions='opaque'
            )
        glview.addItem(side)

    def add_trans_side(fixed_x, Z_bot_col, Z_top_col, is_gradient):
        if is_gradient and gradient_top_color is not None and gradient_bottom_color is not None:
            n_vert = 8
            frac = np.linspace(0, 1, n_vert)
            Z_levels = Z_bot_col[np.newaxis, :] * (1 - frac[:, np.newaxis]) + \
                       Z_top_col[np.newaxis, :] * frac[:, np.newaxis]
            x_param = np.full(n_vert, fixed_x)
            colors = np.zeros((n_vert, len(y), 4))
            for c in range(4):
                colors[:, :, c] = (gradient_bottom_color[c] * (1 - frac[:, np.newaxis]) +
                                   gradient_top_color[c] * frac[:, np.newaxis])
            side = GLSurfacePlotItem(
                x=x_param, y=y, z=Z_levels,
                colors=colors,
                shader='shaded',
                glOptions='opaque'
            )
        else:
            x_param = np.array([fixed_x, fixed_x])
            Z_levels = np.row_stack((Z_bot_col, Z_top_col))
            side = GLSurfacePlotItem(
                x=x_param, y=y, z=Z_levels,
                color=side_color,
                shader='shaded',
                glOptions='opaque'
            )
        glview.addItem(side)

    # 前后侧面
    add_long_side(y[0], Z_bot[:, 0], Z_top[:, 0], use_vertical_gradient)
    add_long_side(y[-1], Z_bot[:, -1], Z_top[:, -1], use_vertical_gradient)
    # 左右侧面
    add_trans_side(x[0], Z_bot[0, :], Z_top[0, :], use_vertical_gradient)
    add_trans_side(x[-1], Z_bot[-1, :], Z_top[-1, :], use_vertical_gradient)


# ==============================
# 主程序（GPU + 动态高度比例尺 + 3D网格表格）
# ==============================
class SimulationApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("海冰-海洋波动三维光影模拟系统（GPU版 - 动态比例尺）")
        self.setGeometry(100, 100, 1500, 850)

        # 默认参数
        self.H = 600
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

        # GPU 颜色（RGBA）
        self.ice_top_color = (0.85, 0.90, 0.95, 1.0)
        self.ice_side_color = (0.60, 0.81, 0.90, 0.95)
        self.ice_under_top_color = (0.10, 0.35, 0.65, 0.90)
        self.deep_water_color = (0.04, 0.21, 0.33, 1.0)
        self.water_top_color = (0.30, 0.55, 0.95, 0.95)
        self.water_bottom_color = (0.00, 0.06, 0.13, 1.0)

        self.init_ui()
        self.recompute_physics()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        sidebar = QVBoxLayout()
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(320)
        sidebar_frame.setLayout(sidebar)
        layout.addWidget(sidebar_frame)

        # 物理参数
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

        # 可视化控制
        vis_group = QGroupBox("可视化控制")
        vis_layout = QFormLayout()
        self.spin_amp = self.create_spin(1, 100, self.amp_scale)
        self.spin_vert = self.create_spin(1, 20, self.vert_exag)
        vis_layout.addRow("振幅放大倍数", self.spin_amp)
        vis_layout.addRow("垂直夸张倍数", self.spin_vert)
        vis_group.setLayout(vis_layout)
        sidebar.addWidget(vis_group)

        # 动画控制
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

        # ======== GPU 3D 渲染（固定视觉比例盒子）========
        self.glview = GLViewWidget()
        self.glview.setBackgroundColor('#101010')
        self.glview.setCameraPosition(distance=6.0, elevation=18, azimuth=-55)
        layout.addWidget(self.glview, 1)

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
        self.k_values = np.linspace(0.01, 50, 400)  # 与参考代码一致
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
        self.y_range = np.linspace(-1, 1, 10)  # 与参考代码一致

    def toggle_animation(self):
        self.is_running = not self.is_running

    def reset_animation(self):
        self.current_t = 0

    def update_frame(self):
        if self.is_running:
            self.current_t += self.spin_dt.value()

        # 保存当前视角（用户旋转不会丢失）
        current_elev = self.glview.opts.get('elevation', 18)
        current_azim = self.glview.opts.get('azimuth', -55)

        self.glview.clear()

        # 3D 网格表格（动态高度比例尺的核心视觉参考）
        grid = GLGridItem(size=Vector(4.2, 1.3, 1.6))  # 匹配参考代码的视觉盒子比例
        self.glview.addItem(grid)

        # ===== 物理计算（与参考代码完全一致）=====
        time_factor = np.exp(-1j * self.k_values * self.current_t) * self.f_hat_k
        dk = self.k_values[1] - self.k_values[0]

        disp_w = dk * np.real(self.dw_store.T @ time_factor)
        disp_s = dk * np.real(self.ds_store.T @ time_factor)

        amp = self.spin_amp.value()
        vert = self.spin_vert.value()

        disp_w *= amp * vert
        disp_s *= amp * vert

        max_disp = max(np.max(np.abs(disp_w)), np.max(np.abs(disp_s))) or 1.0

        # ===== 动态高度比例尺（核心：与参考代码的 set_box_aspect + zlim 完全等效）=====
        z_base = -max(2.0, max_disp * 3.5)
        ice_vis_thick = max(0.5, max_disp * 0.8)

        ny = len(self.y_range)
        Z_top_shelf_phys = np.tile(disp_s[:, np.newaxis], (1, ny))
        Z_top_water_phys = np.tile(disp_w[:, np.newaxis], (1, ny))

        Z_bot_shelf_phys = Z_top_shelf_phys - ice_vis_thick
        Z_bot_under_phys = np.full_like(Z_top_shelf_phys, z_base)
        Z_bot_water_phys = np.full_like(Z_top_water_phys, z_base)

        # 固定视觉盒子比例 (4 : 1.2 : 1.5) —— 彻底解决“高度夸张、长度薄弱”
        x_span = 20.0
        y_span = 4.0
        actual_z_span = max_disp * 2.5 - z_base
        scale_x = 4.0 / x_span
        scale_y = 1.2 / y_span
        scale_z = 1.5 / actual_z_span

        # 应用缩放（x/y/z 全部进入固定比例盒子）
        x_shelf_plot = self.x_shelf * self.L_scale * scale_x
        x_water_plot = self.x_water * self.L_scale * scale_x
        y_plot = self.y_range * scale_y

        Z_top_shelf = Z_top_shelf_phys * scale_z
        Z_bot_shelf = Z_bot_shelf_phys * scale_z
        Z_bot_under = Z_bot_under_phys * scale_z
        Z_top_water = Z_top_water_phys * scale_z
        Z_bot_water = Z_bot_water_phys * scale_z

        # ===== GPU 体积绘制（使用缩放后的坐标）=====
        # 冰架
        plot_volume_gl(
            self.glview, x_shelf_plot, y_plot,
            Z_top=Z_top_shelf, Z_bot=Z_bot_shelf,
            top_color=self.ice_top_color,
            side_color=self.ice_side_color,
            use_vertical_gradient=False
        )

        # 冰下海水
        plot_volume_gl(
            self.glview, x_shelf_plot, y_plot,
            Z_top=Z_bot_shelf, Z_bot=Z_bot_under,
            top_color=self.ice_under_top_color,
            side_color=self.deep_water_color,
            use_vertical_gradient=False
        )

        # 敞水区（垂直渐变）
        plot_volume_gl(
            self.glview, x_water_plot, y_plot,
            Z_top=Z_top_water, Z_bot=Z_bot_water,
            top_color=self.water_top_color,
            side_color=None,
            use_vertical_gradient=True,
            gradient_top_color=self.water_top_color,
            gradient_bottom_color=self.water_bottom_color
        )

        # 恢复用户视角 + 固定距离（盒子大小恒定，画面永远均衡）
        self.glview.setCameraPosition(distance=6.0, elevation=current_elev, azimuth=current_azim)

        t_min = self.current_t * self.T0 / 60
        self.info_label.setText(
            f"模拟时间: {t_min:.2f} 分钟\n"
            f"最大水面位移: {np.max(np.abs(disp_w)):.3f}\n"
            f"最大冰架位移: {np.max(np.abs(disp_s)):.3f}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulationApp()
    window.show()
    sys.exit(app.exec_())