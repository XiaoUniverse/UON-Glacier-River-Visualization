import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def reflection(L, beta, gamma, ratio, k):
    """
    Python version of the MATLAB function:

    function [ vector_c, R ,M, our_roots] = reflection(L,beta,gamma,ratio, k)
    """
    M = np.zeros((7, 7), dtype=complex)

    coeffs = [beta, 0, 0, 0, 1 - gamma * k**2, 0, ratio * k**2]
    our_roots = np.roots(coeffs)

    idx = np.argsort(np.real(our_roots))
    our_roots = our_roots[idx]

    for count1 in range(3):
        r = our_roots[count1]

        M[0, count1] = r
        M[1, count1] = r**2
        M[2, count1] = r**3

        M[3, count1] = r**4 * np.exp(r * L)
        M[4, count1] = r**5 * np.exp(r * L)
        M[5, count1] = np.exp(r * L)
        M[6, count1] = r * np.exp(r * L)

    for count1 in range(3, 6):
        r = our_roots[count1]

        M[0, count1] = r * np.exp(-r * L)
        M[1, count1] = r**2 * np.exp(-r * L)
        M[2, count1] = r**3 * np.exp(-r * L)

        M[3, count1] = r**4
        M[4, count1] = r**5
        M[5, count1] = 1
        M[6, count1] = r

    M[5, 6] = -1
    M[6, 6] = -1j * k * ratio

    C = np.zeros((7,), dtype=complex)
    C[5] = 1
    C[6] = -1j * k * ratio

    LHS = np.linalg.solve(M, C)
    vector_c = LHS[:6]
    R = LHS[6]

    return vector_c, R, M, our_roots


def plot_ice_block(ax, x_shelf_plot, y_shelf, displacement_shelf, h_ice_vis):
    """
    把冰架画成一个有厚度的实体:
    - 顶面
    - 底面
    - 前后侧面
    - 左右端面
    """

    # 顶面 / 底面 mesh
    Xs, Ys = np.meshgrid(x_shelf_plot, y_shelf)
    Zs_top = np.tile(displacement_shelf, (len(y_shelf), 1))
    Zs_bottom = Zs_top - h_ice_vis

    # 1) 顶面
    ax.plot_surface(
        Xs, Ys, Zs_top,
        color='red',
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        shade=True
    )

    # 2) 底面
    ax.plot_surface(
        Xs, Ys, Zs_bottom,
        color='red',
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        shade=True
    )

    # 3) 前侧面: y = y_shelf[0]
    X_front = np.vstack([x_shelf_plot, x_shelf_plot])
    Y_front = np.vstack([
        np.full_like(x_shelf_plot, y_shelf[0]),
        np.full_like(x_shelf_plot, y_shelf[0])
    ])
    Z_front = np.vstack([
        displacement_shelf,
        displacement_shelf - h_ice_vis
    ])

    ax.plot_surface(
        X_front, Y_front, Z_front,
        color='red',
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True
    )

    # 4) 后侧面: y = y_shelf[-1]
    X_back = np.vstack([x_shelf_plot, x_shelf_plot])
    Y_back = np.vstack([
        np.full_like(x_shelf_plot, y_shelf[-1]),
        np.full_like(x_shelf_plot, y_shelf[-1])
    ])
    Z_back = np.vstack([
        displacement_shelf,
        displacement_shelf - h_ice_vis
    ])

    ax.plot_surface(
        X_back, Y_back, Z_back,
        color='red',
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True
    )

    # 5) 左端面: x = x_shelf_plot[0]
    X_left = np.full((2, len(y_shelf)), x_shelf_plot[0])
    Y_left = np.vstack([y_shelf, y_shelf])
    Z_left = np.vstack([
        np.full(len(y_shelf), displacement_shelf[0]),
        np.full(len(y_shelf), displacement_shelf[0] - h_ice_vis)
    ])

    ax.plot_surface(
        X_left, Y_left, Z_left,
        color='red',
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True
    )

    # 6) 右端面: x = x_shelf_plot[-1]
    X_right = np.full((2, len(y_shelf)), x_shelf_plot[-1])
    Y_right = np.vstack([y_shelf, y_shelf])
    Z_right = np.vstack([
        np.full(len(y_shelf), displacement_shelf[-1]),
        np.full(len(y_shelf), displacement_shelf[-1] - h_ice_vis)
    ])

    ax.plot_surface(
        X_right, Y_right, Z_right,
        color='red',
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True
    )


def plot_water_surface(ax, x_water_plot, y_water, displacement_water):
    """
    海水保持为一个表面
    """
    Xw, Yw = np.meshgrid(x_water_plot, y_water)
    Zw = np.tile(displacement_water, (len(y_water), 1))

    ax.plot_surface(
        Xw, Yw, Zw,
        color='blue',
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        shade=True
    )


def main():
    fontsize = 12
    linewidth = 1.5

    name = 'iceshelf_movie_initial_incident'

    H = 500
    H_s = 300
    h = 200
    nu = 0.33
    E = 1.1e9
    g = 9.8066
    rho_i = 922.5
    rho_w = 1025
    L_sheet = 10000

    D = E * h**3 / (12 * (1 - nu**2))
    beta = D / H**4 / rho_w / g
    gamma = rho_i * h * H_s / H**2 / rho_w

    L1 = L_sheet / H
    T_0 = np.sqrt(H / g)

    k_values = np.linspace(0, 50, 10000)

    x_water = np.linspace(0, L1, 1000)
    x_shelf = np.linspace(-L1, 0, 1000)

    R_store = np.ones(len(k_values), dtype=complex)
    displacement_water_store = np.ones((len(k_values), len(x_water)), dtype=complex)
    displacement_shelf_store = np.ones((len(k_values), len(x_shelf)), dtype=complex)
    displacement_shelf_4_derivative_store = np.ones((len(k_values), len(x_shelf)), dtype=complex)

    for count in range(1, len(k_values)):
        print(count + 1)

        vector_c, R, M, our_roots = reflection(L1, beta, gamma, H / H_s, k_values[count])
        R_store[count] = R

        displacement_water = (
            np.exp(-1j * k_values[count] * x_water)
            + R_store[count] * np.exp(1j * k_values[count] * x_water)
        )

        displacement_shelf = np.zeros_like(x_shelf, dtype=complex)
        for count2 in range(3):
            displacement_shelf = (
                displacement_shelf
                - our_roots[count2]**2 / k_values[count]**2
                * vector_c[count2]
                * np.exp(our_roots[count2] * (x_shelf + L1))
                - our_roots[count2 + 3]**2 / k_values[count]**2
                * vector_c[count2 + 3]
                * np.exp(our_roots[count2 + 3] * x_shelf)
            )

        displacement_shelf_4_derivative = np.zeros_like(x_shelf, dtype=complex)
        for count2 in range(3):
            displacement_shelf_4_derivative = (
                displacement_shelf_4_derivative
                - our_roots[count2]**6 / k_values[count]**2
                * vector_c[count2]
                * np.exp(our_roots[count2] * (x_shelf + L1))
                - our_roots[count2 + 3]**6 / k_values[count]**2
                * vector_c[count2 + 3]
                * np.exp(our_roots[count2 + 3] * x_shelf)
            )

        displacement_water_store[count, :] = displacement_water
        displacement_shelf_store[count, :] = H_s / H * displacement_shelf
        displacement_shelf_4_derivative_store[count, :] = H_s / H * displacement_shelf_4_derivative

    eta_water_0 = np.exp(-0.5 * (x_water - 6)**2).reshape(-1, 1)
    eta_shelf_0 = np.exp(-0.5 * (x_shelf + 6)**2).reshape(-1, 1)

    delta_x = x_water[1]
    delta_k = k_values[1]

    k_values = k_values.reshape(-1, 1)

    t_values = np.arange(0, 50 + 0.025, 0.025)

    f_hat_k = (
        1 / np.pi / 2 * delta_x
        * (
            displacement_water_store @ eta_water_0
            + displacement_shelf_store @ eta_shelf_0
            + beta * (displacement_shelf_4_derivative_store @ eta_shelf_0)
        )
    )

    matrix_out = np.exp(-1j * (k_values @ x_water.reshape(1, -1)))
    f_hat_k_out = 1 / np.pi / 2 * delta_x * (matrix_out @ eta_water_0)

    # ===== 3D visualization =====
    plt.ion()
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')

    L_scale = L_sheet / L1 / 1000
    T_scale = T_0 / 60

    # y方向只是为了3D显示，不代表真实第二空间维度
    y_shelf = np.linspace(-0.6, 0.0, 25)
    y_water = np.linspace(0.0, 0.6, 25)

    # 冰块的可视化厚度（z方向）
    h_ice_vis = 0.8

    for count_outer in range(len(t_values)):
        t = t_values[count_outer]

        time_factor = np.exp(-1j * k_values * t) * f_hat_k

        displacement_water = delta_k * np.real(displacement_water_store.T @ time_factor).flatten()
        displacement_shelf = delta_k * np.real(displacement_shelf_store.T @ time_factor).flatten()

        x_shelf_plot = L_scale * x_shelf
        x_water_plot = L_scale * x_water

        ax.clear()

        # 重新设置，否则 clear 后可能恢复默认
        ax.set_box_aspect((5, 2, 1))

        # 冰块实体
        plot_ice_block(
            ax=ax,
            x_shelf_plot=x_shelf_plot,
            y_shelf=y_shelf,
            displacement_shelf=displacement_shelf,
            h_ice_vis=h_ice_vis
        )

        # 海水表面
        plot_water_surface(
            ax=ax,
            x_water_plot=x_water_plot,
            y_water=y_water,
            displacement_water=displacement_water
        )

        # 冰水交界线（x=0）
        z_join = displacement_shelf[-1]
        ax.plot(
            [0, 0],
            [y_shelf[0], y_shelf[-1]],
            [z_join, z_join],
            color='k',
            linewidth=2
        )

        ax.set_xlabel('x / km', fontsize=fontsize)
        ax.set_ylabel('visual width', fontsize=fontsize)
        ax.set_zlabel(r'$\eta$', fontsize=fontsize)

        ax.set_xlim(L_scale * np.min(x_shelf), L_scale * np.max(x_water))
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-2.5, 2.0)

        ax.view_init(elev=25, azim=-55)

        ax.set_title(
            f'Ice shelf (solid block) and seawater surface, t = {t * T_scale:.2f} min',
            fontsize=fontsize + 2
        )

        plt.draw()
        plt.pause(0.03)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()