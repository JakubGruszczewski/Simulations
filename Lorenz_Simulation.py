import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def dxdt(x, y, sigma=10):
    derivative = sigma * (y - x)
    return derivative


def dydt(x, y, z, rho=28):
    derivative = x * (rho - z) - y
    return derivative


def dzdt(x, y, z, beta=8/3):
    derivative = x * y - beta * z
    return derivative


def lorenz_euler(x_0, y_0, z_0, dt, n, sigma, beta, rho):
    t_e = np.zeros(n + 1)
    x_e = np.zeros(n + 1)
    y_e = np.zeros(n + 1)
    z_e = np.zeros(n + 1)

    x_e[0] = x_0
    y_e[0] = y_0
    z_e[0] = z_0
    t_e[0] = 0

    for i in range(n):
        t_e[i + 1] = t_e[i] + dt
        x_e[i + 1] = x_e[i] + dt * dxdt(x_e[i], y_e[i], sigma)
        y_e[i + 1] = y_e[i] + dt * dydt(x_e[i], y_e[i], z_e[i], rho)
        z_e[i + 1] = z_e[i] + dt * dzdt(x_e[i], y_e[i], z_e[i], beta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_e, y_e, z_e)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"EULER, z(x, y), dt = {dt}")
    plt.show()

    return x_e, y_e, z_e


def lorenz_odeint(x_0, y_0, z_0, dt, n, sigma, beta, rho):
    t_o = np.linspace(0, n * dt, n)
    xnynz_0 = np.asarray([x_0, y_0, z_0])
    ode = odeint(lambda m, t: [dxdt(m[0], m[1], sigma), dydt(m[0], m[1], m[2], rho),
                               dzdt(m[0], m[1], m[2], beta)], xnynz_0, t_o)

    x_o = ode[:, 0]
    y_o = ode[:, 1]
    z_o = ode[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_o, y_o, z_o)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"ODEINT, z(x, y), dt = {dt}")
    plt.show()

    return x_o, y_o, z_o


def approx_error(odeint_data, euler_data):
    n = len(odeint_data)
    error_sum = 0

    for i in range(n):
        error_sum += abs(euler_data[i] - odeint_data[i])

    return error_sum / n


x_0_os = 1
y_0_os = 1
z_0_os = 1
sigma_os = 10
rho_os = 28
beta_os = 8 / 3

T_os = 25
dt_os = 0.001
n_os = int(T_os / dt_os)

x_euler, y_euler, z_euler = lorenz_euler(x_0_os, y_0_os, z_0_os, dt_os, n_os, sigma_os, beta_os, rho_os)
x_odeint, y_odeint, z_odeint = lorenz_odeint(x_0_os, y_0_os, z_0_os, dt_os, n_os, sigma_os, beta_os, rho_os)

approx_error_x = approx_error(x_odeint, x_euler)
approx_error_y = approx_error(y_odeint, y_euler)
approx_error_z = approx_error(z_odeint, z_euler)
print(approx_error_x)
print(approx_error_y)
print(approx_error_z)

fig_os = plt.figure()
ax_os = fig_os.add_subplot(111, projection='3d')
ax_os.plot(x_euler, y_euler, z_euler, label='EULERA')
ax_os.plot(x_odeint, y_odeint, z_odeint, label='ODEINT')
ax_os.set_xlabel('X')
ax_os.set_ylabel('Y')
ax_os.set_zlabel('Z')
ax_os.legend()
plt.show()
