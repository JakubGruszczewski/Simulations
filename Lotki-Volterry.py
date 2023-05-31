import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def dxdt(x, y, a=1.2, b=0.6):
    derivative = (a - b * y) * x
    return derivative


def dydt(x, y, c=0.3, d=0.8):
    derivative = (c * x - d) * y
    return derivative


def lv_euler(x_0, y_0, dt, n, a, b, c, d):
    t_e = np.zeros(n + 1)
    x_e = np.zeros(n + 1)
    y_e = np.zeros(n + 1)

    x_e[0] = x_0
    y_e[0] = y_0
    t_e[0] = 0

    for i in range(n):
        t_e[i + 1] = t_e[i] + dt
        x_e[i + 1] = x_e[i] + dt * dxdt(x_e[i], y_e[i], a, b)
        y_e[i + 1] = y_e[i] + dt * dydt(x_e[i], y_e[i], c, d)

    plt.plot(t_e, y_e, label='y(t) - drapieżniki')
    plt.plot(t_e, x_e, label='x(t) - ofiary')
    plt.title(f"EULER, dt = {dt}")
    plt.legend()
    plt.show()

    return x_e, y_e


def lv_odeint(x_0, y_0, dt, n, a, b, c, d):
    t_o = np.linspace(0, n*dt, n)
    xny_0 = np.asarray([x_0, y_0])
    ode = odeint(lambda matrix, time: [dxdt(matrix[0], matrix[1], a, b), dydt(matrix[0], matrix[1], c, d)], xny_0, t_o)

    x_o = ode[:, 0]
    y_o = ode[:, 1]

    plt.plot(t_o, y_o, label='y(t) - drapieżniki')
    plt.plot(t_o, x_o, label='x(t) - ofiary')
    plt.title(f"ODEINT, dt = {dt}")
    plt.legend()
    plt.show()

    return x_o, y_o


def approx_error(odeint_data, euler_data):
    n = len(odeint_data)
    error_sum = 0

    for i in range(n):
        error_sum += abs(euler_data[i] - odeint_data[i])

    return error_sum / n


x_0_os = 2
y_0_os = 1
a_os = 1.2
b_os = 0.6
c_os = 0.3
d_os = 0.8

T_os = 25 
dt_os = 1 
n_os = int(T_os/dt_os)

x_euler, y_euler = lv_euler(x_0_os, y_0_os, dt_os, n_os, a_os, b_os, c_os, d_os)
x_odeint, y_odeint = lv_odeint(x_0_os, y_0_os, dt_os, n_os, a_os, b_os, c_os, d_os)

approx_error_x = approx_error(x_odeint, x_euler)
approx_error_y = approx_error(y_odeint, y_euler)
print(approx_error_x)
print(approx_error_y)
