import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols, plot, init_printing, evalf, sqrt
from sympy.solvers.ode.systems import dsolve_system

plt.style.use('ggplot')
init_printing(use_latex=True)


def pendulum_equation(theta, t_e, g_e, l_e):
    theta1, theta2 = theta
    dtheta1_dt = theta2
    dtheta2_dt = -round(g_e / l_e) * np.sin(theta1)
    return [dtheta1_dt, dtheta2_dt]


def pendulum_odeint(theta0_o, t_o, g_o, l_o):
    theta0_odeint = [theta0_o, -np.pi / 18]
    sol = odeint(pendulum_equation, theta0_odeint, t_o, args=(g_o, l_o))

    plt.plot(t, sol[:, 1])
    plt.title(f"En. wahadełka. ODEINT, dt = {dt}")
    plt.show()

    return sol[:, 1]


def pendulum_sympy(t_s, k_s):
    t = symbols('t')
    x = Function('x')

    eq = Eq(x(t).diff(t, 2), -k_s * x(t))
    sol = dsolve(eq, ics={x(0): np.pi / 18, x(t).diff(t).subs(t, 0): 0})

    plot(sol.rhs, (t, -1, 1), axis_center=(-1, -0.2), title='En. wahadełka. SYMPY')

    return [sol.rhs.subs(t, t_val) for t_val in t_s]


def mean_abs_error(list_sympy, list_odeint, n):
    error_sum = 0
    for i in range(n):
        error_sum += abs(list_sympy[i].evalf() - (list_odeint[i]))
    return error_sum / n


def mean_sqr_error(list_sympy, list_odeint, n):
    error_sum = 0
    for i in range(n):
        error_sum += abs(list_sympy[i].evalf() - abs(list_odeint[i])) ** 2
    return error_sum / n


g = 10
l = 1
k = int(g / l)
theta0 = 0
dt = 0.1
t = np.linspace(-1, 1, int(1 / dt))

odeint_results = pendulum_odeint(theta0, t, g, l)
sympy_results = pendulum_sympy(t, k)

m_a_e = mean_abs_error(sympy_results, odeint_results, int(1 / dt))
m_s_e = mean_sqr_error(sympy_results, odeint_results, int(1 / dt))
print(m_a_e)
print(m_s_e)
