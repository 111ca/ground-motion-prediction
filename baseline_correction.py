import numpy as np


def integrate_acceleration(acc, dt):
    n = len(acc)

    vel = np.zeros(n)
    vel[0] = 0.5 * acc[0] * dt

    for i in range(1, n):
        vel[i] = vel[i - 1] + 0.5 * (acc[i - 1] + acc[i]) * dt

    vel = vel - np.mean(vel)

    disp = np.zeros(n)
    disp[0] = 0.5 * vel[0] * dt

    for i in range(1, n):
        disp[i] = disp[i - 1] + 0.5 * (vel[i - 1] + vel[i]) * dt

    disp = disp - np.mean(disp)

    return vel, disp


def baseline_correction(acc, dt, order=3):
    n = len(acc)
    t = np.linspace(dt, dt * n, n)

    acc_corr = acc.copy()
    vel, _ = integrate_acceleration(acc, dt)

    Gv = np.zeros((n, order + 1))
    for i in range(order + 1):
        Gv[:, i] = t ** (order + 1 - i)

    coef_v = np.linalg.solve(Gv.T @ Gv, Gv.T @ vel)

    for i in range(order + 1):
        acc_corr -= (order + 1 - i) * coef_v[i] * t ** (order - i)

    acc_new = acc_corr.copy()
    _, disp = integrate_acceleration(acc_corr, dt)

    Gd = np.zeros((n, order + 1))
    for i in range(order + 1):
        Gd[:, i] = t ** (order + 2 - i)

    coef_d = np.linalg.solve(Gd.T @ Gd, Gd.T @ disp)

    for i in range(order + 1):
        acc_new -= (
            (order + 2 - i)
            * (order + 1 - i)
            * coef_d[i]
            * t ** (order - i)
        )

    return acc_new


def load_two_column_record(file_path):
    data = np.loadtxt(file_path)
    time = data[:, 0]
    acc = data[:, 1]
    dt = time[1] - time[0]
    return time, acc, dt


def save_two_column_record(file_path, time, acc):
    output = np.column_stack([time, acc])
    np.savetxt(file_path, output, fmt="%.6f %.8e")