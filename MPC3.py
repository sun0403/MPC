import casadi as ca
import numpy as np
import time
# 下一个状态Nächster Zustand
def shift_movement(T, t0, x0, u, f):
    st = x0 + T*f(x0, u[:, 0])
    # 时间增加# Zunahme der Zeit
    t = t0 + T
    # 准备下一个估计的最优控制 Bereiten Sie die nächste geschätzte optimale Steuerung vor.
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t, st, u_end.T


if __name__ == '__main__':
    T = 0.2
    N = 5
    x1=ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x = ca.vertcat(x1,x2,x3)
    n_zustand = x.size()[0]
    u1 = ca.SX.sym('u1')
    u2 = ca.SX.sym('u2')
    u =  ca.vertcat(u1,u2)
    n_steuer = u.size()[0]
    A1 = ca.SX(153.94)
    A2 = ca.SX(153.94)
    A3 = ca.SX(153.94)
    g = ca.SX(981)
    p1 = ca.SX(0.75)
    p2 = ca.SX(0.5)
    qc1 = ca.SX(0.285)
    qc2 = ca.SX(0.285)
    qo1 = ca.SX(0.228)
    qo2 = ca.SX(0.228)
    qo3 = ca.SX(0.228)
    dx1 = ca.SX.sym('dx1')
    dx2 = ca.SX.sym('dx2')
    dx3 = ca.SX.sym('dx3')
    dx1 = (1 / A1) * (u1 - qc1 * ca.sqrt(2 * g *(x1 - x2)) - p1 * qo1 * ca.sqrt(2 * g * x1))
    dx2 = (1 / A2) * (qc1 * ca.sqrt(2 * g * (x1 - x2)) - qc2 * ca.sqrt(2 * g * (x2 - x3)) - p2 * qo2 * ca.sqrt(2 * g * x2))
    dx3 = (1 / A3) * (u2 + qc2 * ca.sqrt(2 * g * (x2 - x3)) - qo3 * ca.sqrt(2 * g * x3))
    dx = ca.vertcat(dx1,dx2,dx3)
    dynamik_system = ca.Function('dynamik_system', [x, u], [dx])#dynamik system


    U = ca.SX.sym('U', n_steuer, N)  # N步内的控制输出Steuerung der Ausgänge innerhalb von N Schritten
    X = ca.SX.sym('X', n_zustand, N + 1)  # N+1步的系统状态 N+1 Schritt des Systemzustands
    P = ca.SX.sym('P', n_zustand + n_zustand)  # 参数Parameter(Anfangs- und Endzustand)
    X[:, 0] = P[:3]
    for i in range(N):
        #dynamik_system_value = dynamik_system(X[:, i], U[:, i])
        X[:, i + 1] = X[:, i] + dynamik_system(X[:, i], U[:, i]) * T
    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    J=0
    for i in range(N):
        J = J + ca.mtimes([(X[:, i] - P[3:]).T, Q, X[:, i] - P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])

    g = []
    for i in range(N + 1):
        g.append(X[0, i])
        g.append(X[1, i])
        g.append(X[2, i])
        g.append(X[0, i]-X[1,i])
        g.append(X[1, i]-X[2, i])
    nlp = {'f': J, 'x':ca.reshape(U,-1,1), 'p': P, 'g': ca.vertcat(*g)}
    opts = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    lbx=0
    ubx=100

    ### 约束Beschränkung
    lbg = []  # 最低约束条件 Beschränkung
    ubg = []  # 最高约束条件
    for _ in range(N+1):
        lbg.append(0.0) #0<x1<100
        ubg.append(100)
        lbg.append(0.0)#0<x2<100
        ubg.append(100)
        lbg.append(0.0)#0<x3<100
        ubg.append(100)
        lbg.append(0.0)#0<x1-x2<inf
        ubg.append(ca.inf)
        lbg.append(0.0)#0<x2-x3<inf
        ubg.append(ca.inf)


    t0=0.0
    x0 = np.array([6,2,1]).reshape(-1, 1)  # 初始状态Anfangszustand

    xs = np.array([37,24, 12]).reshape(-1, 1)  # 末了状态Endzustand
    u0 = np.array([12, 2] * N).reshape(-1, 1)  # 系统初始控制状态Eingang(Steuersvariante)
    x_c = []  # 存储系统的状态 speichern Zustand des Systems
    u_c = []  # 存储控制全部计算后的控制指令 Speichert Eingang
    t_c = []  # 保存时间 Zeit
    sim_time = 100 # 时长 Simzeit
    index_t = []  # 存储时间戳，以便计算每一步求解的时间

    #mpciter = 0
    start_time = time.time()

    while (np.linalg.norm(x0 - xs)> 1) :#and (mpciter - sim_time / T < 0.0):
        c_p = np.concatenate((x0, xs))
        init_u= ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(x0=init_u, p=c_p, lbx=lbx,lbg=lbg,ubx=ubx,ubg=ubg)
        index_t.append(time.time() - t_)
        u_sol = ca.reshape(res['x'], n_steuer, N)
        ff_value = ff(u_sol, c_p)
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, dynamik_system)
        x0 = ca.reshape(x0, -1, 1)
        #mpciter = mpciter + 1

    import matplotlib.pyplot as plt
    import numpy as np

    # Example data (replace with your actual data)
    t_c = np.linspace(0, sim_time, len(x_c))
    u_np_array = np.array([u.full() for u in u_c])

    # Plotting each state variable over time
    plt.figure(figsize=(10, 6))
    for i in range(u_np_array.shape[1]):
        plt.plot(t_c, u_np_array[:, i, 0], label=f'x{i + 1}')

    plt.xlabel('Time')
    plt.ylabel('System State')
    plt.title('System States over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

