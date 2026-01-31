# thermal.py
import numpy as np

def simulate_1r1c_hourly(
    Q_sol_W: np.ndarray,
    T_out_C: np.ndarray,
    occ_mask: np.ndarray,
    C_J_per_K: float,
    H_W_per_K: float,
    T_init_C: float,
    T_set_C: float,
    dt_hours: float = 1.0,
    Q_int_W: np.ndarray | None = None,
):
    """
    1R1C ODE (lumped):
      C dT/dt = (Q_sol + Q_int) - H (T - T_out) - Q_cool
    Control: if occupied and free-floating step exceeds T_set, apply cooling
             to clamp T(t+1)=T_set.

    Returns:
      T_C: temperature (len T+1)
      Q_cool_W: hourly average cooling power (len T)
      L_kWh: hourly cooling energy (len T)
    """
    n = len(Q_sol_W)
    if Q_int_W is None:
        Q_int_W = np.zeros(n, dtype=float)

    dt_s = dt_hours * 3600.0
    tau_s = C_J_per_K / max(H_W_per_K, 1e-12)
    a = np.exp(-dt_s / tau_s) if H_W_per_K > 0 else 1.0
    b = (1.0 - a) / max(H_W_per_K, 1e-12) if H_W_per_K > 0 else dt_s / C_J_per_K

    T = np.zeros(n + 1, dtype=float)
    T[0] = T_init_C

    Q_cool = np.zeros(n, dtype=float)
    L_kWh = np.zeros(n, dtype=float)

    for t in range(n):
        Q_gain = Q_sol_W[t] + Q_int_W[t]

        if H_W_per_K > 0:
            T_free = T_out_C[t] + (T[t] - T_out_C[t]) * a + Q_gain * b
        else:
            # pure capacity model: T_{t+1} = T_t + Q_gain*dt/C
            T_free = T[t] + Q_gain * dt_s / C_J_per_K

        if occ_mask[t] and (T_free > T_set_C):
            # compute average cooling power needed to hit T_set at step end
            if H_W_per_K > 0:
                denom = max(1.0 - a, 1e-12)
                Q_c = H_W_per_K * (T_free - T_set_C) / denom
            else:
                Q_c = C_J_per_K * (T_free - T_set_C) / dt_s

            Q_cool[t] = max(0.0, Q_c)
            T[t + 1] = T_set_C
        else:
            Q_cool[t] = 0.0
            T[t + 1] = T_free

        L_kWh[t] = Q_cool[t] * dt_s / 3.6e6  # W*s to kWh

    return T, Q_cool, L_kWh
