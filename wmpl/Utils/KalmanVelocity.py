#!/usr/bin/env python3

import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt

def kalman_filter_1d_cv(distances, times,
                        pos_err=50.0,
                        beg_speed_frac=0.0001,
                        end_speed_frac=0.01,
                        dynamic_process_noise=True,
                        cutoff_fraction=0.25):
    """
    1D Kalman filter + RTS smoothing (constant-velocity model).
    For the first `cutoff_fraction` of samples, speed process noise is zero (no deceleration).
    After that, speed process noise increases exponentially from beg_speed_frac to end_speed_frac.
    """
    distances = np.asarray(distances, dtype=float)
    times     = np.asarray(times,     dtype=float)
    N = distances.size
    if N < 2:
        raise ValueError("Need at least two measurements.")

    # INITIAL STATE
    s0 = distances[0]
    dt0 = times[1] - times[0]
    v0  = (distances[1] - distances[0]) / dt0
    x0 = np.array([s0, v0])

    # INITIAL COVARIANCE
    P0 = np.diag([1.0**2,
                  (end_speed_frac * abs(v0))**2])
    Q0 = P0.copy()

    # MEASUREMENT MODEL
    H_pos = np.array([[1.0, 0.0]])
    R_pos = np.array([[pos_err**2]])

    # STORAGE
    x_pred = np.zeros((N,2)); P_pred = np.zeros((N,2,2))
    x_filt = np.zeros((N,2)); P_filt = np.zeros((N,2,2))

    # INIT
    x_est, P_est = x0.copy(), P0.copy()
    x_pred[0], P_pred[0] = x_est, P_est
    x_filt[0], P_filt[0] = x_est, P_est

    # Define cutoff index
    cutoff = int(cutoff_fraction * (N - 1))

    # FORWARD FILTER
    for k in range(1, N):
        dt = times[k] - times[k-1]
        A = np.array([[1, dt],
                      [0,  1]])
        x_pr = A @ x_est
        P_pr = A @ P_est @ A.T

        if dynamic_process_noise:
            v_pr = x_pr[1]

            if k <= cutoff:
                frac = beg_speed_frac
                #frac = 0
            else:
                ramp = (k - cutoff) / max(1, (N - 1 - cutoff))
                if beg_speed_frac == 0.0:
                    frac = end_speed_frac * ramp  # fallback to linear
                else:
                    frac = beg_speed_frac * np.exp(ramp * np.log(end_speed_frac / beg_speed_frac))

            Q = np.diag([
                1.0**2,
                (frac * abs(v_pr))**2
            ]) * dt
        else:
            Q = Q0 * dt

        P_pr += Q

        # position update
        z = np.array([distances[k]])
        y = z - (H_pos @ x_pr)
        S = H_pos @ P_pr @ H_pos.T + R_pos
        K = P_pr @ H_pos.T @ np.linalg.inv(S)
        x_up = x_pr + (K @ y).ravel()
        P_up = (np.eye(2) - K @ H_pos) @ P_pr

        # velocity pseudo-meas
        H_vel = np.array([[0.0, 1.0]])
        v_meas = (distances[k] - distances[k-1]) / dt
        R_vel = np.array([[(pos_err/dt)**2]])
        y_vel = np.array([v_meas]) - (H_vel @ x_up)
        S_vel = H_vel @ P_up @ H_vel.T + R_vel
        K_vel = P_up @ H_vel.T @ np.linalg.inv(S_vel)
        x_est = x_up + (K_vel @ y_vel).ravel()
        P_est = (np.eye(2) - K_vel @ H_vel) @ P_up

        x_pred[k], P_pred[k] = x_pr, P_pr
        x_filt[k], P_filt[k] = x_est, P_est

    # RTS SMOOTHING
    x_smooth = np.zeros_like(x_filt); P_smooth = np.zeros_like(P_filt)
    x_smooth[-1], P_smooth[-1] = x_filt[-1], P_filt[-1]
    for k in range(N-2, -1, -1):
        dt = times[k+1] - times[k]
        A = np.array([[1, dt],
                      [0,  1]])
        G = P_filt[k] @ A.T @ np.linalg.inv(P_pred[k+1])
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k+1] - x_pred[k+1])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k+1] - P_pred[k+1]) @ G.T

    return x_smooth, P_smooth

def main():
    parser = argparse.ArgumentParser(
        description="1D Kalman filter on meteor distance data"
    )
    parser.add_argument('input_csv',
                        help='CSV: No,Station ID,Ignore,Time (s),State vect dist (m)')
    args = parser.parse_args()
    
    # READ and SORT
    times, dist, vels = [], [], []
    with open(args.input_csv, newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            if int(row['Ignore']) == 0:
                times.append(float(row['Time (s)']))
                dist.append(float(row['State vect dist (m)']))
                vels.append(float(row['Vel (m/s)']))
    times = np.array(times)
    dist  = np.array(dist)
    vels  = np.array(vels)
    order = np.argsort(times)
    times = times[order]
    dist  = dist[order]
    vels  = vels[order]
    
    # FILTER
    x_smooth, _ = kalman_filter_1d_cv(dist, times, beg_speed_frac=0.001, end_speed_frac=0.05, pos_err=100.0, dynamic_process_noise=True, cutoff_fraction=0.25)
    dist_s, vel_s = x_smooth[:,0], x_smooth[:,1]

    # Print initial and final speed
    print(f"Initial speed: {vel_s[0]/1000:.2f} km/s")
    print(f"Final speed:   {vel_s[-1]/1000:.2f} km/s")
    

    fig, (ax_dist, ax_distres, ax_vel, ax_decel) = plt.subplots(1, 4, sharey=True, figsize=(12, 6))

    # Define a constant velcotiy model using the initial velocity
    def distConstVel(t, v0):
        return dist_s[0] + v0*(t - times[0])
    
    # Plot the lag
    v0 = vel_s[0]
    ax_dist.plot((dist - distConstVel(times, v0))/1000, times, 'o', label='Measured', markersize=2)
    ax_dist.plot((dist_s - distConstVel(times, v0))/1000, times, '-', label='Smoothed', color='red')
    ax_dist.set_ylabel('Time (s)') 
    ax_dist.set_xlabel('Lag (km)')
    ax_dist.legend()

    # Plot the distance residuals
    dist_residuals = dist - dist_s
    ax_distres.plot((dist_residuals)/1000, times, 'o', label='Residuals', markersize=2)
    ax_distres.set_ylabel('Time (s)')
    ax_distres.set_xlabel('Distance residuals (km)')
    ax_distres.set_xlim(np.percentile(dist_residuals, 0.5)/1000, np.percentile(dist_residuals, 99.5)/1000)
    ax_distres.axvline(0, color='black', linestyle='--')
    ax_distres.legend()

    
    # Plot the velocity
    ax_vel.scatter(vels[vels > 0]/1000, times[vels > 0], s=5, label='Measured')
    ax_vel.plot(vel_s/1000, times, '-', label='Velocity', color='red')
    ax_vel.invert_yaxis()
    ax_vel.set_ylabel('Time (s)')
    ax_vel.set_xlabel('Velocity (m/s)')
    ax_vel.legend()

    # Plot the deceleration
    decel = np.zeros_like(vel_s)
    decel[1:] = np.diff(vel_s) / np.diff(times)
    ax_decel.plot(decel[1:]/1000, times[1:], '-', label='Deceleration', color='red')
    ax_decel.set_ylabel('Time (s)')
    ax_decel.set_xlabel('Deceleration (km/s$^2$)')


    # Set X axis to 95th percentile of velocity
    ax_vel.set_xlim(np.percentile(vels[vels > 0], 0.5)/1000, np.percentile(vels[vels > 0], 97.5)/1000)

    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    main()
