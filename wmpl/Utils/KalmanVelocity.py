#!/usr/bin/env python3

import numpy as np

def velocityKalmanFilterCV(
        distances, times,
        pos_err=50.0,
        pos_err_process=1.0, beg_speed_frac=0.0001, end_speed_frac=0.01, dynamic_process_noise=True,
        cutoff_fraction=0.25
        ):
    """
    Taking the time vs distance measurements, apply a 1D Kalman filter + RTS smoothing (constant-velocity 
    model) to smooth the data. This enables deriving a smooth velocity profile and deceleration.
    For the first `cutoff_fraction` of samples, speed process noise is set to a constant value of
    `beg_speed_frac` times the speed. After that, the process noise is exponentially ramped up to 
    `end_speed_frac` times the speed. 
    
    Arguments:
        distances: [ndarray] Distance measurements (m).
        times: [ndarray] Time measurements (s).

    Arguments:
        pos_err: [float] Measurement error in position (m).
        pos_err_process: [float] Process noise in position (m).
        beg_speed_frac: [float] Process noise fraction of initial speed.
        end_speed_frac: [float] Process noise fraction of final speed.
        dynamic_process_noise: [bool] If True, use dynamic process noise.
        cutoff_fraction: [float] Fraction of samples to use for initial process noise.

    Returns:
        x_smooth: [ndarray] Smoothed state vector (position, velocity).
        P_smooth: [ndarray] Smoothed covariance matrix.
    """

    distances = np.asarray(distances, dtype=float)
    times     = np.asarray(times,     dtype=float)
    N = distances.size
    if N < 2:
        raise ValueError("Need at least two measurements.")

    # --- INITIAL STATE ESTIMATION (robust using first 10% of points) ---
    init_frac = 0.10
    n_init = max(2, int(init_frac*N))

    s0 = distances[0]

    # Linear least-squares fit to estimate velocity
    t_init = times[:n_init] - times[0]
    d_init = distances[:n_init]
    A = np.vstack([t_init, np.ones_like(t_init)]).T
    v0, _ = np.linalg.lstsq(A, d_init - s0, rcond=None)[0]

    x0 = np.array([s0, v0])


    # INITIAL COVARIANCE
    P0 = np.diag([pos_err_process**2,
                  (beg_speed_frac*abs(v0))**2])
    Q0 = P0.copy()

    # MEASUREMENT MODEL
    H_pos = np.array([[1.0, 0.0]])
    R_pos = np.array([[pos_err**2]])

    # STORAGE
    x_pred = np.zeros((N,2))
    P_pred = np.zeros((N,2,2))
    x_filt = np.zeros((N,2))
    P_filt = np.zeros((N,2,2))

    # INIT
    x_est, P_est = x0.copy(), P0.copy()
    x_pred[0], P_pred[0] = x_est, P_est
    x_filt[0], P_filt[0] = x_est, P_est

    # Define cutoff index
    cutoff = int(cutoff_fraction*(N - 1))

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
                ramp = (k - cutoff)/max(1, (N - 1 - cutoff))
                if beg_speed_frac == 0.0:
                    frac = end_speed_frac*ramp  # fallback to linear
                else:
                    frac = beg_speed_frac*np.exp(ramp*np.log(end_speed_frac/beg_speed_frac))

            Q = np.diag([
                pos_err_process**2,
                (frac*abs(v_pr))**2
            ])*dt
        else:
            Q = Q0*dt

        P_pr += Q

        # Position update
        z = np.array([distances[k]])
        y = z - (H_pos @ x_pr)
        S = H_pos @ P_pr @ H_pos.T + R_pos
        K = P_pr @ H_pos.T @ np.linalg.inv(S)
        x_up = x_pr + (K @ y).ravel()
        P_up = (np.eye(2) - K @ H_pos) @ P_pr

        # Velocity pseudo-meas
        H_vel = np.array([[0.0, 1.0]])
        v_meas = (distances[k] - distances[k-1])/dt
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



if __name__ == '__main__':

    import os
    import sys
    import argparse

    import matplotlib.pyplot as plt

    from wmpl.Utils.Pickling import loadPickle

    
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="1D Kalman filter on meteor distance data"
    )
    arg_parser.add_argument('traj_path', nargs="?", metavar='TRAJ_PATH', type=str, \
        help="Path to the trajectory pickle file.")
    cml_args = arg_parser.parse_args()

    ### ###
    


    # If the trajectory pickle was given, load the orbital elements from it
    if cml_args.traj_path is None:
        print("No trajectory pickle file given. Exiting.")
        sys.exit(1)

    # Load the trajectory pickle
    traj = loadPickle(*os.path.split(cml_args.traj_path))

    # Dictionary for storing time, height, and velocity data per station
    ht_data_dict = {}

    # Construct an input data array
    ht_data = []
    time_data = []
    dist_data = []
    lat_data = []
    lon_data = []
    vel_data = []
    lag_data = []
    for obs in traj.observations:
        
        if obs.ignore_station:
            continue

        filter_mask = (obs.ignore_list == 0) & (obs.velocities != 0)

        ht_data += obs.model_ht[filter_mask].tolist()
        time_data += obs.time_data[filter_mask].tolist()
        dist_data += obs.state_vect_dist[filter_mask].tolist()
        lat_data += obs.model_lat[filter_mask].tolist()
        lon_data += obs.model_lon[filter_mask].tolist()
        vel_data += obs.velocities[filter_mask].tolist()
        lag_data += obs.lag[filter_mask].tolist()

        # Store the data for each station
        ht_data_dict[obs.station_id] = [obs.time_data[filter_mask], obs.model_ht[filter_mask], 
                                        obs.state_vect_dist[filter_mask],
                                        obs.velocities[filter_mask]]


    ht_data = np.array(ht_data)
    time_data = np.array(time_data)
    dist_data = np.array(dist_data)
    lat_data = np.array(lat_data)
    lon_data = np.array(lon_data)
    vel_data = np.array(vel_data)
    lag_data = np.array(lag_data)


    # Print time, height, velocity for each station
    for station_id, data in ht_data_dict.items():

        t, h, d, v = data

        print()
        print("Station:", station_id)
        print("Time (s)   Height (km)   Velocity (km/s)")
        for i in range(len(t)):
            print("{:8.3f}   {:11.3f}   {:15.2f}".format(t[i], h[i]/1000, v[i]/1000))


    # Sort by time
    sorted_indices = np.argsort(time_data)
    vel_data = vel_data[sorted_indices]
    lag_data  = lag_data[sorted_indices]
    time_data = time_data[sorted_indices]
    dist_data = dist_data[sorted_indices]
    lat_data = lat_data[sorted_indices]
    lon_data = lon_data[sorted_indices]
    ht_data  = ht_data[sorted_indices]

    
    # FILTER
    x_smooth, _ = velocityKalmanFilterCV(dist_data, time_data, beg_speed_frac=0.001, end_speed_frac=0.05, pos_err=100.0, dynamic_process_noise=True, cutoff_fraction=0.25)
    dist_s, vel_s = x_smooth[:,0], x_smooth[:,1]

    # Print initial and final speed
    print(f"Initial speed: {vel_s[0]/1000:.2f} km/s")
    print(f"Final speed:   {vel_s[-1]/1000:.2f} km/s")
    

    fig, (ax_dist, ax_distres, ax_vel, ax_decel) = plt.subplots(1, 4, sharey=True, figsize=(12, 6))

    # Define a constant velcotiy model using the initial velocity
    def distConstVel(t, v0):
        return dist_s[0] + v0*(t - time_data[0])
    
    # Plot the lag
    v0 = vel_s[0]
    ax_dist.plot((dist_data - distConstVel(time_data, v0))/1000, time_data, 'o', label='Measured', markersize=2)
    ax_dist.plot((dist_s - distConstVel(time_data, v0))/1000, time_data, '-', label='Smoothed', color='red')
    ax_dist.set_ylabel('Time (s)') 
    ax_dist.set_xlabel('Lag (km)')
    ax_dist.legend()

    # Plot the distance residuals
    dist_residuals = dist_data - dist_s
    ax_distres.plot((dist_residuals)/1000, time_data, 'o', label='Residuals', markersize=2)
    ax_distres.set_ylabel('Time (s)')
    ax_distres.set_xlabel('Distance residuals (km)')
    ax_distres.set_xlim(np.percentile(dist_residuals, 0.5)/1000, np.percentile(dist_residuals, 99.5)/1000)
    ax_distres.axvline(0, color='black', linestyle='--')
    ax_distres.legend()

    
    # Plot the velocity
    ax_vel.scatter(vel_data[vel_data > 0]/1000, time_data[vel_data > 0], s=5, label='Measured')
    ax_vel.plot(vel_s/1000, time_data, '-', label='Velocity', color='red')
    ax_vel.invert_yaxis()
    ax_vel.set_ylabel('Time (s)')
    ax_vel.set_xlabel('Velocity (m/s)')
    ax_vel.legend()

    # Plot the deceleration
    decel = np.zeros_like(vel_s)
    decel[1:] = np.diff(vel_s)/np.diff(time_data)
    ax_decel.plot(decel[1:]/1000, time_data[1:], '-', label='Deceleration', color='red')
    ax_decel.set_ylabel('Time (s)')
    ax_decel.set_xlabel('Deceleration (km/s$^2$)')


    # Set X axis to 95th percentile of velocity
    ax_vel.set_xlim(np.percentile(vel_data[vel_data > 0], 0.5)/1000, np.percentile(vel_data[vel_data > 0], 97.5)/1000)

    plt.tight_layout()
    
    plt.show()
