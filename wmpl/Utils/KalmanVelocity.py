#!/usr/bin/env python3

import numpy as np

def velocityKalmanFilterCV(
        distances, times,
        pos_err=50.0,
        pos_err_process=1.0, beg_speed_frac=0.001, end_speed_frac=0.05, dynamic_process_noise=True,
        cutoff_fraction=0.25
        ):
    """
    Taking the time vs distance measurements, apply a 1D Kalman filter + RTS smoothing (constant-velocity 
    model) to smooth the data. This enables deriving a smooth velocity profile and deceleration.
    For the first `cutoff_fraction` of samples, speed process noise is set to a constant value of
    `beg_speed_frac` times the speed. After that, the process noise is exponentially ramped up to 
    `end_speed_frac` times the speed. 
    
    This version includes a modification to the RTS smoother to ensure that the
    smoothed velocity is always non-increasing (i.e., the fireball decelerates or maintains constant velocity).

    Arguments:
        distances: [ndarray] Distance measurements (m).
        times: [ndarray] Time measurements (s).

    Keyword Arguments:
        pos_err: [float] Measurement error in position (m).
        pos_err_process: [float] Process noise in position (m).
        beg_speed_frac: [float] Process noise fraction of initial speed.
        end_speed_frac: [float] Process noise fraction of final speed.
        dynamic_process_noise: [bool] If True, use dynamic process noise.
        cutoff_fraction: [float] Fraction of samples to use for initial process noise.

    Returns:
        x_smooth: [ndarray] Smoothed state vector (position, velocity).
        P_smooth: [ndarray] Smoothed covariance matrix.
                     Note: P_smooth may not perfectly reflect the covariance of the
                     velocity component after the clamping constraint is applied.
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
    A_ls = np.vstack([t_init, np.ones_like(t_init)]).T # Design matrix for v0*t + c = d - s0
    v0, _ = np.linalg.lstsq(A_ls, d_init - s0, rcond=None)[0]

    x0 = np.array([s0, v0]) # Initial state: [initial_position, initial_velocity]

    # INITIAL COVARIANCE
    # Assuming v0 is positive for fireballs, abs(v0) is for robustness.
    P0 = np.diag([pos_err_process**2, (beg_speed_frac*abs(v0))**2])
    Q0 = P0.copy() # Used if dynamic_process_noise is False

    # MEASUREMENT MODEL
    H_pos = np.array([[1.0, 0.0]]) # Observation matrix for position
    R_pos = np.array([[pos_err**2]]) # Measurement noise covariance for position

    # STORAGE ARRAYS
    x_pred = np.zeros((N,2))   # Predicted state x(k|k-1)
    P_pred = np.zeros((N,2,2)) # Predicted covariance P(k|k-1)
    x_filt = np.zeros((N,2))   # Filtered state x(k|k)
    P_filt = np.zeros((N,2,2)) # Filtered covariance P(k|k)

    # INITIALIZE FILTER
    x_est, P_est = x0.copy(), P0.copy()
    x_pred[0], P_pred[0] = x_est, P_est # Store initial prediction as initial state
    x_filt[0], P_filt[0] = x_est, P_est # Store initial filter estimate as initial state

    cutoff = int(cutoff_fraction*(N - 1)) # Index for changing process noise calculation

    # FORWARD FILTER PASS
    for k in range(1, N):
        dt = times[k] - times[k-1]
        if dt <= 0:
            # This case should ideally not happen if times are strictly increasing.
            # If it does, one might need to skip the step or use a very small dt.
            # For now, we assume dt > 0.
            print(f"Warning: Non-positive dt ({dt}) at step k={k}. Results may be unreliable.")
            # Defaulting to a small positive dt to avoid division by zero if this happens.
            dt = 1e-6 if dt <=0 else dt


        A_kf = np.array([[1, dt], [0,  1]]) # State transition matrix

        # Prediction step
        x_pr = A_kf @ x_est          # x_k|k-1 = A * x_k-1|k-1
        P_pr = (A_kf @ P_est @ A_kf.T) # P_k|k-1 = A * P_k-1|k-1 * A^T (before adding Q)

        # Process noise covariance Q
        if dynamic_process_noise:
            v_pr = x_pr[1] # Predicted velocity at current step

            if k <= cutoff:
                frac = beg_speed_frac
            else:
                ramp_denominator = max(1, (N - 1 - cutoff))
                ramp = (k - cutoff) / ramp_denominator
                if beg_speed_frac == 0.0:
                    frac = end_speed_frac * ramp
                elif end_speed_frac == 0.0: # Ramp down to zero if end_speed_frac is 0
                     frac = beg_speed_frac * (1.0 - ramp) # Linear ramp down
                     # Original exponential formula handles log(0) correctly if end_speed_frac is 0
                     # frac = beg_speed_frac*np.exp(ramp*np.log(end_speed_frac/beg_speed_frac))
                elif beg_speed_frac > 0: # General exponential ramp
                    frac = beg_speed_frac*np.exp(ramp*np.log(end_speed_frac/beg_speed_frac))
                else: # beg_speed_frac is 0, end_speed_frac could be >0
                    frac = end_speed_frac * ramp
            
            # Ensure frac is non-negative
            frac = max(0.0, frac)

            Q_k = np.diag([
                pos_err_process**2,
                (frac*abs(v_pr))**2 # abs() for robustness if v_pr could be negative
            ]) * dt # Scaling by dt assumes pos_err_process and (frac*v_pr) are related to noise variance rate
        else:
            Q_k = Q0*dt 
            # Note: If Q0 represents variances, Q_k = Q0 might be more standard for discrete time.
            # The multiplication by dt implies Q0 elements are spectral densities or rates.
            # We'll keep original formulation.

        P_pr += Q_k # Add process noise: P_k|k-1 = A*P_k-1|k-1*A^T + Q_k

        # Measurement Update for Position
        z_pos = np.array([distances[k]])   # Position measurement at time k
        y_pos = z_pos - (H_pos @ x_pr)     # Innovation (measurement residual)
        
        S_pos_val = (H_pos @ P_pr @ H_pos.T + R_pos).item() # Innovation covariance (scalar)
        if S_pos_val == 0:
            K_pos = np.zeros((2,1)) # Kalman Gain is zero vector if S is zero
        else:
            K_pos = (P_pr @ H_pos.T) / S_pos_val # Kalman Gain
        
        x_up = x_pr + (K_pos @ y_pos).ravel() # Updated state estimate after position measurement
        P_up = (np.eye(2) - K_pos @ H_pos) @ P_pr # Updated covariance

        # Pseudo-Measurement Update for Velocity
        H_vel = np.array([[0.0, 1.0]]) # Observation matrix for velocity
        v_meas_pseudo = (distances[k] - distances[k-1])/dt # Velocity from finite difference
        
        # Covariance of pseudo-velocity measurement. Original: (pos_err/dt)**2
        # If errors in distances[k] and distances[k-1] are independent with variance pos_err**2:
        # Var((d_k - d_{k-1})/dt) = (pos_err**2 + pos_err**2)/dt**2 = 2*pos_err**2/dt**2
        # Sticking to original formulation for R_vel:
        R_vel = np.array([[(pos_err/dt)**2]]) 
        
        y_vel = np.array([v_meas_pseudo]) - (H_vel @ x_up) # Velocity innovation (using x_up)
        
        S_vel_val = (H_vel @ P_up @ H_vel.T + R_vel).item() # Velocity innovation covariance (scalar)
        if S_vel_val == 0:
            K_vel = np.zeros((2,1)) # Kalman Gain for velocity is zero vector if S_vel is zero
        else:
            K_vel = (P_up @ H_vel.T) / S_vel_val # Kalman Gain for velocity
            
        x_est = x_up + (K_vel @ y_vel).ravel() # Final state estimate for step k
        P_est = (np.eye(2) - K_vel @ H_vel) @ P_up # Final covariance

        x_pred[k], P_pred[k] = x_pr, P_pr # Store prediction
        x_filt[k], P_filt[k] = x_est, P_est # Store filter estimate

    # RTS SMOOTHER PASS
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    
    # Initialize with the last filtered state
    x_smooth[-1], P_smooth[-1] = x_filt[-1], P_filt[-1]

    for k in range(N-2, -1, -1): # Iterate backwards from N-2 down to 0
        dt_smooth = times[k+1] - times[k]
        A_smooth = np.array([[1, dt_smooth], [0,  1]]) # State transition from k to k+1
        
        P_pred_k_plus_1 = P_pred[k+1] # This is P(k+1|k) from the forward pass

        # Calculate smoother gain G_k = P_filt[k] @ A_smooth.T @ P_pred[k+1]^-1
        try:
            # Add a small epsilon for numerical stability if P_pred_k_plus_1 is ill-conditioned
            # P_pred_inv = np.linalg.inv(P_pred_k_plus_1 + np.eye(P_pred_k_plus_1.shape[0]) * 1e-12)
            P_pred_inv = np.linalg.inv(P_pred_k_plus_1)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular or near-singular
            print(f"Warning: P_pred[{k+1}] may be singular. Using pseudo-inverse in RTS smoother.")
            P_pred_inv = np.linalg.pinv(P_pred_k_plus_1)
            
        G = P_filt[k] @ A_smooth.T @ P_pred_inv # Smoother gain
        
        # Smoothed state: x_smooth[k] = x_filt[k] + G * (x_smooth[k+1] - x_pred[k+1])
        x_smooth_k_candidate = x_filt[k] + G @ (x_smooth[k+1] - x_pred[k+1])
        
        # Smoothed covariance: P_smooth[k] = P_filt[k] + G * (P_smooth[k+1] - P_pred[k+1]) * G.T
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k+1] - P_pred[k+1]) @ G.T
        
        x_smooth[k] = x_smooth_k_candidate # Tentatively assign candidate

        # --- START MODIFICATION: Enforce non-increasing velocity ---
        # For deceleration, velocity at earlier time k (x_smooth[k,1]) must be >= velocity at later time k+1 (x_smooth[k+1,1]).
        # x_smooth[k+1,1] is the already smoothed (and potentially clamped) velocity from the previous RTS iteration.
        if x_smooth[k, 1] < x_smooth[k+1, 1]:
            # This implies an acceleration from time k to k+1, which is physically unrealistic for a fireball.
            # Clamp the velocity at time k to be equal to the velocity at time k+1.
            x_smooth[k, 1] = x_smooth[k+1, 1]
            # Note: This clamping directly modifies the state's velocity component. 
            # The covariance P_smooth[k] calculated above does not rigorously account for this deterministic clamping.
            # A fully constrained estimation would also adjust P_smooth[k], which is more complex.
        # --- END MODIFICATION ---

    return x_smooth, P_smooth



if __name__ == '__main__':

    import os
    import sys
    import argparse

    import matplotlib.pyplot as plt

    from wmpl.Utils.Pickling import loadPickle
    from wmpl.Trajectory.Trajectory import jacchiaLengthFunc

    
    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="1D Kalman filter on meteor distance data"
    )
    arg_parser.add_argument('traj_path', nargs="?", metavar='TRAJ_PATH', type=str, \
        help="Path to the trajectory pickle file.")
    

    # Add the optional arguments with Kalman filter parameters
    arg_parser.add_argument('--pos_err', type=float, default=100.0, \
        help="Measurement error in position (m). Default: 100.0 m.")
    arg_parser.add_argument('--pos_err_process', type=float, default=1.0, \
        help="Process noise in position (m). Default: 1.0 m.")
    arg_parser.add_argument('-b', '--beg_speed_perc', type=float, default=0.1, \
        help="Process noise percent of initial speed. Default: 0.1 percent.")
    arg_parser.add_argument('-e', '--end_speed_perc', type=float, default=5.0, \
        help="Process noise percent of final speed. Default: 5 percent.")
    arg_parser.add_argument('--fixed_process_noise', action='store_true', \
        help="If set, use a process noise across the whole trajectory. Default: False.")
    arg_parser.add_argument('--cutoff_fraction', type=float, default=0.25, \
        help="Fraction of samples to use for initial process noise. Default: 0.25.")

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
    
        
    #     # Plot distance vs time for each station
    #     plt.plot(obs.time_data[filter_mask], obs.state_vect_dist[filter_mask], 'o', label=obs.station_id)

    #     # Plot the Jacchia model
    #     a1, a2 = obs.jacchia_fit
    #     model_dist = jacchiaLengthFunc(obs.time_data[filter_mask], a1, a2, traj.v_init, 0)
    #     plt.plot(obs.time_data[filter_mask], model_dist, color='k', linewidth=1)

    #     # Compute the stddev between the Jacchia model and the data
    #     jacchia_std = np.std(obs.state_vect_dist[filter_mask] - model_dist)
    #     print(f"Station {obs.station_id}: Jacchia stddev = {jacchia_std:.2f} m")
        


    # plt.legend()
    # plt.show()


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
    x_smooth, _ = velocityKalmanFilterCV(
        dist_data, time_data, 
        pos_err=cml_args.pos_err, pos_err_process=cml_args.pos_err_process,
        beg_speed_frac=cml_args.beg_speed_perc/100, end_speed_frac=cml_args.end_speed_perc/100,
        dynamic_process_noise=not cml_args.fixed_process_noise,
        cutoff_fraction=cml_args.cutoff_fraction
        )

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
    ax_dist.plot((dist_data - distConstVel(time_data, v0))/1000, ht_data/1000, 'o', label='Measured', markersize=2)
    ax_dist.plot((dist_s - distConstVel(time_data, v0))/1000, ht_data/1000, '-', label='Smoothed', color='red')
    ax_dist.set_ylabel('Height (km)') 
    ax_dist.set_xlabel('Lag (km)')
    ax_dist.legend()

    # Plot the distance residuals
    dist_residuals = dist_data - dist_s
    ax_distres.plot((dist_residuals)/1000, ht_data/1000, 'o', label='Residuals', markersize=2)
    ax_distres.set_xlabel('Distance residuals (km)')
    ax_distres.set_xlim(np.percentile(dist_residuals, 0.5)/1000, np.percentile(dist_residuals, 99.5)/1000)
    ax_distres.axvline(0, color='black', linestyle='--')
    ax_distres.legend()

    
    # Plot the velocity
    ax_vel.scatter(vel_data[vel_data > 0]/1000, ht_data[vel_data > 0]/1000, s=5, label='Measured')
    ax_vel.plot(vel_s/1000, ht_data/1000, '-', label='Velocity', color='red')
    ax_vel.set_xlabel('Velocity (m/s)')
    ax_vel.legend()

    # Plot the deceleration
    decel = np.zeros_like(vel_s)
    decel[1:] = np.diff(vel_s)/np.diff(time_data)
    ax_decel.plot(decel[1:]/1000, ht_data[1:]/1000, '-', label='Deceleration', color='red')
    ax_decel.axvline(0, color='black', linestyle='--')
    ax_decel.set_xlabel('Deceleration (km/s$^2$)')


    # Set X axis to 95th percentile of velocity
    ax_vel.set_xlim(np.percentile(vel_data[vel_data > 0], 0.5)/1000, np.percentile(vel_data[vel_data > 0], 97.5)/1000)

    plt.tight_layout()
    
    plt.show()
