
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# Minimum difference between slider
SLIDER_EPSILON = 0.01


def sineFunc(amp, freq):
    return amp*np.sin(2*np.pi*freq*t)


def polyFunc(amp):
    return (t + amp)**2 - (t**(1.0/((amp + 1)))) + t


### Create grid

# Main gridspec
gs = gridspec.GridSpec(6, 2)
gs.update(hspace=0.5, bottom=0.05, top=0.95, left=0.05, right=0.98)

# Index vs. deviations axes
gs_ind = gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=gs[:3, :], wspace=0.0, hspace=2.0)
ax_ind_1 = plt.subplot(gs_ind[:5, 0])
ax_ind_2 = plt.subplot(gs_ind[:5, 1], sharex=ax_ind_1, sharey=ax_ind_1)

# Mass colorbar axis
ax_cbar = plt.subplot(gs_ind[5, :])

# Velocity vs. deviations axies
gs_vel = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3:5, :], wspace=0.0, hspace=0.2)
ax_vel_1 = plt.subplot(gs_vel[0, 0])
ax_vel_2 = plt.subplot(gs_vel[0, 1], sharex=ax_vel_1, sharey=ax_vel_1)


# Sliders
gs_sl = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[5, :], wspace=0.15, hspace=0.2)

# Slider upper left
ax_sl_11 = plt.subplot(gs_sl[0, 0])

# Slider lower left
ax_sl_12 = plt.subplot(gs_sl[1, 0])

# Slider upper right
ax_sl_21 = plt.subplot(gs_sl[0, 1])

# Slider lower right
ax_sl_22 = plt.subplot(gs_sl[1, 1])

######


### PLOT DATA

t = np.arange(0, 1, 0.01)

a0_min = 0.2
a0_max = 0.8
f0 = 3.0

# Index vs. deviations data
sine_0_min = sineFunc(a0_min, f0)
sine_0_max = sineFunc(a0_max, f0)
masses = np.exp(t*10)

# Velocity vs. deviations data
poly_0_min = polyFunc(a0_min)
poly_0_max = polyFunc(a0_max)


# Index vs. deviations plot 1
scat_ind_dev_1  = ax_ind_1.scatter(t, sine_0_min, c=masses, zorder=3, norm=matplotlib.colors.LogNorm())
ax_ind_1.set_title('Site 1')
ax_ind_1.set_ylabel('Deviation (m)')
ax_ind_1.set_xlabel('Index')
ax_ind_1.grid()


# Index vs. deviations plot 2
scat_ind_dev_2  = ax_ind_2.scatter(t, sine_0_max, c=masses, zorder=3, norm=matplotlib.colors.LogNorm())
ax_ind_2.tick_params(labelleft='off') # Disable left tick labels
ax_ind_2.set_title('Site 2')
ax_ind_2.set_xlabel('Index')
ax_ind_2.grid()


# Colorbar
plt.gcf().colorbar(scat_ind_dev_1, label='Mass (kg)', cax=ax_cbar, orientation='horizontal')


# Velocity vs. deviations plot 1
scat_vel_dev_1,  = ax_vel_1.plot(t, poly_0_min, marker='x', zorder=3)
ax_vel_1.set_xlabel('Velocity (km/s)')
ax_vel_1.set_ylabel('Deviation (m)')
ax_vel_1.grid()


# Velocity vs. deviations plot 1
scat_vel_dev_2,  = ax_vel_2.plot(t, poly_0_max, marker='x', zorder=3)
ax_vel_2.tick_params(labelleft='off') # Disable left tick labels
ax_vel_2.set_xlabel('Velocity (km/s)')
ax_vel_2.grid()


######


### SLIDERS

# Sliders for density
sl_ind_dev_1 = Slider(ax_sl_11, 'Min', 0.0, 1.0, valinit=a0_min)
sl_ind_dev_2 = Slider(ax_sl_12, 'Max', 0.0, 1.0, valinit=a0_max, slidermin=sl_ind_dev_1)
ax_sl_12.set_xlabel('Density')


def update(val):
    """ Update slider values. """

    # Get slider values
    amp_min = sl_ind_dev_1.val
    amp_max = sl_ind_dev_2.val

    # Make sure the sliders do not go beyond one another
    if amp_min > amp_max - SLIDER_EPSILON:
        sl_ind_dev_1.set_val(amp_max - SLIDER_EPSILON)

    if amp_max < amp_min + SLIDER_EPSILON:
        sl_ind_dev_2.set_val(amp_min + SLIDER_EPSILON)

    # Get slider values
    amp_min = sl_ind_dev_1.val
    amp_max = sl_ind_dev_2.val

    # Redraw index vs. deviations plots
    scat_ind_dev_1.set_offsets(np.c_[t, sineFunc(amp_min, f0)])
    scat_ind_dev_2.set_offsets(np.c_[t, sineFunc(amp_max, f0)])

    # Redraw velocity vs. deviations plots
    scat_vel_dev_1.set_ydata(polyFunc(amp_min))
    scat_vel_dev_2.set_ydata(polyFunc(amp_max))

    plt.gcf().canvas.draw_idle()


# Turn on slider updating
sl_ind_dev_1.on_changed(update)
sl_ind_dev_2.on_changed(update)


######

plt.show()