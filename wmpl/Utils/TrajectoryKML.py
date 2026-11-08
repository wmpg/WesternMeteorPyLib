
import numpy as np

from wmpl.Utils.DragFit import usedPoints
from wmpl.Utils.TrajConversions import geo2Cartesian, jd2Date
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory


# URL of the icons that Google Earth Web renders in their colour
WEB_ICON = 'https://earth.google.com/earth/document/icon?'

# Number of frames above which the KML may not load in Google Earth Web (1000 frames, 4.5 MB, loaded and
#   2000 frames, 9 MB, did not, while Google Earth Pro handled 7800 frames, 36 MB)
MAX_FRAMES_WEB = 1000

# Number of dashes in each residual line
N_DASHES = 6



def generateTrajectoryKML(traj, dir_path):
    """
    Arguments:
        traj: [Trajectory object]

    """

    # Sample the trajectory every 100 m by height
    traj_samples = sampleTrajectory(traj, -1, -1, 100.0)

    # Get trajectory time/name
    #traj_name = jd2Date(traj.jdt_ref, dt_obj=True).strftime("%Y%m%d-%H%M%S")
    traj_name = traj.file_name


    polygon_list = []

    # Make the trajectory line by making the begin and end point
    polygon_points = []
    polygon_points.append([np.degrees(traj.rbeg_lat), np.degrees(traj.rbeg_lon), traj.rbeg_ele])
    polygon_points.append([np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.rend_ele])
    polygon_points.append([np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), 0])
    polygon_points.append([np.degrees(traj.rbeg_lat), np.degrees(traj.rbeg_lon), 0])
    polygon_points.append(polygon_points[0]) # close the polygon
    
    # Add the trajectory shading to the polygon list
    polygon_list.append(polygon_points)


    # Mark begin and end points in space and on the ground
    # TBD

    # Mark the trajectory line in sapce and on the ground


    ### MAKE A KML ###

    # Init the KML
    kml = "<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://earth.google.com/kml/2.1'><Folder><name>{:s}</name><open>1</open>".format(traj_name) \

    # Add the trajectory path
    kml += """<Placemark id='{:s}'>""".format(traj_name) \
        + """
                <Style id='trajectory'>
                  <LineStyle>
                <width>1.5</width>
                  </LineStyle>
                  <PolyStyle>
                <color>40000800</color>
                  </PolyStyle>
                </Style>
                <styleUrl>#trajectory</styleUrl>\n""" \
        + "<name>Trajectory profile</name>\n"




    kml += """
    <MultiGeometry>"""


    ### Plot all polygons ###
    for polygon_points in polygon_list:
        kml += \
"""    <Polygon>
        <extrude>0</extrude>
        <altitudeMode>absolute</altitudeMode>
        <outerBoundaryIs>
            <LinearRing>
                <coordinates>\n"""

        # Add the polygon points to the KML
        for p_lat, p_lon, p_elev in polygon_points:
            kml += "                    {:.6f},{:.6f},{:.0f}\n".format(p_lon, p_lat, p_elev)

        kml += \
"""                </coordinates>
            </LinearRing>
        </outerBoundaryIs>
    </Polygon>"""
    ### ###


    kml += \
"""    </MultiGeometry>
    </Placemark>
"""

    ### Add a line representing the trajectory in space
    kml += \
"""
    <Placemark> 
        <name>Trajectory</name>
        <description>"""
    kml += "Beg:Lat={:.6f},Lon={:.6f},Ht={:.0f}\n".format(np.degrees(traj.rbeg_lat), np.degrees(traj.rbeg_lon), traj.rbeg_ele)
    kml += "End:Lat={:.6f},Lon={:.6f},Ht={:.0f}".format(np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.rend_ele)
    kml += \
    """
        </description>
        <LineString>
            <coordinates>"""

    for lat, lon, ht in zip(traj_samples.lat, traj_samples.lon, traj_samples.ht):

        kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(
                    np.degrees(lon), np.degrees(lat), ht)

    kml += \
"""
            </coordinates>
            <altitudeMode>absolute</altitudeMode>
        </LineString>
         
        <Style> 
            <LineStyle>  
                <color>#ff0000ff</color>
                <width>2</width>
            </LineStyle> 
        </Style>
    </Placemark>
"""

    ###


    ### Add the ground projection of the trajectory
    kml += \
"""
    <Placemark> 
        <name>Ground projection</name>
        <LineString>
            <coordinates>"""
    for lat, lon in zip(traj_samples.lat, traj_samples.lon):

        kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(
                    np.degrees(lon), np.degrees(lat), 0)

    kml += \
"""
            </coordinates>
            <altitudeMode>clampToGround</altitudeMode>
        </LineString>
         
        <Style> 
            <LineStyle>  
                <color>#ffffffff</color>
                <width>2</width>
            </LineStyle> 
        </Style>
    </Placemark>
"""

    ###


    ### Add beg/end points ###

    # Begin point in 3D
    kml += \
"""
    <Placemark> 
        <name>Begin</name> 
        <description>"""
    kml += \
"""
            Lat = {:.6f} deg
            Lon = {:.6f} deg
            Ht  = {:.3f} km""".format(np.degrees(traj.rbeg_lat), np.degrees(traj.rbeg_lon), traj.rbeg_ele/1000)
    kml += \
    """
        </description>
        <Point>
            <coordinates>
"""
    kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(
                    np.degrees(traj_samples.lon[0]), np.degrees(traj_samples.lat[0]), traj_samples.ht[0])
    kml += \
"""
            </coordinates>
            <altitudeMode>absolute</altitudeMode>
        </Point> 
    </Placemark>
"""

    # Begin point on the ground
    kml += \
"""
    <Placemark> 
        <name>Begin - ground</name> 
        <description>"""
    kml += \
"""
            Lat = {:.6f} deg
            Lon = {:.6f} deg
            Ht  = {:.3f} km""".format(np.degrees(traj.rbeg_lat), np.degrees(traj.rbeg_lon), traj.rbeg_ele/1000)
    kml += \
    """
        </description>
        <Point>
            <coordinates>
"""
    kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(
                    np.degrees(traj_samples.lon[0]), np.degrees(traj_samples.lat[0]), 0)
    kml += \
"""
            </coordinates>
            <altitudeMode>clampToGround</altitudeMode>
        </Point> 
    </Placemark>
"""

    # End point in 3D
    kml += \
"""
    <Placemark> 
        <name>End</name> 
        <description>"""
    kml += \
"""
            Lat = {:.6f} deg
            Lon = {:.6f} deg
            Ht  = {:.3f} km""".format(np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.rend_ele/1000)
    kml += \
    """
        </description>
        <Point>
            <coordinates>
"""
    kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(
                    np.degrees(traj_samples.lon[-1]), np.degrees(traj_samples.lat[-1]), traj_samples.ht[-1])
    kml += \
"""
            </coordinates>
            <altitudeMode>absolute</altitudeMode>
        </Point> 
    </Placemark>
"""

    # End point on the ground
    kml += \
"""
    <Placemark> 
        <name>End - ground</name> 
        <description>"""
    kml += \
"""
            Lat = {:.6f} deg
            Lon = {:.6f} deg
            Ht  = {:.3f} km""".format(np.degrees(traj.rend_lat), np.degrees(traj.rend_lon), traj.rend_ele/1000)
    kml += \
    """
        </description>
        <Point>
            <coordinates>
"""
    kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(
                    np.degrees(traj_samples.lon[-1]), np.degrees(traj_samples.lat[-1]), 0)
    kml += \
"""
            </coordinates>
            <altitudeMode>clampToGround</altitudeMode>
        </Point> 
    </Placemark>
"""

    ###

    ###


    ### Add stations

    # Add a folder for the stations
    kml += \
"""

    <Folder><name>Stations</name><open>1</open>
"""

    for obs in traj.observations:


        # Add station pin
        kml += \
"""
    <Placemark> 
        <name>{:s}</name>""".format(obs.station_id)

        kml += \
        """
        <description>"""
        kml += \
"""
            Lat = {:.6f} deg
            Lon = {:.6f} deg
            Ht  = {:.0f} m""".format(np.degrees(obs.lat), np.degrees(obs.lon), obs.ele)
        kml += \
    """
        </description>
        <Point>
            <coordinates>
"""
        kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(np.degrees(obs.lon), np.degrees(obs.lat), 0)
        kml += \
"""
            </coordinates>
            <altitudeMode>clampToGround</altitudeMode>
        </Point> 
    </Placemark>
""" 

        # Add station observing extent
        kml += \
"""
    <Placemark id="{:s}">""".format(str(obs.station_id) + " observed extent")
        kml += \
"""
        <LineString>
            <coordinates>"""

        # First observed point
        kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(np.degrees(obs.model_lon[0]), np.degrees(obs.model_lat[0]), obs.model_ht[0])

        # Station coordinates
        kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(np.degrees(obs.lon), np.degrees(obs.lat), obs.ele)

        # Last observed point
        kml += \
"""
                {:.6f},{:.6f},{:.2f}""".format(np.degrees(obs.model_lon[-1]), np.degrees(obs.model_lat[-1]), obs.model_ht[-1])

        kml += \
"""
            </coordinates>
            <altitudeMode>absolute</altitudeMode>
        </LineString>
         
        <Style> 
            <LineStyle>  
                <color>#ff000000</color>
                <width>2</width>
            </LineStyle> 
        </Style>
    </Placemark>
"""

    # Close the stations folder
    kml += \
"""
    </Folder>
"""

    ###



    # Close the KML
    kml += \
"""


    </Folder>
    </kml> 
"""


    # Save the KML file to the directory with the platepar
    kml_path = os.path.join(dir_path, "{:s}_trajectory.kml".format(traj_name))
    with open(kml_path, 'w') as f:
        f.write(kml)

    print("KML saved to:", kml_path)


    return kml_path



    return None



def kmlColor(rgb, alpha=255):
    """ Convert a matplotlib RGB colour to a KML colour string.

    Arguments:
        rgb: [tuple] Red, green and blue between 0 and 1.

    Keyword arguments:
        alpha: [int] Opacity between 0 and 255. 255 by default.

    Return:
        [str] KML colour as 'aabbggrr'.
    """

    r, g, b = [int(round(c*255)) for c in rgb[:3]]

    return "{:02x}{:02x}{:02x}{:02x}".format(alpha, b, g, r)



def trajectoryToKML(traj, output_kml_path, measured=False, n_samples=500):
    """ Save a 3D KML with the trajectory of the drag fit, the vertical plane and the stations, and optionally
        the measured points.

    The trajectory is sampled from the drag fit (SampleTrajectoryPositions.sampleTrajectory()) evenly in time
    over the used picks, which gives smooth times and speeds and the path bent by gravity, also for an
    Earth-grazer. The measured layers add the projected point of every frame on the solved line, the line of
    sight to it and the residual. The vertical plane spans the ends of the drag fit. The measured
    folders are split by station, and the ignored stations and points are skipped. If the drag model cannot
    be fitted, the measured layers are saved instead.

    Arguments:
        traj: [Trajectory object] Solved trajectory.
        output_kml_path: [str] Path of the KML file.

    Keyword arguments:
        measured: [bool] Add the projected points, the lines of sight and the residuals of the measurements.
            False by default.
        n_samples: [int] Number of samples of the drag fit, at least 2. 500 by default.

    Return:
        output_kml_path: [str] Path of the KML file.
    """

    import simplekml
    import matplotlib.pyplot as plt

    colors = plt.get_cmap('tab10').colors

    # Sample the trajectory of the drag fit evenly in time over the used picks
    try:
        time_data, _ = usedPoints(traj)
        sample_step = (np.max(time_data) - np.min(time_data))/(max(n_samples, 2) - 1)
        traj_samples = sampleTrajectory(traj, np.min(time_data), np.max(time_data), sample_step, \
            by_time=True)
        drag_fit = traj_samples.drag_fit

    except ValueError as e:
        print("WARNING: The drag model could not be fitted ({:s}), so the KML gives the measured points " \
            "instead.".format(str(e)))
        traj_samples = None
        drag_fit = None
        measured = True

    # Skip a drag fit with too few samples for a line
    if (traj_samples is not None) and (len(traj_samples.t_est) < 2):
        print("WARNING: The drag fit gave fewer than 2 samples, so the KML gives the measured points " \
            "instead.")
        traj_samples = None
        measured = True

    # Make the folders
    kml = simplekml.Kml(name=traj.file_name)
    folder_names = ['Vertical Plane']
    if traj_samples is not None:
        folder_names.append('Drag Fit Trajectory')
    if measured:
        folder_names += ['Projected Points', 'Lines of Sight', 'Residuals']
    folder_names.append('Stations')
    folders = {name: kml.newfolder(name="{:d}. {:s}".format(i + 1, name)) for i, name \
        in enumerate(folder_names)}

    # Draw the trajectory of the drag fit as a line through its samples
    if traj_samples is not None:

        sample_coords = [(np.degrees(lon), np.degrees(lat), ht) for lat, lon, ht \
            in zip(traj_samples.lat, traj_samples.lon, traj_samples.ele_geo)]

        line = folders['Drag Fit Trajectory'].newlinestring(name="Drag fit", coords=sample_coords)
        line.altitudemode = simplekml.AltitudeMode.absolute
        line.style.linestyle.color = 'ff00ffff'
        line.style.linestyle.width = 3

        # Share one style between the samples
        sample_style = simplekml.Style()
        sample_style.iconstyle.color = 'ff00ffff'
        sample_style.iconstyle.scale = 0.3
        sample_style.iconstyle.icon.href = WEB_ICON + "id=304&scale=4"
        sample_style.labelstyle.scale = 0

        # Add the samples with their times and speeds, without labels
        folder_samples = folders['Drag Fit Trajectory'].newfolder(name="Samples")
        for t, coords, vel, vel_norot in zip(traj_samples.t_est, sample_coords, traj_samples.vel, \
            traj_samples.vel_norot):

            point = folder_samples.newpoint(coords=[coords])
            point.altitudemode = simplekml.AltitudeMode.absolute
            point.style = sample_style
            point.description = "t = {:.4f} s<br>Alt = {:.3f} km<br>v = {:.3f} km/s<br>" \
                "v ground = {:.3f} km/s<br>Lat {:.5f}, Lon {:.5f}".format(t, coords[2]/1000, vel/1000, \
                vel_norot/1000, coords[1], coords[0])

    for i, obs in enumerate(traj.observations):

        station_id = obs.station_id
        color = colors[i%len(colors)]
        color_kml = kmlColor(color)

        # Take the used points, none of an ignored station
        if obs.ignore_station:
            used = np.array([], dtype=int)
        else:
            used = np.where(np.array(obs.ignore_list) == 0)[0]

        # Add the station on the ground
        stat = folders['Stations'].newpoint(name=station_id, coords=[(np.degrees(obs.lon), \
            np.degrees(obs.lat), obs.ele)])
        stat.altitudemode = simplekml.AltitudeMode.clamptoground
        stat.style.iconstyle.icon.href = WEB_ICON + "color={:02x}{:02x}{:02x}&id=2000&scale=4".format( \
            *[int(round(c*255)) for c in color[:3]])
        stat.description = "Lat {:.5f}, Lon {:.5f}, Elev {:.0f} m, {:d} frames".format(np.degrees(obs.lat), \
            np.degrees(obs.lon), obs.ele, len(used))

        if (not measured) or (len(used) == 0):
            continue

        folder_points_stat = folders['Projected Points'].newfolder(name=station_id)
        folder_rays_stat = folders['Lines of Sight'].newfolder(name=station_id)
        folder_res_stat = folders['Residuals'].newfolder(name=station_id)

        # Share one style per layer of the station
        point_style = simplekml.Style()
        point_style.iconstyle.color = color_kml
        point_style.iconstyle.scale = 0.4
        point_style.iconstyle.icon.href = WEB_ICON + "id=304&scale=4"
        point_style.labelstyle.scale = 0

        ray_style = simplekml.Style()
        ray_style.linestyle.color = kmlColor(color, 130)
        ray_style.linestyle.width = 2

        res_style = simplekml.Style()
        res_style.linestyle.color = color_kml
        res_style.linestyle.width = 2

        # Compute the speeds of the drag fit at the lengths it was fitted to
        vel_drag = np.full(len(obs.time_data), np.nan)
        if drag_fit is not None:
            vel_drag[used] = drag_fit.velocityAtLength(np.array(obs.state_vect_dist)[used])

        for k in used:

            # Compute the distance between the end of the line of sight and the projected point
            model_eci = np.array(geo2Cartesian(obs.model_lat[k], obs.model_lon[k], obs.model_ht[k], \
                traj.jdt_ref))
            meas_eci = np.array(geo2Cartesian(obs.meas_lat[k], obs.meas_lon[k], obs.meas_ht[k], traj.jdt_ref))
            res = np.linalg.norm(model_eci - meas_eci)

            model_coords = (np.degrees(obs.model_lon[k]), np.degrees(obs.model_lat[k]), obs.model_ht[k])
            meas_coords = (np.degrees(obs.meas_lon[k]), np.degrees(obs.meas_lat[k]), obs.meas_ht[k])

            # Give the speed of the drag fit, and the point-to-point speed except for the first point,
            #   where the solver sets it to 0
            vel_str = "N/A"
            if np.isfinite(vel_drag[k]):
                vel_str = "{:.3f} km/s (drag fit)".format(vel_drag[k]/1000)

            if (obs.velocities is not None) and (k > 0):
                vel_str += "<br>v point = {:.3f} km/s (noisy)".format(obs.velocities[k]/1000)

            # Add the projected point on the trajectory
            point = folder_points_stat.newpoint(coords=[model_coords])
            point.altitudemode = simplekml.AltitudeMode.absolute
            point.style = point_style
            point.description = "Station {:s}, frame {:d}<br>t = {:.4f} s<br>Alt = {:.3f} km<br>" \
                "Distance = {:.3f} km<br>v = {:s}<br>Lat {:.5f}, Lon {:.5f}".format(station_id, k, \
                obs.time_data[k], obs.model_ht[k]/1000, obs.length[k]/1000, vel_str, \
                np.degrees(obs.model_lat[k]), np.degrees(obs.model_lon[k]))

            # Add the line of sight from the station to its closest point to the trajectory
            ray = folder_rays_stat.newlinestring(name="{:s} #{:d}".format(station_id, k), \
                coords=[(np.degrees(obs.lon), np.degrees(obs.lat), obs.ele), meas_coords])
            ray.altitudemode = simplekml.AltitudeMode.absolute
            ray.style = ray_style
            ray.description = "t = {:.4f} s<br>Azim = {:.4f} deg<br>Elev = {:.4f} deg<br>" \
                "Discrepancy = {:.1f} m".format(obs.time_data[k], np.degrees(obs.azim_data[k]), \
                np.degrees(obs.elev_data[k]), res)

            # Draw the residual as a row of short segments, as KML has no dashed lines
            dash_pts = np.linspace(meas_coords, model_coords, 2*N_DASHES + 1)
            residual = folder_res_stat.newmultigeometry(name="Residual {:s} #{:d}: {:.1f} m".format( \
                station_id, k, res))
            for a in range(0, 2*N_DASHES, 2):
                residual.newlinestring(coords=dash_pts[a:a + 2].tolist(), \
                    altitudemode=simplekml.AltitudeMode.absolute)
            residual.style = res_style

    # Span the vertical plane between the ends of the drag fit, or the observed beginning and end without it
    if traj_samples is not None:
        plane_coords = [sample_coords[0], sample_coords[-1]]
    else:
        plane_coords = [(np.degrees(traj.rbeg_lon), np.degrees(traj.rbeg_lat), traj.rbeg_ele), \
            (np.degrees(traj.rend_lon), np.degrees(traj.rend_lat), traj.rend_ele)]

    # Add the vertical plane as an extruded line, as a polygon hides what is behind it
    wall = folders['Vertical Plane'].newlinestring(name="Vertical plane", coords=plane_coords)
    wall.altitudemode = simplekml.AltitudeMode.absolute
    wall.extrude = 1
    wall.tessellate = 1
    wall.style.linestyle.color = 'ff00aaff'
    wall.style.linestyle.width = 1.5
    wall.style.polystyle.color = simplekml.Color.changealphaint(127, 'ff00aaff')

    kml.save(output_kml_path)

    # Warn if the KML may be too large for Google Earth Web
    n_frames = sum(int(np.sum(np.array(obs.ignore_list) == 0)) for obs in traj.observations)
    if measured and (n_frames > MAX_FRAMES_WEB):
        print("WARNING: The KML has {:d} frames, so it may not load in Google Earth Web, which is limited " \
            "to 10000 map elements. It should open in Google Earth Pro.".format(n_frames))

    return output_kml_path



if __name__ == "__main__":

    import os
    import argparse

    import matplotlib.pyplot as plt

    from wmpl.Utils.Pickling import loadPickle



    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Generate a trajectory KML file showing the trajectory in 3D and the stations.")

    arg_parser.add_argument('traj_path', nargs="?", metavar='TRAJ_PATH', type=str, \
        help="Path to the trajectory pickle file.")

    arg_parser.add_argument('-k', '--kml', action='store_true', \
        help="Save the 3D KML (*_3D.kml) with the trajectory of the drag fit instead of the simple one.")

    arg_parser.add_argument('-m', '--measured', action='store_true', \
        help="Save the 3D KML with the projected points, lines of sight and residuals of the measurements " \
        "as well.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # If the trajectory pickle was given, load the orbital elements from it
    if cml_args.traj_path is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path))

        # Generate the 3D KML next to the pickle, with the name the solver gives it
        if cml_args.kml or cml_args.measured:

            if cml_args.traj_path.endswith('_trajectory.pickle'):
                kml_path = cml_args.traj_path[:-len('_trajectory.pickle')] + '_3D.kml'
            else:
                kml_path = os.path.splitext(cml_args.traj_path)[0] + '_3D.kml'

            trajectoryToKML(traj, kml_path, measured=cml_args.measured)

        else:
            generateTrajectoryKML(traj, os.path.dirname(cml_args.traj_path))


