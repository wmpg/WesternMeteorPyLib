
import numpy as np

from wmpl.Utils.TrajConversions import jd2Date
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory

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

def _kmlColor(rgb, alpha=255):
    """ Matplotlib RGB(A) (0-1) -> KML 'aabbggrr' string. """
    r, g, b = [int(round(c*255)) for c in rgb[:3]]
    return "{:02x}{:02x}{:02x}{:02x}".format(alpha, b, g, r)


# Icons that Google Earth Web renders with their colour
WEB_ICON = "https://earth.google.com/earth/document/icon?"

# Above this many frames the KML may not load in Google Earth Web (tested: ~1000 frames, 4.5 MB, loads;
#   ~2000 frames, 9 MB, fails; Google Earth Pro handled 7800 frames, 36 MB)
MAX_FRAMES_WEB = 1000

# Number of dashes in each residual line
N_DASHES = 6


def trajectoryToKML(traj, output_kml_path):
    """ Write a 3D KML with the projected points, raw lines of sight, residuals and stations.

    Arguments:
        traj: [Trajectory] Solved trajectory.
        output_kml_path: [str] Path of the KML file to write.
    """

    import simplekml
    import matplotlib.pyplot as plt

    from wmpl.Utils.TrajConversions import geo2Cartesian

    colors = plt.get_cmap('tab10').colors
    deg = np.degrees
    xyz = lambda lat, lon, h: np.array(geo2Cartesian(lat, lon, h, traj.jdt_ref))
    llh = lambda lat, lon, h: (deg(lon), deg(lat), h)

    kml = simplekml.Kml(name=traj.file_name)
    f_plane = kml.newfolder(name="1. Vertical Plane")
    f_pts, f_rays, f_res, f_stat = [kml.newfolder(name=n) for n in ("2. Projected Points",
        "3. Lines of Sight", "4. Residuals", "5. Stations")]

    for i, obs in enumerate(traj.observations):

        sid = obs.station_id
        col = _kmlColor(colors[i%len(colors)])
        col_ray = _kmlColor(colors[i%len(colors)], 130)
        fp = f_pts.newfolder(name=sid)
        fr = f_rays.newfolder(name=sid)
        fd = f_res.newfolder(name=sid)

        # Station placemark
        pt = f_stat.newpoint(name=sid, coords=[llh(obs.lat, obs.lon, obs.ele)])
        pt.altitudemode = simplekml.AltitudeMode.clamptoground
        pt.style.iconstyle.icon.href = WEB_ICON + "color={:02x}{:02x}{:02x}&id=2000&scale=4".format(
            *[int(round(c*255)) for c in colors[i%len(colors)][:3]])
        pt.description = "Lat {:.5f}, Lon {:.5f}, Elev {:.0f} m, {:d} frames".format(deg(obs.lat), 
            deg(obs.lon), obs.ele, int(np.sum(obs.ignore_list == 0)))

        for k in np.where(obs.ignore_list == 0)[0]:

            tp = (obs.model_lat[k], obs.model_lon[k], obs.model_ht[k])
            rp = (obs.meas_lat[k], obs.meas_lon[k], obs.meas_ht[k])
            res = np.linalg.norm(xyz(*tp) - xyz(*rp))
            vel = "{:.3f} km/s".format(obs.velocities[k]/1000) if obs.velocities is not None else "N/A"

            # Projected point on the trajectory
            p = fp.newpoint(coords=[llh(*tp)])
            p.altitudemode = simplekml.AltitudeMode.absolute
            p.style.iconstyle.color = col
            p.style.iconstyle.scale = 0.4
            p.style.iconstyle.icon.href = WEB_ICON + "id=304&scale=4"  # round marker, tinted by its colour
            p.style.labelstyle.scale = 0  # points are unnamed so Google Earth Web shows no label
            p.description = ("Station {}, frame {}<br>t = {:.4f} s<br>Alt = {:.3f} km<br>"
                "Distance = {:.3f} km<br>v = {}<br>Lat {:.5f}, Lon {:.5f}").format(sid, k, obs.time_data[k], 
                tp[2]/1000, obs.length[k]/1000, vel, deg(tp[0]), deg(tp[1]))

            # Raw line of sight, from the station to its closest point to the trajectory
            ray = fr.newlinestring(name="{} #{}".format(sid, k), 
                coords=[llh(obs.lat, obs.lon, obs.ele), llh(*rp)])
            ray.altitudemode = simplekml.AltitudeMode.absolute
            ray.style.linestyle.color = col_ray
            ray.style.linestyle.width = 2
            ray.description = "t = {:.4f} s<br>Azim = {:.4f} deg<br>Elev = {:.4f} deg<br>Discrepancy = {:.1f} m"\
                .format(obs.time_data[k], deg(obs.azim_data[k]), deg(obs.elev_data[k]), res)

            # Residual: ray end -> projected point
            # (KML has no dashed lines, so draw it as a row of short segments)
            pts = np.linspace(llh(*rp), llh(*tp), 2*N_DASHES + 1)
            r = fd.newmultigeometry(name="Residual {} #{}: {:.1f} m".format(sid, k, res))
            for a in range(0, 2*N_DASHES, 2):
                r.newlinestring(coords=pts[a:a + 2].tolist(), altitudemode=simplekml.AltitudeMode.absolute)
            r.style.linestyle.color = col
            r.style.linestyle.width = 2

    # Vertical plane
    # It spans the first and last observed points and their projection to 0 m
    b, e = (traj.rbeg_lat, traj.rbeg_lon, traj.rbeg_ele), (traj.rend_lat, traj.rend_lon, traj.rend_ele)
    # An extruded line (not a polygon) lets Google Earth show what is behind the curtain
    wall = f_plane.newlinestring(name="Vertical plane", coords=[llh(*b), llh(*e)])
    wall.altitudemode = simplekml.AltitudeMode.absolute
    wall.extrude = 1
    wall.tessellate = 1
    wall.style.linestyle.color = "ff00aaff"
    wall.style.linestyle.width = 1.5
    wall.style.polystyle.color = simplekml.Color.changealphaint(127, "ff00aaff")

    kml.save(output_kml_path)

    n_frames = sum(int(np.sum(obs.ignore_list == 0)) for obs in traj.observations)
    if n_frames > MAX_FRAMES_WEB:
        print("WARNING: the KML has {:d} frames (3 elements each); it may not load in Google Earth Web "
            "(limit of 10,000 map elements and file size), but should open in Google Earth Pro.".format(n_frames))

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

    arg_parser.add_argument('-k', '--kml', action='store_true', 
        help="Write the enriched 3D KML (*_3D.kml) instead of the simple one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # If the trajectory pickle was given, load the orbital elements from it
    if cml_args.traj_path is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path))

        # Generate the KML
        if cml_args.kml:
            trajectoryToKML(traj, cml_args.traj_path.replace('_trajectory.pickle', '') + '_3D.kml')
        else:
            generateTrajectoryKML(traj, os.path.dirname(cml_args.traj_path))


