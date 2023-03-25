
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

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # If the trajectory pickle was given, load the orbital elements from it
    if cml_args.traj_path is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path))

        # Generate the KML
        generateTrajectoryKML(traj, os.path.dirname(cml_args.traj_path))


