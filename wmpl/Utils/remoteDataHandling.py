# The MIT License

# Copyright (c) 2024 Mark McIntyre

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import paramiko
import logging
import glob
import shutil

from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Pickling import loadPickle


log = logging.getLogger("traj_correlator")


def collectRemoteTrajectories(remotehost, max_trajs, output_dir):
    """
    Collect trajectory pickles from a remote server for local phase2 (monte-carlo) processing
    NB: do NOT use os.path.join here, as it will break on Windows
    """
    ftpcli, remote_dir, sshcli = getSFTPConnection(remotehost) 
    if ftpcli is None:
        return 
    remote_phase1_dir = os.path.join(remote_dir, 'phase1').replace('\\','/')
    log.info(f'Looking in {remote_phase1_dir} on remote host for up to {max_trajs} trajectories')
    try:
        files = ftpcli.listdir(remote_phase1_dir)
        files = [f for f in files if '.pickle' in f and 'processing' not in f]
        files = files[:max_trajs]
        if len(files) == 0:
            log.info('no data available at this time')
            ftpcli.close()
            sshcli.close()
            return
        for trajfile in files:
            fullname = os.path.join(remote_phase1_dir, trajfile).replace('\\','/')
            localname = os.path.join(output_dir, trajfile)
            ftpcli.get(fullname, localname)
            ftpcli.rename(fullname, f'{fullname}_processing')
        log.info(f'Obtained {len(files)} trajectories')
    except Exception as e:
        log.warning('Problem with download')
        log.info(e)
    ftpcli.close()
    sshcli.close()
    return 


def uploadTrajToRemote(remotehost, trajfile, output_dir):
    """
    At the end of MC phase, upload the trajectory pickle and report to a remote host for integration
    into the solved dataset
    """
    ftpcli, remote_dir, sshcli = getSFTPConnection(remotehost) 
    if ftpcli is None:
        return 

    remote_phase2_dir = os.path.join(remote_dir, 'remoteuploads').replace('\\','/')
    try:
        ftpcli.mkdir(remote_phase2_dir)
    except Exception:
        pass

    localname = os.path.join(output_dir, trajfile)
    remotename = os.path.join(remote_phase2_dir, trajfile).replace('\\','/')
    ftpcli.put(localname, remotename)
    
    localname = localname.replace('_trajectory.pickle', '_report.txt')
    remotename = remotename.replace('_trajectory.pickle', '_report.txt')
    if os.path.isfile(localname):
        ftpcli.put(localname, remotename)

    ftpcli.close()
    sshcli.close()
    return


def moveRemoteTrajectories(output_dir):
    """
    Move remotely processed pickle files to their target location in the trajectories area,
    making sure we clean up any previously-calculated trajectory and temporary files
    """
    phase2_dir = os.path.join(output_dir, 'remoteuploads')
    if os.path.isdir(phase2_dir):
        log.info('Checking for remotely calculated trajectories...')
        pickles = glob.glob1(phase2_dir, '*.pickle')
        for pick in pickles:
            traj = loadPickle(phase2_dir, pick)
            phase1_name = traj.pre_mc_longname
            traj_dir = f'{output_dir}/trajectories/{phase1_name[:4]}/{phase1_name[:6]}/{phase1_name[:8]}/{phase1_name}'
            if os.path.isdir(traj_dir):
                shutil.rmtree(traj_dir)
            processed_traj_file = os.path.join(output_dir, 'phase1', phase1_name + '_trajectory.pickle_processing')
            if os.path.isfile(processed_traj_file):
                log.info(f'  Moving {phase1_name} to processed folder...')
                dst = os.path.join(output_dir, 'phase1', 'processed', phase1_name + '_trajectory.pickle')
                shutil.copyfile(processed_traj_file, dst)
                os.remove(processed_traj_file)

            phase2_name = traj.longname
            traj_dir = f'{output_dir}/trajectories/{phase2_name[:4]}/{phase2_name[:6]}/{phase2_name[:8]}/{phase2_name}'
            mkdirP(traj_dir)
            log.info(f'  Moving {phase2_name} to {traj_dir}...')
            src = os.path.join(phase2_dir, pick)
            dst = os.path.join(traj_dir, pick[:15]+'_trajectory.pickle')
            shutil.copyfile(src, dst)
            os.remove(src)
            report_file = src.replace('_trajectory.pickle','_report.txt')
            if os.path.isfile(report_file):
                dst = dst.replace('_trajectory.pickle','_report.txt')
                shutil.copyfile(report_file, dst)
                os.remove(report_file)
        log.info(f'Moved {len(pickles)} trajectories.')
    return


def getSFTPConnection(remotehost):
    hostdets = remotehost.split(':')
    if len(hostdets) < 2 or '@' not in hostdets[0]:
        log.warning(f'{remotehost} malformed, should be user@host:port:/path/to/dataroot')
        return None, None, None
    if len(hostdets) == 3:
        port = int(hostdets[1])
        remote_data_dir = hostdets[2]
    else:
        port = 22
        remote_data_dir = hostdets[1]
    user,host = hostdets[0].split('@')
    log.info(f'Connecting to {host}....')
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if not os.path.isfile(os.path.expanduser('~/.ssh/trajsolver')):
        log.warning('ssh keyfile ~/.ssh/trajsolver missing')
        ssh_client.close()
        return None, None, None
    pkey = paramiko.RSAKey.from_private_key_file(os.path.expanduser('~/.ssh/trajsolver')) 
    try:
        ssh_client.connect(hostname=host, username=user, port=port, pkey=pkey, look_for_keys=False)
        ftp_client = ssh_client.open_sftp()
        return ftp_client, remote_data_dir, ssh_client
    except Exception as e:
        log.warning('sftp connection to remote host failed')
        log.warning(e)
        ssh_client.close()
        return None, None, None
