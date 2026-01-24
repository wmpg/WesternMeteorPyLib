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
from configparser import ConfigParser

from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Pickling import loadPickle



log = logging.getLogger("traj_correlator")


class RemoteDataHandler():
    def __init__(self, cfg_file):
        self.initialised = False
        if not os.path.isfile(cfg_file):
            log.warning(f'unable to find {cfg_file}, aborting remote processing')
            return 
        
        cfg = ConfigParser()
        cfg.read(cfg_file)
        self.mode = cfg['mode']['mode']
        if self.mode not in ['master', 'child']:
            log.warning('remote cfg: mode must be master or child, aborting remote processing')
            return 
        if self.mode == 'master':
            if 'children' not in cfg.sections() or 'capacity' not in cfg.sections():
                log.warning('remote cfg: capacity or children sections missing, aborting remote processing')
                return 
            
            self.nodes = [k for k in cfg['children'].values()]
            self.capacity = [int(k) for k in cfg['capacity'].values()]
            if len(self.nodes) != len(self.capacity):
                log.warning('remote cfg: capacity and children not same length, aborting remote processing')
                return
        else:
            if 'key' not in cfg['sftp'] or 'host' not in cfg['sftp'] or 'user' not in cfg['sftp']:
                log.warning('remote cfg: child user, key or host missing, aborting remote processing')
                return
            
            self.remotehost = cfg['sftp']['host']
            self.user = cfg['sftp']['user']
            self.key = os.path.normpath(os.path.expanduser(cfg['sftp']['key']))
            if 'port' not in cfg['sftp']:
                self.port = 22
            else: 
                self.port = int(cfg['sftp']['port'])

        self.initialised = True
        self.ssh_client = None
        self.sftp_client = None
        return 
    
    def getSFTPConnection(self):
        if not self.initialised:
            return False
        log.info(f'Connecting to {self.host}:{self.port} as {self.user}....')

        if not os.path.isfile(os.path.expanduser(self.key)):
            log.warning(f'ssh keyfile {self.key} missing')
            return False
        
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pkey = paramiko.RSAKey.from_private_key_file(self.key) 
        try:
            self.ssh_client.connect(hostname=self.host, username=self.user, port=self.port, pkey=pkey, look_for_keys=False)
            self.ftp_client = self.ssh_client.open_sftp()
            return True
        
        except Exception as e:

            log.warning('sftp connection to remote host failed')
            log.warning(e)
            self.ssh_client.close()
            return False
        
    def closeSFTPConnection(self):
        if self.sftp_client: 
            self.sftp_client.close()
        if self.ssh_client: 
            self.ssh_client.close()
        return

    def getRemoteCandidates(self):
        return 
    

    def collectRemotePhase1(self, max_trajs, output_dir):
        """
        Collect trajectory or candidate pickles from a remote server for local processing
        NB: do NOT use os.path.join here, as it will break on Windows
        """

        if not self.initialised or not self.getSFTPConnection():
            return 
        
        try:
            files = self.ftp_client.listdir('phase1')
            files = [f for f in files if '.pickle' in f and 'processing' not in f]
            files = files[:max_trajs]

            if len(files) == 0:
                log.info('no data available at this time')
                self.closeSFTPConnection()
                return
            
            for trajfile in files:
                fullname = os.path.join('phase1', trajfile).replace('\\','/')
                localname = os.path.join(output_dir, trajfile)
                self.ftp_client.get(fullname, localname)
                self.ftp_client.rename(fullname, f'{fullname}_processing')
            log.info(f'Obtained {len(files)} trajectories')


        except Exception as e:
            log.warning('Problem with download')
            log.info(e)

        self.closeSFTPConnection()
        return 


    def uploadToRemote(self, trajfile, output_dir, operation_mode=None):
        """
        upload the trajectory pickle and report to a remote host for integration
        into the solved dataset
        """

        if not self.initialised or not self.getSFTPConnection():
            return 

        remote_phase2_dir = ''
        try:
            self.sftp_client.mkdir(remote_phase2_dir)
        except Exception:
            pass

        localname = os.path.join(output_dir, trajfile)
        remotename = os.path.join(remote_phase2_dir, trajfile).replace('\\','/')
        self.ftp_client.put(localname, remotename)
        
        localname = localname.replace('_trajectory.pickle', '_report.txt')
        remotename = remotename.replace('_trajectory.pickle', '_report.txt')
        if os.path.isfile(localname):
            self.ftp_client.put(localname, remotename)

        self.closeSFTPConnection()
        return


def moveRemoteData(output_dir, datatype='traj'):
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




def putPhase1Trajectories():
    return 
