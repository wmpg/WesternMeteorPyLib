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
import shutil
import uuid
import time

from configparser import ConfigParser


log = logging.getLogger("traj_correlator")


class RemoteNode():
    def __init__(self, nodename, dirpath, capacity, mode, active=False):
        self.nodename = nodename
        self.dirpath = dirpath
        self.capacity = int(capacity)
        self.mode = int(mode)
        self.active = active


class RemoteDataHandler():
    def __init__(self, cfg_file):
        self.initialised = False
        if not os.path.isfile(cfg_file):
            log.warning(f'unable to find {cfg_file}, not enabling remote processing')
            return 
        
        self.nodenames = None
        self.nodes = None
        self.capacity = None

        self.host = None
        self.user = None
        self.key = None

        self.ssh_client = None
        self.sftp_client = None
        
        cfg = ConfigParser()
        cfg.read(cfg_file)
        self.mode = cfg['mode']['mode'].lower()
        if self.mode not in ['master', 'child']:
            log.warning('remote cfg: mode must be master or child, not enabling remote processing')
            return 
        if self.mode == 'master':
            if 'children' not in cfg.sections():
                log.warning('remote cfg: children section missing, not enabling remote processing')
                return 
            
            # create a list of available nodes, disabling any that are malformed in the config file
            self.nodenames = [k for k in cfg['children'].keys()]
            self.nodes = [k.split(',') for k in cfg['children'].values()]
            self.nodes = [RemoteNode(nn,x[0],x[1],x[2]) for nn,x in zip(self.nodenames,self.nodes) if len(x)==3]
            self.nodes.append(RemoteNode('localhost', None, -1, -1))
            activenodes = [n.nodename for n in self.nodes if n.capacity!=0]
            log.info(f' using nodes {activenodes}')
        else:
            # 'child' mode
            if 'sftp' not in cfg.sections() or 'key' not in cfg['sftp'] or 'host' not in cfg['sftp'] or 'user' not in cfg['sftp']:
                log.warning('remote cfg: sftp user, key or host missing, not enabling remote processing')
                return
            
            self.host = cfg['sftp']['host']
            self.user = cfg['sftp']['user']
            self.key = os.path.normpath(os.path.expanduser(cfg['sftp']['key']))
            if 'port' not in cfg['sftp']:
                self.port = 22
            else: 
                self.port = int(cfg['sftp']['port'])

        self.initialised = True
        return 
    
    def getSFTPConnection(self, verbose=False):
        if not self.initialised:
            return False
        
        if self.sftp_client:
            return True
        
        log.info(f'Connecting to {self.host}:{self.port} as {self.user}....')

        if not os.path.isfile(os.path.expanduser(self.key)):
            log.warning(f'ssh keyfile {self.key} missing')
            return False
        
        self.ssh_client = paramiko.SSHClient()
        if verbose:
            log.info('created paramiko ssh client....')
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pkey = paramiko.RSAKey.from_private_key_file(self.key) 
        try:
            if verbose:
                log.info('connecting....')
            self.ssh_client.connect(hostname=self.host, username=self.user, port=self.port, 
                pkey=pkey, look_for_keys=False, timeout=10)
            if verbose:
                log.info('connected....')
            self.sftp_client = self.ssh_client.open_sftp()
            if verbose:
                log.info('created client')
            return True
        
        except Exception as e:

            log.warning('sftp connection to remote host failed')
            log.warning(e)
            self.ssh_client.close()
            return False
        
    def closeSFTPConnection(self):
        if self.sftp_client:
            self.sftp_client.close()
            self.sftp_client = None
        if self.ssh_client: 
            self.ssh_client.close()
            self.ssh_client = None
        return
    
    def putWithRetry(self, local_name, remname):
        for i in range(10):
            try:
                self.sftp_client.put(local_name, remname)
                break
            except Exception:
                time.sleep(1)
        if i == 10:
            log.warning(f'upload of {local_name} failed after 10 retries')
        return 

    ########################################################    
    # functions used by the client nodes

    def collectRemoteData(self, datatype, output_dir, verbose=False):
        """
        Collect trajectory or candidate pickles from a remote server for local processing

        parameters:
        datatype = 'candidates' or 'phase1'
        output_dir = folder to put the pickles into generally dh.output_dir
        """

        if not self.initialised or not self.getSFTPConnection(verbose=verbose):
            return False

        for pth in ['files', 'files/candidates', 'files/phase1', 'files/trajectories', 
                    'files/candidates/processed','files/phase1/processed']:
            try:
                self.sftp_client.mkdir(pth)
            except Exception:
                pass
        
        try:
            rem_dir = f'files/{datatype}'
            files = self.sftp_client.listdir(rem_dir)
            files = [f for f in files if '.pickle' in f and 'processing' not in f]
            if len(files) == 0:
                log.info('no data available at this time')
                self.closeSFTPConnection()
                return False
            
            local_dir = os.path.join(output_dir, datatype)
            if not os.path.isdir(local_dir):
                os.makedirs(local_dir, exist_ok=True)
            for trajfile in files:
                fullname = f'{rem_dir}/{trajfile}'
                localname = os.path.join(local_dir, trajfile)
                if verbose:
                    log.info(f'downloading {fullname} to {localname}')
                for i in range(10):
                    try:
                        self.sftp_client.get(fullname, localname)
                        break
                    except Exception: 
                        time.sleep(1)
                try:
                    self.sftp_client.rename(fullname, f'{rem_dir}/processed/{trajfile}')
                except:
                    try:
                        self.sftp_client.remove(fullname)
                    except:
                        pass

            log.info(f'Obtained {len(files)} {"trajectories" if datatype=="phase1" else "candidates"}')

        except Exception as e:
            log.warning('Problem with download')
            log.info(e)

        self.closeSFTPConnection()
        return True

    def uploadToMaster(self, source_dir, verbose=False):
        """
        upload the trajectory pickle and report to a remote host for integration
        into the solved dataset

        parameters:
        source_dir = root folder containing data, generally dh.output_dir
        """

        if not self.initialised or not self.getSFTPConnection(verbose=verbose):
            return 

        for pth in ['files', 'files/candidates', 'files/phase1', 'files/trajectories', 
                    'files/candidates/processed','files/phase1/processed']:
            try:
                self.sftp_client.mkdir(pth)
            except Exception:
                pass
        phase1_dir = os.path.join(source_dir, 'phase1')
        if os.path.isdir(phase1_dir):
            # upload any phase1 trajectories
            i=0
            proc_dir = os.path.join(phase1_dir, 'processed')
            os.makedirs(proc_dir, exist_ok=True)
            for fil in os.listdir(phase1_dir):
                local_name = os.path.join(phase1_dir, fil)
                if os.path.isdir(local_name):
                    continue
                remname = f'files/phase1/{fil}'
                if verbose:
                    log.info(f'uploading {local_name} to {remname}')
                self.putWithRetry(local_name, remname)
                if os.path.isfile(os.path.join(proc_dir, fil)):
                    os.remove(os.path.join(proc_dir, fil))
                shutil.move(local_name, proc_dir)
                i += 1
            if i > 0:
                log.info(f'uploaded {i} phase1 solutions')
        # now upload any data in the 'trajectories' folder, flattening it to make it simpler
        i=0
        if os.path.isdir(os.path.join(source_dir, 'trajectories')):
            traj_dir = f'{source_dir}/trajectories'
            for (dirpath, dirnames, filenames) in os.walk(traj_dir):
                if len(filenames) > 0:
                    rem_path = f'files/trajectories/{os.path.basename(dirpath)}'
                    try:
                        self.sftp_client.mkdir(rem_path)
                    except Exception:
                        pass
                    for fil in filenames:
                        local_name = os.path.join(dirpath, fil)
                        rem_file = f'{rem_path}/{fil}'
                        if verbose:
                            log.info(f'uploading {local_name} to {rem_file}')
                        self.putWithRetry(local_name, rem_file)
                        i += 1
            shutil.rmtree(traj_dir, ignore_errors=True)
        if i > 0:
            log.info(f'uploaded {int(i/2)} trajectories')

        # finally the databases
        uuid_str = str(uuid.uuid4())
        for fname in ['observations', 'trajectories']:
            local_name = os.path.join(source_dir, f'{fname}.db')
            if os.path.isfile(local_name):
                rem_file = f'files/{fname}-{uuid_str}.db'
                if verbose:
                    log.info(f'uploading {local_name} to {rem_file}')
                self.putWithRetry(local_name, rem_file)

        log.info('uploaded databases')
        self.closeSFTPConnection()
        return
    
    def setStopFlag(self, verbose=False):
        if not self.initialised or not self.getSFTPConnection():
            return 
        try:
            readyfile = os.path.join(os.getenv('TMP', default='/tmp'),'stop')
            open(readyfile,'w').write('stop')
            self.sftp_client.put(readyfile, 'files/stop')                      
        except Exception:
            log.warning('unable to set stop flag, master will not continue to assign data')
        time.sleep(2)
        self.closeSFTPConnection()
        log.info('set stop flag')
        return

    def clearStopFlag(self, verbose=False):
        if not self.initialised or not self.getSFTPConnection():
            return 
        try:
            self.sftp_client.remove('files/stop')
            log.info('removed stop flag')
        except:
            pass
        self.closeSFTPConnection()
        return
