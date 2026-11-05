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
import platform
import glob

from configparser import ConfigParser


log = logging.getLogger("wmpl_logger")


def getKey(fname):
    key = None
    for pkey_class in (paramiko.RSAKey, paramiko.ECDSAKey, paramiko.Ed25519Key):
        try:
            key = pkey_class.from_private_key_file(fname)
            print(f'keytype was', pkey_class.__name__)
            break
        except Exception as e:
            pass
    return key

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
        if not cfg.has_option('mode', 'mode'):
            log.warning('remote cfg: [mode] section/key missing, not enabling remote processing')
            self.mode = 'none'
            return        
        self.mode = cfg['mode']['mode'].lower()
        self.mode = 'parent' if self.mode=='master' else self.mode
        if self.mode not in ['master', 'child', 'parent']:
            log.warning('remote cfg: mode must be parent or child, not enabling remote processing')
            return 
        if self.mode == 'master' or self.mode == 'parent':
            if 'children' not in cfg.sections():
                log.warning('remote cfg: children section missing, not enabling remote processing')
                return 
            
            # create a list of available nodes, disabling any that are malformed in the config file
            self.nodenames = [k for k in cfg['children'].keys()]
            self.nodes = [k.split(',') for k in cfg['children'].values()]
            self.nodes = [RemoteNode(nn,x[0],x[1],x[2]) for nn,x in zip(self.nodenames,self.nodes) if len(x)==3]
            for i in range(0, len(self.nodes)):
                # make sure the node's files are accessible
                sts = False
                try:
                    sts = os.path.isdir(os.path.join(self.nodes[i].dirpath, 'files'))
                except Exception:
                    # node is inaccessible, consider it unusable
                    self.nodes[i].capacity = 0
                if not sts:
                    self.nodes[i].capacity = 0
                    
            self.nodes.append(RemoteNode('localhost', None, -1, -1))
            self.nodes = [n for n in self.nodes if n.capacity!=0 and n.mode!=0]
            activenodes = [n.nodename for n in self.nodes]
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
        pkey = None
        for pkey_class in (paramiko.RSAKey, paramiko.ECDSAKey, paramiko.Ed25519Key):
            try:
                pkey = pkey_class.from_private_key_file(self.key)
                break
            except Exception as e:
                pass
        if pkey is None:
            log.warning(f'ssh keyfile {self.key} type unknown, cannot be used')
            return False
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
            self.closeSFTPConnection()
            return False
        
    def closeSFTPConnection(self):
        if self.sftp_client:
            self.sftp_client.close()
            self.sftp_client = None
        if self.ssh_client: 
            self.ssh_client.close()
            self.ssh_client = None
        return
    
    ########################################################    
    # functions used by the client nodes

    def putWithRetry(self, local_name, rem_name):
        """
        Upload a file to the parent node. 
        We use a temporary name because the remote node may be running several processes in parallel.

        arguments:
        local_name: [string] the file to upload
        rem_name:   [string] the target name for the file
        """

        if not os.path.isfile(local_name):
            return False
        temp_rem_name = f'{rem_name}.filepart'
        errmsg = ''
        for i in range(10): 
            try:
                # try to put the file with the temporary name, then rename it. 
                # If this fails, catch the exceptions and retry 
                self.sftp_client.put(local_name, temp_rem_name)

                # now try to rename the file to the required final name, returning immediately if we succeed
                # we have to remove any existing file with that name first. We don't mind if remove() fails 
                # because we'll catch any issues when rename()ing.
                try:
                    self.sftp_client.remove(rem_name)
                except Exception:
                    pass
                try:                    
                    self.sftp_client.rename(temp_rem_name, rem_name)
                    return True
                except Exception as e:
                    errmsg = e
            except Exception as e:
                errmsg = e
            time.sleep(1)
        log.warning(f'upload of {local_name} failed after 10 retries, will be re-attempted on the next loop')
        log.warning(f'reason: {errmsg}')
        return False

    def getWithRetry(self, rem_name, local_name):
        """
        Retrieve a file from the remote server, retrying up to ten times.
        The function does not remove the remote file. Thats done when uploading solution data.
        This ensures that if the client crashes then it can be restarted without loss. 

        Arguments:
        rem_name:   [string] remote filename to collect.
        local_name: [string] local name to save to.
        """
        errmsg = ''
        for i in range(10): 
            try:
                self.sftp_client.get(rem_name, local_name)
                return True
            except Exception as e:
                errmsg = e
            time.sleep(1)
        log.warning(f'download of {rem_name} failed after 10 retries')
        log.warning(f'reason: {errmsg}')
        return False
    
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
                self.sftp_client.chmod(pth, 0o777)
            except Exception:
                pass
        
        try:
            download_list = []
            rem_dir = f'files/{datatype}'
            files = self.sftp_client.listdir(rem_dir)
            files = [f for f in files if '.pickle' in f]
            if len(files) == 0:
                log.info('no data available at this time')
                self.closeSFTPConnection()
                return False
            
            local_dir = os.path.join(output_dir, datatype)
            if not os.path.isdir(local_dir):
                os.makedirs(local_dir, exist_ok=True)
            num_received = 0
            for trajfile in files:
                fullname = f'{rem_dir}/{trajfile}'
                localname = os.path.join(local_dir, trajfile)
                if verbose:
                    log.info(f'downloading {fullname} to {localname}')

                if self.getWithRetry(fullname, localname):
                    num_received += 1
                    download_list.append(trajfile)

            if num_received > 0:
                open(os.path.join(output_dir, f'processed_{datatype}.txt'),'w').write('\n'.join(download_list))
        except Exception as e:
            log.warning('Problem with download')
            log.info(e)

        self.closeSFTPConnection()
        return True

    def uploadToRemote(self, source_dir, verbose=False):
        """
        upload the trajectory pickle and report to a remote host for integration
        into the solved dataset

        parameters:
        source_dir = root folder containing data, generally dh.output_dir
        """

        if not self.initialised or not self.getSFTPConnection(verbose=verbose):
            return 

        # flag to indicate success. Any upload failures will set this to False
        success_flag = True

        for pth in ['files', 'files/candidates', 'files/phase1', 'files/trajectories', 
                    'files/candidates/processed','files/phase1/processed']:
            try:
                self.sftp_client.mkdir(pth)
                self.sftp_client.chmod(pth, 0o777)
            except Exception:
                pass

        # push processed data to the remote 'processed' folder, and remove them from the local and remote source folder
        for datatype in ['candidates', 'phase1']:
            data_dir = os.path.join(source_dir, datatype)
            if os.path.isdir(data_dir):
                i = 0
                if not os.path.isfile(os.path.join(source_dir, f'processed_{datatype}.txt')):
                    continue
                fils = open(os.path.join(source_dir, f'processed_{datatype}.txt'),'r').readlines()
                for local_name in fils:
                    local_name = local_name.strip()
                    local_full_name = os.path.join(data_dir, 'processed', local_name)
                    rem_name = f'files/{datatype}/processed/{local_name}'
                    if self.putWithRetry(local_full_name, rem_name): 
                        try:
                            self.sftp_client.remove(f'files/{datatype}/{local_name}')
                        except Exception:
                            log.warning(f'unable to remove {local_name} from server, it will be processed again on the next pass')
                        if os.path.isfile(local_full_name):
                            try:
                                os.remove(local_full_name)
                            except Exception:
                                log.warning(f'unable to remove local {local_name}, it will be processed again on the next pass')
                        i += 1
                if i > 0:
                    log.info(f'uploaded {i} processed {datatype}')
                os.remove(os.path.join(source_dir, f'processed_{datatype}.txt'))

        # check the phase1 dir for data that was generated by phase1 local processing. Upload the files 
        # and then remove the local copy. 
        phase1_dir = os.path.join(source_dir, 'phase1')
        if os.path.isdir(phase1_dir):
            i=0
            fils = os.listdir(phase1_dir)
            for fil in [f for f in fils if '.pickle' in f]:
                local_ph1_name = os.path.join(phase1_dir, fil)
                if os.path.isdir(local_ph1_name):
                    continue
                rem_ph1_name = f'files/phase1/{fil}'
                if verbose:
                    log.info(f'uploading {local_name} to {rem_ph1_name}')
                # If the upload is successful, we can delete the local file
                if self.putWithRetry(local_ph1_name, rem_ph1_name): 
                    os.remove(local_ph1_name)
                    i += 1
                else:
                    success_flag = False

            if i > 0:
                log.info(f'uploaded {i} phase1 solutions')
            if not success_flag:
                log.info('some files didnt upload, will retry on next loop')

        # now upload any data in the 'trajectories' folder, flattening it to make it simpler to handle
        i=0
        
        traj_dir = os.path.join(source_dir, 'trajectories')
        if os.path.isdir(traj_dir):
            for (dirpath, dirnames, filenames) in os.walk(traj_dir):
                if len(filenames) > 0:

                    # flag to indicate whether this specific trajectory upload succeeded
                    traj_success_flag = True

                    rem_path = f'files/trajectories/{os.path.basename(dirpath)}'
                    try:
                        self.sftp_client.mkdir(rem_path)
                        self.sftp_client.chmod(rem_path, 0o777)
                    except Exception:
                        pass

                    # upload all files in the folder. If any upload fails, set the traj sucess flag to false
                    for fil in filenames:

                        local_name = os.path.join(dirpath, fil)
                        rem_file = f'{rem_path}/{fil}'

                        if verbose:
                            log.info(f'uploading {local_name} to {rem_file}')

                        if self.putWithRetry(local_name, rem_file):
                            if 'pickle' in local_name:
                                i += 1
                        else:
                            traj_success_flag = False

                    # if this trajectory uploaded, remove the local files
                    # Otherwise set the overall status to False
                    if traj_success_flag: 
                        shutil.rmtree(dirpath, ignore_errors=True)
                    else: 
                        log.info('some files didnt upload, will retry on next loop')
                        success_flag = traj_success_flag

            
            if i > 0:
                log.info(f'uploaded {i} trajectories')

            # if everything uploaded we can remove the entire 'trajectories' folder
            if success_flag: 
                shutil.rmtree(traj_dir, ignore_errors=True)

        # finally the databases - upload these with a random name for uniqueness at the server side
        # Again, if any upload fails mark the status False
        uuid_str = str(uuid.uuid4())

        db_success_flag = True
        for fname in ['observations', 'trajectories']:
            local_name = os.path.join(source_dir, f'{fname}.db')

            if os.path.isfile(local_name):
                rem_file = f'files/{fname}-{uuid_str}.db'

                if verbose:
                    log.info(f'uploading {local_name} to {rem_file}')

                if not self.putWithRetry(local_name, rem_file):
                    db_success_flag = False

        if db_success_flag:
            log.info('uploaded databases')
        else:
            log.warning('unable to upload at least one of the databases, will retry in next loop')
            success_flag = db_success_flag

        self.closeSFTPConnection()

        return success_flag
    
    def setStopFlag(self, verbose=False):
        if not self.initialised or not self.getSFTPConnection():
            return 
        try:
            readyfile = os.path.join(os.getenv('TMP', default='/tmp'),'stop')
            open(readyfile,'w').write('stop')
            self.sftp_client.put(readyfile, 'files/stop')                      
        except Exception:
            log.warning('unable to set stop flag, parent will continue to assign data')
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
