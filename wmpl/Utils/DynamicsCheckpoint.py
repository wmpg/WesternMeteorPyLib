""" Checkpoint of the clones of the Jupiter-family dynamics of wmpl.Utils.OrbitClassification.

The integrations take hours, so each clone (or batch of clones) is saved as soon as it ends, and a run
started again with the same parameters integrates only the rest. The file has the layout of the
checkpoint of wmpl.Rebound.REBOUND: consecutive pickle records, a header with the run parameters and then
one record at a time, each flushed to disk on its own. A record cut while it was being written is dropped
when the file is read back.

As any pickle, a checkpoint runs code when it is read, so read only your own.
"""

from __future__ import print_function, division, absolute_import

import hashlib
import json
import os
import pickle


CHECKPOINT_FORMAT = 'wmpl_jfc_dynamics'
CHECKPOINT_VERSION = 1


class DynamicsCheckpointMismatch(ValueError):
    """ The checkpoint was written by a run with different parameters, so it cannot be continued.

    Attributes:
        differences: [list] (parameter, value in the checkpoint, value of this run) tuples.
    """

    def __init__(self, path, differences):

        self.path = path
        self.differences = differences

        text = ", ".join("{:s} ({} in the checkpoint, {} now)".format(k, a, b) for k, a, b in differences)
        super(DynamicsCheckpointMismatch, self).__init__("The checkpoint {:s} belongs to a run with different "
            "parameters: {:s}. Give another file, or delete it to start again.".format(path, text))


def fingerprint(obj):
    """ SHA-1 of a structure of numbers, lists and strings, with the floats at full precision, to compare
        what is too long to report by value.

    Arguments:
        obj: [object] Anything json can write, with numpy values as floats.

    Return:
        [str] The hash.
    """

    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def readCheckpoint(path):
    """ Read a checkpoint file.

    Arguments:
        path: [str] Path to the file.

    Return:
        (header, records, valid_size):
            header: [dict] With the run parameters under 'signature', or None if the file is missing or
                is not a checkpoint of this kind.
            records: [dict] {key: data} of every complete record.
            valid_size: [int] Size in bytes of the readable part. Anything after it is a record cut while
                it was being written.
    """

    header = None
    records = {}
    valid_size = 0

    if not os.path.isfile(path):
        return header, records, valid_size

    with open(path, 'rb') as f:

        while True:

            # A record cut while it was written raises many kinds of errors while unpickling
            try:
                entry = pickle.load(f)
            except Exception:
                break

            if not isinstance(entry, dict):
                break

            if (header is None) and (entry.get('kind') == 'header') \
                    and (entry.get('format') == CHECKPOINT_FORMAT):
                header = entry

            elif (header is not None) and (entry.get('kind') == 'record'):
                records[entry['key']] = entry['data']

            else:
                break

            valid_size = f.tell()

    if header is None:
        return None, {}, 0

    return header, records, valid_size


def checkpointSeed(path):
    """ The seed of the clones a checkpoint was written with, so that a run that did not set one goes on
        with the same clones.

    Arguments:
        path: [str] Path to the file.

    Return:
        [int] The seed, or None if there is no checkpoint there.
    """

    header = readCheckpoint(path)[0]

    return None if header is None else header['signature'].get('seed')


class DynamicsCheckpoint(object):
    """ A checkpoint opened to continue a run: what it already holds, and where to add the rest.

    Arguments:
        path: [str] Path to the file. It is created if it does not exist. An existing file that is not a
            checkpoint raises a ValueError instead of being overwritten, as it may be anything, such as
            the trajectory pickle given by mistake. So does one written by a run with other parameters
            (DynamicsCheckpointMismatch).
        signature: [dict] The parameters of this run that change the clones or their integration.
    """

    def __init__(self, path, signature):

        self.path = path

        header, self.records, valid_size = readCheckpoint(path)

        if header is None:

            if os.path.isfile(path) and (os.path.getsize(path) > 0):
                raise ValueError("{:s} exists and is not a checkpoint of the dynamics. Give another file, or "
                    "delete it if it is one that was cut before its first record.".format(path))

            self._write({'kind': 'header', 'format': CHECKPOINT_FORMAT, 'version': CHECKPOINT_VERSION,
                'signature': signature}, mode='wb')

            return

        saved = header['signature']
        differences = [(key, saved.get(key), signature.get(key)) for key in sorted(set(saved) | set(signature))
            if saved.get(key) != signature.get(key)]

        if differences:
            raise DynamicsCheckpointMismatch(path, differences)

        # Cut off a record left incomplete by the interruption
        with open(path, 'r+b') as f:
            f.truncate(valid_size)

    def _write(self, entry, mode='ab'):

        # The file is opened for each record, so nothing is left open if the run is cut
        with open(self.path, mode) as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

    def add(self, key, data):
        """ Append the record of a clone or a batch, and keep it in records.

        Arguments:
            key: [str] Key of the record.
            data: [object] What to keep, picklable.
        """

        self._write({'kind': 'record', 'key': key, 'data': data})
        self.records[key] = data
