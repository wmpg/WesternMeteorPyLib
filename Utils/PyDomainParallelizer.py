from __future__ import print_function

import multiprocessing

def DomainParallelizer(domain, function, cores, kwarg_dict=None):
    """ Runs N (cores) functions as separate processes with parameters given in the domain list.

    Arguments:
        domain: [list] a list of separate data (arguments) to feed individual function calls
        function: [function object] function that will be called with one entry from the domain
        cores: [int] number of CPU cores, or number of parallel processes to run simultaneously
    
    Keyword arguments:
        kwarg_dict: [dictionary] a dictionary of keyword arguments to be passed to the function, None by default

    Return:
        results: [list] a list of function results

    """

    def _logResult(result):
        """ Save the result from the async multiprocessing to a results list. """
        
        results.append(result)


    if kwarg_dict is None:
        kwarg_dict = {}


    results = []    


    # Special case when running on only one core, run without multiprocessing
    if cores == 1:
        for args in domain:
            results.append(function(*args, **kwarg_dict))

    # Run real multiprocessing if more than one core
    elif cores > 1:

        # Generate a pool of workers
        pool = multiprocessing.Pool(cores)

        # Give workers things to do
        for args in domain:
            pool.apply_async(function, args, kwarg_dict, callback=_logResult)

        # Clean up
        pool.close()
        pool.join()

    else:
        print('The number of CPU cores defined is not in an expected range (1 or more.)')
        print('Use cpu_cores = 1 as a fallback value.')

    return results



##############################
## USAGE EXAMPLE


import time
import sys

def mp_worker(inputs, wait_time):
    """ Example worker function. This function will print out the name of the worker and wait 'wait_time
        seconds. 

    """

    print(" Processs %s\tWaiting %s seconds" % (inputs, wait_time))

    time.sleep(int(wait_time))

    print(" Process %s\tDONE" % inputs)

    # Must use if you want print to be visible on the screen!
    sys.stdout.flush()

    return int(wait_time)



if __name__ == '__main__':

    # List of function arguments for every run
    data = [
        ['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'],
        ['e', '1'], ['f', '3'], ['g', '5'], ['h', '7']
            ]

    # Get the number of cpu cores available
    cpu_cores = multiprocessing.cpu_count()

    # Run the parallelized function
    results = DomainParallelizer(data, mp_worker, cpu_cores)

    print('Results:', results)

