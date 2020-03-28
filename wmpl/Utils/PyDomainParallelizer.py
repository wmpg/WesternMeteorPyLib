from __future__ import print_function

import multiprocessing





def parallelComputeGenerator(generator, workerFunc, resultsCheckFunc, req_num, n_proc=None, 
    results_check_kwagrs=None, max_runs=None):
    """ Given a generator which generates inputs for the workerFunc function, generate and process results 
        until req_num number of results satisfies the resultsCheckFunc function.

    Arguments:
        generator: [generator] Generator function which creates inputs for the workerFunc. It should
            return a list of arguments that will be fed into the workerFunc.
        workerFunc: [function] Worker function.
        resultsCheckFunc: [function] A function which takes a lists of results and returns only those which
            satisfy some criteria.
        req_num: [int] Number of good results required. A good results is the one that passes the 
            resultsCheckFunc check.

    Keyword arguments:
        n_proc: [int] Number of processes to use. None by default, in which case all available processors
            will be used.
        results_check_kwargs: [dict] Keyword arguments for resultsCheckFunc. None by default.
        max_runs: [int] Maximum number of runs. None by default, which will limit the runs to 10x req_num.

    Return:
        [list] A list of results.
    """


    # If the number of processes was not given, use all available CPUs
    if n_proc is None:
        n_proc = multiprocessing.cpu_count()


    if results_check_kwagrs is None:
        results_check_kwagrs = {}

    # Limit the maxlimum number or runs
    if max_runs is None:
        max_runs = 10*req_num


    # Init the pool
    with multiprocessing.Pool(processes=n_proc) as pool:

        results = []

        total_runs = 0

        # Generate an initial input list
        input_list = [next(generator) for i in range(req_num)]

        # Run the initial list
        results = pool.map(workerFunc, input_list)

        total_runs += len(input_list)
            
        # Only take good results
        results = resultsCheckFunc(results, **results_check_kwagrs)


        # If there are None, do not continue, as there is obviously a problem
        if len(results) == 0:
            print("No successful results after the initial run!")
            return results


        # Run the processing until a required number of good values is returned
        while len(results) < req_num:

            # Generate an input for processing
            input_list = [next(generator) for i in range(n_proc)]

            # Map the inputs
            results_temp = pool.map(workerFunc, input_list)

            total_runs += len(input_list)

            # Only take good results
            results += resultsCheckFunc(results_temp, **results_check_kwagrs)

            # Check if the number of runs exceeded the maximum
            if total_runs >= max_runs:
                print("Total runs exceeded! Stopping...")
                break

        # Make sure that there are no more results than needed
        if len(results) > req_num:
            results = results[:req_num]

        return results



def unpackDecorator(func):
    def dec(args):
        return func(*args)

    return dec


def domainParallelizer(domain, function, cores=None, kwarg_dict=None):
    """ Runs N (cores) functions as separate processes with parameters given in the domain list.

    Arguments:
        domain: [list] a list of separate data (arguments) to feed individual function calls
        function: [function object] function that will be called with one entry from the domain
    
    Keyword arguments:
        cores: [int] Number of CPU cores, or number of parallel processes to run simultaneously. None by 
            default, in which case all available cores will be used.
        kwarg_dict: [dictionary] a dictionary of keyword arguments to be passed to the function, None by default

    Return:
        results: [list] a list of function results

    """


    def _logResult(result):
        """ Save the result from the async multiprocessing to a results list. """
        
        results.append(result)


    if kwarg_dict is None:
        kwarg_dict = {}


    # If the number of cores was not given, use all available cores
    if cores is None:
        cores = multiprocessing.cpu_count()


    results = []    


    # Special case when running on only one core, run without multiprocessing
    if cores == 1:
        for args in domain:
            results.append(function(*args, **kwarg_dict))

    # Run real multiprocessing if more than one core
    elif cores > 1:

        # Generate a pool of workers
        pool = multiprocessing.Pool(cores)

        # Maximum number of jobs in waiting
        max_jobs = 10*cores

        # Give workers things to do
        count = 0
        for args in domain:

            # Give job to worker
            last_job = pool.apply_async(function, args, kwarg_dict, callback=_logResult)

            # Limit the amount of jobs in wawiting
            count += 1
            if count%cores == 0:
                if len(pool._cache) > max_jobs:
                    last_job.wait()

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

def mpWorker(inputs, wait_time):
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
    results = domainParallelizer(data, mpWorker, cores=(cpu_cores - 1))

    print('Results:', results)

