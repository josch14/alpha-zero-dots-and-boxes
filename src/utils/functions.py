import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    When using multiple threads to perform a loop of the same calculation, tqdm is not able to display actual progress
    correctly when starmap is used. Therefore, implement the patch istarmap() based on the code for imap().
    see: https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put((
        self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length
    ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap



def print_parameters(config: dict):
    print("####################### Parameters #######################")
    print("game_size: " + str(config["game_size"]) + " | n_iterations: " + str(config["n_iterations"]))
    print("[MCTS] " + parameters_to_string(config["mcts_parameters"]))
    print("[Neural Network] " + parameters_to_string(config["model_parameters"]))
    print("[Optimizer] " + parameters_to_string(config["optimizer_parameters"]))
    print("[Data] " + parameters_to_string(config["data_parameters"]))
    print("[Evaluator] " + parameters_to_string(config["evaluator_parameters"]))
    print("##########################################################")


def parameters_to_string(model_parameters: dict):
    s = ""
    for key, value in model_parameters.items():
        s += (key + ": " + str(model_parameters[key]) + " | ")
    s = s[:-2]
    return s
