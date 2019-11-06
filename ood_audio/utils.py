import time


LABELS = [
    'Acoustic_guitar',
    'Bass_guitar',
    'Clapping',
    'Coin_(dropping)',
    'Crash_cymbal',
    'Dishes_and_pots_and_pans',
    'Engine',
    'Fart',
    'Fire',
    'Fireworks',
    'Glass',
    'Hi-hat',
    'Piano',
    'Rain',
    'Slam',
    'Squeak',
    'Tearing',
    'Walk_or_footsteps',
    'Wind',
    'Writing',
]


def statistics(x, axis=(0, 1, 2)):
    """Return the mean and standard deviation of a dataset.

    Args:
        x (np.ndarray): Dataset for which statistics are computed.
        axis (None or int or tuple of ints): Axis or axes along which
            the statistics are computed.

    Returns:
        tuple: A tuple containing the mean and standard deviation.
    """
    mean = x.mean(axis)
    std = x.std(axis)
    return mean, std


def standardize(data, mean, std):
    """Standardize datasets using the given statistics.

    Args:
        data (np.ndarray or list of np.ndarray): Dataset or list of
            datasets to standardize.
        mean (number): Mean statistic.
        std (number): Standard deviation statistic.

    Returns:
        np.ndarray or list of np.ndarray: The standardized dataset(s).
    """
    if isinstance(data, list):
        return [(x - mean) / std for x in data]
    return (data - mean) / std


def timeit(callback, message):
    """Measure the time taken to execute a function.

    This function measures the amount of time it takes to execute the
    specified callback and prints a message afterwards regarding the
    time taken. The `message` parameter provides part of the message,
    e.g. if `message` is 'Executed', the printed message is 'Executed in
    1.234567 seconds'.

    Args:
        callback: Function to execute and time.
        message (str): Message to print after executing the callback.

    Returns:
        The return value of the callback.
    """
    onset = time.time()
    x = callback()

    print('{} in {:.3f} seconds'.format(message, time.time() - onset))

    return x
