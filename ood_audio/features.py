import os.path
import datetime as dt

import h5py
import librosa
import numpy as np
from tqdm import tqdm


def extract_dataset(dataset_path,
                    file_names,
                    extractor,
                    output_path,
                    recompute=False,
                    ):
    """Extract feature vectors from an audio dataset.

    Feature vectors are extracted using the given `extractor` and then
    stored in a single HDF5 file. The extractor must be an instance of
    a class containing two functions: ``extractor.output_shape()`` and
    ``extractor.extract()``. Refer to :class:`.LogmelExtractor` for an
    example of such a class.

    Args:
        dataset_path (str): Path to directory containing dataset.
        file_names (array_like): List of file names for the audio clips.
        extractor: Instance of a feature extractor class.
        output_path (str): Path to the output HDF5 file.
        recompute (bool): Whether to extract features that already exist
            in the HDF5 file.
    """
    # Create/load HDF5 file to store feature vectors
    with h5py.File(output_path, 'a') as f:
        size = len(file_names)  # Size of dataset

        # Create/load feature vector dataset and timestamp dataset
        feats = f.require_dataset('F', (size,),
                                  dtype=h5py.special_dtype(vlen=float))
        timestamps = f.require_dataset('timestamps', (size,),
                                       dtype=h5py.special_dtype(vlen=bytes))

        # Record shape of reference feature vector. Used to infer the
        # original shape of a vector prior to flattening.
        feats.attrs['shape'] = extractor.output_shape(1)[1:]

        for i, name in enumerate(tqdm(file_names)):
            # Skip if existing feature vector should not be recomputed
            if timestamps[i] and not recompute:
                continue

            path = os.path.join(dataset_path, name)
            x, sample_rate = librosa.load(path, sr=None)

            # Extract and save to dataset as flattened array
            feats[i] = extractor.extract(x, sample_rate).flatten()
            # Record timestamp in ISO format
            timestamps[i] = dt.datetime.now().isoformat()


def load_features(path, block_size=128, r_threshold=16):
    """Load feature vectors from the specified HDF5 file.

    Since the original feature vectors are of variable length, this
    function partitions them into blocks of length `block_size`. If they
    cannot be partitioned exactly, one of three things occurs:

      * If the length of the vector is less than the block size, the
        vector is simply padded.
      * If the remainder, `r`, is less than `r_threshold`, the start and
        end of the vector are truncated so that it can be partitioned.
      * If the remainder, `r`, is greater than `r_threshold`, an
        additional block is included to align with the end of the
        feature vector. This means it will overlap with the prior block.

    Args:
        path (str): Path to the HDF5 file.
        block_size (int): Size of a block.
        r_threshold (int): Threshold for `r` (see above).

    Returns:
        tuple: Tuple of the form ``(x, n_blocks)``, where ``x`` is the
        array of feature vectors, and ``n_blocks`` is a list of the
        number of blocks for each audio clip.
    """
    blocks = []
    n_blocks = []
    with h5py.File(path, 'r') as f:
        feats = f['F']
        shape = feats.attrs['shape']
        for i, feat in enumerate(tqdm(feats)):
            # Reshape flat array to original shape
            feat = np.reshape(feat, (-1, *shape))

            # Split feature vector into blocks along time axis
            q = len(feat) // block_size
            r = len(feat) % block_size
            if not q and r:
                split = [_pad_truncate(feat, block_size,
                                       pad_value=np.min(feat))]
            elif r:
                off = r // 2 if r < r_threshold else 0
                split = np.split(feat[off:q * block_size + off], q)
                if r >= r_threshold:
                    split.append(feat[-block_size:])
            else:
                split = np.split(feat, q)

            n_blocks.append(len(split))
            blocks += split

    return np.array(blocks), n_blocks


def _pad_truncate(x, length, pad_value=0):
    """Pad or truncate an array to a specified length.

    Args:
        x (array_like): Input array.
        length (int): Target length.
        pad_value (number): Padding value.

    Returns:
        array_like: The array padded/truncated to the specified length.
    """
    x_len = len(x)
    if x_len > length:
        x = x[:length]
    elif x_len < length:
        padding = np.full((length - x_len,) + x.shape[1:], pad_value)
        x = np.concatenate((x, padding))

    return x


class LogmelExtractor(object):
    """Feature extractor for logmel representations.

    A logmel feature vector is a spectrogram representation that has
    been scaled using a Mel filterbank and a log non-linearity.

    Args:
        sample_rate (number): Target sample rate.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of audio samples between frames.
        n_mels (int): Number of Mel bands.

    Attributes:
        sample_rate (number): Target sample rate.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of audio samples between frames.
        mel_fb (np.ndarray): Mel fitlerbank matrix.
    """

    def __init__(self,
                 sample_rate=32000,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Create Mel filterbank matrix
        self.mel_fb = librosa.filters.mel(sr=sample_rate,
                                          n_fft=n_fft,
                                          n_mels=n_mels,
                                          )

    def output_shape(self, clip_duration):
        """Determine the shape of a logmel feature vector.

        Args:
            clip_duration (number): Duration of the input time-series
                signal given in seconds.

        Returns:
            tuple: The shape of a logmel feature vector.
        """
        n_samples = clip_duration * self.sample_rate
        n_frames = n_samples // self.hop_length + 1
        return (n_frames, self.mel_fb.shape[0])

    def extract(self, x, sample_rate):
        """Transform the given signal into a logmel feature vector.

        Args:
            x (np.ndarray): Input time-series signal.
            sample_rate (number): Sample rate of signal.

        Returns:
            np.ndarray: The logmel feature vector.
        """
        # Resample to target sample rate
        x = librosa.resample(x, sample_rate, self.sample_rate)

        # Compute mel-scaled spectrogram
        D = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.dot(self.mel_fb, np.abs(D) ** 2).T
        # Apply log non-linearity
        return librosa.power_to_db(S, ref=0., top_db=None)
