Learning with OOD Audio
=======================

This is the source code for an audio classification system that
demonstrates learning with out-of-distribution (OOD) data, which is
defined as data that does not belong to any of the target classes. More
specifically, the training data is labeled, but some of the instances
are OOD (hence incorrectly labeled) due to annotation errors.

The system uses a pseudo-labeling method to relabel the instances that
are believed to be OOD yet also similar enough to the target classes
that pseudo-labeling is reasonable. This is achieved by first training
an auxiliary classifier on a subset of the dataset that has been
manually verified (i.e. known to be in-distribution). The auxiliary
classifier is then used to detect and relabel the appropriate instances.

For more details about the method, consider reading the `paper`__ [1]_.

__ https://arxiv.org/abs/2002.04683


.. contents::


Requirements
------------

This software requires Python >=3.6. To install the dependencies, run::

    poetry install

or::

    pip install -r requirements.txt

Currently, this software works with the FSDnoisy18k dataset, which may
be downloaded `here`__. For convenience, a bash script is provided to
download the dataset automatically. The dependencies are bash, curl, and
unzip. Simply run the following command from the root directory of the
project::

    $ scripts/download_dataset.sh

This will download the dataset to a directory called ``_dataset/``,
which should then contain the following directories::

    FSDnoisy18k.audio_test
    FSDnoisy18k.audio_train

When running the software, use the ``--dataset_path`` option (refer to
the `Usage`_ section) to specify the path of the dataset directory. This
is only necessary if the dataset path is different from the default:
``_dataset/``.

__ https://zenodo.org/record/2529934#.Xc71bNHLdrk


Quick Start
-----------

If you want to run the experiments that were presented in the paper
accompanying this code [1]_, there are several bash scripts available that
automate this. Assuming the FSDnoisy18k dataset has been downloaded in
``_dataset/``, run::

    $ scripts/icassp/extract.sh
    $ scripts/icassp/train.sh
    $ scripts/icassp/evaluate.sh

The last command evaluates the systems and prints the results.

Note that various files will be created in a directory called
``_workspace/``, which itself is created in the current working
directory. Ensure that enough hard disk space is available (at least 20
GiB). To change the path of the workspace directory, the easiest way is
to modify ``default.conf``. Alternatively, the configuration files in
``scripts/icassp/`` can also be modified. More details about configuring
the software can be found in the next section.


Usage
-----

The general usage pattern is::

    python ood_audio/main.py [general-options...] <command> [command-options...]

The various options can also be specified in a configuration file. Using
the ``--config_file`` (or ``-f``) command-line option, the path of the
configuration file can be specified to the program. Options that are
passed in the command-line override those in the config file. See the
``default.conf`` file for an example of a config file. It also includes
descriptions of each option. Note that this file is generally not
intended to be modified, with the exception being the paths.

In the following subsections, the various commands are described. Using
this program, the user is able to extract feature vectors, train the
network, compute predictions, and evaluate the predictions.

Feature Extraction
^^^^^^^^^^^^^^^^^^

To extract feature vectors, run::

    python ood_audio/main.py extract <training/test> [--recompute BOOL] [--sample_rate RATE] [--n_fft N] [--hop_length N] [--n_mels N]

This extracts log-mel feature vectors and stores them in a HDF5 file.

Training
^^^^^^^^

To train a model, run::

    python ood_audio/main.py train [--training_id ID] [--model MODEL] [--mask MASK] [--seed N] [--block_size N] [--batch_size N] [--n_epochs N] [--lr NUM] [--lr_decay NUM] [--lr_decay_rate N] [--relabel BOOL] [--relabel_threshold NUM] [--relabel_weight NUM] [--relabel_weight NUM] [--augment BOOL]

The ``--model`` option accepts the following values:

* ``vgg`` - Use the randomly-initialized VGG9 model.
* ``densenet`` - Use the pre-trained DenseNet model.

The ``--training_id`` option is used to differentiate training runs, and
partially determines where the models are saved. When running multiple
trials, use the ``--seed`` option to specify different random seeds.
Otherwise, the learned models will be identical across the different
trials.

Use the ``--relabel`` option to enable the pseudo-labeling algorithm.
The path to the CSV file containing the pseudo-labels (generated using
the *predict* sub-command described below) is specified using the
``--pseudolabel_path`` option.

Other notable options are ``--augment``, which enables data
augmentation, and the ``--mask`` option, which allows training with a
subset of the training set. For example, to train with the
manually-verified subset of the training set only, run::

    python ood_audio/main.py train --mask manually_verified=1 [options...]

Prediction
^^^^^^^^^^

To compute predictions, run::

    python ood_audio/main.py predict <training/test> [--training_id ID] [--block_size N] [--epochs EPOCHS] [--odin BOOL]

The ``--odin`` option enables the ODIN algorithm.

Evaluation
^^^^^^^^^^

To evaluate the predictions, run::

    python ood_audio/main.py evaluate [--training_id ID [ID ...]]

The ``--training_id`` option can be passed more than once, which allows
evaluating the performance over multiple trials.


Citing
------
If you wish to cite this work, please cite the following paper:

.. [1] \T. Iqbal, Y. Cao, Q. Kong, M. D. Plumbley, and W. Wang, "Learning
       with Out-of-Distribution Data for Audio Classification", arXiv
       preprint arXiv:2002.04683, 2020
