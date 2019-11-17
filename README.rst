Learning with OOD Audio
=======================

This is the source code for an audio classification system that
demonstrates learning with out-of-distribution (OOD) data, which is
defined as data that does not belong to any of the target classes. More
specifically, the training data is labelled, but some of the instances
are OOD (hence incorrectly labelled) due to annotation errors.

The system uses a pseudo-labeling method to relabel the instances that
are believed to be OOD yet also similar enough to the target classes
that pseudo-labeling is reasonable. This is achieved by first training
an auxilliary classifier on a subset of the dataset that has been
manually verified (i.e. known be in-distribution). The auxilliary
classifier is then used to detect and relabel the appropriate instances.


Requirements
------------

This software requires Python >=3.6. To install the dependencies, run::

    poetry install

or::

    pip install -r requirements.txt

Currenty, this software works with the FSDnoisy18k dataset, which may be
downloaded `here`__. Under the root directory of the dataset, the
following directories should be present::

    FSDnoisy18k.audio_test
    FSDnoisy18k.audio_train

When running the software, use the ``--dataset_path`` option (see later
sections for details on how to specify options) to specify the path of
the dataset directory.

__ https://zenodo.org/record/2529934#.Xc71bNHLdrk


Usage
-----

The general usage pattern is::

    python ood_audio/main.py [general-options...] <command> [command-options...]

The various options can also be specified in a configuration file. Using
the ``--config_file`` (or ``-f``) command-line option, the path of the
configuration file can be specified to the program. Options that are
passed in the command-line override those in the config file. See the
``default.conf`` file for an example of a config file. It also includes
descriptions of each option. Note that it is not intended to be
modified, as it specifies the default values for each option.

In the following subsections, the various commands are described. Using
this program, the user is able to extract feature vectors, train the
network, compute predictions, and evaluate the predictions.

Feature Extraction
^^^^^^^^^^^^^^^^^^

To extract feature vectors, run::

    python ood_audio/main.py extract <training/test> [--recompute] [--sample_rate RATE] [--n_fft N] [--hop_length N] [--n_mels N]

This extracts log-mel feature vectors and stores them in a HDF5 file.

Training
^^^^^^^^

To train a model, run::

    python ood_audio/main.py train [--training_id ID] [--model MODEL] [--mask MASK] [--seed N] [--block_size N] [--batch_size N] [--n_epochs N] [--lr NUM] [--lr_decay NUM] [--lr_decay_rate N] [--relabel] [--relabel_threshold NUM] [--relabel_weight NUM] [--relabel_weight NUM] [--augmentation]

The ``--model`` option accepts the following values:

* ``vgg`` - Use the randomly-initialized VGG9 model.
* ``densenet`` - Use the pre-trained DenseNet model.

The ``--training_id`` option is used to differentiate training runs, and
partially determines where the models are saved. When running multiple
trials, use the ``--seed`` option to specify different random seeds.
Otherwise. the outputs of the trials will be identical.

Use the ``--relabel`` option to enable the pseudo-labeling algorithm.
The path to the CSV file containing the pseudo-labels (generated using
the *predict* sub-command described below) is specified using the
``--pseudolabel_path`` option.

Other notable options are ``--augmentation``, which enables data
augmentation, and the ``--mask`` option, which allows training with a
subset of the training set. For example, to train with the
manually-verified subset of the training set only, run::

    python ood_audio/main.py train --mask manually_verified=1 [options...]

Prediction
^^^^^^^^^^

To compute predictions, run::

    python ood_audio/main.py predict <training/test> [--training_id ID] [--block_size N] [--epochs EPOCHS] [--odin]

The ``--odin`` option enables the ODIN algorithm.

Evaluation
^^^^^^^^^^

To evaluate the predictions, run::

    python task2/main.py evaluate [--training_id ID [ID ...]]

The ``--training_id`` option can be passed more than once, which allows
evaluating the performance over multiple trials.
