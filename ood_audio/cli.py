import argparse
import configparser


def parse_args():
    """Parse the command-line arguments.

    This function implements the command-line interface of the program.
    The interface accepts general command-line arguments as well as
    arguments that are specific to a sub-command. The sub-commands are
    *extract*, *train*, *predict*, and *evaluate*. Specifying a
    sub-command is required, as it specifies the task that the program
    should carry out.

    A notable feature is that the user can specify a config file using
    the ``--config_file`` option. The config file can be used to specify
    the same options that the command-line interface accepts. Whether
    the user specifies a config file or not, the program will also read
    from a default config file as a fallback. If an option is specified
    by multiple sources, the source with the highest precedence is
    given preference. The command-line has the highest precedence.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    # Parse the command-line arguments, but separate the `--config_file`
    # option from the other arguments. This way, options can be parsed
    # from the config file(s) first and then overidden by the other
    # command-line arguments later.
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('-f', '--config_file', metavar='FILE')
    args, remaining_args = conf_parser.parse_known_args()

    # Parse the config file(s). The default config file is a fallback
    # for options that are not specified by the user.
    config = configparser.ConfigParser()
    try:
        config.read_file(open('default.conf'))
        if args.config_file:
            config.read_file(open(args.config_file))
    except FileNotFoundError:
        raise FileNotFoundError(f'Config file not found: {args.config_file}')

    # Initialize the main parser and the sub-command parsers
    parser = argparse.ArgumentParser(parents=[conf_parser])
    subparsers = parser.add_subparsers(dest='command')
    parser_extract = subparsers.add_parser('extract')
    parser_train = subparsers.add_parser('train')
    parser_predict = subparsers.add_parser('predict')
    parser_evaluate = subparsers.add_parser('evaluate')

    # Require the user to specify a sub-command
    subparsers.required = True

    # Extract options from the config file(s)
    args_default = dict(config.items('Default'))
    args_extraction = dict(config.items('Extraction'))
    args_logmel = dict(config.items('Extraction.Logmel'))
    args_training = dict(config.items('Training'))
    args_relabel = dict(config.items('Training.Relabel'))
    args_prediction = dict(config.items('Prediction'))
    training_id = args_training['training_id']
    block_size = args_training['block_size']

    # Set them as defaults for the relevant parsers
    parser.set_defaults(**args_default)
    parser_extract.set_defaults(**args_extraction, **args_logmel)
    parser_train.set_defaults(**args_training, **args_relabel)
    parser_predict.set_defaults(**args_prediction,
                                training_id=training_id,
                                block_size=block_size)
    parser_evaluate.set_defaults(training_id=training_id)

    # Specify the general command-line arguments
    parser.add_argument('--work_path', metavar='PATH')
    parser.add_argument('--dataset_path', metavar='PATH')
    parser.add_argument('--extraction_path', metavar='PATH')
    parser.add_argument('--model_path', metavar='PATH')
    parser.add_argument('--log_path', metavar='PATH')
    parser.add_argument('--prediction_path', metavar='PATH')
    parser.add_argument('--result_path', metavar='PATH')

    # Specify the command-line arguments of the 'extract' sub-command
    parser_extract.add_argument('dataset', choices=['training', 'test'])
    parser_extract.add_argument('--recompute', type=_bool, metavar='BOOL')
    parser_extract.add_argument('--sample_rate', type=int, metavar='RATE')
    parser_extract.add_argument('--n_fft', type=int, metavar='N')
    parser_extract.add_argument('--hop_length', type=int, metavar='N')
    parser_extract.add_argument('--n_mels', type=int, metavar='N')

    # Specify the command-line arguments of the 'train' sub-command
    parser_train.add_argument('--training_id', metavar='ID')
    parser_train.add_argument('--model', metavar='MODEL',
                              choices=['vgg', 'densenet'])
    parser_train.add_argument('--mask', type=_mask)
    parser_train.add_argument('--seed', type=int, metavar='N')
    parser_train.add_argument('--block_size', type=int, metavar='N')
    parser_train.add_argument('--batch_size', type=int, metavar='N')
    parser_train.add_argument('--n_epochs', type=int, metavar='N')
    parser_train.add_argument('--lr', type=float, metavar='NUM')
    parser_train.add_argument('--lr_decay', type=float, metavar='NUM')
    parser_train.add_argument('--lr_decay_rate', type=int, metavar='N')
    parser_train.add_argument('--relabel', type=_bool, metavar='BOOL')
    parser_train.add_argument('--relabel_threshold', type=float, metavar='NUM')
    parser_train.add_argument('--relabel_weight', type=float, metavar='NUM')
    parser_train.add_argument('--pseudolabel_path', metavar='PATH')
    parser_train.add_argument('--augment', type=_bool, metavar='BOOL')
    parser_train.add_argument('--overwrite', type=_bool, metavar='BOOL')

    # Specify the command-line arguments of the 'predict' sub-command
    parser_predict.add_argument('dataset', choices=['training', 'test'])
    parser_predict.add_argument('--training_id', metavar='ID')
    parser_predict.add_argument('--block_size', type=int, metavar='N')
    parser_predict.add_argument('--epochs', type=_epochs, metavar='EPOCHS')
    parser_predict.add_argument('--odin', type=_bool, metavar='BOOL')
    parser_predict.add_argument('--clean', type=_bool, metavar='BOOL')

    # Specify the command-line arguments of the 'evaluate' sub-command
    parser_evaluate.add_argument('--training_id', metavar='ID', nargs='+')

    return parser.parse_args(remaining_args)


def _bool(arg):
    """Convert a string into a boolean."""
    if arg.lower() == 'true':
        return True
    if arg.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('boolean value expected')


def _mask(arg):
    """Convert the ``--mask`` argument.

    The string must be in the format ``key1=value1,key2=value2,...``,
    and is converted into a Python dict.
    """
    if not arg:
        return dict()
    return {k: int(v) for k, v in [spec.split('=') for spec in arg.split(',')]}


def _epochs(arg):
    """Convert the ``--epochs`` argument.

    The string is either a list of epoch numbers or a string of the form
    ``'metric:n_epochs'``. The former is converted into a list, while
    the latter is converted into a tuple.
    """
    split = arg.split(':')

    if len(split) == 1:
        return list(map(int, arg.split(',')))

    metric, n_epochs = split
    if metric in ['val_loss',
                  'val_acc',
                  'val_mAP',
                  ]:
        return metric, int(n_epochs)
    raise argparse.ArgumentTypeError(f"unrecognized metric: '{metric}'")
