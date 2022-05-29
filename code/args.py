import argparse
import pytorch_lightning as pl


def parse_global_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-R', '--root', type=str, default='../data')
    parser.add_argument('-M', '--model_name', type=str, default='DH_GEM')
    parser.add_argument('-D', '--dataset', type=str, choices=['it', 'fin', 'cons'], default='it')
    parser.add_argument('-E', '--epochs', type=int, default=1024, help='epoch size')
    parser.add_argument('-TrainBS', '--train_batch_size', type=int, default=128, help='train batch size.')
    parser.add_argument('-ValBS', '--val_batch_size', type=int, default=128, help='validation batch size.')
    parser.add_argument('-TestBS', '--test_batch_size', type=int, default=128, help='test batch size.')
    parser.add_argument('-C', '--class_num', type=int, default=5, help='class number')
    parser.add_argument('--max_length', type=int, default=37, help='trend series max length')
    parser.add_argument('--min_length', type=int, default=12, help='trend series min length')
    parser.add_argument('-MNT', '--monitor', type=str, default='Total Weighted-F1', help='training monitor')
    global_args, _ = parser.parse_known_args()
    return parser, global_args


def parse_args(parent_parser: argparse.ArgumentParser, model: str, dataset: str):
    parser = parent_parser.add_argument_group(model)

    # ----------------------IT----------------------

    if dataset == 'it':

        if model == 'DH_GEM':
            # meta
            parser.add_argument('--meta_lr', type=int, default=1e-3, help='meta learning learning rate')
            parser.add_argument('--meta_step', type=float, default=4, help='for lr scheduler')
            parser.add_argument('--meta_gamma', type=float, default=0.9, help='reduce learning rate factor.')
            parser.add_argument('--meta_epochs', type=int, default=8, help='meta learning pre-train epoch size')
            parser.add_argument('--meta_sample_prop', type=float, default=0.5, help='sample size prop of total')
            # model
            parser.add_argument('--embed_dim', type=int, default=16, help='amplifier embedding dim')
            parser.add_argument('--com_pos_embed_dim', type=int, default=4, help='company and position embedding dim')
            parser.add_argument('--hidden_dim', type=int, default=4, help='decoder hidden dim')
            parser.add_argument('--nhead', type=int, default=4, help='transformer multi head attention number')
            parser.add_argument('--nhid', type=int, default=16, help='transformer encoder layer hidden dim')
            parser.add_argument('--nlayers', type=int, default=2, help='transformer layers number')
            # train
            parser.add_argument('-S', '--seed', type=int, default=4499, help='random seed')
            parser.add_argument('-LR', '--lr', type=float, default=0.01, help='initial learning rate.')
            parser.add_argument('-Gamma', '--gamma', type=float, default=0.9, help='reduce learning rate factor.')
            parser.add_argument('-Step', '--step', type=float, default=4, help='for lr scheduler')
            parser.add_argument('-Dropout', '--dropout', type=float, default=0.1, help='dropout probability')
            parser.add_argument('-WD', '--weight_decay', type=float, default=1e-6, help='for optimizer')
            parser.add_argument('-P', '--patience', type=int, default=16, help='early stopping patience')

    # ----------------------FIN----------------------

    elif dataset == 'fin':

        if model == 'DH_GEM':
            # meta
            parser.add_argument('--meta_lr', type=int, default=1e-3, help='meta learning learning rate')
            parser.add_argument('--meta_step', type=float, default=4, help='for lr scheduler')
            parser.add_argument('--meta_gamma', type=float, default=0.9, help='reduce learning rate factor.')
            parser.add_argument('--meta_epochs', type=int, default=8, help='meta learning pre-train epoch size')
            parser.add_argument('--meta_sample_prop', type=float, default=0.5, help='sample size prop of total')
            # model
            parser.add_argument('--embed_dim', type=int, default=16, help='amplifier embedding dim')
            parser.add_argument('--com_pos_embed_dim', type=int, default=4, help='company and position embedding dim')
            parser.add_argument('--hidden_dim', type=int, default=4, help='decoder hidden dim')
            parser.add_argument('--nhead', type=int, default=4, help='transformer multi head attention number')
            parser.add_argument('--nhid', type=int, default=16, help='transformer encoder layer hidden dim')
            parser.add_argument('--nlayers', type=int, default=2, help='transformer layers number')
            # train
            parser.add_argument('-S', '--seed', type=int, default=4653, help='random seed')
            parser.add_argument('-LR', '--lr', type=float, default=1e-2, help='initial learning rate.')
            parser.add_argument('-Gamma', '--gamma', type=float, default=0.9, help='reduce learning rate factor.')
            parser.add_argument('-Step', '--step', type=float, default=4, help='for lr scheduler')
            parser.add_argument('-Dropout', '--dropout', type=float, default=0.1, help='dropout probability')
            parser.add_argument('-WD', '--weight_decay', type=float, default=1e-6, help='for optimizer')
            parser.add_argument('-P', '--patience', type=int, default=32, help='early stopping patience')

    # ----------------------CONS----------------------

    elif dataset == 'cons':

        if model == 'DH_GEM':
            # meta
            parser.add_argument('--meta_lr', type=int, default=1e-3, help='meta learning learning rate')
            parser.add_argument('--meta_step', type=float, default=16, help='for lr scheduler')
            parser.add_argument('--meta_gamma', type=float, default=0.9, help='reduce learning rate factor.')
            parser.add_argument('--meta_epochs', type=int, default=8, help='meta learning pre-train epoch size')
            parser.add_argument('--meta_sample_prop', type=float, default=0.5, help='sample size prop of total')
            # model
            parser.add_argument('--embed_dim', type=int, default=16, help='amplifier embedding dim')
            parser.add_argument('--com_pos_embed_dim', type=int, default=4, help='company and position embedding dim')
            parser.add_argument('--hidden_dim', type=int, default=4, help='decoder hidden dim')
            parser.add_argument('--nhead', type=int, default=4, help='transformer multi head attention number')
            parser.add_argument('--nhid', type=int, default=16, help='transformer encoder layer hidden dim')
            parser.add_argument('--nlayers', type=int, default=2, help='transformer layers number')
            # train
            parser.add_argument('-S', '--seed', type=int, default=5645, help='random seed')
            parser.add_argument('-LR', '--lr', type=float, default=0.01, help='initial learning rate.')
            parser.add_argument('-Gamma', '--gamma', type=float, default=0.9, help='reduce learning rate factor.')
            parser.add_argument('-Step', '--step', type=float, default=4, help='for lr scheduler')
            parser.add_argument('-Dropout', '--dropout', type=float, default=0.1, help='dropout probability')
            parser.add_argument('-WD', '--weight_decay', type=float, default=1e-6, help='for optimizer')
            parser.add_argument('-P', '--patience', type=int, default=32, help='early stopping patience')

    return parent_parser.parse_args()
