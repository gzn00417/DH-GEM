import os
import torch
import dgl
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datamodule import TalentTrendDataModule
from dataset import PostingDataset, WorkExperienceDataset, TalentTrendDataset, Taskset
from trainmodule import *
from model import *
from args import *


if __name__ == '__main__':

    # hyperparameters
    print('Setting parameters...')
    parser, global_args = parse_global_args()
    MODEL: torch.nn.Module = locals()[global_args.model_name]
    args = parse_args(parser, global_args.model_name, global_args.dataset)
    pl.seed_everything(args.seed)
    dgl.seed(args.seed)

    # dataset
    print('Loading data...')
    dataset = TalentTrendDataset(
        demand=PostingDataset(os.path.join(args.root, 'jd_%s.csv' % args.dataset)),
        supply=WorkExperienceDataset(os.path.join(args.root, 'we_%s.csv' % args.dataset)),
        **vars(args)
    )
    n_com = len(dataset.companies)
    n_pos = len(dataset.positions)

    # data module
    datamodule: pl.LightningDataModule = TalentTrendDataModule(dataset, **vars(args))

    # if meta learning
    if global_args.model_name in TrainMeta.model_names:
        # init tasks
        print('Preparing tasks...')
        taskset = Taskset(n_com, n_pos, dataset.data, args.max_length)

        # inti model & algorithm
        print('Initiating algorithm...')
        model: torch.nn.Module = MODEL(dataset.com_pos_hg, n_com, n_pos, **vars(args))
        algorithm = TrainMeta(model, taskset, n_com, n_pos, **vars(args))

        # meta-train
        trainer = Trainer(
            gpus=1,
            auto_select_gpus=True,
            accelerator='gpu',
            max_epochs=args.meta_epochs,
            logger=TensorBoardLogger(
                save_dir='../',
                name='log/meta',
            ),
            enable_checkpointing=False,
        )
        print('Pre-training model...')
        trainer.fit(algorithm)

    # training module
    module: pl.LightningModule = TrainTalentTrendPredictModel(model, **vars(args))

    # train & test
    early_stopping_callback = EarlyStopping(monitor=global_args.monitor, mode='max', min_delta=0.0, patience=args.patience, verbose=False, check_finite=True)
    model_checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor=global_args.monitor, mode='max')
    trainer = Trainer(
        gpus=1,
        auto_select_gpus=True,
        accelerator='gpu',
        max_epochs=args.epochs,
        logger=TensorBoardLogger(
            save_dir='../',
            name='log',
        ),
        weights_summary='top',  # 'full'
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )
    print('Training model...')
    trainer.fit(module, datamodule=datamodule)
    print('Testing model...')
    trainer.test(module, datamodule=datamodule)
