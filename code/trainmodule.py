import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc, f1, confusion_matrix
import learn2learn as l2l
import gc


class TrainTalentTrendPredictModel(pl.LightningModule):

    '''Lightning training module for talent demand and supply trend prediction model
    '''

    ignore_params = ['model']

    def __init__(self, model: nn.Module, *args, **kwargs):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=self.ignore_params)

    def forward(self, x, l, c, p, t_s, t_e):
        try:
            return self.model(x, l)
        except:
            return self.model(x, l, c, p, t_s, t_e)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.step, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        (d_x, s_x), (d_y, s_y), l, c, p, t_s, t_e = batch
        (d_output, s_output) = self((d_x, s_x), l, c, p, t_s, t_e)
        demand_loss = self.model.loss(pred=d_output.view(-1, self.hparams.class_num), true=d_y.view(-1))
        supply_loss = self.model.loss(pred=s_output.view(-1, self.hparams.class_num), true=s_y.view(-1))
        loss = demand_loss + supply_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (d_x, s_x), (d_y, s_y), l, c, p, t_s, t_e = batch
        (d_output, s_output) = self((d_x, s_x), l, c, p, t_s, t_e)
        return (d_output, s_output), (d_y, s_y)

    def validation_epoch_end(self, val_step_outputs):
        # merge outputs and labels
        d_os = []
        s_os = []
        d_ys = []
        s_ys = []
        for (d_o, s_o), (d_y, s_y) in val_step_outputs:
            d_os.append(d_o)
            s_os.append(s_o)
            d_ys.append(d_y)
            s_ys.append(s_y)
        d_os = torch.concat(d_os, axis=0).cpu()
        s_os = torch.concat(s_os, axis=0).cpu()
        d_ys = torch.concat(d_ys, axis=0).cpu()
        s_ys = torch.concat(s_ys, axis=0).cpu()
        # demand eval
        demand_acc, demand_f1, demand_auroc = _eval_metrics(d_os, d_ys, self.hparams.class_num)
        # supply eval
        supply_acc, supply_f1, supply_auroc = _eval_metrics(s_os, s_ys, self.hparams.class_num)
        # overall
        total_output = torch.concat([d_os, s_os], axis=0).cpu()
        total_labels = torch.concat([d_ys, s_ys], axis=0).cpu()
        total_acc, total_f1, total_auroc = _eval_metrics(total_output, total_labels, self.hparams.class_num)
        # log
        _log_metrcis_values(self, demand_acc, demand_f1, demand_auroc, supply_acc, supply_f1, supply_auroc, total_acc, total_f1, total_auroc)
        return (d_os, s_os), (d_ys, s_ys)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        (d_os, s_os), (d_ys, s_ys) = self.validation_epoch_end(test_step_outputs)
        _, d_os = d_os.view(-1, self.hparams.class_num).max(dim=-1, keepdim=False)
        _, s_os = s_os.view(-1, self.hparams.class_num).max(dim=-1, keepdim=False)
        # show confusion matrices
        print()
        print('Demand Confusion Matrix\n', confusion_matrix(d_os, d_ys, num_classes=self.hparams.class_num))
        print('Supply Confusion Matrix\n', confusion_matrix(s_os, s_ys, num_classes=self.hparams.class_num))
        print('Total Confusion Matrix\n', confusion_matrix(torch.concat([d_os, s_os], axis=0), torch.concat([d_ys, s_ys], axis=0), num_classes=self.hparams.class_num))


class TrainMeta(pl.LightningModule):

    '''Lightning training module for talent demand and supply trend prediction model in a meta-learning strategy
    '''

    from dataset import Taskset

    model_names = ['DH_GEM']
    ignore_params = ['model', 'taskset']

    def __init__(self, model: nn.Module, taskset: Taskset, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=self.ignore_params)
        self.model = model
        self.taskset = taskset
        self.algorithm = l2l.algorithms.MAML(self.model, lr=self.hparams.meta_lr, allow_nograd=True)
        self.loss_func = nn.NLLLoss()
        self.sample_n = int(len(self.taskset) * self.hparams.meta_sample_prop)

    def forward(self, x, l, c, p, t_s, t_e, learner):
        try:
            return learner(x, l)
        except:
            return learner(x, l, c, p, t_s, t_e)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.meta_lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.meta_step, gamma=self.hparams.meta_gamma)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(range(self.sample_n), batch_size=1, pin_memory=True)

    def val_dataloader(self):
        return DataLoader([0], batch_size=1, pin_memory=True)

    def training_step(self, batch, batch_idx):
        company_index, dataloader = self.taskset.sample()

        learner = self.algorithm.clone()
        learner.train()

        # fast adapt
        for epoch_i in range(1):
            for data_batch in dataloader:
                (d_x, s_x), (d_y, s_y), l, c, p, t_s, t_e = super().transfer_batch_to_device(data_batch, self.device, -1)
                (d_output, s_output) = self((d_x, s_x), l, c, p, t_s, t_e, learner)
                demand_loss = self.loss_func(d_output.view(-1, self.hparams.class_num), d_y.view(-1))
                supply_loss = self.loss_func(s_output.view(-1, self.hparams.class_num), s_y.view(-1))
                loss = demand_loss + supply_loss
                learner.adapt(loss)

        # eval
        total_loss = []
        for data_batch in dataloader:
            (d_x, s_x), (d_y, s_y), l, c, p, t_s, t_e = super().transfer_batch_to_device(data_batch, self.device, -1)
            (d_output, s_output) = self((d_x, s_x), l, c, p, t_s, t_e, learner)
            demand_loss = self.loss_func(d_output.view(-1, self.hparams.class_num), d_y.view(-1))
            supply_loss = self.loss_func(s_output.view(-1, self.hparams.class_num), s_y.view(-1))
            loss: torch.Tensor = demand_loss + supply_loss
            total_loss.append(loss.div_(l.size(0)))
        total_loss = sum(total_loss) / len(total_loss)

        # free memory
        del dataloader, learner
        gc.collect()

        self.log('train_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):

        # eval
        total_loss = []
        d_os = []
        s_os = []
        d_ys = []
        s_ys = []
        for company_index, dataloader in enumerate(self.taskset):
            assert(len(dataloader) == 1)
            for data_batch in dataloader:
                (d_x, s_x), (d_y, s_y), l, c, p, t_s, t_e = super().transfer_batch_to_device(data_batch, self.device, -1)
                (d_output, s_output) = self((d_x, s_x), l, c, p, t_s, t_e, self.model)
                demand_loss = self.loss_func(d_output.view(-1, self.hparams.class_num), d_y.view(-1))
                supply_loss = self.loss_func(s_output.view(-1, self.hparams.class_num), s_y.view(-1))
                loss: torch.Tensor = demand_loss + supply_loss
                loss.div_(l.size(0))
                total_loss.append(loss)
                self.taskset.record(company_index, loss)
                d_os.append(d_output)
                s_os.append(s_output)
                d_ys.append(d_y)
                s_ys.append(s_y)
        total_loss = sum(total_loss) / len(total_loss)
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True, logger=True)
        d_os = torch.concat(d_os, axis=0).cpu()
        s_os = torch.concat(s_os, axis=0).cpu()
        d_ys = torch.concat(d_ys, axis=0).cpu()
        s_ys = torch.concat(s_ys, axis=0).cpu()
        # demand eval
        demand_acc, demand_f1, demand_auroc = _eval_metrics(d_os, d_ys, self.hparams.class_num)
        # supply eval
        supply_acc, supply_f1, supply_auroc = _eval_metrics(s_os, s_ys, self.hparams.class_num)
        # overall
        total_output = torch.concat([d_os, s_os], axis=0).cpu()
        total_labels = torch.concat([d_ys, s_ys], axis=0).cpu()
        total_acc, total_f1, total_auroc = _eval_metrics(total_output, total_labels, self.hparams.class_num)
        # log
        _log_metrcis_values(self, demand_acc, demand_f1, demand_auroc, supply_acc, supply_f1, supply_auroc, total_acc, total_f1, total_auroc)

        # free memory
        # del dataloader, learner, total_loss, d_os, s_os, d_ys, s_ys, total_output, total_labels
        gc.collect()

    def on_validation_epoch_end(self):
        self.taskset.update_sample_prob()
        return super().on_validation_epoch_end()


# ----------------------Utilities----------------------

def _eval_metrics(output: torch.Tensor, labels: torch.Tensor, class_num):
    '''evaluate model in three classification metrics: accuracy, weighted f1 score and auroc
    '''
    _, output_labels = output.view(-1, class_num).max(dim=-1, keepdim=False)
    acc = accuracy(output_labels, labels)
    weighted_f1 = f1(output_labels, labels, average='weighted', num_classes=class_num)
    au = auroc(output, labels, num_classes=class_num)
    return acc, weighted_f1, au


def _log_metrcis_values(self: pl.LightningModule, demand_acc, demand_f1, demand_auroc, supply_acc, supply_f1, supply_auroc, total_acc, total_f1, total_auroc):
    '''logging operation for Lightning module
    '''
    self.log('Demand Accuracy', demand_acc, on_epoch=True, prog_bar=False, logger=True)
    self.log('Demand Weighted-F1', demand_f1, on_epoch=True, prog_bar=True, logger=True)
    self.log('Demand AUROC', demand_auroc, on_epoch=True, prog_bar=False, logger=True)
    self.log('Supply Accuracy', supply_acc, on_epoch=True, prog_bar=False, logger=True)
    self.log('Supply Weighted-F1', supply_f1, on_epoch=True, prog_bar=True, logger=True)
    self.log('Supply AUROC', supply_auroc, on_epoch=True, prog_bar=False, logger=True)
    self.log('Total Accuracy', total_acc, on_epoch=True, prog_bar=True, logger=True)
    self.log('Total Weighted-F1', total_f1, on_epoch=True, prog_bar=True, logger=True)
    self.log('Total AUROC', total_auroc, on_epoch=True, prog_bar=True, logger=True)
