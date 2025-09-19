import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict
from nfrl.components import BinarizeLayer
from nfrl.components import  LRLayer, Selection_Layer, Selection_Layer_mask
import time

TEST_CNT_MOD = 500
class NFRL_NET(nn.Module):
    def __init__(self, dim_list, left=None, right=None,):
        super(NFRL_NET, self).__init__()
        self.dim_list = dim_list
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])
        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            if i == 4:
                num += self.layer_list[-2].output_dim
                
            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.left, self.right)
                layer_name = 'binary{}'.format(i)
            elif i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)
            elif i == 2:
                layer = Selection_Layer(dim_list[i], num)
                layer_name = 'selection{}'.format(i)
            elif i == 3:
                layer = Selection_Layer_mask(dim_list[i], num)
                layer_name = 'selection{}'.format(i)
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, x):
        x_res = None

        for i, layer in enumerate(self.layer_list):
            if i == 0:
                x = layer(x)
            elif i == 1:
                x, prev_w_op = layer(x)
                x_res = x
            elif i == 2:
                x = layer(x, prev_w_op=prev_w_op)
            else:
                if len(x_res.shape) == 1:
                    x_res = x_res.unsqueeze(0)
                    x = x.unsqueeze(0)
                x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                x = layer(x_cat)
        return x


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list


class NFRL:
    def __init__(self, dim_list, device_id, is_rank0=False, log_file=None, writer=None, left=None,
                 right=None, save_best=False, save_path=None, distributed=True):
        super(NFRL, self).__init__()
        self.dim_list = dim_list
        self.best_f1 = -1.
        self.device_id = device_id
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.save_path = save_path
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        self.writer = writer
        self.net = NFRL_NET(dim_list, left=left, right=right, )

        self.net.cuda(self.device_id)
        if distributed:
            self.net = MyDistributedDataParallel(self.net, device_ids=[self.device_id])

    def clip(self):
        for layer in self.net.layer_list[: -1]:
            layer.clip()

    def data_transform(self, X, y):
        X = X.astype(np.float)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float)
        return torch.tensor(X), torch.tensor(y)

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, X=None, y=None, X_validation=None, y_validation=None, data_loader=None, valid_loader=None,  class_weights=None, 
                    epoch=50, lr=0.01, lr_decay_epoch=100, lr_decay_rate=0.75, batch_size=64, weight_decay=0.0,
                    log_iter=50):

        if (X is None or y is None) and data_loader is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        accuracy = []
        f1_score = []
        # criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        cnt = -1
        avg_batch_loss = 0.0
        epoch_histc = defaultdict(list)
        nweights = 0
        for name,weights in self.net.named_parameters():
            if 'bias' not in name:
                nweights = nweights + weights.numel()
        print(f'Total number of weights in the model = {nweights}')
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            epoch_loss = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0
            ba_cnt = 0
            for X, y in data_loader:
                ba_cnt += 1
                X = X.cuda(self.device_id, non_blocking=True)
                y = y.cuda(self.device_id, non_blocking=True)
                optimizer.zero_grad()
                y_pred = self.net.forward(X)
                with torch.no_grad():
                    y_prob = torch.softmax(y_pred, dim=1)
                    y_arg = torch.argmax(y, dim=1)
                    loss = criterion(y_prob, y_arg)
                    ba_loss = loss.item()
                    epoch_loss += ba_loss
                    avg_batch_loss += ba_loss
                y_pred.backward((y_prob - y) / y.shape[0])  # for CrossEntropy Loss
                cnt += 1
                if self.is_rank0 and cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                    self.writer.add_scalar('Avg_Batch_Loss', avg_batch_loss / log_iter, cnt)
                    avg_batch_loss = 0.0
                optimizer.step()
                if self.is_rank0:
                    for i, param in enumerate(self.net.parameters()):
                        abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                        abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (param.grad.numel())
                self.clip()

                if self.is_rank0 and cnt % TEST_CNT_MOD == 0:
                    if X_validation is not None and y_validation is not None:
                        acc, f1 = self.test(X_validation, y_validation, batch_size=batch_size,
                                                         need_transform=False, set_name='Validation')
                    elif valid_loader is not None:
                        acc, f1 = self.test(test_loader=valid_loader, need_transform=False,
                                                         set_name='Validation')
                    elif data_loader is not None:
                        acc, f1 = self.test(test_loader=data_loader, need_transform=False,
                                                         set_name='Training')
                    else:
                        acc, f1 = self.test(X, y, batch_size=batch_size, need_transform=False,
                                                         set_name='Training')
                    
                    if self.save_best and f1 > self.best_f1:
                        self.best_f1 = f1
                        self.save_model()

                    accuracy.append(acc)
                    f1_score.append(f1)
                    if self.writer is not None:
                        self.writer.add_scalar('Accuracy', acc, cnt // TEST_CNT_MOD)

                        self.writer.add_scalar('F1_Score', f1, cnt // TEST_CNT_MOD)
            if self.is_rank0:
                logging.info('epoch: {}, loss: {}'.format(epo, epoch_loss,))
        if self.is_rank0 and not self.save_best:
            self.save_model()
        return epoch_histc

    def test(self, X=None, y=None, test_loader=None, batch_size=32, need_transform=True, set_name='Validation'):
        if X is not None and y is not None and need_transform:
            X, y = self.data_transform(X, y)
        with torch.no_grad():
            if X is not None and y is not None:
                test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

            y_list = []
            for X, y in test_loader:
                y_list.append(y)
            y_true = torch.cat(y_list, dim=0)
            y_true = y_true.cpu().numpy().astype(np.int64)
            y_true = np.argmax(y_true, axis=1)
            data_num = y_true.shape[0]
            slice_step = data_num // 40 if data_num >= 40 else 1
            logging.debug('y_true: {} {}'.format(y_true.shape, y_true[:: slice_step]))
            y_pred_list = []
            for X, y in test_loader:
                X = X.cuda(self.device_id, non_blocking=True)
                output = self.net.forward(X)
                y_pred_list.append(output)
            y_pred = torch.cat(y_pred_list).cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            logging.debug('y: {} {}'.format(y_pred.shape, y_pred[:: slice_step]))
            accuracy = metrics.accuracy_score(y_true, y_pred)
            f1_score = metrics.f1_score(y_true, y_pred, average='macro')

            logging.info('-' * 60)
            logging.info('On {} Set:\n\tAccuracy of NFRL  Model: {}'
                         '\n\tF1 Score of NFRL  Model: {}'.format(set_name, accuracy, f1_score))
            logging.info('On {} Set:\nPerformance of  NFRL Model: \n{}\n{}'.format(
                 set_name, metrics.confusion_matrix(y_true, y_pred), metrics.classification_report(y_true, y_pred)))
            logging.info('-' * 60)
        return accuracy, f1_score


    def save_model(self):
        nfrl_args = {'dim_list': self.dim_list,}
        torch.save({'model_state_dict': self.net.state_dict(), 'nfrl_args': nfrl_args}, self.save_path)

