import os
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold

from nfrl.utils import read_csv, DBEncoder
from nfrl.models import NFRL

DATA_DIR = 'dataset/'


def get_data_loader(dataset, world_size, rank, batch_size,  k=0, pin_memory=False, save_best=True):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = list(kf.split(X_df))[k]
    X_train = X[train_index]

    y_train = y[train_index]
    
    uniques, counts = np.unique(y_train[:,0], return_counts=True)
    counts = counts[::-1]

    percentages = dict(zip(uniques, counts * 100 / len(y_train[:,0])))
    weights = [len(y_train[:,0])/ i for i in counts]
    # for weight in percentages.values():
    #     weights.append(weight)
    class_weights = torch.tensor(weights).float()

    X_test = X[test_index]
    y_test = y[test_index]
    uniques, counts = np.unique(y_test[:,0], return_counts=True)
    percentages = dict(zip(uniques, counts * 100 / len(y_test[:,0])))

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    train_len = int(len(train_set) * 0.90)
    train_sub, valid_sub = random_split(train_set, [train_len, len(train_set) - train_len])
    if not save_best:
        train_sub = train_set
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_sub, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=train_sampler)
    valid_loader = DataLoader(valid_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return db_enc, train_loader, valid_loader, test_loader,  class_weights


def train_model(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    rank = torch.distributed.get_rank()
    torch.manual_seed(args.r_seed)
    device_id = args.device_ids[gpu]
    
    torch.cuda.set_device(device_id)

    if gpu == 0:
        writer = SummaryWriter(args.folder_path)
        is_rank0 = True
    else:
        writer = None
        is_rank0 = False

    dataset = args.data_set
    db_enc, train_loader, valid_loader, _ , class_weights = get_data_loader(dataset, args.world_size, rank, args.batch_size,
                                                            k=args.ith_kfold, pin_memory=True, save_best=args.save_best)

    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen

    nfrl = NFRL(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
              device_id=device_id,
              is_rank0=is_rank0,
              log_file=args.log,
              writer=writer,
              save_best=args.save_best,
              save_path=args.model)

    nfrl.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        class_weights=class_weights, 
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter)


def load_model(path, device_id, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['nfrl_args']
    nfrl = NFRL(
        dim_list=saved_args['dim_list'],
        device_id=device_id,
        is_rank0=True,
        log_file=log_file,
        distributed=distributed,)
    stat_dict = checkpoint['model_state_dict']
    print(stat_dict.keys())
    for key in list(stat_dict.keys()):
        # remove 'module.' prefix
        print(key)
        stat_dict[key[7:]] = stat_dict.pop(key)
    nfrl.net.load_state_dict(checkpoint['model_state_dict'])
    return nfrl


def test_model(args):
    # print(args.model)
    nfrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)
    dataset = args.data_set
    db_enc, train_loader, _, test_loader,  class_weights= get_data_loader(dataset, 1, 0, args.batch_size, args.ith_kfold, save_best=False)
    nfrl.test(test_loader=test_loader, set_name='Test')


def train_main(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    from args import nfrl_args

    train_main(nfrl_args)
    test_model(nfrl_args)