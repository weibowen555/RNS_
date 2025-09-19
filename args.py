import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data_set', type=str, default='bank',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-i', '--device_ids', type=str, default="0", help='Set the device (GPU ids). Split by @.'
                                                                       ' E.g., 0@2@3.')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('-e', '--epoch', type=int, default=41, help='Set the total epoch.')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Set the batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Set the initial learning rate.')
parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.76, help='Set the learning rate decay rate.')
parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=20, help='Set the learning rate decay epoch.')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Set the weight decay (L2 penalty).')
parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th 5-fold validation, 0 <= ki < 5.')
parser.add_argument('-rc', '--round_count', type=int, default=0, help='Count the round of experiments.')
parser.add_argument('-ma', '--master_address', type=str, default='127.0.0.1', help='Set the master address.')
parser.add_argument('-mp', '--master_port', type=str, default='12345', help='Set the master port.')
parser.add_argument('-li', '--log_iter', type=int, default=50, help='The number of iterations (batches) to log once.')
parser.add_argument('--save_best', action="store_true",default=True)
parser.add_argument('-s', '--structure', type=str, default='5@64',
                    help='Set the number of nodes in the binarization layer and logical layers. '
                         'E.g., 10@32@32.')
parser.add_argument('-bin', '--bins',type=int,  default=5)
parser.add_argument('-seed', '--r_seed', type = int, default=42)
nfrl_args = parser.parse_args()
nfrl_args.folder_name = 'seed_{}_{}_bins{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_ki_{}_saveBest{}'.format( nfrl_args.r_seed, 
    nfrl_args.data_set,nfrl_args.bins, nfrl_args.epoch, nfrl_args.batch_size, nfrl_args.learning_rate, nfrl_args.lr_decay_rate,
    nfrl_args.lr_decay_epoch, nfrl_args.weight_decay, nfrl_args.ith_kfold, nfrl_args.save_best)

if not os.path.exists('log_folder'):
    os.mkdir('log_folder')
nfrl_args.folder_name = nfrl_args.folder_name + '_L' + nfrl_args.structure
nfrl_args.set_folder_path = os.path.join('log_folder', nfrl_args.data_set)
if not os.path.exists(nfrl_args.set_folder_path):
    os.mkdir(nfrl_args.set_folder_path)
nfrl_args.folder_path = os.path.join(nfrl_args.set_folder_path, nfrl_args.folder_name)
if not os.path.exists(nfrl_args.folder_path):
    os.mkdir(nfrl_args.folder_path)
nfrl_args.model = os.path.join(nfrl_args.folder_path, 'model.pth')
nfrl_args.nfrl_file = os.path.join(nfrl_args.folder_path, 'nfrl.txt')
nfrl_args.plot_file = os.path.join(nfrl_args.folder_path, 'plot_file.pdf')
nfrl_args.log = os.path.join(nfrl_args.folder_path, 'log.txt')
nfrl_args.test_res = os.path.join(nfrl_args.folder_path, 'test_res.txt')
nfrl_args.device_ids = list(map(int, nfrl_args.device_ids.strip().split('@')))
nfrl_args.gpus = len(nfrl_args.device_ids) 
nfrl_args.nodes = 1
nfrl_args.world_size = (nfrl_args.gpus * nfrl_args.nodes )
nfrl_args.batch_size = int(nfrl_args.batch_size / nfrl_args.gpus)
