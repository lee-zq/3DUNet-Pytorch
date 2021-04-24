import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=2,help='number of classes') # 分割肝脏则置为2（二类分割），分割肝脏和肿瘤则置为3（三类分割）
parser.add_argument('--upper', type=int, default=200, help='')
parser.add_argument('--lower', type=int, default=200, help='')
parser.add_argument('--norm_factor', type=float, default=200.0, help='')
parser.add_argument('--expand_slice', type=int, default=20, help='')
parser.add_argument('--min_slices', type=int, dnorm_factorefault=48, help='')
parser.add_argument('--xy_down_scale', type=int, default=0.5, help='')
parser.add_argument('--slice_down_scale', type=int, default=1.0, help='')

# data in/out and dataset
parser.add_argument('--dataset_path',default = '/ssd/lzq/dataset/fixed_lits2',help='fixed trainset root path')
parser.add_argument('--save',default='model1',help='save path of trained model')
parser.add_argument('--train_resize_scale', type=float, default=1.0,help='resize scale for input data')
parser.add_argument('--test_resize_scale', type=float, default=1.0,help='resize scale for input data')
parser.add_argument('--crop_size', type=list, default=[32, 144, 144],help='patch size of train samples after resize')
parser.add_argument('--batch_size', type=list, default=4,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=200, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--early-stop', default=20, type=int, help='early stopping (default: 20)')

args = parser.parse_args()


