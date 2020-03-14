import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# data in/out
parser.add_argument('--dataset_path',default = r'E:\Files\pycharm\MIS\3DUnet\fixed',
                    help='trainset root path')
parser.add_argument('--save_name',default='test_model',
                    help='save path of trained model')

# train
parser.add_argument('--epochs', type=int, default=12, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

args = parser.parse_args()


