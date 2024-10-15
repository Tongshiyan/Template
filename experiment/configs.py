import argparse

parser = argparse.ArgumentParser(description='Paper EXP')


#model

parser.add_argument('--img_size', type=int, default=150,help='size of img')

parser


#data loader
parser.add_argument('--root_path', type=str, default=r'F:\SF-Net', help='root_path')
parser.add_argument('--data_path', type=str, default='dataset', help='data_path')


#model hyperparameter settings
parser.add_argument('--opt_name', type=str, default='AdamW')
parser.add_argument('--train_epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--batch_size', type=int, default=64,help='batch_size')
parser.add_argument('--patience', type=int, default=30,help='number of rounds terminated early')
parser.add_argument('--itr', type=int, default=1,help='number of experiments')
parser.add_argument('--checkpoints', type=str, default='./output/checkpoints/', help='location of model checkpoints')
parser.add_argument('--is_training', type=bool, default=True)

#exp
parser.add_argument('--exp_name', type=str, default='2024-4-29')
parser.add_argument('--model', type=str, default='NAS')

args = parser.parse_args()
print('Args in experiment:')
print(args)

