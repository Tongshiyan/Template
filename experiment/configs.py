import argparse

parser = argparse.ArgumentParser(description='Paper EXP')


#model
parser.add_argument('--img_size', type=int, default=150,help='size of img')


#data loader
parser.add_argument('--root_path', type=str, default=r'T:\Paper\2024\elpv-dataset-master\src\elpv_dataset\data', help='root_path')
parser.add_argument('--data_path', type=str, default='images', help='data_path')
parser.add_argument('--labels_path', type=str, default='labels.csv', help='label_path')
parser.add_argument('--train_set_ratio', type=float, default=0.75)

#model hyperparameter settings
parser.add_argument('--task_name', type=str, default='VGG16_EXP')
parser.add_argument('--opt_name', type=str, default='SDG')
parser.add_argument('--train_epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.0025)
parser.add_argument('--weight_decay', type=float, default=0.007)
parser.add_argument('--batch_size', type=int, default=64,help='batch_size')
parser.add_argument('--patience', type=int, default=10,help='number of rounds terminated early')
parser.add_argument('--itr', type=int, default=1,help='number of experiments')
parser.add_argument('--checkpoints', type=str, default='./output/checkpoints/', help='location of model checkpoints')
parser.add_argument('--train_condition', type=bool, default=True,help='True is train False is test')

#exp
parser.add_argument('--exp_name', type=str, default='2024-10-15')
parser.add_argument('--model', type=str, default='VGG16')

args = parser.parse_args()
print('Args in experiment:')
print(args)

