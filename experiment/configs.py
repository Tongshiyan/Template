import argparse

parser = argparse.ArgumentParser(description='GASF-GAN-ALOFT')


#model

parser.add_argument('--window_size', type=int, default=80,help='window_size for GASF')
parser.add_argument('--img_size', type=int, default=2560,help='size of img')
parser.add_argument('--patch_size', type=int, default=10,help='size of patch >=3')
parser.add_argument('--embed_dim', type=int, default=64,help='size of embed / dim')
parser.add_argument('--in_chans', type=int, default=1,help='number of in_channel')
parser.add_argument('--block_num', type=int, default=6,help='number of blocklayer')
parser.add_argument('--mlp_ratio', type=float, default=4.,help='mlp_ratio*dim=hidden layer number')
parser.add_argument('--drop', type=float, default=0.5,help='dropout of mlp')
parser.add_argument('--drop_path', type=float, default=0.5,help='droppath of net')
parser.add_argument('--h', type=int, default=16,help='h of filter')
parser.add_argument('--w', type=int, default=9,help='w of filter')
parser.add_argument('--mask_radio', type=float, default=0.5)
parser.add_argument('--mask_alpha', type=float, default=0.5)
parser.add_argument('--noise_mode', type=int, default=3,help='1 amplitude; 2: phase 3:both')
parser.add_argument('--uncertainty_model', type=int, default=2,help='1 batch; 2: channel 3:gan-batch')
parser.add_argument('--perturb_prob', type=float, default=0.5)
parser.add_argument('--uncertainty_factor', type=float, default=1.)
parser.add_argument('--gauss_or_uniform', type=int, default=0,help='0 gauss')


#data loader
parser.add_argument('--root_path', type=str, default=r'F:\SF-Net', help='root_path')
parser.add_argument('--data_path', type=str, default='dataset', help='data_path')
parser.add_argument('--train_bearings', type=list, default=[['1_1','1_2','2_1','2_2'],['1_1','1_2','3_1','3_2'],['2_1','2_2','3_1','3_2']])
parser.add_argument('--test_bearings', type=list, default=[['3_1','3_2'],['2_1','2_2'],['1_1','1_2']])
parser.add_argument('--condition_mode', type=int, default=0,help='0:1,2-3  1:1,3-2 2:2,3-1')
parser.add_argument('--suffix', type=str, default='.csv',help='suffixes for data')
parser.add_argument('--readmode', type=bool, default=True,help='True:all bearings False:read bearing one by one')
parser.add_argument('--readitem', type=int, default=0,help='If readmode is False it indicates the serial number of the bearing set')
parser.add_argument('--bearing_name', type=str, default='1_1',help='bearing name') #

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
parser.add_argument('--model', type=str, default='GAN-ALOFT')

args = parser.parse_args()
print('Args in experiment:')
print(args)

