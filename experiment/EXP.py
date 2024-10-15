# from models.model import *
from models.model1D import *
from experiment.configs import *
from util.warehouse import *
from util.metrics import *
from util.tools import *
import warnings
warnings.filterwarnings('ignore')

class Dataset_loader(Dataset):
    def __init__(self,args):
        self.root_path=os.path.join(args.root_path,args.data_path)
        self.listdir=os.listdir(self.root_path)
        self.condition=args.condition_mode
        if args.is_training:
            self.bearings=args.train_bearings[self.condition]
        else:
            self.bearings = args.test_bearings[self.condition]
        if args.readmode:
            self.data=torch.tensor(np.genfromtxt(os.path.join(self.root_path,'Bearing'+self.bearings[0]+args.suffix), delimiter=','))
            self.health=torch.linspace(1, 0, self.data.shape[0]).reshape(-1, 1)
            for i in range(1,len(self.bearings)):
                self.newdata=torch.tensor(np.genfromtxt(os.path.join(self.root_path,'Bearing'+self.bearings[i]+args.suffix), delimiter=','))
                self.newhealth=torch.linspace(1, 0, self.newdata.shape[0]).reshape(-1, 1)
                self.data=torch.cat([self.data,self.newdata],dim=0)
                self.health=torch.cat([self.health,self.newhealth],dim=0)
        else:
            self.data=torch.tensor(np.genfromtxt(os.path.join(self.root_path,'Bearing'+self.bearings[args.readitem]+args.suffix), delimiter=','))
            self.health = torch.linspace(1, 0, self.data.shape[0]).reshape(-1, 1)
        self.data=self.data.unsqueeze(1)
    def __getitem__(self, item):
        return self.data[item],self.health[item]
    def __len__(self):
        return len(self.data)

class EXP_model():
    def __init__(self,args):
        self.args=args
        self.model=ALOFT(self.args)
        self.model=self.model.to(device)
        self.model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        self.criterion=nn.MSELoss()
        self.criterion=self.criterion.to(device)
        self.device=device

    def data_provider(self,shuffle_flag = True):
        dataset=Dataset_loader(self.args)
        data_loader=DataLoader(dataset,batch_size=self.args.batch_size,shuffle=shuffle_flag)
        return dataset,data_loader

    def get_data(self,flag='train',shuffle_flag=True):
        if flag=='train':
            data_set, data_loader = self.data_provider(shuffle_flag)
        else:
            self.args.is_training = False
            data_set, data_loader = self.data_provider(shuffle_flag)
            self.args.is_training = True
        return data_set, data_loader

    def plot_img(self,preds,trues,bearing_name):

        plt.figure()
        plt.plot(range(len(trues)), trues, color='b') #真实曲线（blue）
        plt.plot(range(len(preds)), preds, color='r') #测试曲线（red）
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Health', fontsize=12)
        plt.title('Bearing '+bearing_name)
        plt.show()

    def vail(self,vali_loader,criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self,setting):
        train_data, train_loader = self.get_data()
        test_data, test_loader = self.get_data(flag='test')


        self.args.is_training=True
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        time_now = time.time()
        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                self.model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                self.model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vail(train_loader, self.criterion)
            test_loss = self.vail(test_loader, self.criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss,test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.args.is_training = False
        return self.model

    def test(self, setting,flag='train', test=0):

        test_data, test_loader = self.get_data(flag=flag,shuffle_flag=False)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
            self.model=self.model.to(device)
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.extend(pred)
                trues.extend(true)
                #

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1)
        trues = trues.reshape(-1)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './output/test_results/' + setting + '/' + self.args.bearing_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{} ,rmse:{}'.format(mse, mae,rmse))
        f = open(folder_path+'metric_process.txt', 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{} ,rmse{}'.format(mse, mae,rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.savetxt(folder_path + 'metrics.csv', np.array([mae, mse, rmse, mape, mspe]),delimiter=',')
        np.savetxt(folder_path + 'pred.csv', preds,delimiter=',')
        np.savetxt(folder_path + 'true.csv', trues,delimiter=',')

        return preds,trues
    def vail_img_plot(self,setting):

        self.args.readmode=False
        # self.is_training=True

        # for self.args.readitem in range(len(self.args.train_bearings[self.args.condition_mode])):
        #     self.args.bearing_name= self.args.train_bearings[self.args.condition_mode][self.args.readitem]
        #     print('Bearing'+self.args.bearing_name+':\n')
        #     preds,trues=self.test(setting,test=1)
        #     self.plot_img(preds,trues,self.args.bearing_name)

        for self.args.readitem in range(len(self.args.test_bearings[self.args.condition_mode])):
            self.args.bearing_name= self.args.test_bearings[self.args.condition_mode][self.args.readitem]
            print('Bearing'+self.args.bearing_name+':\n')
            preds,trues=self.test(setting,flag='test',test=1)
            self.plot_img(preds,trues,self.args.bearing_name)





if __name__ == '__main__':
    a=Dataset_loader(args)
    b=DataLoader(a,batch_size=args.batch_size)
    c=[]
    for x, y in b:
        c.extend(y.detach().cpu().numpy())
    print(np.array(c).shape)