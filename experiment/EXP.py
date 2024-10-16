from models.model import *
from experiment.configs import *
from util.metrics import *
from util.tools import *
import warnings
warnings.filterwarnings('ignore')


class Dataset_loader(Dataset):
    def __init__(self,args):
        self.args=args
        self.labels=pd.read_csv(os.path.join(self.args.root_path,self.args.labels_path))
        self.length=self.labels.shape[0]
        self.shuffle_list=list(range(self.length))
        random.shuffle(self.shuffle_list)
        self.ratio=self.args.train_set_ratio
        self.set_ratio = self.args.set_ratio
        self.trans= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.args.img_size,self.args.img_size))
        ])
        self.test_start=round(self.ratio*self.length)
        self.vail_start=self.test_start-round(self.set_ratio*self.length)

    def __getitem__(self, item):
        # 0 is functional and 1 is defective
        if self.args.set_select==0:
            img=Image.open(os.path.join(self.args.root_path,self.labels.iloc[self.shuffle_list[item],0])).convert('RGB')
            if self.labels.iloc[self.shuffle_list[item],1]>0.5:
                return self.trans(img),torch.tensor([0,1]).float(),self.labels.iloc[self.shuffle_list[item],2]
            else:
                return self.trans(img), torch.tensor([1,0]).float(), self.labels.iloc[self.shuffle_list[item], 2]
        elif self.args.set_select==1:
            img = Image.open(os.path.join(self.args.root_path, self.labels.iloc[self.shuffle_list[item+self.vail_start], 0])).convert('RGB')
            if self.labels.iloc[self.shuffle_list[item+self.vail_start], 1] > 0.5:
                return self.trans(img), torch.tensor([0, 1]).float(), self.labels.iloc[self.shuffle_list[item+self.vail_start], 2]
            else:
                return self.trans(img), torch.tensor([1, 0]).float(), self.labels.iloc[self.shuffle_list[item+self.vail_start], 2]

        else:
            img = Image.open(os.path.join(self.args.root_path, self.labels.iloc[self.shuffle_list[item+self.test_start], 0])).convert('RGB')
            if self.labels.iloc[self.shuffle_list[item+self.test_start], 1] > 0.5:
                return self.trans(img), torch.tensor([0,1]), self.labels.iloc[self.shuffle_list[item+self.test_start], 2]
            else:
                return self.trans(img), torch.tensor([1,0]).float(), self.labels.iloc[self.shuffle_list[item+self.test_start], 2]

    def __len__(self):
        if self.args.set_select==0:
            return self.vail_start
        elif self.args.set_select==1:
            return self.test_start-self.vail_start
        else:
            return self.length-self.test_start

class EXP_model():
    def __init__(self,args):
        self.args=args
        self.model=Vgg16_net()
        self.model=self.model.to(device)
        self.model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        self.criterion=nn.CrossEntropyLoss()
        self.criterion=self.criterion.to(device)
        self.device=device
        self.dataset=Dataset_loader(self.args)

    def vail(self,vali_loader,criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,_) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self,setting):
        self.dataset.args.set_select = 0
        print('The length of train set is {}'.format(len(self.dataset)))
        train_loader = DataLoader(self.dataset,self.args.batch_size,shuffle=False)
        self.dataset.args.set_select = 1
        print('The length of vail set is {}'.format(len(self.dataset)))
        vail_loader = DataLoader(self.dataset, self.args.batch_size, shuffle=False)
        self.dataset.args.set_select=2
        print('The length of test set is {}'.format(len(self.dataset)))
        test_loader = DataLoader(self.dataset,self.args.batch_size,shuffle=False)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        time_now = time.time()
        train_steps = len(train_loader)

        train_loss_list=[]
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y,_) in enumerate(train_loader):
                iter_count += 1
                self.model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())


                if (i + 1) % 10 == 0:
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
            train_loss_list.append(train_loss)
            vali_loss = self.vail(vail_loader, self.criterion)
            test_loss = self.vail(test_loader, self.criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss,test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        plt.figure()
        plt.plot(range(len(train_loss_list)),train_loss_list)
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.show()
        plt.savefig(path + '/' + 'train_loss.png')
        return self.model

    def test(self, setting):
        self.dataset.args.set_select=2
        test_loader = DataLoader(self.dataset, self.args.batch_size, shuffle=False)

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        self.model=self.model.to(device)

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,_) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = np.argmax(outputs,axis=1) #(batch_size,1)
                true = np.argmax(batch_y,axis=1)

                preds.extend(pred)
                trues.extend(true)
                #

        # preds = np.array(preds)
        # trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './output/test_results/' + setting + '/' + self.args.task_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        acc,b_acc,prec,rec,f1 = metric(preds, trues)
        print('acc:{}, b_acc:{} ,prec:{} ,rec:{} ,f1:{}'.format(acc,b_acc,prec,rec,f1))
        f = open(folder_path+'metric_process.txt', 'a')
        f.write(setting + "  \n")
        f.write('acc:{}, b_acc:{} ,prec:{} ,rec:{} ,f1:{}'.format(acc,b_acc,prec,rec,f1))
        f.write('\n')
        f.write('\n')
        f.close()

        np.savetxt(folder_path + 'metrics.csv', np.array([acc,b_acc,prec,rec,f1]),delimiter=',')

        return preds,trues






if __name__ == '__main__':
    a=Dataset_loader(args)
    b=a[:50]
    print(len(b))
    # b=DataLoader(a,batch_size=args.batch_size)
    # c=[]
    # for x, y in b:
    #     c.extend(y.detach().cpu().numpy())
    # print(np.array(c).shape)
    # for x,y,z in a:
    #     print(x.shape)