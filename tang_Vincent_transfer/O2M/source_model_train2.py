import torch
import os
import math
import data_loader
import models
from config import CFG
import utils
import numpy as np
# import KMM_Lin
from sklearn.metrics import confusion_matrix
import torch.utils.data as Data
import call_resnet18_multi
# import estimate_mu as es
from torch.utils.tensorboard import SummaryWriter
import LabelSmoothing as LS
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import backbone_multi

torch.backends.cudnn.benchmark = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="training source model and testing performance")
parser.add_argument("--extracted_layer",type=str, default="None",help='Which point of feature map want to extract')
parser.add_argument('--train_root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/")
parser.add_argument('--test_root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/")
parser.add_argument('--source_dir', type=str, default="UCM")
# parser.add_argument('--target_train_dir', type=str, default="UCM")
parser.add_argument('--target_test_dir', type=str, default="RSSCN7")
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--epoch", type=int, default=150, help="Number of training epochs")
parser.add_argument('--source_class', type=str, default="None")
parser.add_argument('--test_class', type=str, default="None")
parser.add_argument("--save_parameter_path_name", type=str, default="None", help='path to save models and log files')
# parser.add_argument("--save_test_name", type=str, default='test_log1.csv', help='path to save models and log files')
# parser.add_argument("--save_train_loss_name", type=str, default='train_log.csv', help='path to save models and log files')
parser.add_argument("--gpu_id", type=str, default='cuda:1', help='GPU id')

opt = parser.parse_args()
extracted_layer = opt.extracted_layer
backbone_multi.extracted_layer = extracted_layer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(opt.gpu_id)
logtrain = [] #創建一個空的list
logtest = [] #創建一個空的list
#writer = SummaryWriter()

def test(model, target_test_loader, test_flag):
    model.eval() # 讓模型變成測試模式，主要針對Dropout與Batch Normalization在train與eval的不同設置模式
    test_loss = utils.AverageMeter() # Computes and stores the average and current value
    correct_total = 0.
    
    count_stack = 0
    #import pdb;pdb.set_trace()
    test_flag = 1
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = LS.LabelSmoothingCrossEntropy(reduction='sum') # 定義一個標準準則，用來計算loss (讓輸出經過softmax，再進入Cross Entropy)
    len_target_dataset = len(target_test_loader.dataset) #所有test資料集的總數
    with torch.no_grad(): # 在做evaluation時，關閉計算導數來增加運行速度
        for data, target in target_test_loader: # data為test資料，target為test label
            data, target = data.to(DEVICE), target.to(DEVICE)
            #import pdb; pdb.set_trace()
            #_, _, s_output, t_output, test_mmd_loss = model(data, data, target, mu) # 將data放入模型得到預測的輸出
            test_output = model.predict(data, test_flag)
            loss = criterion(test_output, target) #計算loss
            test_loss.update(loss.item()) # 更新值到紀錄中

            # torch.max(a,1) 返回每一列中最大值的那個元素，且返回其索引
            # troch.max()[1] 只返回最大值的每個索引
            pred = torch.max(test_output, 1)[1]
            
            correct_total += torch.sum(pred == target)

            pred_matrix = pred.data
            target_matrix = target.data
            # print("pred_matrix、target_matrix")
            # print(pred_matrix,target_matrix)
            if count_stack == 0:
                pred_matrix_total = pred_matrix
                target_matrix_total = target_matrix
                count_stack =1
            elif count_stack == 1:
                pred_matrix_total = torch.cat((pred_matrix_total,pred_matrix))
                target_matrix_total = torch.cat((target_matrix_total,target_matrix))

            # print("pred_matrix_total、target_matrix_total")
            # print(pred_matrix_total, target_matrix_total)
        target_matrix_total = target_matrix_total.cpu().numpy()
        pred_matrix_total = pred_matrix_total.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(target_matrix_total, pred_matrix_total,labels=[0,1]).ravel()
    test_total = 100*correct_total.type(torch.float32)/len_target_dataset
    print('{} --> {}: test max correct: {}, test accuracy: {:.3f} % \n'.format(source_name, target_name, correct_total,test_total))
    #print("test_loss = {}".format(test_loss.avg))

    with open('D:\\Tang\\source_model_training_record'+str(opt.epoch)+opt.save_parameter_path_name+'.txt', 'a') as f:
    	f.write('\ntest accuracy = ')
    	f.write(str(test_total))

    logtest.append([test_total,tn, fp, fn, tp])
    np_test = np.array(logtest, dtype=float)

    # delimiter : 分隔浮號 ； %.6f 浮點型保留六位小數
    #np.savetxt(opt.save_test_name, np_log, delimiter=',', fmt='%.6f')
    test_flag = 0
    return np_test

def train(source_loader, test_flag,  target_test_loader, model, CFG, optimizer):
    len_source_loader = len(source_loader) # 訓練資料來源的batch數量
    # len_target_loader = len(target_train_loader) # 訓練資料目標batch的數量
    # mu = 0.5

    for e in range(opt.epoch):
        test_flag = 0
        train_loss_clf = utils.AverageMeter()
        # train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        LEARNING_RATE = opt.lr / math.pow((1 + 10 * (e - 1) / opt.epoch), 0.75) ##
        #print(LEARNING_RATE)
        
        #scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85)
        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        #print(scheduler.get_last_lr())
        #print("model.base_network.conv1.weight")
        #print(model.base_network.conv1.weight)
        #print("model.base_network.layer4[0].conv1.weight")
        # print(model.base_network.layer4[0].conv1.weight)

        # 讓pytorch知道切換到training mode
        model.train()

        # dataloader是可迭代對象，需先使用iter()訪問，返回一個迭代器後，再使用next()訪問
        iter_source = iter(source_loader)

        n_batch = len_source_loader # 取train資料source與target的最小值 (batch_size設置的參數不是在這傳入)
        # n_batch = len_source_loader
        #criterion = torch.nn.CrossEntropyLoss()

        criterion = LS.LabelSmoothingCrossEntropy(reduction = 'sum') # 使用Loss為CrossEntropy (pytorch crossentropy會自動先經過softmax function)
        #scheduler.step()
        #print(scheduler.get_last_lr())
        tStart = time.time()
        for i in range(n_batch):
            
            data_source, label_source = iter_source.next()
            #print(data_source.size())
            # data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            # data_target = data_target.to(DEVICE)
            #writer.add_images("image", data_source)
            #print(data_target.size())
            #print(label_source)

            label_source = torch.squeeze(label_source)
            #print(label_source)

            optimizer.zero_grad() # set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.

            source, label_source_pred = model(data_source, label_source, test_flag)
            #source, target, label_source_pred, target_pred, transfer_loss = model(data_source, data_target, mu) # 返回 model.forward 的 source_clf 與 transfer_loss
            #print(target)
            clf_loss = criterion(label_source_pred, label_source)
            # gamma = 2 / (1 + math.exp(-10 * (e) / opt.epoch)) - 1 
            loss = clf_loss # + gamma * transfer_loss 
            #loss = clf_loss + gamma * transfer_loss 

            # classification loss + lambda * transfer loss
            loss.backward()
             # computes dloss/dx for every parameter x which has requires_grad=True
            optimizer.step() #updates the value of x using the gradient x.grad
             
            train_loss_clf.update(clf_loss.item()) # 更新值到紀錄中
            # train_loss_transfer.update(transfer_loss.item()) # 更新值到紀錄中
            train_loss_total.update(loss.item()) # 更新值到紀錄中

            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    opt.epoch,
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_total.avg))
            
        scheduler.step()

        logtrain.append([train_loss_clf.avg,])
        np_log = np.array(logtrain, dtype=float)
        # save_log_add = os.path.join(save_log_path, str(opt.save_train_loss_name))
        # delimiter : 分隔浮號 ； %.6f 浮點型保留六位小數
        # np.savetxt(save_log_add, np_log, delimiter=',', fmt='%.12f')
        #mu_last = mu
        
        #mu = es.estimate_mu(source.detach().cpu().numpy(), label_source.detach().cpu().numpy(),
        #target.detach().cpu().numpy(), target_pred.detach().cpu().numpy()) 
        
        tEnd = time.time()
        print (tEnd - tStart)

        #writer.add_scalar('Loss/cls', train_loss_clf.avg, e)
        #writer.add_scalar('Loss/transfer', train_loss_transfer.avg, e)

        #print(mu)
        #return np_log

        #Test
        np_test = test(model, target_test_loader, test_flag)
        #print(data_source.shape)

        # save_test_add = os.path.join(save_test_path, str (opt.save_test_name))
        # np.savetxt(save_test_add, np_test, delimiter=',', fmt='%.6f')

if __name__ == '__main__':
    torch.manual_seed(0) # 為CPU设置隨機種子讓參數是從某一隨機種子初始化
    torch.cuda.manual_seed_all(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    #random.seed(0)
    test_flag = 0
    source_name = opt.source_class
    target_name = opt.test_class
    extracted_layer = opt.extracted_layer

    print('Src: %s, Tar: %s' % (source_name, target_name))
    
    kwargs = {'num_workers': 2, 'pin_memory': False, 'persistent_workers' : True} if opt.gpu_id == 0 or 1 else {}

    path_source_train = os.path.join(opt.train_root_path, opt.source_dir, 'source_train_feature.npy')
    path_source_train_label = os.path.join(opt.train_root_path, opt.source_dir, 'source_train_feature_label.npy')
    # path_target_train = os.path.join(opt.train_root_path, opt.target_train_dir, 'target_train_feature.npy')
    # path_target_train_label = os.path.join(opt.train_root_path, opt.target_train_dir, 'target_train_feature_label.npy')

    source_train = torch.from_numpy(np.load(path_source_train))
    source_train_label = torch.from_numpy(np.load(path_source_train_label))
    # target_train = torch.from_numpy(np.load(path_target_train))
    # target_train_label = torch.from_numpy(np.load(path_target_train_label))

    source_dataset = Data.TensorDataset(source_train, source_train_label)
    # target_dataset = Data.TensorDataset(target_train, target_train_label)

    source_loader = Data.DataLoader(
    dataset=source_dataset,      # torch TensorDataset format
    batch_size=opt.batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,   # 多线程来读数据
    drop_last = True,
    persistent_workers=True, # 未到batch_size數量之樣本丟棄
    )

    # target_train_loader = Data.DataLoader(
    # dataset=target_dataset,      # torch TensorDataset format
    # batch_size=opt.batch_size,      # mini batch size
    # shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=2,   # 多线程来读数据
    # drop_last = True,
    # persistent_workers=True, # 未到batch_size數量之樣本丟棄
    # )

    target_test_loader = data_loader.load_testing(opt.test_root_path, opt.target_test_dir, opt.batch_size, kwargs)

    save_parameter_path = 'D:\\Tang\\create_datasets\\source_model_para\\' + opt.extracted_layer + '\\' + opt.source_class + '\\' +str(opt.lr) + '\\'
    # save_test_path = './accuracy/' + opt.source_class+ '/' + 'to' + '/' + opt.test_class + '/' +str(opt.lr) 
    # save_log_path = './log/' + opt.source_class + '/' + 'to' + '/' + opt.test_class + '/' +str(opt.lr) 

    if not os.path.exists(save_parameter_path):
        os.makedirs(save_parameter_path)
    # if not os.path.exists(save_test_path):
    #     os.makedirs(save_test_path)
    # if not os.path.exists(save_log_path):
    #     os.makedirs(save_log_path)
    
    model = models.Transfer_Net(CFG['n_class'])
    model = model.cuda()

    # KMM_weight = KMM_Lin.compute_kmm(path_source_train, path_target_train, path_source_train_label)
    # KMM_weight = torch.from_numpy(KMM_weight).float().to(DEVICE)
    # print(KMM_weight)


    optimizer = torch.optim.Adam([
            {'params': model.base_network.parameters(), 'lr' : 100 * opt.lr},
            {'params': model.base_network.avgpool.parameters(), 'lr' : 100 * opt.lr},
            {'params': model.bottle_layer.parameters(), 'lr' : 10 *  opt.lr},
            {'params': model.classifier_layer.parameters(), 'lr' :10 *  opt.lr},
        ], lr=opt.lr , betas=CFG['betas'], weight_decay=CFG['l2_decay'])

    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.16)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.85, verbose = False)
    #for params in model.parameters():
        #print(params.data)
    
    '''
    optimizer = torch.optim.SGD([
        {'params': model.base_network.convm1_layer.parameters(), 'lr' : 500 * opt.lr},
        {'params': model.classifier_layer.parameters(), 'lr' : 100 * opt.lr},
        {'params': model.bottleneck_layer.parameters(), 'lr' : 100 * opt.lr},
    ], lr=opt.lr, momentum=0.9, weight_decay=CFG['l2_decay'])
    '''
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.16)
    with open('D:\\Tang\\source_model_training_record'+str(opt.epoch)+opt.save_parameter_path_name+'.txt', 'a') as f:
    	f.write('\nfruit = ')
    	f.write(source_name)
    	f.write('\nlayer = ')
    	f.write(opt.extracted_layer)
    np_log = train(source_loader, test_flag, target_test_loader, model, CFG, optimizer)
    #np.savetxt(opt.save_train_loss_name, np_log, delimiter=',', fmt='%.12f')

    # Test

    #np_test = test(model, target_test_loader, test_flag, opt.mu)
    #np.savetxt(opt.save_test_name, np_log, delimiter=',', fmt='%.6f')
 
    save_parameter_add = os.path.join(save_parameter_path + str(opt.save_parameter_path_name))
    torch.save(model.state_dict(), save_parameter_add)