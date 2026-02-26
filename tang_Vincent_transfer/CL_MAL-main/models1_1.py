import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import backbone_multi
import call_resnet18_multi as cl
import estimate_mu as es


# ResNet-18在最初時沒有使用bottleneck
class Transfer_Net(nn.Module):


    def __init__(self, num_class, base_net='resnet18_multi_new', transfer_loss='cmmd', use_bottleneck=True, bottleneck_width=128, width=512):
        super(Transfer_Net, self).__init__()

        self.base_network = backbone_multi.network_dict[base_net]() # 從backbone已經定義好的resnet-18到fully-connected之前
        self.use_bottleneck = use_bottleneck # 是否用bottleneck (這邊指的不是resnet-18的bottleneck，而是論文DDC的fully-connected)
        self.transfer_loss = transfer_loss
        
        if base_net == 'resnet18_multi_new':
            self.base_network = cl.load_resnet18_multi()
        
        
        #classifier_layer_list = [nn.Linear(256, width), nn.ReLU(), nn.Dropout(0.15), nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, 2)]
        bottle_list = [nn.Linear(256, 64), nn.Linear(64, 64), nn.Linear(64, 256), nn.Linear(256, 128)]
        classifier_list = [nn.Dropout(0.25), nn.Linear(bottleneck_width, num_class)]

        self.bottle_layer = nn.Sequential(*bottle_list)
        self.classifier_layer = nn.Sequential(*classifier_list)

        
        self.bottle_layer[0].weight.data.normal_(0, 0.01)
        self.bottle_layer[0].bias.data.fill_(0.0)
        self.bottle_layer[1].weight.data.normal_(0, 0.01)
        self.bottle_layer[1].bias.data.fill_(0.0)
        self.bottle_layer[2].weight.data.normal_(0, 0.01)
        self.bottle_layer[2].bias.data.fill_(0.0)
        self.bottle_layer[3].weight.data.normal_(0, 0.01)
        self.bottle_layer[3].bias.data.fill_(0.0)
        self.classifier_layer[1].weight.data.normal_(0, 0.01)
        self.classifier_layer[1].bias.data.fill_(0.0)


    def forward(self, source, s_label, test_flag):

        source = self.base_network(source, test_flag)
        source = self.bottle_layer(source)
        with torch.no_grad():
            source_bottle = source
        source_clf = self.classifier_layer(source)
        
        
        
        
        return source,source_clf

    def predict(self, x, test_flag):

        features = self.base_network(x, test_flag)
        features = self.bottle_layer(features)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss, s_label, t_label):

        if adapt_loss == 'mmd':
            transfer_loss = mmd.mmd_rbf_noaccelerate(X, Y)
        elif adapt_loss == 'coral':
            transfer_loss = CORAL(X, Y)
        elif adapt_loss == 'cmmd':
            cmmd_loss = 0
            mmd_loss = mmd.mmd_rbf_noaccelerate(X, Y)
            if self.training:
                cmmd_loss = Variable(torch.Tensor([0]))
                cmmd_loss = cmmd_loss.cuda()
                cmmd_loss = mmd.cmmd(X, Y, s_label, t_label)
                mu = mu = es.estimate_mu(X.detach().cpu().numpy(), s_label.detach().cpu().numpy(),Y.detach().cpu().numpy(), t_label.detach().cpu().numpy())
                                                                                                

            transfer_loss = (1- mu) * cmmd_loss + mu * mmd_loss
        else:
            transfer_loss = 0
        return transfer_loss
