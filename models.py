import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torch.nn.init as init
import torch
from collections import OrderedDict
import torchvision.models as models_tv
from network.pytorch_i3d import Unit3D


class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


'''class feature_extractor(nn.Module):

    def __init__(self):
        super(feature_extractor, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.feature.add_module('f_pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_relu1', nn.ReLU())
        self.feature.add_module('f_conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.feature.add_module('f_pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        self.feature.add_module('f_relu2', nn.ReLU())
        self.feature.add_module('f_conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.feature.add_module('f_relu3', nn.ReLU())
        self.feature.add_module('f_pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.initialize_params()

    def forward(self, input_data):
        features = self.feature(input_data)

        return features

    def initialize_params(self):

        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()'''


class task_classifier(nn.Module):

    def __init__(self, spatial_squeeze = True, dropout_keep_prob=0.5, num_classes = 8):
        super(task_classifier, self).__init__()
        self._spatial_squeeze = spatial_squeeze
        self.dropout_keep_prob = dropout_keep_prob
        self._num_classes = num_classes
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits_task_classifier')
    def forward(self, x):
        logits = self.logits(self.dropout(x))
        if self._spatial_squeeze:
            logits = logits.squeeze(3).squeeze(3)
        return logits



class domain_discriminator(nn.Module):

    def __init__(self, rp_size, optimizer, lr, momentum, weight_decay, n_outputs=512):
        super(domain_discriminator, self).__init__()
        rp_size = 512
        n_outputs = 1024
        fl = 3 #512

        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc1', nn.Linear(rp_size, 512))
        self.domain_discriminator.add_module('d_relu1', nn.ReLU())
        self.domain_discriminator.add_module('d_drop1', nn.Dropout(0.2))

        self.domain_discriminator.add_module('d_fc2', nn.Linear(512, 256))
        self.domain_discriminator.add_module('d_relu2', nn.ReLU())
        self.domain_discriminator.add_module('d_drop2', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d_fc3', nn.Linear(256, 2))
        self.domain_discriminator.add_module('d_sfmax', nn.LogSoftmax(dim=1))
        # self.domain_discriminator.add_module('d_relu2', nn.ReLU())
        # self.domain_discriminator.add_module('d_drop2', nn.Dropout())
        # self.domain_discriminator.add_module('d_fc3', nn.Linear(1024, 1))

        self.optimizer = optimizer(list(self.domain_discriminator.parameters()), lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)

        self.initialize_params()

        # TODO Check the RP size
        self.projection = nn.Linear(n_outputs, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

    def forward(self, input_data):
        # reverse_feature = ReverseLayer.apply(input_data, alpha)	# Make sure there will be no problem when updating discs params
        #print('on est dans forward')
        #print(input_data.shape)
        feature = input_data.view(input_data.size(0), -1)
        #print(feature.shape)
        feature_proj = self.projection(feature)
        #print(feature_proj.shape)

        domain_output = self.domain_discriminator(feature_proj)
        #print(domain_output.shape)

        #print('\n\n\n')

        return domain_output

    def initialize_params(self):

        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class domain_discriminator_ablation_RP(nn.Module):

    def __init__(self, optimizer, lr, momentum, weight_decay):
        super(domain_discriminator_ablation_RP, self).__init__()
        n_outputs = 512
        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc1', nn.Linear(n_outputs, 512))
        self.domain_discriminator.add_module('d_relu1', nn.ReLU())
        self.domain_discriminator.add_module('d_drop1', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d_fc2', nn.Linear(512, 256))
        self.domain_discriminator.add_module('d_relu2', nn.ReLU())
        self.domain_discriminator.add_module('d_drop2', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d_fc3', nn.Linear(256, 2))
        # self.domain_discriminator.add_module('d_drop2', nn.Dropout(0.2))
        self.domain_discriminator.add_module('d1_sfmax', nn.LogSoftmax(dim=1))

        self.optimizer = optimizer(list(self.domain_discriminator.parameters()), lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)

        self.initialize_params()

    def forward(self, input_data):
        feature = input_data.view(input_data.size(0), -1)
        domain_output = self.domain_discriminator(feature)

        return domain_output

    def initialize_params(self):

        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()


class AlexNet(nn.Module):
    def __init__(self, baseline=True, num_classes=7):
        super(AlexNet, self).__init__()

        self.num_classes = num_classes
        self.baseline = baseline
        '''
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(5, 1e-4, 0.75),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.LocalResponseNorm(5, 1e-4, 0.75),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),			
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

        #self.classifier_added = nn.Sequential(
            #nn.ReLU(),
        #	nn.Dropout(),
        #	nn.Linear(4096, self.num_classes),
        #)
        '''
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))

        if self.baseline:
            self.classifier = nn.Sequential(OrderedDict([
                ("fc6", nn.Linear(256 * 6 * 6, 4096)),
                ("relu6", nn.ReLU(inplace=True)),
                ("drop6", nn.Dropout()),
                ("fc7", nn.Linear(4096, 4096)),
                ("relu7", nn.ReLU(inplace=True)),
                ("drop7", nn.Dropout()),
                ("fc8", nn.Linear(4096, self.num_classes))]))

        else:
            self.classifier = nn.Sequential(OrderedDict([
                ("fc6", nn.Linear(256 * 6 * 6, 4096)),
                ("relu6", nn.ReLU(inplace=True)),
                ("drop6", nn.Dropout()),
                ("fc7", nn.Linear(4096, 4096)),
                ("relu7", nn.ReLU(inplace=True)),
                ("drop7", nn.Dropout())]))

        self.initialize_params()

    def initialize_params(self):

        for layer in self.modules():
            # if isinstance(layer, torch.nn.Conv2d):
            # init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            # layer.bias.data.zero_()
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.1)
                layer.bias.data.zero_()

    # elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
    # layer.weight.data.fill_(1)
    # layer.bias.data.zero_()

    def forward(self, x):
        x = self.features(x * 57.6)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


## Add resnet18

# bypass layer
class Identity(nn.Module):
    def __init__(self, n_inputs):
        super(Identity, self).__init__()
        self.in_features = n_inputs

    def forward(self, x):
        return x


from network.pytorch_i3d import InceptionI3d 
from network.stam.models  import create_model as create_stam
from network.x3d import make_x3d_model as create_x3d


def get_pretrained_model(args):
    model_name = args.model_name

    if model_name == 'r2p1d':
        model = models_tv.resnet18(pretrained=True)
        # n_classes= 7
        n_inputs = model.fc.in_features
        n_outputs = model.fc.in_features
        # print(n_inputs)
        model.fc = Identity(n_inputs)
    elif model_name == 'x3d':
        model = AlexNet(num_classes=7, baseline=False)
        state_dict = torch.load("./alexnet_caffe.pth.tar")
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
        not_loaded = model.load_state_dict(state_dict, strict=False)
        n_outputs = 4096
    else:

        if args.middle_mode == 'flow':
            i3d = InceptionI3d(400, in_channels=2)
            i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
        else:
            i3d = InceptionI3d(400, in_channels=3)
            i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'), strict=False)

        i3d.replace_logits(args.num_classes)
        model = i3d

    return model


#if __name__ == '__main__':
    # state_dict = torch.load("./alexnet_caffe.pth.tar")
    # feature_extractor = AlexNet(baseline=True)

    #feature_extractor = get_pretrained_model('resnet18')
    #
    # not_loaded = feature_extractor.load_state_dict(state_dict, strict = False)

    # print(not_loaded)

    #task_classifier = task_classifier()

    # domain_discriminator_list = []
    # for i in range(3):
    #    	disc = domain_discriminator(optim.SGD, 0.1, 0.9, 0.001).train()
    #        domain_discriminator_list.append(disc)

    #batch = torch.rand(3, 3, 225, 225)

    #z = feature_extractor(batch)
    # task_y = task_classifier(z)

    # domain_y = []

    # for disc in domain_discriminator_list:
    #	domain_y.append(disc(z))

    #print(z.size())
