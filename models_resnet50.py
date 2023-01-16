import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
# from torchvision.models.resnet import resnet50

# # model = resnet50(pretrained = False)
# # model.fc = torch.nn.Linear(2048,43)
# # model.conv1 = torch.nn.Conv2d(3,64,kernel_size=5,stride=1)

# # model()
# # net_name = "Resnet50-Trafficsign"
# # net = model()

# Conv NN constructors:
class ConvNN(nn.Module):
    name = ""

    def __init__(self, params):
        """
        The purpose in using this class in the way it built is to make the process of creating CNNs with the ability
        to control its capacity in efficient way (programming efficiency - not time efficiency).
        I found it very useful in constructing experiments. I tried to make this class general as possible.
        :param params: a dictionary with the following attributes:
            capacity influence:
            - channels_lst: lst of the channels sizes. the network complexity is inflected mostly by this parameter
              * for efficiency channels_lst[0] is the number of input channels
            - #FC_Layers: number of fully connected layers
            - extras_blocks_components: in case we want to add layers from the list ["dropout", "max_pool", "batch norm"]
                                        to each block we can do it. Their parameters are attributes of this dict also.
              * notice that if max_pool in extras_blocks_components then we reduce dims using max_pool instead conv
                layer (the conv layer will be with stride 1 and padding)
            - p_dropout: the dropout parameter

            net structure:
            - in_wh: input width and height
        """
        super().__init__()
        self.params = params
        channels_lst = params["channels_lst"]
        extras_block_components = params["extras_blocks_components"]

        assert 2 <= len(channels_lst) <= 5
        conv_layers = []
        for i in range(1, len(channels_lst)):
            """
            Dims calculations: next #channels x (nh-filter_size/2)+1 x (nw-filter_size/2)+1
            """
            filter_size, stride, padding = (4, 2, 1) if "max_pool" not in extras_block_components else (5, 1, 2)
            conv_layers.append(nn.Conv2d(channels_lst[i - 1], channels_lst[i], filter_size, stride, padding, bias=False))
            conv_layers.append(params["activation"]())

            for comp in extras_block_components:
                if comp == "dropout":
                    conv_layers.append(nn.Dropout(params["p_dropout"]))
                if comp == "max_pool":
                    conv_layers.append(nn.MaxPool2d(2, 2))
                if comp == "batch_norm":
                    conv_layers.append(nn.BatchNorm2d(channels_lst[i]))

        out_channels = channels_lst[-1]
        if params["CNN_out_channels"] is not None:
            conv_layers.append(nn.Conv2d(channels_lst[-1], params["CNN_out_channels"], 1))
            conv_layers.append(params["activation"]())
            out_channels = params["CNN_out_channels"]

        self.cnn = nn.Sequential(*conv_layers)

        lin_layers = []
        wh = params["in_wh"] // (2 ** (len(channels_lst) - 1))  # width and height of last layer output
        lin_layer_width = out_channels * (wh ** 2)
        for _ in range(params["#FC_Layers"] - 1):
            lin_layers.append(nn.Linear(lin_layer_width, lin_layer_width))
        lin_layers.append(nn.Linear(lin_layer_width, params["out_size"]))
        self.linear_nn = nn.Sequential(*lin_layers)

        """use CE loss so we don't need to apply softmax (for test loss we also use the same CE. for accuracy
            we choose the highest value - this property is saved under softmax)"""

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(1, *x.shape)
        assert x.shape[2] == x.shape[3] == self.params["in_wh"]
        assert x.shape[1] == self.params["channels_lst"][0]

        cnn_output = self.cnn(x).view((x.shape[0], -1))
        lin_output = self.linear_nn(cnn_output)
        return lin_output


def create_conv_nn(params):
    return ConvNN(params)

class Resnet50TrafficSign(nn.Module):
    name = "Resnet50-TrafficSign"
    def __init__(self, in_channels = 3, out_channels=43):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.lin = nn.Sequential(
            nn.Linear(128, 84),
            nn.Tanh(),
            nn.Linear(84, 43)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.lin(x)
        return x


            # nn.Conv2d(in_channels, 32, 3, padding=1),  # 32
            # nn.ReLU(),
            # # nn.MaxPool2d(2, stride=2),  # 16
            # nn.Conv2d(32, 32, 3, padding=1),  # 16
            # nn.ReLU(),            
            # # nn.MaxPool2d(2, stride=2),  # 8
            # nn.Conv2d(32, 32, 3, padding = 1),  # 4
            # nn.Dropout(0.02),
            # nn.ReLU(),            

            # nn.Conv2d(32, 64, 3, padding=1),  # 4
            # nn.ReLU(),            
            # nn.Conv2d(64, 64,3, 1),
            # nn.ReLU(),           
            # nn.Conv2d(64, 64, 3, padding=1),  # 32
            # nn.ReLU(),
            # # nn.MaxPool2d(2, stride=2),  # 16
            # nn.Conv2d(64, 128, 3, padding=1),  # 16
            # nn.ReLU(),            # nn.MaxPool2d(2, stride=2),  # 8
            # nn.Conv2d(128, 128, 3, padding = 1),  # 4
            # nn.ReLU(),            
            # nn.Conv2d(128, 128, 3, padding=1),
            # nn.Dropout(0.02),
            # nn.ReLU(),
            # # nn.Conv2d(128, 256, 3, padding=1),  # 16
            # # nn.ReLU(),            
            # # nn.MaxPool2d(2, stride=2),  # 8
            # # nn.Conv2d(256, 256, 3, padding = 1),  # 4
            # # nn.ReLU(),            
            # # nn.Conv2d(256, 256, 3, padding=1),
            # # nn.Dropout(0.02),
            # # nn.ReLU(),            
        

    #     self.lin = nn.Sequential(
    #         nn.Linear(256, 84),
    #         nn.ReLU(),
    #         # nn.Dropout(0.1),
    #         nn.Linear(84, 43)
    #     )

    # def forward(self, x):
    #     x = self.cnn(x)  
    #     x = torch.flatten(x, 1)
    #     x = self.lin(x)
    #     return x

# class resnet50TrafficSign(nn.Module):
#     name = "resnet50-TrafficSign"
#     def __init__(self, in_channels=3, out_channels=43):
#         super().__init__()
#         self.cnn = nn.Sequential(            
#             nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(6, 16, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(16, 120, kernel_size=5, stride=1),
#             nn.Tanh()
#         )
#         self.lin = nn.Sequential(
#             nn.Linear(120, 84),
#             nn.Tanh(),
#             nn.Linear(84, 43)
#         )

#     def forward(self, x):
#         x = self.relu(self.batch_norm1(self.conv1(x)))
#         x = self.max_pool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)
        
#         return x


# class CNNMNISTNet(nn.Module):
#     name = "MNIST-NET"

#     def __init__(self):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1, stride=1),  # 28 -> 14
#             nn.ReLU(),
#             nn.Conv2d(32, 128, 3, padding=1, stride=2),  # 28 -> 14
#             nn.ReLU(),
#             nn.Dropout(p=0.05),
#             nn.Conv2d(128, 128, 3, padding=1, stride=1),  # 14 -> 7
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 3, padding=1, stride=2),  # 14 -> 7
#             nn.ReLU(),

#             nn.Dropout(p=0.05),
#             nn.Conv2d(256, 100, 4),  # 7 -> 4
#             nn.ReLU(),
#         )

#         self.lin = nn.Sequential(
#             nn.Linear(100 * 4 * 4, 100),
#             nn.ReLU(),
#             nn.Linear(100, 10)
#         )

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(-1, 100 * 4 * 4)
#         x = self.lin(x)
#         return x

#
#
# class STN1D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#     # Spatial transformer network forward function
#     def forward(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         return x
#
# class STN_MNISTNet(nn.Module):
#     name = "MNIST-NET"
#
#     def __init__(self):
#         super().__init__()
#         self.stn = STN1D()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 100, 3, padding=1),  # 28
#             nn.ReLU(),
#             nn.Conv2d(100, 100, 3, padding=1),  # 28
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.MaxPool2d(2, stride=2),  # 14
#             nn.Conv2d(100, 200, 3, padding=1),  # 14
#             nn.ReLU(),
#             nn.Conv2d(200, 200, 3, padding=1),  # 14
#             nn.Dropout(0.1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),  # 7
#             nn.Conv2d(200, 400, 5),  # 3
#             nn.ReLU(),
#             nn.Conv2d(400, 250, 3, padding=1),  # 3
#             nn.ReLU(),
#             nn.Conv2d(250, 100, 1),
#             nn.ReLU(),
#         )
#
#         self.lin = nn.Sequential(
#             nn.Linear(100 * 3 * 3, 350),
#             nn.Dropout(0.1),
#             nn.ReLU(),
#             nn.Linear(350, 10)
#         )
#
#     def forward(self, x):
#         x = self.stn(x)
#         x = self.cnn(x)
#         x = x.view(-1, 100 * 3 * 3)
#         x = self.lin(x)
#         return x
#
# class Stn(nn.Module):
#     """ from https://github.com/wolfapple/traffic-sign-recognition/blob/master/notebook.ipynb"""

#     def __init__(self):
#         super().__init__()
#         # localization network
#         self.loc_net = nn.Sequential(
#             nn.Conv2d(3, 50, 7),
#             nn.MaxPool2d(2, 2),
#             nn.ELU(),
#             nn.Conv2d(50, 100, 5),
#             nn.MaxPool2d(2, 2),
#             nn.ELU()
#         )
#         # regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(100 * 4 * 4, 100),
#             nn.ELU(),
#             nn.Linear(100, 3 * 2)
#         )
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#     def forward(self, x):
#         xs = self.loc_net(x)
#         xs = xs.view(-1, 100 * 4 * 4)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#         return x

# class STNTrafficsign(nn.Module):
#     name = "STN-TrafficSignNet"

#     def __init__(self, in_channels=3, out_channels=43):
#         super().__init__()
#         self.stn = Stn()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels, 100, 3, padding=1),  # 32
#             nn.ReLU(),
#             nn.Conv2d(100, 100, 3, padding=1),  # 32
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.MaxPool2d(2, stride=2),  # 16
#             nn.Conv2d(100, 200, 3, padding=1),  # 16
#             nn.ReLU(),
#             nn.Conv2d(200, 200, 3, padding=1),  # 16
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.MaxPool2d(2, stride=2),  # 8
#             nn.Conv2d(200, 400, 5),  # 4
#             nn.ReLU(),
#             nn.Conv2d(400, 250, 3, padding=1),  # 4
#             nn.ReLU(),
#             nn.Conv2d(250, 100, 1),
#             nn.ReLU(),
#         )

#         self.lin = nn.Sequential(
#             nn.Linear(100 * 4 * 4, 350),
#             nn.ReLU(),
#             nn.Linear(350, out_channels)
#         )

#     def forward(self, x):
#         x = self.stn(x)
#         x = self.cnn(x)
#         x = x.view(-1, 100 * 4 * 4)
#         x = self.lin(x)
#         return x
