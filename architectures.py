import torch
import torch.nn as nn
    
class basic(nn.Module):
    def __init__(self):
        super(basic, self).__init__()
        self.fc1 = nn.Linear(1190, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class cnn(nn.Module):
    conv1_size = 16
    conv2_size = 32
    conv3_size = 64
    conv4_size = 128
    reduction_size = 37

    def __init__(self):
        super(cnn, self).__init__()
    
        self.hex_cnn_layer = nn.Sequential(
            nn.Conv3d(1, cnn.conv1_size, kernel_size=(5, 5, 13), padding=(2, 2, 0)),
            nn.BatchNorm3d(cnn.conv1_size),
            nn.Tanh(),
            nn.Conv3d(cnn.conv1_size, cnn.conv2_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(cnn.conv2_size),
            nn.ReLU(),
            nn.Conv3d(cnn.conv2_size , cnn.conv3_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(cnn.conv3_size),
            nn.ReLU(),
            nn.Conv3d(cnn.conv3_size , cnn.conv4_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(cnn.conv4_size),
            nn.ReLU()
        )
        self.num_cnn_layer = nn.Sequential(
            nn.Conv2d(1, cnn.conv1_size, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(cnn.conv1_size),
            nn.Tanh(),
            nn.Conv2d(cnn.conv1_size, cnn.conv2_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn.conv2_size),
            nn.ReLU(),
            nn.Conv2d(cnn.conv2_size, cnn.conv3_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(cnn.conv3_size),
            nn.ReLU(),
            nn.Conv2d(cnn.conv3_size, cnn.conv4_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(cnn.conv4_size),
            nn.ReLU()
        )
        self.reduce_num = nn.Linear(cnn.conv4_size * 3 * 3, cnn.reduction_size)
        self.reduce_hex = nn.Linear(cnn.conv4_size * 3 * 3, cnn.reduction_size * 13)

        self.fc1 = nn.Linear(14 * cnn.reduction_size + 4 * (54 + 72), 4096)
        #self.fc1 = nn.Linear(2 * 4608 + 16, 8000)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[13 * 49, 14 * 49], dim=1)

        hex = hex.view(-1, 1, 7, 7, 13)
        hex = self.hex_cnn_layer(hex).view(-1, cnn.conv4_size * 3 * 3)
        hex = self.reduce_hex(hex)

        num = num.view(-1, 1, 7, 7)
        num = self.num_cnn_layer(num).view(-1, cnn.conv4_size * 3 * 3)
        num = self.reduce_num(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
#The code below was used to test different architectures during development 
#It server no purpose other than reminding me of all my suboptimal attemps
'''
class ffnn1(nn.Module):
    def __init__(self):
        super(ffnn1, self).__init__()
        self.fc1 = nn.Linear(534, 1000)
        self.fc2 = nn.Linear(1000, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ffnn2(nn.Module):
    def __init__(self):
        super(ffnn2, self).__init__()
        self.fc1 = nn.Linear(534, 1000)
        self.fc2 = nn.Linear(1000, 2000)
        self.fc3 = nn.Linear(2000, 4000)
        self.fc4 = nn.Linear(4000, 2000)
        self.fc5 = nn.Linear(2000, 1000)
        self.fc6 = nn.Linear(1000, 500)
        self.fc7 = nn.Linear(500, 200)
        self.fc8 = nn.Linear(200, 100)
        self.fc9 = nn.Linear(100, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = self.fc9(x)
        return x
    
class cnn1(nn.Module):
    conv_size = 1000
    reduction_size = 37
    f = nn.CrossEntropyLoss()

    def __init__(self):
        super(cnn1, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, cnn1.conv_size, 3),
            #nn.BatchNorm2d(cnn1.conv_size),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.reduce = nn.Linear(cnn1.conv_size * 3 * 3, cnn1.reduction_size)
        self.fc1  = nn.Linear(2 * cnn1.reduction_size + 16,   1000)
        self.fc2  = nn.Linear(1000, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[49, 98], dim=1)

        hex = hex.view(-1, 1, 7, 7)
        num = num.view(-1, 1, 7, 7)
        hex = self.cnn_layer(hex).view(-1, cnn1.conv_size * 3 * 3)
        num = self.cnn_layer(num).view(-1, cnn1.conv_size * 3 * 3)
        hex = self.reduce(hex)
        num = self.reduce(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class cnn2(nn.Module):
    conv1_size = 20
    conv2_size = 1000
    reduction_size = 37
    f = nn.CrossEntropyLoss()

    def __init__(self):
        super(cnn2, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, cnn2.conv1_size, 3, padding=1),
            #nn.BatchNorm2d(cnn2.conv1_size),
            nn.ReLU(),
            nn.Conv2d(cnn2.conv1_size, cnn2.conv2_size, 3),
            #nn.BatchNorm2d(cnn2.conv2_size),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.reduce = nn.Linear(cnn2.conv2_size * 3 * 3, cnn2.reduction_size)
        self.fc1  = nn.Linear(2 * cnn2.reduction_size + 16,   1000)
        self.fc2  = nn.Linear(1000, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[49, 98], dim=1)

        hex = hex.view(-1, 1, 7, 7)
        num = num.view(-1, 1, 7, 7)
        hex = self.cnn_layer(hex).view(-1, cnn2.conv2_size * 3 * 3)
        num = self.cnn_layer(num).view(-1, cnn2.conv2_size * 3 * 3)
        hex = self.reduce(hex)
        num = self.reduce(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class cnn3(nn.Module):
    conv1_size = 10
    conv2_size = 50
    conv3_size = 100
    reduction_size = 37
    f = nn.CrossEntropyLoss()

    def __init__(self):
        super(cnn3, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, cnn3.conv1_size, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn3.conv1_size, cnn3.conv2_size, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn3.conv2_size, cnn3.conv3_size, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )
        self.reduce = nn.Linear(cnn3.conv3_size * 3 * 3, cnn3.reduction_size)
        self.fc1  = nn.Linear(2 * cnn3.reduction_size + 16,   1000)
        self.fc2  = nn.Linear(1000, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[49, 98], dim=1)

        hex = hex.view(-1, 1, 7, 7)
        num = num.view(-1, 1, 7, 7)
        hex = self.cnn_layer(hex).view(-1, cnn3.conv3_size * 3 * 3)
        num = self.cnn_layer(num).view(-1, cnn3.conv3_size * 3 * 3)
        hex = self.reduce(hex)
        num = self.reduce(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class cnn4(nn.Module):
    conv1_size = 10
    conv2_size = 20
    conv3_size = 30
    conv4_size = 50
    reduction_size = 37
    f = nn.CrossEntropyLoss()

    def __init__(self):
        super(cnn4, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, cnn4.conv1_size, 5, padding=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(cnn4.conv1_size, cnn4.conv2_size, 4),
            nn.ReLU(),
            nn.Conv2d(cnn4.conv2_size, cnn4.conv3_size, 3),
            nn.ReLU(),
            nn.Conv2d(cnn4.conv3_size, cnn4.conv4_size, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.reduce = nn.Linear(cnn4.conv4_size * 3 * 3, cnn4.reduction_size)
        self.fc1  = nn.Linear(2 * cnn4.reduction_size + 16,   1000)
        self.fc2  = nn.Linear(1000, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[49, 98], dim=1)

        hex = hex.view(-1, 1, 7, 7)
        num = num.view(-1, 1, 7, 7)
        hex = self.cnn_layer(hex).view(-1, cnn4.conv4_size * 3 * 3)
        num = self.cnn_layer(num).view(-1, cnn4.conv4_size * 3 * 3)
        hex = self.reduce(hex)
        num = self.reduce(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class num_cnn(nn.Module):
    conv1_size = 10
    conv2_size = 20
    conv3_size = 30
    reduction_size = 37
    f = nn.CrossEntropyLoss()

    def __init__(self):
        super(num_cnn, self).__init__()
        
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, num_cnn.conv1_size, 5, padding=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(num_cnn.conv1_size, num_cnn.conv2_size, 4),
            nn.ReLU(),
            nn.Conv2d(num_cnn.conv2_size, num_cnn.conv3_size, 3),
            nn.ReLU(),
            nn.BatchNorm2d(num_cnn.conv3_size)
        )
        self.reduce = nn.Linear(num_cnn.conv3_size * 6 * 6, num_cnn.reduction_size)
        self.fc1 = nn.Linear(13 * 37 + num_cnn.reduction_size + 16, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[13 * 37, 13 * 37 + 49], dim=1)


        num = num.view(-1, 1, 7, 7)
        num = self.cnn_layer(num).view(-1, num_cnn.conv3_size * 6 * 6)
        num = self.reduce(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class ultimate(nn.Module):
    conv1_size = 50#50
    conv2_size = 100#100
    conv3_size = 200#400
    reduction_size = 37
    f = nn.CrossEntropyLoss()

    def __init__(self):
        super(ultimate, self).__init__()
    
        self.hex_cnn_layer = nn.Sequential(
            nn.Conv3d(1, ultimate.conv1_size, (5, 5, 13), padding=(4, 4, 0)),
            nn.ReLU(),
            nn.Conv3d(ultimate.conv1_size, ultimate.conv2_size, (4, 4, 1)),
            nn.ReLU(),
            nn.Conv3d(ultimate.conv2_size, ultimate.conv3_size, (3, 3, 1)),
            nn.ReLU(),
            #nn.BatchNorm3d(ultimate.conv3_size)
        )
        self.num_cnn_layer = nn.Sequential(
            nn.Conv2d(1, ultimate.conv1_size, 5, padding=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(ultimate.conv1_size, ultimate.conv2_size, 4),
            nn.ReLU(),
            nn.Conv2d(ultimate.conv2_size, ultimate.conv3_size, 3),
            nn.ReLU(),
            #nn.BatchNorm2d(ultimate.conv3_size)
        )
        self.reduce_num = nn.Linear(ultimate.conv3_size * 6 * 6, ultimate.reduction_size)
        self.reduce_hex = nn.Linear(ultimate.conv3_size * 6 * 6, ultimate.reduction_size * 13)

        self.fc1 = nn.Linear(14 * ultimate.reduction_size + 16, 2500)
        self.fc2 = nn.Linear(2500, 800)
        self.fc3 = nn.Linear(800, 250)
        self.fc4 = nn.Linear(250, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[13 * 49, 14 * 49], dim=1)

        hex = hex.view(-1, 1, 7, 7, 13)
        hex = self.hex_cnn_layer(hex).view(-1, ultimate.conv3_size * 6 * 6)
        hex = self.reduce_hex(hex)

        num = num.view(-1, 1, 7, 7)
        num = self.num_cnn_layer(num).view(-1, ultimate.conv3_size * 6 * 6)
        num = self.reduce_num(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
class too_basic(nn.Module):
    def __init__(self):
        super(too_basic, self).__init__()
        self.fc1 = nn.Linear(702, 2500)
        self.fc2 = nn.Linear(2500, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class final(nn.Module):
    conv1_size = 16
    conv2_size = 32
    conv3_size = 64
    conv4_size = 128
    reduction_size = 37

    def __init__(self):
        super(final, self).__init__()
    
        self.hex_cnn_layer = nn.Sequential(
            nn.Conv3d(1, final.conv1_size, kernel_size=(5, 5, 13), padding=(2, 2, 0)),
            nn.BatchNorm3d(final.conv1_size),
            nn.Tanh(),
            nn.Conv3d(final.conv1_size, final.conv2_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(final.conv2_size),
            nn.ReLU(),
            nn.Conv3d(final.conv2_size , final.conv3_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(final.conv3_size),
            nn.ReLU(),
            nn.Conv3d(final.conv3_size , final.conv4_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(final.conv4_size),
            nn.ReLU()
        )
        self.num_cnn_layer = nn.Sequential(
            nn.Conv2d(1, final.conv1_size, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(final.conv1_size),
            nn.Tanh(),
            nn.Conv2d(final.conv1_size, final.conv2_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(final.conv2_size),
            nn.ReLU(),
            nn.Conv2d(final.conv2_size, final.conv3_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(final.conv3_size),
            nn.ReLU(),
            nn.Conv2d(final.conv3_size, final.conv4_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(final.conv4_size),
            nn.ReLU()
        )
        self.reduce_num = nn.Linear(final.conv4_size * 3 * 3, final.reduction_size)
        self.reduce_hex = nn.Linear(final.conv4_size * 3 * 3, final.reduction_size * 13)

        self.fc1 = nn.Linear(14 * final.reduction_size + 16, 4096)
        #self.fc1 = nn.Linear(2 * 4608 + 16, 8000)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[13 * 49, 14 * 49], dim=1)

        hex = hex.view(-1, 1, 7, 7, 13)
        hex = self.hex_cnn_layer(hex).view(-1, final.conv4_size * 3 * 3)
        hex = self.reduce_hex(hex)

        num = num.view(-1, 1, 7, 7)
        num = self.num_cnn_layer(num).view(-1, final.conv4_size * 3 * 3)
        num = self.reduce_num(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class final2(nn.Module):
    conv1_size = 32
    conv2_size = 64
    conv3_size = 64
    conv4_size = 64
    conv5_size = 64
    conv6_size = 64
    conv7_size = 64
    conv8_size = 128
    conv9_size = 256
    reduction_size = 37

    def __init__(self):
        super(final2, self).__init__()
    
        self.hex_cnn_layer = nn.Sequential(
            nn.Conv3d(1, final2.conv1_size, kernel_size=(5, 5, 13), padding=(2, 2, 0)),
            nn.BatchNorm3d(final2.conv1_size),
            nn.Tanh(),
            nn.Conv3d(final2.conv1_size, final2.conv2_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(final2.conv2_size),
            nn.ReLU(),
            nn.Conv3d(final2.conv2_size , final2.conv3_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(final2.conv3_size),
            nn.ReLU(),
            nn.Conv3d(final2.conv3_size , final2.conv4_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(final2.conv4_size),
            nn.ReLU(),
            nn.Conv3d(final2.conv4_size , final2.conv5_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(final2.conv5_size),
            nn.ReLU(),
            nn.Conv3d(final2.conv5_size , final2.conv6_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(final2.conv6_size),
            nn.ReLU(),
            nn.Conv3d(final2.conv6_size , final2.conv7_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(final2.conv7_size),
            nn.ReLU(),
            nn.Conv3d(final2.conv7_size, final2.conv8_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(final2.conv8_size),
            nn.ReLU(),
            nn.Conv3d(final2.conv8_size, final2.conv9_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(final2.conv9_size),
            nn.ReLU()
        )
        self.num_cnn_layer = nn.Sequential(
            nn.Conv2d(1, final2.conv1_size, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(final2.conv1_size),
            nn.Tanh(),
            nn.Conv2d(final2.conv1_size, final2.conv2_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(final2.conv2_size),
            nn.ReLU(),
            nn.Conv2d(final2.conv2_size, final2.conv3_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(final2.conv3_size),
            nn.ReLU(),
            nn.Conv2d(final2.conv3_size, final2.conv4_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(final2.conv4_size),
            nn.ReLU(),
            nn.Conv2d(final2.conv4_size, final2.conv5_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(final2.conv5_size),
            nn.ReLU(),
            nn.Conv2d(final2.conv5_size, final2.conv6_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(final2.conv6_size),
            nn.ReLU(),
            nn.Conv2d(final2.conv6_size, final2.conv7_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(final2.conv7_size),
            nn.ReLU(),
            nn.Conv2d(final2.conv7_size, final2.conv8_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(final2.conv8_size),
            nn.ReLU(),
            nn.Conv2d(final2.conv8_size, final2.conv9_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(final2.conv9_size),
            nn.ReLU()
        )
        self.reduce_num = nn.Linear(final2.conv9_size * 3 * 3, final2.reduction_size)
        self.reduce_hex = nn.Linear(final2.conv9_size * 3 * 3, final2.reduction_size * 13)

        self.fc1 = nn.Linear(14 * final2.reduction_size + 16, 4096)
        #self.fc1 = nn.Linear(2 * 4608 + 16, 8000)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[13 * 49, 14 * 49], dim=1)

        hex = hex.view(-1, 1, 7, 7, 13)
        hex = self.hex_cnn_layer(hex).view(-1, final2.conv9_size * 3 * 3)
        hex = self.reduce_hex(hex)

        num = num.view(-1, 1, 7, 7)
        num = self.num_cnn_layer(num).view(-1, final2.conv9_size * 3 * 3)
        num = self.reduce_num(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class newest(nn.Module):
    conv1_size = 16
    conv2_size = 32
    conv3_size = 64
    conv4_size = 128
    reduction_size = 37

    def __init__(self):
        super(newest, self).__init__()
    
        self.hex_cnn_layer = nn.Sequential(
            nn.Conv3d(1, newest.conv1_size, kernel_size=(5, 5, 13), padding=(2, 2, 0)),
            nn.BatchNorm3d(newest.conv1_size),
            nn.Tanh(),
            nn.Conv3d(newest.conv1_size, newest.conv2_size, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(newest.conv2_size),
            nn.ReLU(),
            nn.Conv3d(newest.conv2_size , newest.conv3_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(newest.conv3_size),
            nn.ReLU(),
            nn.Conv3d(newest.conv3_size , newest.conv4_size, kernel_size=(3, 3, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(newest.conv4_size),
            nn.ReLU()
        )
        self.num_cnn_layer = nn.Sequential(
            nn.Conv2d(1, newest.conv1_size, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(newest.conv1_size),
            nn.Tanh(),
            nn.Conv2d(newest.conv1_size, newest.conv2_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(newest.conv2_size),
            nn.ReLU(),
            nn.Conv2d(newest.conv2_size, newest.conv3_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(newest.conv3_size),
            nn.ReLU(),
            nn.Conv2d(newest.conv3_size, newest.conv4_size, kernel_size=(3, 3), padding=(0, 0)),
            nn.BatchNorm2d(newest.conv4_size),
            nn.ReLU()
        )
        self.reduce_num = nn.Linear(newest.conv4_size * 3 * 3, newest.reduction_size)
        self.reduce_hex = nn.Linear(newest.conv4_size * 3 * 3, newest.reduction_size * 13)

        self.fc1 = nn.Linear(14 * newest.reduction_size + 32, 4096)
        #self.fc1 = nn.Linear(2 * 4608 + 16, 8000)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 4)
    
    def forward(self, x):
        hex, num, placements = torch.tensor_split(input=x, indices=[13 * 49, 14 * 49], dim=1)

        hex = hex.view(-1, 1, 7, 7, 13)
        hex = self.hex_cnn_layer(hex).view(-1, newest.conv4_size * 3 * 3)
        hex = self.reduce_hex(hex)

        num = num.view(-1, 1, 7, 7)
        num = self.num_cnn_layer(num).view(-1, newest.conv4_size * 3 * 3)
        num = self.reduce_num(num)

        x = torch.cat(tensors=(hex, num, placements), dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
'''