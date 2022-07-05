import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

# from model import Net
# from mobilenet import MobileNetv2
# from ghost_net import ghost_net
from ShuffleNetV2 import shufflenet_v2_x0_5

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",
                    default=r'C:\Users\user\Desktop\projects\Yolov5_DeepSort_Pytorch-master\deep_sort\deep_sort\deep\market1501',
                    type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True


def load_model(model_path, net, gpu_id=None):  # 加载预训练权重
    if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = torch.device("cpu")
    if model_path is not None:
        pretrained_params = torch.load(model_path, map_location=device)

        pretrained_params = \
            {k: v for k, v in pretrained_params.items() if
             k in net.state_dict().keys() and net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(pretrained_params, strict=False)


# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64, shuffle=True
)
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))

# net definition
start_epoch = 0
# net = Net(num_classes=num_classes)
# net = MobileNetv2(num_classes=num_classes)
net = shufflenet_v2_x0_5(num_classes=num_classes)

# if args.resume:
#     assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
#     print('Loading from checkpoint/ckpt.t7')
#     checkpoint = torch.load("./checkpoint/ckpt.t7")
#     # import ipdb; ipdb.set_trace()
#     net_dict = checkpoint['net_dict']
#     net.load_state_dict(net_dict)
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

net.to(device)

# model_path = r"deep/checkpoint/mobilenet_v2-b0353104.pth"
# load_model(model_path, net, 0)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.


# train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.
            start = time.time()

    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for id, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (id + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total,
            100. * correct / total
        ))

    return train_loss / len(trainloader), 1. - correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total,
            100. * correct / total
        ))

    # saving checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t8")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t8')

    return test_loss / len(testloader), 1. - correct / total


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="Loss")
ax1 = fig.add_subplot(122, title="Top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train00.jpg")


# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range(start_epoch, start_epoch + 40):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch + 1) % 20 == 0:
            lr_decay()


if __name__ == '__main__':
    main()