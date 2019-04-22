import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def avg_rates(y_test, y_pred, num_classes = 10):
    total_correct = 0
    num_samples, num_classified_correctly = np.zeros((num_classes, 1)), np.zeros((num_classes, 1))

    for i, y in enumerate(y_test):

        num_samples[y]+=1
        if y==y_pred[i]:
            total_correct+=1
            num_classified_correctly[y] += 1

    avg_classification_rate = total_correct/np.float(len(y_test))
    avg_class_classification_rate = num_classified_correctly/num_samples

    return avg_classification_rate, avg_class_classification_rate

def main():
    def getAccuracy():
        y_pred = np.array([])
        correct = 0
        total = 0    
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(testloader, 0):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                labels = torch.tensor(labels, dtype=torch.long, device=device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred = np.append(y_pred, predicted.cpu().detach().numpy())
                # y_pred.append(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            net.train()
        print('Accuracy of the network on the 10000 test images: %.3f %% ' % (100 * correct / total))
        return y_pred

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    x_train = np.load("data/x_train.npy")
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    y_train = np.load("data/y_train.npy")

    x_test = np.load("data/x_test.npy")
    x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
    y_test = np.load("data/y_test.npy")

    x_train_list = [np.array(x) for x in x_train]
    y_train_list = [np.array(y) for y in y_train]
    x_test_list = [np.array(x) for x in x_test]
    y_test_list = [np.array(y) for y in y_test]

    x_t_train = torch.stack([torch.Tensor(i.reshape(-1, 28, 28)) for i in x_train_list]) # transform to torch tensors
    y_t_train = torch.stack([torch.from_numpy(i) for i in y_train_list])
    x_t_test = torch.stack([torch.Tensor(i.reshape(-1, 28, 28)) for i in x_test_list]) # transform to torch tensors
    y_t_test = torch.stack([torch.from_numpy(i) for i in y_test_list])

    trainset = utils.TensorDataset(x_t_train,y_t_train) # create your datset
    testset = utils.TensorDataset(x_t_test,y_t_test) # create your datset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                            shuffle=False, num_workers=2)

    classes = ('T-shirt/top','Trouser','Pullover','Dress',
            'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    batch_size = 200
    epochs = 30

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            
            outputs = net(inputs)
            #print(type(outputs), type(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000
            optimizer.step()        
            # print statistics
            running_loss += loss.item()
            interval = int(50000/(6 * batch_size))
            if i % interval == (interval-1):    # print every interval mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / interval))
                running_loss = 0.0
        #torch.save(net, 'model.ckpt')
        if epoch % 2 == 1:
            y_pred = getAccuracy()
            
    class_names = np.array(["T-shirt/top","Trouser","Pullover","Dress",
        "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
    
    #print('Average Classification Rate', avg_class_rate)
    
    v1, v2 = avg_rates(y_test, y_pred)
    print(v1, v2)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show()



if __name__ == '__main__':
    main()
    pass
    