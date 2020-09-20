import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import random
import glob
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import mygrad

thumbs_up_p = glob.glob(r"C:\Users\fkora\GitHub\data\hand_data\thumbs_up\*HEIC")
flat_p = glob.glob(r"C:\Users\fkora\GitHub\data\hand_data\flat\*HEIC")

flat_x = [np.array(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0) / torch.norm(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0))) for path in flat_p]
flat_y = torch.zeros(len(flat_x), dtype=torch.long)

thumbs_up_x = [np.array(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0) / torch.norm(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0))) for path in thumbs_up_p]
thumbs_up_y = torch.ones(len(thumbs_up_x), dtype=torch.long)

x_train = np.concatenate((flat_x, thumbs_up_x)).reshape(110, 3, 160, 160)
y_train = torch.cat((flat_y, thumbs_up_y))

resnet = InceptionResnetV1(classify=True, num_classes=2).eval()
optim = torch.optim.Adam(resnet.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

from livelossplot import PlotLosses
plotlosses = PlotLosses()

batch_size = 100

for _ in range(1):
    idxs = np.arange(len(x_train))
    np.random.shuffle(idxs)

    for batch_cnt in range(0, len(x_train)//batch_size):
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        batch = x_train[batch_indices]

        optim.zero_grad()

        pred = resnet(torch.Tensor(batch))
        truth = y_train[batch_indices]
        #print(pred)
        #print(truth)

        loss = loss_fn(pred, truth)
        #print(loss)

        loss.backward()
        optim.step()

        plotlosses.update({'loss': loss.item()})
        plotlosses.send()
