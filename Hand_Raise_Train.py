#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import random
import glob
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import mygrad


# In[2]:


thumbs_up_p = glob.glob(r"C:\Users\fkora\GitHub\data\hand_data\thumbs_up\*HEIC")
flat_p = glob.glob(r"C:\Users\fkora\GitHub\data\hand_data\flat\*HEIC")


# In[3]:


flat_x = [np.array(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0) / torch.norm(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0))) for path in flat_p]
flat_y = torch.zeros(len(flat_x), dtype=torch.long)

thumbs_up_x = [np.array(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0) / torch.norm(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0))) for path in thumbs_up_p]
thumbs_up_y = torch.ones(len(thumbs_up_x), dtype=torch.long)


# In[4]:


x_train = np.concatenate((flat_x, thumbs_up_x)).reshape(110, 3, 160, 160)
y_train = torch.cat((flat_y, thumbs_up_y))


# In[5]:


x_train.shape


# In[6]:


y_train.shape


# In[20]:


animal_p = glob.glob(r"C:\Users\fkora\GitHub\data\dogs-vs-cats\train\*JPG")


# In[22]:


x_train = np.array([np.array(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0) / torch.norm(transforms.ToTensor()(transforms.Resize((160, 160))(Image.open(path))).unsqueeze(0))) for path in animal_p])


# In[31]:


x_train = x_train.reshape(25000, 3, 160, 160)


# In[23]:


y_train = torch.cat((torch.zeros(12500, dtype=torch.long), torch.ones(12500, dtype=torch.long)))


# In[24]:


resnet = InceptionResnetV1(classify=True, num_classes=2).eval()
optim = torch.optim.Adam(resnet.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()


# In[14]:


from noggin import create_plot
plotter, fig, ax = create_plot(metrics=["loss"])


# In[32]:


from livelossplot import PlotLosses
plotlosses = PlotLosses()


# In[35]:


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
        
        #plotter.set_train_batch({"loss" : loss.item()}, batch_size=20, plot=True)
    #plotter.set_train_epoch()
#plotter.show()


# In[52]:


pred = resnet(torch.Tensor(x_train[0]).reshape(1, 3, 160, 160))
pred


# In[ ]:




