import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import os

original_model = models.alexnet(pretrained=True)

class AlexNetConv3(nn.Module):
            def __init__(self):
                super(AlexNetConv3, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv3
                    *list(original_model.features.children())[:-6]
                )
            def forward(self, x):
                x = self.features(x)
                return x

model = AlexNetConv3()
model.eval()

print('model initiated')


###########################################


N_database = 446
N_query = 760
w = h = 231

database = torch.Tensor(N_database,3,w,h)
query = torch.Tensor(N_query,3,w,h)

transform = transforms.Compose([transforms.Resize((w,h)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

for i, filename in enumerate(sorted(os.listdir("./slice4/database/class1/"))):
    if i < N_database:
        im=Image.open("./slice4/database/class1/" + filename)
        database[i] = transform(im)
        im.close()

for i, filename in enumerate(sorted(os.listdir("./slice4/query/class1/"))):
    if i < N_query:
        im=Image.open("./slice4/query/class1/" + filename)
        query[i] = transform(im)
        im.close()

print('images loaded')


###########################################


with torch.no_grad():
    output_database = model(database)
    output_query = model(query)

print('Feed forwarded')

database_reshaped = torch.reshape(output_database,(N_database,64896))
query_reshaped = torch.reshape(output_query,(N_query,64896))

cos = nn.CosineSimilarity(dim=0)
simularity_matrix = np.zeros([N_query,N_database])

for i in range(N_query):  
    for j in range(N_database):
        simularity_matrix[i,j] = cos(query_reshaped[i], database_reshaped[j])

print(database_reshaped.shape, 'reshaped database with', N_database , 'images')
print(query_reshaped.shape, 'shape of query with', N_query , 'images')
print(simularity_matrix.shape, 'shape of simularity matrix')

print(simularity_matrix)

with open('similarityMatrix.npy', 'wb') as f:
    np.save(f, simularity_matrix)