# model.py

# Once we have engine.py, we are ready to create a new file: model.py. model.py will
# consist of our model. It’s a good idea to keep it separate because that allows us to
# easily experiment with different models and different architectures


# A PyTorch
# library called pretrainedmodels has a lot of different model architectures, such as
# AlexNet, ResNet, DenseNet, etc.

# There are different model architectures which
# have been trained on large image dataset called ImageNet. We can use them with
# their weights after training on ImageNet, and we can also use them without these
# weights. If we train without the ImageNet weights, it means our network is learning
# everything from scratch. This is what model.py looks like.

# model.py
import torch.nn as nn
import pretrainedmodels
def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["alexnet"](pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__["alexnet"](pretrained=None)
    # print the model here to know whats going on.
        model.last_linear = nn.Sequential(
        nn.BatchNorm1d(4096),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1),)
    return model



# resnet18 with pretrained weights



# model.py
# import torch.nn as nn
# import pretrainedmodels
# def get_model(pretrained):
#     if pretrained:
#         model = pretrainedmodels.__dict__["resnet18"](pretrained='imagenet')
#     else:
#         model = pretrainedmodels.__dict__["resnet18"](pretrained=None)
#         # print the model here to know whats going on.
#         model.last_linear = nn.Sequential(
#         nn.BatchNorm1d(512),
#         nn.Dropout(p=0.25),
#         nn.Linear(in_features=512, out_features=2048),
#         nn.ReLU(),
#         nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
#         nn.Dropout(p=0.5),
#         nn.Linear(in_features=2048, out_features=1),)
#     return model