#!/usr/bin/env python
import copy
import warnings
import logging


from flcore.trainmodel.models import *
from flcore.trainmodel.transformer import *
from torch import nn

from flcore.trainmodel.models import Mclr_Logistic, FedAvgCNN, Digit5CNN, DNN, BaseHeadSplit

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)

vocab_size = 98635
max_len = 200
emb_dim = 32


def ensure_model(args, model_str):
    # print(args.device)
    args.model = model_str
    if model_str == "mlr":  # convex
        if "mnist" in args.dataset:
            args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
        else:
            args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

    elif model_str == "cnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "omniglot" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
        elif "Digit5" in args.dataset:
            args.model = Digit5CNN().to(args.device)
        else:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

    elif model_str == "dnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = DNN(1 * 28 * 28, 512, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = DNN(3 * 32 * 32, 512, num_classes=args.num_classes).to(args.device)
        else:
            args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
    else:
        raise NotImplementedError


    if args.algorithm == "FedAvg":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)

    elif args.algorithm == "FedADA":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)

    elif args.algorithm == "FedGNN":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)

    elif args.algorithm == "FedHete":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)

    # print("Personalized arg.model struvture!!")
    # print(arg.model.base)
    # print(arg.model.head)
    return args.model, args.model.head, args.model.base