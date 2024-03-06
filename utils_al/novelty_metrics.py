import torch
import numpy as np
import math

'''
metric the novelty of the AL selected data
more novel data and more uniform novel classes are better
'''


def novel_coverage(labels, num_old_classes, num_novel_classes):
    num_select = len(labels)
    labels_new = labels[labels>=num_old_classes]
    coverage = len(labels_new.unique())
    coverage /= num_novel_classes

    return coverage


def novel_ratio(labels, num_old_classes, num_novel_classes):
    num_select = len(labels)
    labels_new = labels[labels>=num_old_classes]
    ratio = len(labels_new) / num_select

    return ratio


def novel_uniformity(labels, num_old_classes, num_novel_classes):
    #epsilon = 1e-6
    labels_new = labels[labels>=num_old_classes]
    num_new = len(labels_new)
    count = torch.unique(labels_new, return_counts=True)[1]
    prob = count.float() / num_new
    entropy = -(prob * torch.log(prob)).sum().item()
    upper_bound = math.log(num_novel_classes)

    return entropy, upper_bound
