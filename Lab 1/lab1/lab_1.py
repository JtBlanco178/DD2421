import monkdata as m
import dtree as d
#from drawtree_qt5 import *
#import matplotlib.pyplot as plt
import random

""" Assignment 1"""


def entropies(data):
    ent = d.entropy(data)
    print("Entropy: ", ent)

    return ent


def average_gains(data):
    average = []
    for i in range(6):
        average.append(d.averageGain(data, m.attributes[i]))

    print(average)

    return average


def build_tree(train, test):
    t = d.buildTree(train, m.attributes)
    print("Train: ", d.check(t, train))
    print("Test: ", d.check(t, test))

    return t


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


# def prune(tree, train, val, test, err):
#     current_val_err = d.check(tree, val)
#     best_tree = tree
#     best_val_err = d.check(tree, val)
#
#     print("First val error: ", best_val_err)
#
#     flag = True
#     count = 0
#     while flag:
#         count += 1
#         flag_2 = False
#         trees = d.allPruned(best_tree)
#         print("All trees: ", trees)
#         print("Pruning Iteration ", count)
#         for t in trees:
#             print("Tree: ", t)
#             if len(str(t)) != 1:
#                 current_val_err = d.check(t, val)
#             print("Current val error: ", current_val_err)
#             if current_val_err <= best_val_err:
#                 print("Best val error: ", best_val_err)
#                 best_val_err = current_val_err
#                 best_tree = t
#                 print("Best tree: ", best_tree)
#                 flag_2 = True
#         if not flag_2:
#             flag = False
#     err.append(best_val_err)
  #  return error
   # print("Final Tree: ", best_tree)
  #  print(count)

def prune(dataset, fraction, testset):
    training_set, validation_set = partition(dataset, fraction)
    newtree = buildTree(training_set, m.attributes)
    bestVal = 10000
    while  True:
        alternativeTrees = allPruned(newtree)
        minVal = 10000
        bestTree = 10000
        if len(alternativeTrees) == 1:
            break
        for i in range (1, len(alternativeTrees)):
            tempVal = (1 - check(alternativeTrees[i], validation_set))
            if tempVal < minVal:
                minVal = tempVal
                bestTree = i
        if (minVal <= bestVal):
            bestVal = minVal
        else:
            break
        newtree = alternativeTrees[bestTree]
    return newtree, 1 - d.check(newtree, testset) # returns the new tree and the error rate for it

if __name__ == '__main__':
    data = m.monk1
    test = m.monk1test
    fraction = []
    val = []
    err = []
    j = 0
    print("break")
    for i in range(6):
        fraction.append(round(0.3 + (i * 0.1),1))
        #print("asdfasdfsdfsdfsdf: ", fraction)
        train, val = partition(data, fraction[j])
        entropy = entropies(data)
        average_gain = average_gains(data)
        tree = build_tree(train, test)
       # print("First tree: ", tree)
        finaltree, new_err = prune(tree, train, val)
        err.append(new_err)
#        print("asdasdasdasdasdasd: ", prune(tree, train, val, test))
        j += 1
#    plt.plot(fraction, val)
    print(err)
    print(fraction)