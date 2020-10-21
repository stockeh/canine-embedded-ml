import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs = axs.ravel()
    
    axs[0].plot(history['accuracy'], label='train')
    axs[0].plot(history['val_accuracy'], label = 'val')
    axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Accuracy')
    axs[0].set_ylim([0.4, 1])
    axs[0].legend(loc='lower right')
    
    axs[1].plot(history['loss'], label='train')
    axs[1].plot(history['val_loss'], label = 'val')
    axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Loss')
    
    plt.show();
    
    
def confusion_matrix(actual, predicted, classes):
    """Print a confusion matrix of the results."""
    nc = len(classes)
    confmat = np.zeros((nc, nc))
    for ri in range(nc):
        trues = (actual==classes[ri]).squeeze()
        predictedThisClass = predicted[trues]
        keep = trues
        predictedThisClassAboveThreshold = predictedThisClass
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predictedThisClassAboveThreshold == classes[ci]) / float(np.sum(keep))
    _print_confusion_matrix(confmat, classes)
    return confmat


def _print_confusion_matrix(confmat, classes):
    """Helper function for `confusion_matrix`"""
    print('   ',end='')
    for i in classes:
        print('%5d' % (i), end='')
    print('\n    ',end='')
    print('{:s}'.format('------'*len(classes)))
    for i,t in enumerate(classes):
        print('{:2d} |'.format(t), end='')
        for i1,t1 in enumerate(classes):
            if confmat[i,i1] == 0:
                print('  0  ',end='')
            else:
                print('{:5.1f}'.format(100*confmat[i,i1]), end='')
        print()