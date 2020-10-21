import numpy as np
import pandas as pd
import glob
import os
import cv2

def save_data(vol, npy_image_path, npy_label_path, dsize=(32, 32)):
    """Save the images and labels to respective np.array's.
    ---
    params:
        vol : str
            Parent directory for the the `Images` and `labels`
        npy_image_path : str
            Path for where to save the np.array of images
        npy_label_path : str
            Path for where to save the np.array of labels
        dsize : tuple
            Specific size to resize all of the images to
    """
    # join all csv's to a single dataframe
    data_labels = glob.glob(os.path.join(vol, 'labels/*'))

    li = []
    for filename in data_labels:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['breed'] = filename.split('-')[-1][:-4]
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    X = np.zeros((len(df), dsize[0], dsize[1], 3))
    T = np.zeros((len(df), 1), dtype=object)

    for i, row in df.iterrows():
        im_path = glob.glob(os.path.join(vol, f"Images/*/{row['id']}"))[0]
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB) # BGR -> RGB
        im = cv2.resize(im, dsize)
        X[i] = im
        T[i] = row['label']

    with open(npy_image_path, 'wb') as f:
        np.save(f, X)
    print(f'finished saving images to {npy_image_path}')
    
    with open(npy_label_path, 'wb') as f:
        np.save(f, T)
    print(f'finished saving labels to {npy_label_path}')

    
def load_data(npy_image_path, npy_label_path, normalize=True, remove_undefined=False):
    """Assumes an np.array for the images X and labels T have been saved.
    ---
    params:
        npy_image_path : str
            Path for where to save the np.array of images
        npy_label_path : str
            Path for where to save the np.array of labels
        normalize : str
            Normalize the images by dividing by 255.
        remove_undefined : bool
            Remove the `undefined` class from X and T
    ---
    returns:
        X : np.array
        T : np.array
        class_labels : np.array
    """
    with open(npy_image_path, 'rb') as f:
        X = np.load(f, allow_pickle=True)
    
    if normalize:
        X /= 255.0
        
    with open(npy_label_path, 'rb') as f:
        T = np.load(f, allow_pickle=True)
    
    class_labels, T = np.unique(T, return_inverse=True)
    T = T.reshape((-1,1))
    
    if remove_undefined:
        remove = np.where(class_labels == 'undefined')[0][0]
        class_labels = np.delete(class_labels, remove)
        undefined_is = np.where(T == remove)[0]
        X = np.delete(X, undefined_is, 0)
        T = np.delete(T, undefined_is, 0)
    
    return X, T, class_labels


def partition_data(X, T, partition=(0.6,0.2,0.2), shuffle=False):
    """Load the data and parition it to training, validation (optional), and test set 
    according to class proportions.
    ---
    params:
        X : np.array
        T : np.array
        partition : list
            Of the form (train, val, test) or (train, test)
        shuffle : bool
            Shuffle the data after partitioning
    ---
    return: 
        Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest :: where validation is optional
    
    """
    assert sum(partition) == 1, 'partion must sum to 1.0, e.g. (0.6, 0.2, 0.2) or (0.8, 0.2)'
    
    trainFraction = partition[0]
    
    if len(partition) == 2:
        validateFraction = 0
        testFraction = partition[1]
    if len(partition) == 3:
        validateFraction = partition[1]
        testFraction = partition[2]
        
    rowIndices = np.arange(X.shape[0])
    
    classes = np.unique(T)
    trainIndices = []
    validateIndices = []
    testIndices = []
    for c in classes:
        # row indices for class c
        cRows = np.where(T[rowIndices,:] == c)[0]
        # collect row indices for class c for each partition
        n = len(cRows)
        nTrain = round(trainFraction * n)
        nValidate = round(validateFraction * n)
        nTest = round(testFraction * n)
        
        if nTrain + nValidate + nTest > n:
            nTest = n - nTrain - nValidate
        trainIndices += rowIndices[cRows[:nTrain]].tolist()
        
        if nValidate > 0:
            validateIndices += rowIndices[cRows[nTrain:nTrain+nValidate]].tolist()
        testIndices += rowIndices[cRows[nTrain+nValidate:nTrain+nValidate+nTest]].tolist()
    
    if shuffle:
        np.random.shuffle(trainIndices)
        if nValidate > 0:
            np.random.shuffle(validateIndices)
        np.random.shuffle(testIndices)
        
    Xtrain = X[trainIndices,:]
    Ttrain = T[trainIndices,:]
    if nValidate > 0:
        Xvalidate = X[validateIndices,:]
        Tvalidate = T[validateIndices,:]
    Xtest = X[testIndices,:]
    Ttest = T[testIndices,:]
        
    if nValidate > 0:
        return Xtrain,Ttrain,Xvalidate,Tvalidate,Xtest,Ttest
    else:
        return Xtrain,Ttrain,Xtest,Ttest