from sklearn.datasets import load_digits
#from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from tsne import bh_sne
from sklearn.manifold import TSNE
import pandas as pd

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import torch
import pickle as pkl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn
import time

import glob

#import scipy.spatial.distance.pdist as pom


def plot_t():
    
    dir_name = 'checkpoint_cme_sep_16_8_sobel_16_0_3_True_5fps'
    data_dir = os.path.join('./intermediary_sub', dir_name)
    
    list_data = sorted(glob.glob(os.path.join(data_dir, 'data_*.pkl')))
    list_target = sorted(glob.glob(os.path.join(data_dir, 'target_*.pkl')))
    
    
    nb_videos = 16 * (len(list_data)-1)
    
    
    all_data = np.zeros((nb_videos, 100352))
    all_target = np.zeros((nb_videos))
    
    
    for cp in range(len(list_data)-1):
        #print('data %s  target %s' %(list_data[cp], list_target[cp]))
        with open(list_data[cp], "rb") as fout:
            data = pkl.load(fout)
        
        with open(list_target[cp], "rb") as fout:
            target = pkl.load(fout)
            
        data = data.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        #print(data.shape)
        
        new_target = np.zeros((16))
        
        for i in range(target.shape[0]):
            tp = target[i]
            for j in range(tp.shape[0]):
                if np.max(tp[j]) != 0 :
                    new_target[i] = j
                    break
        
        target = new_target
        
        
        #print(all_data[cp:cp+16,:].shape)
        #print(data.shape)
        
        all_data[cp:cp+16,:] = data
        all_target[cp:cp+16] = target
        
    
    print(all_data.shape)
    print(all_target.shape)
    
    # perform t-SNE embedding
    #vis_data = bh_sne(all_data)
    
    
    
    vis_data = TSNE(n_components=2, random_state=0, verbose=1).fit_transform(all_data)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    plt.scatter(vis_x, vis_y, c=all_target, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig(dir_name +  '.png')
    plt.show()


def plot_with_pca(dir_name):
    
    #dir_name = 'checkpoint_cme_sep_16_8_sobel_16_0_3_True_5fps'
    data_dir = os.path.join('./intermediary_sub', dir_name)
    
    list_data = sorted(glob.glob(os.path.join(data_dir, 'data_*.pkl')))
    list_target = sorted(glob.glob(os.path.join(data_dir, 'target_*.pkl')))
    
    
    
    
    #all_data = np.zeros((1, 100352))
    #all_target = np.zeros((1))
    

    
    for cp in range(len(list_data)-1):
        #print('data %s  target %s' %(list_data[cp], list_target[cp]))g
        with open(list_data[cp], "rb") as fout:
            data = pkl.load(fout)
        
        with open(list_target[cp], "rb") as fout:
            target = pkl.load(fout)
            
        data = data.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        #print(data.shape)
        
        new_target = np.zeros((16))
        
        for i in range(target.shape[0]):
            tp = target[i]
            for j in range(tp.shape[0]):
                if np.max(tp[j]) != 0 :
                    new_target[i] = j
                    break
        
        target = new_target
        
        #print(target)
        
        
        #print(all_data[cp:cp+16,:].shape)
        #print(data.shape)
        
        if cp == 0:
            all_data = data
            all_target = target
            
        all_data = np.append(all_data, data, axis = 0)
        all_target = np.append(all_target, target)
        
        #all_data[cp:cp+16,:] = data
        #all_target[cp:cp+16] = target
        
        #print( np.count_nonzero( new_target))
      
    
    #print( np.count_nonzero( all_target))
    print(all_data.shape)
    print(all_target.shape)
    
    
        
    nb_features = all_data.shape[1] 
    feat_cols = [ 'feature_'+str(i) for i in range(nb_features) ]
    
    df = pd.DataFrame(all_data, columns=feat_cols)
    
    df['y'] = all_target
    
    print( np.count_nonzero( all_target))
    
    df['label'] = df['y'].apply(lambda i: str(i))
    
    print(sorted(sklearn.neighbors.VALID_METRICS['brute']))
    
    
    distance_metric = 'l2'
    # perform t-SNE embedding
    #vis_data = bh_sne(all_data)
    
    
    
    #nb_videos = 1000
    
    nb_videos = all_data.shape[0]
    
    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(nb_videos)
    
    df_subset = df.loc[rndperm[:nb_videos],:].copy()
    data_subset = df_subset[feat_cols].values
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)
    


    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1] 
    df_subset['pca-three'] = pca_result[:,2]
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40,
                n_iter=400, n_jobs = 8, metric=distance_metric)
    
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    print(tsne_results.shape)
    
    
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    
    plt.figure(figsize=(16,10))
    

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("Paired", 8),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.savefig('./results_tsne/sub' + dir_name + '_' + distance_metric + '.png', dpi = 600)
    plt.show()

if __name__ == '__main__':

    list_dir = glob.glob('./intermediary_sub/check*True_True_5fps')
    for l in list_dir:
        dir_name = l.split('/')[-1]
        print(dir_name)
        plot_with_pca(dir_name)