import numpy as np

def find_center(cluster):
    center_x_coo = cluster[:, 0].mean()
    center_y_coo = cluster[:, 1].mean()
    return center_x_coo, center_y_coo

def find_weighted_center(cluster):
    
    x_coos = cluster[:, 0]
    y_coos = cluster[:, 1]
    
    charges = cluster[:, -1]
    
    center_x_coo = x_coos*charges/charges.sum()
    center_y_coo = y_coos*charges/charges.sum()
    return center_x_coo, center_y_coo

def update_hit_coordinates(cluster):
    center_x, center_y = find_weighted_center(cluster)
    cluster[:, 0] -= center_x
    cluster[:, 1] -= center_y
     
    return cluster 

def transform_train_set(X):
    for x in X:
        x = update_hit_coordinates(x)
    return X