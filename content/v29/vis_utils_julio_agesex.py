#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:56:30 2023

@author: Julio Villalón Reina. Modified from @seykia. 
"""

import pickle
import numpy as np
import os
from pcntoolkit.normative import predict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_dummy_inputs_sexbatch(processing_dir, site_num, age_range = [10,90], 
                                 output_dim=None):
    """
    This is the original function from Mosi (2020). In this code sex is added as
    a batch effect.
    This function is used to create dummy inputs in a certain age range. The dummy
    inputs can be used as inputs to already trained normative models in order to 
    visualize the normative ranges.
    
    Parameters
    ----------
    processing_dir: a string that contains the address to processing 
                    directory.
    site_num: An integer that indicates the number of sites to create the 
              dummy data for. It may be the total number of sites or less.
    age_range: A list of two integers indicating the start and end age for
               dummy subjects. For example [10,90] says create dummy subjects for age 
               10 to 90. [10,90] is the default value.
    output_dim: an integer that decides the dimensionality of the output 
                space, i.e., the number of brain measures in the normative model. 
    
    Returns
    -------
    The outputs are all in the file format including:
    X_test_dummy.pkl: dummy ages.
    y_test_dummy.pkl: dummy brain measures (all zeros).
    tsbefile_dummy.pkl: dummy batch effects.
            
    """
    
    r = age_range[1] - age_range[0]
    X_test_dummy = np.concatenate([np.arange(age_range[0],age_range[1]), 
                                    np.arange(age_range[0],age_range[1])], axis=0)
    batch_effects_test_dummy = np.zeros([r*2, 2], dtype=int)
    batch_effects_test_dummy[0:r,1] = 1
    for i in range(1,site_num):
        X_test_dummy = np.concatenate([X_test_dummy, np.arange(age_range[0],age_range[1]), 
                                        np.arange(age_range[0],age_range[1])], axis=0)
        batch_effects_test_dummy = np.concatenate([batch_effects_test_dummy, 
                                                    np.zeros([r*2, 2], dtype=int)], axis=0) 
        batch_effects_test_dummy[i*r*2:(i+1)*r*2,0] = i
        batch_effects_test_dummy[i*r*2:(i)*r*2+r,1] = 1
    
    if output_dim is not None:
        y_test_dummy = np.zeros([X_test_dummy.shape[0], output_dim])
    
    X_test_dummy = X_test_dummy/100
    with open(processing_dir + 'X_test_dummy.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_test_dummy), file)
    
    with open(processing_dir + 'tsbefile_dummy.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_test_dummy), file)
    
    with open(processing_dir + 'y_test_dummy.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(y_test_dummy), file)
    
    return X_test_dummy, batch_effects_test_dummy 


def create_dummy_inputs_sexfixed(processing_dir, site_num, age_range = [10,90], 
                                 output_dim=None, sex_batch=False):
    """
    This is a modified version that Julio Villalón wrote: sex is a fixed variable.
    This function is used to create dummy inputs in a certain age range. The dummy
    inputs can be used as inputs to already trained normative model in order to 
    visualize the normative ranges.
    
    Parameters
    ----------
    processing_dir: a string that contains the address to processing 
                    directory.
    site_num: an integer that indicates the number of sites to create the 
              dummy data for. It may be the total number of sites or less.
    age_range: A list of two integers indicating the start and end ages for
               dummy subjects. For example [10,90] says create dummy subjects for age 
               10 to 90. [10,90] is the default value.
    output_dim: an integer that decides the dimensionality of the output 
                space, i.e., the number of brain measures in the normative model.
    sex_batch: Whether the variable Sex should be left as a batch effect
               or it should be set as a fixed variable. If set to True it'll do the
               same thing as Mosis's original function (see above).
    
    Returns
    -------
    The outputs are all in the file format including:
    X_test_dummy.pkl: dummy ages and sexes
    y_test_dummy.pkl: dummy brain measures (all zeros).
    tsbefile_dummy.pkl: dummy batch effects.
            
    """
    
    r = age_range[1] - age_range[0]
    X_test_dummy = np.concatenate([np.arange(age_range[0], age_range[1]),
                                   np.arange(age_range[0],age_range[1])], axis=0)
    batch_effects_test_dummy = np.zeros([r*2, 2], dtype=int)
    batch_effects_test_dummy[0:r,1] = 1
    
    for i in range(1,site_num):
        X_test_dummy = np.concatenate([X_test_dummy, np.arange(age_range[0],age_range[1]), 
                                        np.arange(age_range[0],age_range[1])], axis=0)
        batch_effects_test_dummy = np.concatenate([batch_effects_test_dummy, 
                                                   np.zeros([r*2, 2], dtype=int)], axis=0) 
        batch_effects_test_dummy[i*r*2:(i+1)*r*2,0] = i
        batch_effects_test_dummy[i*r*2:(i)*r*2+r,1] = 1


    if output_dim is not None:
        y_test_dummy = np.zeros([X_test_dummy.shape[0], output_dim])
    
    X_test_dummy = X_test_dummy/100
    
    #because sex may not be included as a batch we remove that column.
    if sex_batch is False:
        sex_cov = batch_effects_test_dummy[:,1][:,np.newaxis]
        X_test_dummy = X_test_dummy[:,np.newaxis]
        X_test_dummy = np.concatenate([X_test_dummy, sex_cov],axis=1)
        batch_effects_test_dummy = batch_effects_test_dummy[:,0]   
    
    batch_effects_test_dummy = batch_effects_test_dummy
    
    with open(processing_dir + 'X_test_dummy.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(X_test_dummy), file)
    
    with open(processing_dir + 'tsbefile_dummy.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(batch_effects_test_dummy), file)
    
    with open(processing_dir + 'y_test_dummy.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(y_test_dummy), file)
    
    return X_test_dummy, batch_effects_test_dummy, y_test_dummy 


def predict_dummy(dir_out, dir_model, site_num, age_min, age_max, roi_id, sex_batch):
    """
    This function makes the actual model prediction on the dummy data created
    with the 'create_dummy' functions (see above). The prediction is based on 
    a previously estimated HBR model.
    
    Parameters
    ----------
    dir_out: Path to output folder of the model prediction.
    dir_model: Path to the outputs of the already estimated model.
    site_num: Integer that indicates the number of sites to create the 
              dummy data for and consequently. It may be the total number
              of sites or less.
    age_min: Integer - Minimum age.
    age_max: Integer - Maximum age.
    roi_id: Name of the ROI to be predicted.
    sex_batch: Whether the variable Sex should be left as a batch effect
               or it should be set as a fixed variable. If set to True it'll do the
               same thing as Mosis's original function (see above).
    
    Returns
    -------
    yhat_te: The predicted y_hat for the dummy data.
    s2_te: The predicted s2 for the dummy data.
    
    """
    
    # First we create the dummy data. Here we consider sex as a fixed effect.    
    create_dummy_inputs_sexfixed(dir_out, site_num, age_range = [age_min, age_max],
                                 output_dim = 1, sex_batch=sex_batch)
    
    # These are the outputs that the function writes by default. 
    X_test_dummy = os.path.join(dir_out, 'X_test_dummy.pkl')
    tsbefile_dummy = os.path.join(dir_out, 'tsbefile_dummy.pkl')
        
    # This is the path to the output folder where the dummy predicted
    # model will go for this particular ROI
    processing_dir = os.path.join(dir_out, roi_id)
    
    # This is the full path to the original precomputed model for this
    # specific ROI
    print("This is the path to the pre-estimated model:")
    print(os.path.join(dir_model, roi_id, 'Models'))
    
    try:
        os.makedirs(processing_dir, exist_ok = True)
        print("Directory '%s' created successfully" %roi_id)
    except OSError as error:
        print("Directory '%s' can not be created")
    
    # Changing position to folder with dummies
    os.chdir(processing_dir)

    # Predict on dummy data
    predicted = predict(X_test_dummy,
                        respfile=None,
                        tsbefile=tsbefile_dummy,
                        alg='hbr',
                        model_path=os.path.join(dir_model, roi_id, 'Models'),
                        outputsuffix=''.join(['_', roi_id]),
                        outputpath=dir_out
                        )

    # The algorithm returns a tuple: predicted=(Yhat, S2)
    # Z is not returned because respfile=None
    yhat_te = predicted[0]
    s2_te = predicted[1]

    np.savetxt(os.path.join(processing_dir, 'yhat_dummy.txt'), yhat_te)
    np.savetxt(os.path.join(processing_dir, 'ys2_dummy.txt'), s2_te)
    
    # Saving these pickles was giving an error. 
    # with open(processing_dir + '/' + 'yhat_dummy.pkl', 'wb') as file:
    #     pickle.dump(pd.DataFrame(yhat_te), file)
    # with open(processing_dir + '/', 'ys2_dummy.pkl', 'wb') as file:
    #     pickle.dump(pd.DataFrame(s2_te), file)
    
    return yhat_te, s2_te


def plot_normative_models_multi(processing_dir, num_sites, site_id, area_id, metric, age_min, age_max, sex_batch):
    
    """
    A utility function to plot resulting normative models. It pulls from the
    predictions on dummy data (20 times). The dummy predictions must have been
    previously done as this function just goes to the folder to get the model 
    parameters. 
    
    Parameters
    ----------
    processing_dir: a string that contains the path to the dummy
                    predicted models.
    num_sites: the total number of sites that were used in the prediction.
               Usually a number less or equal than the total number of sites.
    site_id: an integer between 0 and num_sites that indicates the site you 
             want to plot the normative model for. Zero indexed!
    area_id: a string for the JHU-WM ENIGMA ROIs that indicates the WM
             region you want to plot the normative model for.
    metric: the diffusion MRI metric.
    age_min: an integer for the minimum age.
    age_max: an integer for the maximum age.
    
    Returns
    -------
    
    
    """
    
    with open(processing_dir + 'Dummy1/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_1 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy1/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_1 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy2/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_2 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy2/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_2 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy3/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_3 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy3/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_3 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy4/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_4 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy4/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_4 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy5/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_5 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy5/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_5 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy6/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_6 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy6/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_6 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy7/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_7 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy7/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_7 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy8/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_8 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy8/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_8 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy9/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_9 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy9/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_9 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy10/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_10 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy10/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_10 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy11/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_11 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy11/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_11 = pickle.load(file).to_numpy()
           
    with open(processing_dir + 'Dummy12/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_12 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy12/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_12 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy13/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_13 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy13/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_13 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy14/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_14 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy14/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_14 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy15/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_15 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy15/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_15 = pickle.load(file).to_numpy()

    with open(processing_dir + 'Dummy16/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_16 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy16/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_16 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy17/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_17 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy17/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_17 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy18/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_18 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy18/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_18 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy19/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_19 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy19/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_19 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy20/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_20 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy20/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_20 = pickle.load(file).to_numpy()

   ########################################################################### 
   # This is opening the dummy X_test and the dummy batch files
    with open(processing_dir + 'Dummy1/' + 'X_test_dummy.pkl', 'rb') as file:
        X_dummy = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy1/' + 'tsbefile_dummy.pkl', 'rb') as file:
        batch_effects_dummy = pickle.load(file).to_numpy()
 
    X_dummy[:,0] = X_dummy[:,0] * 100  # Las edades deben aparecer en decenas
    
    # This is masking the dummy generated data. 
    # Here we can mask based on only the site_id (sex_batch=False) or the
    # site_id + sex (sex_batch=True), which is the sex column X_dummy[:,1]
    
    if sex_batch is False:
        mask = (batch_effects_dummy[:,0] == site_id) & (X_dummy[:,1] == 0)
        mask2 = (batch_effects_dummy[:,0] == site_id) & (X_dummy[:,1] == 1) # This is for males=1.
    else:
        mask = (batch_effects_dummy[:,0] == site_id) & (batch_effects_dummy[:,1] == 0)
        mask2 = (batch_effects_dummy[:,0] == site_id) & (batch_effects_dummy[:,1] == 1) # This is for males=1.
    
    # Concatenating all predictions
    yhat = np.concatenate((yhat_1, yhat_2, yhat_3, yhat_4, yhat_5, yhat_6,
                           yhat_7, yhat_8, yhat_9, yhat_10, yhat_11, yhat_12,
                           yhat_13, yhat_14, yhat_15, yhat_16, yhat_17,
                           yhat_18, yhat_19, yhat_20), axis=1)
    ys2 = np.concatenate((s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9,
                          s2_10, s2_11, s2_12, s2_13, s2_14, s2_15, s2_16,
                          s2_17, s2_18, s2_19, s2_20), axis=1)
    
    yhat_dummy = np.mean(yhat, axis=1)
    yhat_women = yhat_dummy[mask]
    yhat_men = yhat_dummy[mask2]
    yhat_dummy_bothsex = np.mean((yhat_women, yhat_men), axis=0)
    
    ys2_dummy = np.mean(ys2, axis=1)
    ys2_women = ys2_dummy[mask]
    ys2_men = ys2_dummy[mask2]
    ys2_dummy_bothsex = np.mean((ys2_women, ys2_men), axis=0)
    
    age = X_dummy[:,0][mask]
    
    # Plotting figures
    sns.set(style='whitegrid')
    plt.figure()
    plt.subplot()
    plt.plot(age, yhat_dummy_bothsex)
    # plt.scatter(site_data_roi[:,0], site_data_roi[:,1])
    
    plt.fill_between(age, yhat_dummy_bothsex - 1.0 * np.sqrt(ys2_dummy_bothsex), 
                     yhat_dummy_bothsex + 1.00 * np.sqrt(ys2_dummy_bothsex), 
                     alpha=0.1, color='C0', zorder=0.6)
    plt.fill_between(age, yhat_dummy_bothsex - 2.0 * np.sqrt(ys2_dummy_bothsex), 
                     yhat_dummy_bothsex + 2.0 * np.sqrt(ys2_dummy_bothsex), 
                     alpha=0.1, color='C0', zorder=0.3)

    plt.xlabel('Age')
    plt.ylabel(metric)    
    plt.title(area_id)
    plt.xlim((0,95))
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90'])
    
    ##################
    plt.figure()
    plt.subplot()
    plt.plot(age, yhat_women)
    plt.plot(age, yhat_men)
    
    # plt.fill_between(age, yhat_dummy_bothsex - 1.0 * np.sqrt(ys2_dummy_bothsex), 
    #                  yhat_dummy_bothsex + 1.00 * np.sqrt(ys2_dummy_bothsex), 
    #                  alpha=0.1, color='C0', zorder=0.6)
    # plt.fill_between(age, yhat_dummy_bothsex - 2.0 * np.sqrt(ys2_dummy_bothsex), 
    #                  yhat_dummy_bothsex + 2.0 * np.sqrt(ys2_dummy_bothsex), 
    #                  alpha=0.1, color='C0', zorder=0.3)
    
    plt.xlabel('Age')
    plt.ylabel(metric)    
    plt.title(area_id)
    plt.xlim((0,95))
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90'])


    return X_dummy, batch_effects_dummy, yhat_dummy_bothsex, ys2_dummy_bothsex, yhat_women, yhat_men


def plot_normative_models_multi_predict(processing_dir, model_dir, num_sites, area_id, metric, age_min, age_max, sex_batch):
    
    """
    This function runs the predicitons 20 times and saves the predicted yhat
    and s2 for each of the 20 iterations.
    
    Parameters
    ----------
    processing_dir: a string that contains the path to the output of the
                    predicitions. This folder will be created.
    model_dir: a string with the path to the previously estimated HBR model.
    num_sites: the total number of sites that were used in the prediction.
               Usually a number less or equal than the total number of sites.
    area_id: a string for the JHU-WM ENIGMA ROIs that indicates the WM
             region you want to plot the normative model for.
    metric: the diffusion MRI metric.
    age_min: an integer for the minimum age.
    age_max: an integer for the maximum age.
        
    Returns
    -------
    All files will be saved in the 'processing_dir' path.
        
    """
    
    os.makedirs(processing_dir + '/Dummy1/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy2/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy3/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy4/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy5/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy6/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy7/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy8/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy9/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy10/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy11/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy12/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy13/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy14/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy15/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy16/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy17/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy18/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy19/', exist_ok = True)
    os.makedirs(processing_dir + '/Dummy20/', exist_ok = True)
    
    [yhat_1, s2_1] = predict_dummy(processing_dir + '/Dummy1/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_2, s2_2] = predict_dummy(processing_dir + '/Dummy2/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_3, s2_3] = predict_dummy(processing_dir + '/Dummy3/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_4, s2_4] = predict_dummy(processing_dir + '/Dummy4/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_5, s2_5] = predict_dummy(processing_dir + '/Dummy5/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_6, s2_6] = predict_dummy(processing_dir + '/Dummy6/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_7, s2_7] = predict_dummy(processing_dir + '/Dummy7/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_8, s2_8] = predict_dummy(processing_dir + '/Dummy8/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_9, s2_9] = predict_dummy(processing_dir + '/Dummy9/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_10, s2_10] = predict_dummy(processing_dir + '/Dummy10/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_11, s2_11] = predict_dummy(processing_dir + '/Dummy11/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_12, s2_12] = predict_dummy(processing_dir + '/Dummy12/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_13, s2_13] = predict_dummy(processing_dir + '/Dummy13/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_14, s2_14] = predict_dummy(processing_dir + '/Dummy14/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_15, s2_15] = predict_dummy(processing_dir + '/Dummy15/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_16, s2_16] = predict_dummy(processing_dir + '/Dummy16/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_17, s2_17] = predict_dummy(processing_dir + '/Dummy17/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_18, s2_18] = predict_dummy(processing_dir + '/Dummy18/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_19, s2_19] = predict_dummy(processing_dir + '/Dummy19/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    [yhat_20, s2_20] = predict_dummy(processing_dir + '/Dummy20/', model_dir, num_sites, age_min, age_max, area_id, sex_batch)
    
    print("Done with predictions!")


def plot_normative_models_multisite(processing_dir, site_id, area_id, metric, age_min, age_max):
    
    """
    A utility function to plot resulting normative models for as many sites as
    desired. It pulls from the predictions on dummy data (20 times). 
    The dummy predictions must have been previeouly done as this function just
    goes to the folder to get the model parameters. 
    
    Parameters
    ----------
    processing_dir: a string that contains the path to the dummy
                    predicted models.
    site_id: an integer between 0 and num_sites that indicates the site you 
             want to plot the normative model for. Zero indexed!
             This number must be less than or equal to the max number of sites
             predicted.
    area_id: a string for the JHU-WM ENIGMA ROIs that indicates the WM
             region you want to plot the normative model for.
    metric: the diffusion MRI metric.
    age_min: an integer for the minimum age.
    age_max: an integer for the maximum age.
        
    """
    
    with open(processing_dir + 'Dummy1/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_1 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy1/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_1 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy2/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_2 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy2/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_2 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy3/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_3 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy3/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_3 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy4/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_4 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy4/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_4 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy5/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_5 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy5/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_5 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy6/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_6 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy6/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_6 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy7/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_7 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy7/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_7 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy8/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_8 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy8/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_8 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy9/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_9 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy9/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_9 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy10/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_10 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy10/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_10 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy11/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_11 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy11/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_11 = pickle.load(file).to_numpy()
            
    with open(processing_dir + 'Dummy12/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_12 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy12/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_12 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy13/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_13 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy13/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_13 = pickle.load(file).to_numpy()
     
    with open(processing_dir + 'Dummy14/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_14 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy14/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_14 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy15/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_15 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy15/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_15 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy16/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_16 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy16/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_16 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy17/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_17 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy17/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_17 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy18/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_18 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy18/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_18 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy19/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_19 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy19/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_19 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy20/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_20 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy20/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_20 = pickle.load(file).to_numpy()
    
   ############################################################################
    with open(processing_dir + 'Dummy1/' + 'X_test_dummy.pkl', 'rb') as file:
        X_dummy = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy1/' + 'tsbefile_dummy.pkl', 'rb') as file:
        batch_effects_dummy = pickle.load(file).to_numpy()

    X_dummy[:,0] = X_dummy[:,0] * 100  # Las edades deben aparecer en decenas
    age = np.arange(age_min, age_max)
    yhat_dummy = np.empty((len(age), 0))
    ys2_dummy = np.empty((len(age), 0))
    
    yhat_all = np.concatenate((yhat_1, yhat_2, yhat_3, yhat_4, yhat_5, yhat_6,
                               yhat_7, yhat_8, yhat_9, yhat_10, yhat_11, yhat_12,
                               yhat_13, yhat_14, yhat_15, yhat_16, yhat_17,
                               yhat_18, yhat_19, yhat_20), axis=1)
    s2_all = np.concatenate((s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9,
                             s2_10, s2_11, s2_12, s2_13, s2_14, s2_15, s2_16,
                             s2_17, s2_18, s2_19, s2_20), axis=1)
    
    yhat = np.mean(yhat_all, axis=1)
    s2 = np.mean(s2_all, axis=1)
    
    # Going thorugh all the desired sites.
    for i in range(0,site_id):
        # This is masking the dummy generated data. 
        # Here we can mask based on only the site_id or the site + sex, which is the sex column X_dummy[:,0]
        mask = (batch_effects_dummy[:,0] == i) & (X_dummy[:,1] == 0)
        mask2 = (batch_effects_dummy[:,0] == i) & (X_dummy[:,1] == 1) # This is for males=1.
        
        yhat_women = yhat[mask]
        yhat_men = yhat[mask2]
        yhat_dummy_bothsex = np.mean((yhat_women, yhat_men), axis=0)
        yhat_dummy_bothsex = yhat_dummy_bothsex[:, np.newaxis]
        
        ys2_women = s2[mask]
        ys2_men = s2[mask2]
        ys2_dummy_bothsex = np.mean((ys2_women, ys2_men), axis=0)
        ys2_dummy_bothsex = ys2_dummy_bothsex[:, np.newaxis]
        
        yhat_dummy = np.append(yhat_dummy, yhat_dummy_bothsex, axis=1)
        ys2_dummy = np.append(ys2_dummy, ys2_dummy_bothsex, axis=1)
    
    plt.figure()
    plt.subplot()
    for j in range(0,site_id):
        plt.plot(age, yhat_dummy[:,j])
    
    ## Don't need the centile curves.
    # plt.fill_between(age, yhat_dummy_bothsex - 1.0 * np.sqrt(ys2_dummy_bothsex), 
    #                  yhat_dummy_bothsex + 1.00 * np.sqrt(ys2_dummy_bothsex), 
    #                  alpha=0.1, color='C0', zorder=0.6)
    # plt.fill_between(age, yhat_dummy_bothsex - 2.0 * np.sqrt(ys2_dummy_bothsex), 
    #                  yhat_dummy_bothsex + 2.0 * np.sqrt(ys2_dummy_bothsex), 
    #                  alpha=0.1, color='C0', zorder=0.3)
    
    sns.set(style='whitegrid')
    plt.xlabel('Age [years]')
    plt.ylabel(metric)    
    plt.title('Average_WM')
    plt.xlim((0,95))
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']) 

    return X_dummy, batch_effects_dummy


def plot_normative_models_disease(processing_dir, site_id, area_id, metric):
    
    with open(processing_dir + 'Dummy1/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_1 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy1/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_1 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy2/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_2 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy2/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_2 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy3/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_3 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy3/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_3 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy4/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_4 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy4/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_4 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy5/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_5 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy5/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_5 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy6/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_6 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy6/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_6 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy7/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_7 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy7/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_7 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy8/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_8 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy8/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_8 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy9/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_9 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy9/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_9 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy10/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_10 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy10/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_10 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy11/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_11 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy11/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_11 = pickle.load(file).to_numpy()
           
    with open(processing_dir + 'Dummy12/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_12 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy12/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_12 = pickle.load(file).to_numpy()
       
    with open(processing_dir + 'Dummy13/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_13 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy13/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_13 = pickle.load(file).to_numpy()
    
    with open(processing_dir + 'Dummy14/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_14 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy14/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_14 = pickle.load(file).to_numpy()
   
    with open(processing_dir + 'Dummy15/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_15 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy15/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_15 = pickle.load(file).to_numpy()

    with open(processing_dir + 'Dummy16/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_16 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy16/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_16 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy17/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_17 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy17/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_17 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy18/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_18 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy18/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_18 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy19/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_19 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy19/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_19 = pickle.load(file).to_numpy()
        
    with open(processing_dir + 'Dummy20/' + area_id + '/yhat_' + area_id + '.pkl', 'rb') as file:
        yhat_20 = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy20/' + area_id + '/ys2_' + area_id + '.pkl', 'rb') as file:
        s2_20 = pickle.load(file).to_numpy()

   ########################################################################### 
   # This is opening the dummy X_test and the dummy batch files
    with open(processing_dir + 'Dummy1/' + 'X_test_dummy.pkl', 'rb') as file:
        X_dummy = pickle.load(file).to_numpy()
    with open(processing_dir + 'Dummy1/' + 'tsbefile_dummy.pkl', 'rb') as file:
        batch_effects_dummy = pickle.load(file).to_numpy()
 
    X_dummy[:,0] = X_dummy[:,0] * 100  # Las edades deben aparecer en decenas
    
    # This is masking the dummy generated data. 
    # Here we can mask based on only the site_id or the site + sex, which is the sex column X_dummy[:,1]
    mask = (batch_effects_dummy[:,0] == site_id) & (X_dummy[:,1] == 0)
    mask2 = (batch_effects_dummy[:,0] == site_id) & (X_dummy[:,1] == 1) # This is for males=1.
    
    yhat = np.concatenate((yhat_1, yhat_2, yhat_3, yhat_4, yhat_5, yhat_6,
                           yhat_7, yhat_8, yhat_9, yhat_10, yhat_11, yhat_12,
                           yhat_13, yhat_14, yhat_15, yhat_16, yhat_17,
                           yhat_18, yhat_19, yhat_20), axis=1)
    ys2 = np.concatenate((s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7, s2_8, s2_9,
                          s2_10, s2_11, s2_12, s2_13, s2_14, s2_15, s2_16,
                          s2_17, s2_18, s2_19, s2_20), axis=1)
    
    yhat_dummy = np.mean(yhat, axis=1)
    yhat_women = yhat_dummy[mask]
    yhat_men = yhat_dummy[mask2]
    yhat_dummy_bothsex = np.mean((yhat_women, yhat_men), axis=0)
    
    ys2_dummy = np.mean(ys2, axis=1)
    ys2_women = ys2_dummy[mask]
    ys2_men = ys2_dummy[mask2]
    ys2_dummy_bothsex = np.mean((ys2_women, ys2_men), axis=0)
    
    age = X_dummy[:,0][mask]
    
    #Reading in the table for FA/MD/RD/AD for the cases
    df_ = pd.read_csv('/Users/julio/Documents/IGC/HBR_paper/Denoising_data/MD_ADNI3_OASIS3_ROItable_Alzheimers_4test.csv', sep=',', index_col=False)
    site_data = df_.loc[df_['SID_2'] == site_id]
    site_data_roi = site_data[['age', area_id]]
    site_data_roi = site_data_roi.to_numpy()
    
    sns.set(style='whitegrid')
    plt.figure()
    plt.subplot()
    plt.plot(age, yhat_dummy_bothsex)
    plt.scatter(site_data_roi[:,0], site_data_roi[:,1])
    
    plt.fill_between(age, yhat_dummy_bothsex - 1.0 * np.sqrt(ys2_dummy_bothsex), 
                     yhat_dummy_bothsex + 1.00 * np.sqrt(ys2_dummy_bothsex), 
                     alpha=0.1, color='C0', zorder=0.6)
    plt.fill_between(age, yhat_dummy_bothsex - 2.0 * np.sqrt(ys2_dummy_bothsex), 
                     yhat_dummy_bothsex + 2.0 * np.sqrt(ys2_dummy_bothsex), 
                     alpha=0.1, color='C0', zorder=0.3)
    
    # sns.set(style='whitegrid')
    plt.xlabel('Age')
    plt.ylabel(metric)    
    plt.title(area_id)
    plt.xlim((0,95))
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']) 

    return X_dummy, batch_effects_dummy, yhat_dummy_bothsex, ys2_dummy_bothsex
    
