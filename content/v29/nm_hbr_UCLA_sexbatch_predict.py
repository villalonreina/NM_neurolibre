# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:19:41 2024

@author: julio
"""

import os
import pandas as pd
import numpy as np
from pcntoolkit.normative import predict, evaluate
from scipy.stats import norm
import argparse
__author__ = 'Julio Villalón'
 
parser = argparse.ArgumentParser(description='This run the hbr normative modeling with HBR to predict the Z-scores from a newly given dataset.')
parser.add_argument('-dirM','--dirModel', help='The full path to the already trained model that will be used for prediction.',required=True)
parser.add_argument('-patients', '--patients_csv', help='Input table with patients only. Include the site ID', required=True)
parser.add_argument('-dirO','--dirOutput', help='Ouput directory of the predicted Z-scores.',required=True)
parser.add_argument('-site_column','--site_column', help='Name of the site column header.',required=True)
parser.add_argument('-age_column','--age_column', help='Name of the age column header.',required=True)
parser.add_argument('-sex_column','--sex_column', help='Name of the sex column header.',required=True)
args = parser.parse_args()

## show the inputs ##
print("Path to the pre-estimated model : %s" % args.dirModel)
print("Input patients file: %s" % args.patients_csv)
print("Output directory: %s" % args.dirOutput)
print("Name of the site column: %s" % args.site_column)
print("Name of the age column: %s" % args.age_column)
print("Name of the sex column: %s" % args.sex_column)

data_dir = args.dirOutput
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# These are all the ROIs (bilateral). Removed CST!!!!
roi_ids = ['ACR', 'ALIC', 'Average', 'BCC', 'CGC', 'CGH', 'EC', 'FX', 'FXST', 'GCC', 'UNC', 'PCR', 'PLIC', 'PTR', 'RLIC', 'SCC', 'SCR', 'SFO', 'SLF', 'SS', 'TAP']


### Reading in the patients table ####
patients_path = args.patients_csv
patients = pd.read_csv(patients_path)
# The following line is not needed anymore because the column is indexed starting with 0.
#patients[args.site_column] -= 1  # Here I am subtracting 1 from the Site IDs, as the index starts at 0.

patients['site'] = patients[args.site_column] 
patients_cov = pd.get_dummies(patients, columns=['site'])
patients_cov['site_ID_bin'] = patients['SID']  # this is the numeric version of Protocol


patients_cov = patients_cov[['subjectID',
                             args.sex_column,
                             args.age_column,
                             'site_ID_bin',
                             'site_UCLA_0',
                             'site_UCLA_1',
                             ]]

UCLA_0_te = patients_cov.index[patients_cov['site_UCLA_0'] == 1].to_list()
UCLA_1_te = patients_cov.index[patients_cov['site_UCLA_1'] == 1].to_list()

# Creating batch/site file
dfbatch_te = patients_cov[['site_ID_bin', args.sex_column]]
dfbatch_te.to_csv(os.path.join(data_dir, 'patients_batch_te.txt'),
                  na_rep='NaN', index=False, header=False, sep=' ')

dfsubject_te = patients_cov[['subjectID']]
dfsubject_te.to_csv(os.path.join(data_dir, 'patients_subjects_te.txt'),
                    na_rep='NaN', index=False, header=False, sep=' ')

patients_cov = patients_cov.drop(['subjectID', args.sex_column, 'site_ID_bin', 'site_UCLA_0', 'site_UCLA_1'], axis=1)
patients_cov.loc[:, args.age_column] = patients_cov[args.age_column] / 100

sites = [UCLA_0_te, UCLA_1_te]
site_names = ['UCLA_0_te', 'UCLA_1_te']


### Here we create the text files to be called by the normative modeling ###
for roi in roi_ids: 
    print('Saving the tables for ROI:', roi)
    roi_dir = os.path.join(data_dir, roi)
    
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)
    
    # Saving the X (covariates) for each roi
    np.savetxt(os.path.join(roi_dir, 'cov_adapt_patients_te.txt'), patients_cov)                                 
    
    # Saving the Y for each roi
    np.savetxt(os.path.join(roi_dir, 'resp_adapt_patients_te.txt'), patients[roi])


# Create pandas dataframes with header names to save out the overall and per-site model evaluation metrics
hbr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])
hbr_site_metrics = pd.DataFrame(columns = ['ROI', 'site', 'y_mean', 'y_var', 'yhat_mean', 'yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

# the folder where the previously esimated model is. 
dir_model = args.dirModel
### From here on is the actual model estimation ###
# Loop through ROIs for controls 80/20 train/test split
for roi in roi_ids: 
    print('Running ROI:', roi)
    roi_dir = os.path.join(data_dir, roi)
    os.chdir(roi_dir)
     
    # configure the test covariates to use.
    cov_file_te = os.path.join(roi_dir, 'cov_adapt_patients_te.txt')
    print(os.path.join(roi_dir, 'cov_adapt_patients_te.txt'))
    
    # load test response files
    resp_file_te = os.path.join(roi_dir, 'resp_adapt_patients_te.txt')
    print(os.path.join(roi_dir, 'resp_adapt_patients_te.txt'))
    
    #load the test batch file
    batch_te = os.path.join(data_dir, 'patients_batch_te.txt')
    print(os.path.join(data_dir, 'patients_batch_te.txt'))
    
    print(os.path.join(dir_model, roi, 'Models'))
    # run a predict model
    predicted = predict(cov_file_te,
                        alg='hbr',
                        respfile=resp_file_te,
                        tsbefile=batch_te,
                        model_path=os.path.join(dir_model, roi,'Models'),
                        outputsuffix=''.join(['_', roi]),
                        outscaler='standardize')
    
    # The algorithm returns a tuple: predicted=(Yhat, S2, Z)
    yhat_te = predicted[0]
    s2_te = predicted[1]
    Z = predicted[2]
    
    p_mosi = 1 - norm.sf(np.abs(Z)) * 2
    Z_p = np.concatenate((Z, p_mosi), axis=1)
    np.savetxt(os.path.join(roi_dir, 'Z_p_predicted.txt'), Z_p)
    np.savetxt(os.path.join(roi_dir, 'Yhat_predicted.txt'), yhat_te)
    np.savetxt(os.path.join(roi_dir, 'Ys2_predicted.txt'), s2_te)

    # Read csv files with model evaluation metric files saved automatically
    # by normative.predict
    MSLL = pd.read_csv(''.join([roi_dir, '/MSLL_' , roi,'.txt']), header=None)
    EXPV = pd.read_csv(''.join([roi_dir, '/EXPV_' , roi,'.txt']), header=None)
    SMSE = pd.read_csv(''.join([roi_dir, '/SMSE_' , roi,'.txt']), header=None)
    RMSE = pd.read_csv(''.join([roi_dir, '/RMSE_' , roi,'.txt']), header=None)
    Rho = pd.read_csv(''.join([roi_dir, '/Rho_' , roi,'.txt']), header=None)
    
    # Stacking the evaluation metrics by ROI in a big table
    hbr_metrics.loc[len(hbr_metrics)] = [roi, MSLL[0][0], EXPV[0][0],
                                         SMSE[0][0], RMSE[0][0], Rho[0][0]]
    
    # Compute metrics per site in test set, save to pandas df
    # load true test data
    X_te = np.loadtxt(cov_file_te)
    y_te = np.loadtxt(resp_file_te)
    y_te = y_te[:, np.newaxis]  # make sure it is a 2-d array
    
    # # load training data (required to compute the MSLL)
    # y_tr = np.loadtxt(resp_file_tr)
    # y_tr = y_tr[:, np.newaxis]  # make sure it is a 2-d array
    
    for num, site in enumerate(sites):     
        y_mean_te_site = np.array([[np.mean(y_te[site])]])
        y_var_te_site = np.array([[np.var(y_te[site])]])
        yhat_mean_te_site = np.array([[np.mean(yhat_te[site])]])
        yhat_var_te_site = np.array([[np.var(yhat_te[site])]])
        
        metrics_te_site = evaluate(y_te[site], yhat_te[site], s2_te[site], y_mean_te_site, y_var_te_site)
        
        site_name = site_names[num]
        hbr_site_metrics.loc[len(hbr_site_metrics)] = [roi, site_names[num],
                                                       y_mean_te_site[0],
                                                       y_var_te_site[0],
                                                       yhat_mean_te_site[0],
                                                       yhat_var_te_site[0],
                                                       metrics_te_site['MSLL'][0],
                                                       metrics_te_site['EXPV'][0],
                                                       metrics_te_site['SMSE'][0],
                                                       metrics_te_site['RMSE'][0],
                                                       metrics_te_site['Rho'][0]]

os.chdir(data_dir)
# Save per site test set metrics variable to CSV file
hbr_site_metrics.to_csv(os.path.join(data_dir, 'hbr_UCLA_patients_site_metrics_f1.csv'), index=False, index_label=None)

# Save overall test set metrics to CSV file
hbr_metrics.to_csv(os.path.join(data_dir, 'hbr_UCLA_patients_metrics_f1.csv'), index=False, index_label=None)
