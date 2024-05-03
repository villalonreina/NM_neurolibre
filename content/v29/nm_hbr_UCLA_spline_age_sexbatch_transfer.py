# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:52:31 2021

@author: Julio
"""

import os
import pandas as pd
import numpy as np
from pcntoolkit.normative import transfer, evaluate
from scipy.stats import norm
import argparse
import pickle
__author__ = 'Julio Villalón'
 
parser = argparse.ArgumentParser(description='This runs the HBR model extension.')
parser.add_argument('-controls','--controls_csv', help='Input table of the new datasets to be added, including the site IDs.',required=True)
parser.add_argument('-patients','--patients_csv', help='Input table of the new test datasets to be added (usually patients).',required=True)
parser.add_argument('-dirM','--dirModel', help='The full path to the already trained model that will be used for prediction.',required=True)
parser.add_argument('-dirO','--dirOutput', help='Ouput directory. Put slash at the end.',required=True)
parser.add_argument('-age_column','--age_column', help='Name of the age column header.',required=True)
parser.add_argument('-site_column','--site_column', help='Name of the site column header.',required=True)
parser.add_argument('-sex_column','--sex_column', help='Name of the sex column header.',required=True)
args = parser.parse_args()


## show the inputs ##
print("Input training controls file: %s" % args.controls_csv)
print("Input test patients file: %s" % args.patients_csv)
print("Model directory: %s" % args.dirModel)
print("Output directory: %s" % args.dirOutput)
print("Name of the site column: %s" % args.site_column)
print("Name of the age column: %s" % args.age_column)
print("Name of the sex column: %s" % args.sex_column)

out_dir = args.dirOutput
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

### Reading in the training new data ####
controls = pd.read_csv(args.controls_csv)
controls['site'] = controls[args.site_column]
controls_cov = pd.get_dummies(controls, columns=['site'])
controls_cov['site_ID_bin'] = controls['SID']  # this is the numeric version of Protocol

controls_cov = controls_cov[['subjectID',
                             args.age_column,
                             args.sex_column,
                             'site_ID_bin',
                             'site_UCLA_0',
                             'site_UCLA_1'
                             ]]

# Creating batch/site file
dfsubject_tr = controls_cov[['subjectID']]
dfsubject_tr.to_csv(os.path.join(out_dir, 'controls_subjects_tr.txt'),
                    na_rep='NaN', index=False, header=False, sep=' ')

dfbatch_adapt_tr = controls_cov[['site_ID_bin', args.sex_column]]
dfbatch_adapt_tr.to_csv(os.path.join(out_dir, 'controls_batch_tr.txt'),
                        na_rep='NaN', index=False, header=False, sep=' ')
batch_effects_adapt = dfbatch_adapt_tr.to_numpy(dtype=int)
with open(out_dir + 'controls_batch_tr.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_adapt), file)

controls_cov = controls_cov.drop(['subjectID', args.sex_column, 'site_ID_bin', 'site_UCLA_0', 'site_UCLA_1'], axis=1)
controls_cov[args.age_column] = pd.to_numeric(controls_cov[args.age_column])
controls_cov.loc[:, args.age_column] = controls_cov[args.age_column] / 100
controls_cov.to_csv(os.path.join(out_dir, 'cov_adapt_controls_tr.txt'),
                    na_rep='NaN', index=False, header=False, sep=' ')
X_adapt = controls_cov.to_numpy(dtype=float)
with open(out_dir + 'cov_adapt_controls_tr.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_adapt), file)

# These are all the ROIs (bilateral). Removed the CST!!!!!!!!!!!!!!!!!!!!!!!!!!
roi_ids = ['ACR', 'ALIC', 'Average', 'BCC', 'CGC', 'CGH', 'EC', 'FX',
           'FXST', 'GCC', 'UNC', 'PCR', 'PLIC', 'PTR', 'RLIC', 'SCC', 'SCR',
           'SFO', 'SLF', 'SS', 'TAP']


##############
# Test data (new dataset)
patients = pd.read_csv(args.patients_csv)

dfsubject_te = patients[['subjectID']]
dfsubject_te.to_csv(os.path.join(out_dir, 'patients_subjects_te.txt'),
                    na_rep='NaN', index=False, header=False, sep=' ')

batch_effects_test_txfr = patients[['SID', args.sex_column]]
X_test_txfr = patients[[args.age_column]]
X_test_txfr.loc[:,args.age_column] = pd.to_numeric(X_test_txfr[args.age_column])
X_test_txfr.loc[:, args.age_column] = X_test_txfr[args.age_column] / 100

X_test_txfr.to_csv(os.path.join(out_dir, 'cov_adapt_patients_te.txt'),
                   na_rep='NaN', index=False, header=False, sep=' ')
with open(out_dir + 'cov_adapt_patients_te.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(X_test_txfr), file)

batch_effects_test_txfr.to_csv(os.path.join(out_dir, 'patients_batch_te.txt'),
                               na_rep='NaN', index=False, header=False, sep=' ')
with open(out_dir + 'patients_batch_te.pkl', 'wb') as file:
    pickle.dump(pd.DataFrame(batch_effects_test_txfr), file)

# Indexing sites for output metrics
patients['site'] = patients[args.site_column]
patients_cov = pd.get_dummies(patients, columns=['site'])
UCLA_0_te = patients_cov.index[patients_cov['site_UCLA_0'] == 1].to_list()
UCLA_1_te = patients_cov.index[patients_cov['site_UCLA_1'] == 1].to_list()
sites = [UCLA_0_te, UCLA_1_te]
site_names = ['UCLA_0_te', 'UCLA_1_te']


### Here we create the ROI text files to be called by the normative modeling ###
for roi in roi_ids: 
    print('Saving the tables for ROI:', roi)
    roi_dir = os.path.join(out_dir, roi)
    
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)
       
    # Saving the Y for each roi
    np.savetxt(os.path.join(roi_dir, 'resp_adapt_controls_tr.txt'), controls[roi])
    Y_adapt = controls[roi].to_numpy(dtype=float)
    with open(roi_dir + '/resp_adapt_controls_tr.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(Y_adapt), file)
    
    np.savetxt(os.path.join(roi_dir, 'resp_adapt_patients_te.txt'), patients[roi])
    Y_test_txfr = patients[roi].to_numpy(dtype=float)
    with open(roi_dir + '/resp_adapt_patients_te.pkl', 'wb') as file:
        pickle.dump(pd.DataFrame(), file)

# Create pandas dataframes with header names to save out the overall and per-site model evaluation metrics
hbr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])
hbr_site_metrics = pd.DataFrame(columns = ['ROI', 'site', 'y_mean', 'y_var', 'yhat_mean', 'yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])


### From here on is the actual model extension ###
for roi in roi_ids: 
    print('Running ROI:', roi)
    roi_dir = os.path.join(out_dir, roi)
    os.chdir(roi_dir)
     
    # load train covariates to use (controls).
    covfile = os.path.join(out_dir, 'cov_adapt_controls_tr.pkl')
    print(os.path.join(out_dir, 'cov_adapt_controls_tr.pkl'))
    
    # load train response files (controls).
    respfile = os.path.join(roi_dir, 'resp_adapt_controls_tr.pkl')
    print(os.path.join(roi_dir, 'resp_adapt_controls_tr.pkl'))

    # load the train batch file (controls).
    trbefile = os.path.join(out_dir + 'controls_batch_tr.pkl')
    print(os.path.join(out_dir + 'controls_batch_tr.pkl'))
    
    model_path = os.path.join(args.dirModel, roi, 'Models') # path to the previously trained models
    print(os.path.join(args.dirModel, roi, 'Models'))
    
    # load test response files (patients).
    testcovfile_path = os.path.join(out_dir, 'cov_adapt_patients_te.pkl')
    print(os.path.join(out_dir, 'cov_adapt_patients_te.pkl'))
    
    # load test response files (patients).
    testrespfile_path = os.path.join(roi_dir, 'resp_adapt_patients_te.txt')
    print(os.path.join(roi_dir, 'resp_adapt_patients_te.txt'))
    
    # load the test batch file (patients).
    tsbefile = os.path.join(out_dir, 'patients_batch_te.pkl')
    print(os.path.join(out_dir, 'patients_batch_te.pkl'))

    output_path = os.path.join(roi_dir, 'Models/')
    outputsuffix = '_transfer'  # suffix added to the output files from the transfer function


    predicted = transfer(covfile=covfile,
                         respfile=respfile,
                         trbefile=trbefile,
                         model_path=model_path,
                         alg='hbr',
                         output_path=output_path,
                         testcov=testcovfile_path,
                         testresp=testrespfile_path,
                         tsbefile=tsbefile,
                         outputsuffix=outputsuffix,
                         savemodel=True
                         )
    
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
    MSLL = pd.read_pickle(''.join([roi_dir, '/MSLL_transfer.pkl']))
    EXPV = pd.read_pickle(''.join([roi_dir, '/EXPV_transfer.pkl']))
    SMSE = pd.read_pickle(''.join([roi_dir, '/SMSE_transfer.pkl']))
    RMSE = pd.read_pickle(''.join([roi_dir, '/RMSE_transfer.pkl']))
    Rho = pd.read_pickle(''.join([roi_dir, '/Rho_transfer.pkl']))
    
    # Stacking the evaluation metrics by ROI in a big table
    hbr_metrics.loc[len(hbr_metrics)] = [roi, MSLL[0][0], EXPV[0][0],
                                         SMSE[0][0], RMSE[0][0], Rho[0][0]]
    
    # Compute metrics per site in test set, save to pandas df
    # load true test data
    y_te = np.loadtxt(testrespfile_path)
    y_te = y_te[:, np.newaxis]  # make sure it is a 2-d array    
    # y_te_df = pd.read_pickle(testrespfile_path)  # For v29 it can be pickle
    # y_te = y_te_df.to_numpy()
      
    for num, site in enumerate(sites):     
        y_mean_te_site = np.array([[np.mean(y_te[site])]])
        y_var_te_site = np.array([[np.var(y_te[site])]])
        yhat_mean_te_site = np.array([[np.mean(yhat_te[site])]])
        yhat_var_te_site = np.array([[np.var(yhat_te[site])]])
        
        metrics_te_site = evaluate(y_te[site], yhat_te[site], s2_te[site], y_mean_te_site, y_var_te_site,
                                    metrics=['Rho', 'RMSE', 'SMSE', 'EXPV', 'MSLL'])
        
        if 'MSLL' in metrics_te_site:
            print("MSLL in final metrics")
        else:
            metrics_te_site.update({'MSLL': [0]})
        if 'SMSE' in metrics_te_site:
            print("SMSE in final metrics")
        else:
            metrics_te_site.update({'SMSE': [0]})
        
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
os.chdir(out_dir)
# Save per site test set metrics variable to CSV file
hbr_site_metrics.to_csv(os.path.join(out_dir, 'hbr_UCLA_site_metrics_f1.csv'), index=False, index_label=None)

# Save overall test set metrics to CSV file
hbr_metrics.to_csv(os.path.join(out_dir, 'hbr_UCLA_metrics_f1.csv'), index=False, index_label=None)
