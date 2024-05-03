# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:52:31 2021

@author: Julio
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pcntoolkit.normative import estimate, evaluate
from scipy.stats import norm
import argparse
import pickle
__author__ = 'Julio Villalón'
 
parser = argparse.ArgumentParser(description='This runs the HBR normative modeling with HBR.')
parser.add_argument('-controls','--controls_csv', help='Input table with controls, including the site ID.',required=True)
parser.add_argument('-dirO','--dirOutput', help='Ouput directory. Put slash at the end.',required=True)
parser.add_argument('-age_column','--age_column', help='Name of the age column header.',required=True)
parser.add_argument('-site_column','--site_column', help='Name of the site column header.',required=True)
parser.add_argument('-sex_column','--sex_column', help='Name of the sex column header.',required=True)
parser.add_argument('-outscaler','--outscaler', help='Scaling approach for output responses,\
                    could be None (Default), standardize, minmax, or robminmax.',required=True)
args = parser.parse_args()

## show the inputs ##
print("Input controls file: %s" % args.controls_csv)
print("Output directory: %s" % args.dirOutput)
print("Name of the site column: %s" % args.site_column)
print("Name of the age column: %s" % args.age_column)
print("Name of the sex column: %s" % args.sex_column)
print("Scaling approach for output response, outscaler: %s" % args.outscaler)

data_dir = args.dirOutput
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 21 ROIS + Average WM = 22 total rois. Bilateral, not left and right.
rois =['ACR','ALIC','Average','BCC','CGC','CGH','CST','EC','FX','FXST','GCC',
       'UNC','PCR','PLIC','PTR','RLIC','SCC','SCR','SFO','SLF','SS','TAP']

#Reading in the controls table. Making sure 
dmri = args.controls_csv
train_all = pd.read_csv(dmri, dtype={'subjectID': str, 'SID': str, 'Protocol_No': int, 'Protocol': str, 'Study': str})
train_all["site"] = train_all[args.site_column]  # Creating a new column copied from the original site column.
                                                    # This makes the new columns called site_HCP, site_ABCD, site_XXX

train_all_site = pd.get_dummies(train_all, columns=['site'])
train_all_site['site_ID'] = train_all['site']  #adding the columns with site names to the new frame
train_all_site['site_ID_bin'] = train_all['Protocol_No']  # this is the numeric version of Protocol

controls_big = train_all_site.copy()
train_site = controls_big['site_ID_bin']


controls_cov_big = controls_big[['subjectID',
                                 args.age_column,
                                 args.sex_column,
                                 'site_ID',
                                 'site_ID_bin',
                                 'site_ABCD_SIEMENS',	'site_ABCD_GE',	
                                 'site_ABCD_PHILIPS',	'site_ADNI3_GE36',	
                                 'site_ADNI3_GE54',	'site_ADNI3_P33',	
                                 'site_ADNI3_P36',	'site_ADNI3_S127',	
                                 'site_ADNI3_S31',	'site_ADNI3_S55',	
                                 'site_AOMIC_ID1000',	'site_AOMIC_PIOP1',	
                                 'site_AOMIC_PIOP2',	'site_CAMCAN',	'site_CHBMP',	
                                 'site_CHCP',	'site_HBN_CBIC',	'site_HBN_CUNY',	
                                 'site_HBN_RUBIC',	'site_HBN_SI',	'site_HCP_A',	
                                 'site_HCP_D',	'site_HCP_YA',	
                                 'site_NIH_Peds_dti04_SIEMENS',	'site_NIH_Peds_dti04_GE',	
                                 'site_NIH_Peds_edti02_SIEMENS',	'site_NIH_Peds_edti02_GE',	
                                 'site_OASIS3',	'site_PING_GE',	'site_PING_SIEMENS',	
                                 'site_PING_PHILIPS',	'site_PNC',	'site_PPMI',	
                                 'site_QTAB',	'site_QTIM',	'site_SLIM',	'site_UKBB']]

controls_features_big = controls_big[rois]                               
                                
                                
X_train8020_f1, X_test8020_f1, y_train8020_f1, y_test8020_f1 = train_test_split(controls_cov_big, 
                                                                                controls_features_big, stratify=train_site, 
                                                                                test_size=0.2, random_state=54639)

X_train_f1 = X_train8020_f1.copy()
X_test_f1 = X_test8020_f1.copy()
y_train_f1 = y_train8020_f1.copy()
y_test_f1 = y_test8020_f1.copy()

X_train_f1.reset_index(drop=True, inplace=True)
X_test_f1.reset_index(drop=True, inplace=True)
y_train_f1.reset_index(drop=True, inplace=True)
y_test_f1.reset_index(drop=True, inplace=True)

ABCD_SIEMENS_te= X_test_f1.index[X_test_f1['site_ABCD_SIEMENS'] == 1].to_list()
ABCD_GE_te= X_test_f1.index[X_test_f1['site_ABCD_GE'] == 1].to_list()
ABCD_PHILIPS_te= X_test_f1.index[X_test_f1['site_ABCD_PHILIPS'] == 1].to_list()
ADNI3_GE36_te= X_test_f1.index[X_test_f1['site_ADNI3_GE36'] == 1].to_list()
ADNI3_GE54_te= X_test_f1.index[X_test_f1['site_ADNI3_GE54'] == 1].to_list()
ADNI3_P33_te= X_test_f1.index[X_test_f1['site_ADNI3_P33'] == 1].to_list()
ADNI3_P36_te= X_test_f1.index[X_test_f1['site_ADNI3_P36'] == 1].to_list()
ADNI3_S127_te= X_test_f1.index[X_test_f1['site_ADNI3_S127'] == 1].to_list()
ADNI3_S31_te= X_test_f1.index[X_test_f1['site_ADNI3_S31'] == 1].to_list()
ADNI3_S55_te= X_test_f1.index[X_test_f1['site_ADNI3_S55'] == 1].to_list()
AOMIC_ID1000_te= X_test_f1.index[X_test_f1['site_AOMIC_ID1000'] == 1].to_list()
AOMIC_PIOP1_te= X_test_f1.index[X_test_f1['site_AOMIC_PIOP1'] == 1].to_list()
AOMIC_PIOP2_te= X_test_f1.index[X_test_f1['site_AOMIC_PIOP2'] == 1].to_list()
CAMCAN_te= X_test_f1.index[X_test_f1['site_CAMCAN'] == 1].to_list()
CHBMP_te= X_test_f1.index[X_test_f1['site_CHBMP'] == 1].to_list()
CHCP_te= X_test_f1.index[X_test_f1['site_CHCP'] == 1].to_list()
HBN_CBIC_te= X_test_f1.index[X_test_f1['site_HBN_CBIC'] == 1].to_list()
HBN_CUNY_te= X_test_f1.index[X_test_f1['site_HBN_CUNY'] == 1].to_list()
HBN_RUBIC_te= X_test_f1.index[X_test_f1['site_HBN_RUBIC'] == 1].to_list()
HBN_SI_te= X_test_f1.index[X_test_f1['site_HBN_SI'] == 1].to_list()
HCP_A_te= X_test_f1.index[X_test_f1['site_HCP_A'] == 1].to_list()
HCP_D_te= X_test_f1.index[X_test_f1['site_HCP_D'] == 1].to_list()
HCP_YA_te= X_test_f1.index[X_test_f1['site_HCP_YA'] == 1].to_list()
NIH_Peds_dti04_SIEMENS_te= X_test_f1.index[X_test_f1['site_NIH_Peds_dti04_SIEMENS'] == 1].to_list()
NIH_Peds_dti04_GE_te= X_test_f1.index[X_test_f1['site_NIH_Peds_dti04_GE'] == 1].to_list()
NIH_Peds_edti02_SIEMENS_te= X_test_f1.index[X_test_f1['site_NIH_Peds_edti02_SIEMENS'] == 1].to_list()
NIH_Peds_edti02_GE_te= X_test_f1.index[X_test_f1['site_NIH_Peds_edti02_GE'] == 1].to_list()
OASIS3_te= X_test_f1.index[X_test_f1['site_OASIS3'] == 1].to_list()
PING_GE_te= X_test_f1.index[X_test_f1['site_PING_GE'] == 1].to_list()
PING_SIEMENS_te= X_test_f1.index[X_test_f1['site_PING_SIEMENS'] == 1].to_list()
PING_PHILIPS_te= X_test_f1.index[X_test_f1['site_PING_PHILIPS'] == 1].to_list()
PNC_te= X_test_f1.index[X_test_f1['site_PNC'] == 1].to_list()
PPMI_te= X_test_f1.index[X_test_f1['site_PPMI'] == 1].to_list()
QTAB_te= X_test_f1.index[X_test_f1['site_QTAB'] == 1].to_list()
QTIM_te= X_test_f1.index[X_test_f1['site_QTIM'] == 1].to_list()
SLIM_te= X_test_f1.index[X_test_f1['site_SLIM'] == 1].to_list()
UKBB_te= X_test_f1.index[X_test_f1['site_UKBB'] == 1].to_list()


dfbatch_tr = X_train_f1[['site_ID_bin']]
dfbatch_te = X_test_f1[['site_ID_bin']]

dfbatch_tr.to_csv(os.path.join(data_dir, 'batch_tr.txt'),
                  na_rep='NaN', index=False, header=False, sep=' ')
with open(data_dir + '/batch_tr.pkl', 'wb') as file:
    pickle.dump(dfbatch_tr, file)

dfbatch_te.to_csv(os.path.join(data_dir, 'batch_te.txt'),
                  na_rep='NaN', index=False, header=False, sep=' ')
with open(data_dir + '/batch_te.pkl', 'wb') as file:
    pickle.dump(dfbatch_te, file)

dfsubject_tr = X_train_f1[['subjectID']]
dfsubject_te = X_test_f1[['subjectID']]
dfsubject_tr.to_csv(os.path.join(data_dir, 'subjects_tr.txt'),
                    na_rep='NaN', index=False, header=False, sep=' ')
dfsubject_te.to_csv(os.path.join(data_dir, 'subjects_te.txt'),
                    na_rep='NaN', index=False, header=False, sep=' ')


X_train_f1 = X_train_f1.drop(['subjectID', 'site_ID', 'site_ID_bin',
                                'site_ABCD_SIEMENS',	'site_ABCD_GE',	
                                'site_ABCD_PHILIPS',	'site_ADNI3_GE36',	
                                'site_ADNI3_GE54',	'site_ADNI3_P33',	
                                'site_ADNI3_P36',	'site_ADNI3_S127',	
                                'site_ADNI3_S31',	'site_ADNI3_S55',	
                                'site_AOMIC_ID1000',	'site_AOMIC_PIOP1',	
                                'site_AOMIC_PIOP2',	'site_CAMCAN',	'site_CHBMP',	
                                'site_CHCP',	'site_HBN_CBIC',	'site_HBN_CUNY',	
                                'site_HBN_RUBIC',	'site_HBN_SI',	'site_HCP_A',	
                                'site_HCP_D',	'site_HCP_YA',	
                                'site_NIH_Peds_dti04_SIEMENS',	'site_NIH_Peds_dti04_GE',	
                                'site_NIH_Peds_edti02_SIEMENS',	'site_NIH_Peds_edti02_GE',	
                                'site_OASIS3',	'site_PING_GE',	'site_PING_SIEMENS',	
                                'site_PING_PHILIPS',	'site_PNC',	'site_PPMI',	
                                'site_QTAB',	'site_QTIM',	'site_SLIM',	'site_UKBB'], axis=1)

X_test_f1 = X_test_f1.drop(['subjectID', 'site_ID', 'site_ID_bin',
                            'site_ABCD_SIEMENS',	'site_ABCD_GE',	
                            'site_ABCD_PHILIPS',	'site_ADNI3_GE36',	
                            'site_ADNI3_GE54',	'site_ADNI3_P33',	
                            'site_ADNI3_P36',	'site_ADNI3_S127',	
                            'site_ADNI3_S31',	'site_ADNI3_S55',	
                            'site_AOMIC_ID1000',	'site_AOMIC_PIOP1',	
                            'site_AOMIC_PIOP2',	'site_CAMCAN',	'site_CHBMP',	
                            'site_CHCP',	'site_HBN_CBIC',	'site_HBN_CUNY',	
                            'site_HBN_RUBIC',	'site_HBN_SI',	'site_HCP_A',	
                            'site_HCP_D',	'site_HCP_YA',	
                            'site_NIH_Peds_dti04_SIEMENS',	'site_NIH_Peds_dti04_GE',	
                            'site_NIH_Peds_edti02_SIEMENS',	'site_NIH_Peds_edti02_GE',	
                            'site_OASIS3',	'site_PING_GE',	'site_PING_SIEMENS',	
                            'site_PING_PHILIPS',	'site_PNC',	'site_PPMI',	
                            'site_QTAB',	'site_QTIM',	'site_SLIM',	'site_UKBB'], axis=1)

X_train_f1.loc[:, args.age_column] = X_train_f1[args.age_column] / 100
X_test_f1.loc[:, args.age_column] = X_test_f1[args.age_column] / 100

roi_ids = ['ACR','ALIC','Average','BCC','CGC','CGH','CST','EC','FX','FXST',
            'GCC','UNC','PCR','PLIC','PTR','RLIC','SCC','SCR','SFO','SLF','SS',
            'TAP']

# Run normative model for controls train/test split
sites = [ABCD_SIEMENS_te,	ABCD_GE_te,	ABCD_PHILIPS_te,	ADNI3_GE36_te,
         ADNI3_GE54_te,	ADNI3_P33_te,	ADNI3_P36_te,	ADNI3_S127_te,
         ADNI3_S31_te,	ADNI3_S55_te,	AOMIC_ID1000_te,	AOMIC_PIOP1_te,
         AOMIC_PIOP2_te,	CAMCAN_te,	CHBMP_te,	CHCP_te,	HBN_CBIC_te,	
         HBN_CUNY_te,	HBN_RUBIC_te,	HBN_SI_te,	HCP_A_te,	HCP_D_te,	
         HCP_YA_te,	NIH_Peds_dti04_SIEMENS_te,	NIH_Peds_dti04_GE_te,	
         NIH_Peds_edti02_SIEMENS_te,	NIH_Peds_edti02_GE_te,	OASIS3_te,	
         PING_GE_te,	PING_SIEMENS_te,	PING_PHILIPS_te,	PNC_te,	PPMI_te,	QTAB_te,	
         QTIM_te,	SLIM_te,	UKBB_te]

site_names = ['ABCD_SIEMENS_te',	'ABCD_GE_te',	'ABCD_PHILIPS_te',
              'ADNI3_GE36_te',	'ADNI3_GE54_te',	'ADNI3_P33_te',	'ADNI3_P36_te',
              'ADNI3_S127_te',	'ADNI3_S31_te',	'ADNI3_S55_te',	'AOMIC_ID1000_te',
              'AOMIC_PIOP1_te',	'AOMIC_PIOP2_te',	'CAMCAN_te',	'CHBMP_te',
              'CHCP_te',	'HBN_CBIC_te',	'HBN_CUNY_te',	'HBN_RUBIC_te',	
              'HBN_SI_te',	'HCP_A_te',	'HCP_D_te',	'HCP_YA_te',	
              'NIH_Peds_dti04_SIEMENS_te',	'NIH_Peds_dti04_GE_te',	
              'NIH_Peds_edti02_SIEMENS_te',	'NIH_Peds_edti02_GE_te',	'OASIS3_te',	
              'PING_GE_te',	'PING_SIEMENS_te',	'PING_PHILIPS_te',	'PNC_te',	
              'PPMI_te',	'QTAB_te',	'QTIM_te',	'SLIM_te',	'UKBB_te']


for roi in roi_ids: 
    print('Saving the tables for ROI:', roi)
    roi_dir = os.path.join(data_dir, roi)
    
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    np.savetxt(os.path.join(roi_dir, 'cov_int_controls_tr.txt'), X_train_f1)
    np.savetxt(os.path.join(roi_dir, 'cov_int_controls_te.txt'), X_test_f1)
    
    # Saving the Y for each roi
    np.savetxt(os.path.join(roi_dir, 'resp_controls_tr.txt'), y_train_f1[roi])
    np.savetxt(os.path.join(roi_dir, 'resp_controls_te.txt'), y_test_f1[roi])


# Create pandas dataframes with header names to save out the overall and per-site model evaluation metrics
hbr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'NLL'])
hbr_site_metrics = pd.DataFrame(columns = ['ROI', 'site', 'y_mean', 'y_var', 'yhat_mean', 'yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

# b-spline configuration
configs_spline = {
  "order": 3,
  "nknots": 5
}

# Loop through ROIs for controls 80/20 train/test split
for roi in roi_ids: 
    print('Running ROI:', roi)
    roi_dir = os.path.join(data_dir, roi)
    os.chdir(roi_dir)
     
    # configure the covariates to use. Change *_bspline_* to *_int_* to 
    cov_file_tr = os.path.join(roi_dir, 'cov_int_controls_tr.txt')
    cov_file_te = os.path.join(roi_dir, 'cov_int_controls_te.txt')
    
    # load train & test response files
    resp_file_tr = os.path.join(roi_dir, 'resp_controls_tr.txt')
    resp_file_te = os.path.join(roi_dir, 'resp_controls_te.txt')
    
    batch_tr = os.path.join(data_dir, 'batch_tr.pkl')
    batch_te = os.path.join(data_dir, 'batch_te.pkl')
    
    # run a basic model
    yhat_te, s2_te, nm, Z, metrics_te = estimate(cov_file_tr,
                                                 resp_file_tr,
                                                 testresp=resp_file_te,
                                                 testcov=cov_file_te,
                                                 alg = 'hbr',
                                                 trbefile=batch_tr,
                                                 tsbefile=batch_te,
                                                 model_type='bspline',
                                                 savemodel = True,
                                                 saveoutput = False,
                                                 linear_mu ='True',
                                                 linear_sigma='True',
                                                 random_intercept_mu='True',
                                                 random_intercept_sigma='True',
                                                 random_slope_mu='False',
                                                 random_slope_sigma='False',
                                                 outscaler=args.outscaler
                                                 )
    
    p_mosi = 1 - norm.sf(np.abs(Z)) * 2
    Z_p = np.concatenate((Z, p_mosi), axis=1)
    np.savetxt(os.path.join(roi_dir, 'Z_p_estimate.txt'), Z_p)
    np.savetxt(os.path.join(roi_dir, 'Yhat_estimate.txt'), yhat_te)
    np.savetxt(os.path.join(roi_dir, 'Ys2_estimate.txt'), s2_te)

    # display and save metrics
    keys = list(metrics_te.keys())
    print(keys)
    print(metrics_te.items())
    print('EV=', metrics_te['EXPV'][0])
    print('RHO=', metrics_te['Rho'][0])
    print('MSLL=', metrics_te['MSLL'][0])
    print('SMSE=', metrics_te['SMSE'][0])
    
    hbr_metrics.loc[len(hbr_metrics)] = [roi, metrics_te['MSLL'][0], metrics_te['EXPV'][0], metrics_te['SMSE'][0], 
                                         metrics_te['RMSE'][0], metrics_te['Rho'][0], metrics_te['NLL'][0]]
    
    # Compute metrics per site in test set, save to pandas df
    # load true test data
    X_te = np.loadtxt(cov_file_te)
    y_te = np.loadtxt(resp_file_te)
    y_te = y_te[:, np.newaxis]  # make sure it is a 2-d array
    
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
hbr_site_metrics.to_csv(os.path.join(data_dir, 'hbr_controls_site_metrics_f1.csv'), index=False, index_label=None)

# Save overall test set metrics to CSV file
hbr_metrics.to_csv(os.path.join(data_dir, 'hbr_controls_metrics_f1.csv'), index=False, index_label=None)
