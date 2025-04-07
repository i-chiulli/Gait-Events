#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:55:42 2024
BME 3740
Unit 2, Gait Analysis

@author: bella
"""

# %% Import Packages

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import gait_metrics as gm
from statsmodels.stats.power import TTestIndPower
from scipy.stats import pearsonr

#%% Define Parameters

fs = 125    # sampling frequency
subject_count = 10  #number of subjects
data_types = ['chest_accel', 'chest_gyro', 'shank_accel', 'shank_gyro'] # types of data collected for each subject
crop_start_time = 60 # seconds 
crop_stop_time = 75  # seconds
time = np.linspace(0,15,fs*15)
subject_range = range(1,11) # range of subject count

# %% Load Data

# create dict
all_data = {}
# import data for each subject
for subject in range (1,subject_count +1):
    # get subject id's
    subject_id = f'subject{subject}'
    # create lists within each subject dict
    all_data[subject_id] = {'raw_data': {}, 'gait_metrics': []}
    # import each data type, for each subject
    for file_type in data_types:
        # define file path
        file_path = f"RawData/s{subject}_{file_type}.csv"  
        # import data file
        data_file = pd.read_csv(file_path)
        # crop data
        cropped_data_file = data_file.iloc[crop_start_time*fs:crop_stop_time*fs,:]
        # add to dictionary
        all_data[subject_id]['raw_data'][f's{subject}_{file_type}'] = cropped_data_file
        

# define keys
subject_keys = list(all_data.keys())
#%%
for subject in subject_range:
    plt.figure(100+subject, clear=True)
    gyro_time_length = len(all_data[f'subject{subject}']['raw_data'][f's{subject}_shank_gyro'].iloc[:, 3])
    gyro_time_seconds = np.arange(0, gyro_time_length/fs, 1/fs)

    accel_time_length = len(all_data[f'subject{subject}']['raw_data'][f's{subject}_chest_accel'].iloc[:, 2])
    accel_time_seconds = np.arange(0, accel_time_length/fs, 1/fs)
    
    plt.subplot(2,1,1)
    plt.plot(gyro_time_seconds, all_data[f'subject{subject}']['raw_data'][f's{subject}_shank_gyro'].iloc[:, 3])
    # annotate plot
    plt.title(f'Subject {subject}, ML Shank Angular Velocity')
    plt.xlabel('time (s)')
    plt.ylabel('angular velocity (deg/s)')
    
    plt.subplot(2,1,2)
    plt.plot(accel_time_seconds, all_data[f'subject{subject}']['raw_data'][f's{subject}_chest_accel'].iloc[:, 2])
    # annotate plot
    plt.title(f'Subject {subject}, CC Chest Acceleration')
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (g)')
# %% Question 1

# create figure
plt.figure(100, clear=True)

# create time array in seconds
gyro_time_length = len(all_data['subject1']['raw_data']['s1_shank_gyro'].iloc[:, 3])
gyro_time_seconds = np.arange(0, gyro_time_length/fs, 1/fs)

accel_time_length = len(all_data['subject1']['raw_data']['s1_chest_accel'].iloc[:, 2])
accel_time_seconds = np.arange(0, accel_time_length/fs, 1/fs)

# plot ML shank angular velocity
plt.subplot(1,2,1)
plt.plot(gyro_time_seconds, all_data['subject1']['raw_data']['s1_shank_gyro'].iloc[:, 3])
# annotate plot
plt.title('Subject 1, ML Shank Angular Velocity')
plt.xlabel('time (s)')
plt.ylabel('angular velocity (deg/s)')

# label toe-off
plt.scatter(6.43,-230, color='red', marker='.', label='Toe-off', s=150)

# label heelstrike
plt.scatter(6.863,-225, color='pink', marker='^', label='Heelstrike', s=100)

# label swing
# define swing section
start_swing_gyro = 6.43
end_swing_gyro = 6.865
start_swing_index_gyro = np.where(gyro_time_seconds >= start_swing_gyro)[0][0]
end_swing_index_gyro = np.where(gyro_time_seconds >= end_swing_gyro)[0][0]
# change line color for swing
plt.plot(gyro_time_seconds[start_swing_index_gyro:end_swing_index_gyro], all_data['subject1']['raw_data']['s1_shank_gyro'].iloc[start_swing_index_gyro:end_swing_index_gyro, 3], 
          color='orange', label='Swing')

# label stance
start_stance_gyro = 6.86
end_stance_gyro = 7.42
start_stance_index_gyro = np.where(gyro_time_seconds >= start_stance_gyro)[0][0]
end_stance_index_gyro = np.where(gyro_time_seconds >= end_stance_gyro)[0][0]
# change line color for swing
plt.plot(gyro_time_seconds[start_stance_index_gyro:end_stance_index_gyro], all_data['subject1']['raw_data']['s1_shank_gyro'].iloc[start_stance_index_gyro:end_stance_index_gyro, 3], 
          color='purple', label='Stance')

# adjust layout
# plt.xlim(6.25,7.5)
plt.tight_layout
plt.legend()


# plot CC chest accel
plt.subplot(1,2,2)
plt.plot(accel_time_seconds, all_data['subject1']['raw_data']['s1_chest_accel'].iloc[:, 2])
# annotate plot
plt.title('Subject 1, CC Chest Acceleration')
plt.xlabel('time (s)')
plt.ylabel('acceleration (g)')

# label toe-off
plt.scatter(6.35, 1.75, color='red', marker='.', label='Toe-off', s=150)

# label heelstrike
plt.scatter(6.8, 1.3, color='pink', marker='^', label='Heelstrike', s=100)

# Label swing
# Define swing section
start_swing_accel = 6.35
end_swing_accel = 6.8
start_swing_index_accel = np.where(accel_time_seconds >= start_swing_accel)[0][0]
end_swing_index_accel = np.where(accel_time_seconds >= end_swing_accel)[0][0]
# Change line color for swing
plt.plot(accel_time_seconds[start_swing_index_accel:end_swing_index_accel], all_data['subject1']['raw_data']['s1_chest_accel'].iloc[start_swing_index_accel:end_swing_index_accel, 2], 
          color='orange', label='Swing')

# Label stance
start_stance_accel = 6.8
end_stance_accel = 7.321
start_stance_index_accel = np.where(accel_time_seconds >= start_stance_accel)[0][0]
end_stance_index_accel = np.where(accel_time_seconds >= end_stance_accel)[0][0]
# Change line color for stance
plt.plot(accel_time_seconds[start_stance_index_accel:end_stance_index_accel], all_data['subject1']['raw_data']['s1_chest_accel'].iloc[start_stance_index_accel:end_stance_index_accel, 2], 
          color='purple', label='Stance')

# adjust layout
# plt.xlim(6.25,7.5)
# plt.ylim(0,2.75)
plt.tight_layout
plt.legend()

#%% Questions 2-5, Q6 Pt.1


# iterate through each subject
for subject in subject_range:
    
    # pull full r x 4 array for each sub
    data_accel = all_data[f'subject{subject}']['raw_data'][f's{subject}_chest_accel']
    data_gyro = all_data[f'subject{subject}']['raw_data'][f's{subject}_shank_gyro']
    
    # get HS and TO indices -- in tuple
    HS_inds_accel, TO_inds_accel = gm.get_chest_accel_events(data_accel, subject)
    HS_inds_gyro, TO_inds_gyro = gm.get_shank_gyro_events(data_gyro, subject)
    
    # add to metrics list for each subject
    all_data[f'subject{subject}']['gait_metrics'].append(HS_inds_accel) # gait_metrics[0] = HS accel
    all_data[f'subject{subject}']['gait_metrics'].append(TO_inds_accel) # gait_metrics[1] = TO accel
    all_data[f'subject{subject}']['gait_metrics'].append(HS_inds_gyro) # gait_metrics[2] = HS gyro 
    all_data[f'subject{subject}']['gait_metrics'].append(TO_inds_gyro) # gait_metrics[3] = TO gyro

    
# QUESTION 6
    
    # create lists of stance, swing, and stride time for each gait cycle, for each subject
    subject_stance_times_accel = []
    subject_swing_times_accel = []
    subject_stride_times_accel = []
    subject_stance_times_gyro = []
    subject_swing_times_gyro = []
    subject_stride_times_gyro = []
    

    
    # get number of gait cycles in order to get parameters for each
    num_cycles = min(len(HS_inds_accel[0]), len(TO_inds_accel[0]), len(HS_inds_gyro[0]), len(TO_inds_gyro[0]))
    
    '''
    TO detected first
    stance = HS-->TO = TO[i+1] - HS[i]
    swing = TO-->HS = HS[i] - TO[i]
    stride = HS-->HS = HS[i+1] - HS[i]
    ACCEL: keep indexes of only one leg
    '''
    
    # get stance
    for i in range(num_cycles-1):
        # compute accel stance time and append to list
        stance_time_accel =  TO_inds_accel[0][i+1] - HS_inds_accel[0][i]
        subject_stance_times_accel.append(stance_time_accel)
        # compute gyro swing time and append to list
        stance_time_gyro = TO_inds_gyro[0][i+1] - HS_inds_gyro[0][i]
        subject_stance_times_gyro.append(stance_time_gyro)
    
        # compute accel swing time and append to list
        swing_time_accel = HS_inds_accel[0][i+1] - TO_inds_accel[0][i]
        subject_swing_times_accel.append(swing_time_accel)
        # compute gyro swing time and append to list
        swing_time_gyro = HS_inds_gyro[0][i+1] - TO_inds_gyro[0][i]
        subject_swing_times_gyro.append(swing_time_gyro)
        
        # compute accel swing time and append to list
        stride_time_accel = HS_inds_accel[0][i+1] - HS_inds_accel[0][i]
        subject_stride_times_accel.append(stride_time_accel)
        # compute gyro swing time and append to list
        stride_time_gyro = HS_inds_gyro[0][i+1] - HS_inds_gyro[0][i]
        subject_stride_times_gyro.append(stride_time_gyro)
    
    # get rid of opposite foot for accelerometer data
    
    # Append parameters for the subject to the list
    all_data[f'subject{subject}']['gait_metrics'].append(subject_stance_times_accel[::2]) # gait_metrics[4] = stance accel
    all_data[f'subject{subject}']['gait_metrics'].append(subject_swing_times_accel[::2]) # gait_metrics[5] = swing accel
    all_data[f'subject{subject}']['gait_metrics'].append(subject_stride_times_accel[::2]) # gait_metrics[6] = stride accel
    all_data[f'subject{subject}']['gait_metrics'].append(subject_stance_times_gyro)  # gait_metrics[7] = stance gyro
    all_data[f'subject{subject}']['gait_metrics'].append(subject_swing_times_gyro)  # gait_metrics[8] = swing gyro
    all_data[f'subject{subject}']['gait_metrics'].append(subject_stride_times_gyro)  # gait_metrics[9] = stride gyro
 

#%% Question 6 Pt.2

# create lists to store plotting data
accel_stance_times = []
gyro_stance_times = []
accel_swing_times = []
gyro_swing_times = []
accel_stride_times = []
gyro_stride_times = []

# iterate through each subject 
for subject in subject_range:
    # sort accel and gyro metrics
    accel_metrics = all_data[f'subject{subject}']['gait_metrics'][4:7]
    gyro_metrics = all_data[f'subject{subject}']['gait_metrics'][7:]
    # store metrics in lists for plotting
    accel_stance_times.extend(accel_metrics[0])
    gyro_stance_times.extend(gyro_metrics[0])
    accel_swing_times.extend(accel_metrics[1])
    gyro_swing_times.extend(gyro_metrics[1])
    accel_stride_times.extend(accel_metrics[2])
    gyro_stride_times.extend(gyro_metrics[2])


# stance plot
plt.figure(200, clear=True)
# get same length for plotting
min_length_stance = min(len(gyro_stance_times), len(accel_stance_times))
gyro_stance_times_plot = gyro_stance_times[:min_length_stance]
accel_stance_times_plot = accel_stance_times[:min_length_stance]
plt.scatter(gyro_stance_times_plot, accel_stance_times_plot, color='blue')
# annotate plot
plt.title('Gyroscope vs. Accelerometer - Stance')
plt.xlabel('Gyroscope Data')
plt.ylabel('Accelerometer Data')
plt.grid(True)
plt.show()

# stance correlation
stance_corr, stance_p_val = pearsonr(gyro_stance_times_plot, accel_stance_times_plot)
# correlation strength
if stance_corr <= 0.6 and stance_corr > 0.3:
    print(f'There is a moderate correlation for stance between accelerometer and gyroscope data, with a correlation coefficient of {stance_corr}.')
elif stance_corr > 0.6:
    print(f'There is a strong correlation for stance between accelerometer and gyroscope data, with a correlation coefficient of {stance_corr}.')
elif stance_corr <= 0.3:
    print(f'There is a weak correlation for stance between accelerometer and gyroscope data, with a correlation coefficient of {stance_corr}.')
# statistical significance?
if stance_p_val <= 0.05:
    print(f'There is statistically significant correlation of stance times between gyroscope and accelerometer data (p-value = {stance_p_val}).')
else:
    print('There is no statistically significant correlation for stance.')

# swing plot
plt.figure(201, clear=True)
# get same length for plotting
min_length_swing = min(len(gyro_swing_times), len(accel_swing_times))
gyro_swing_times_plot = gyro_swing_times[:min_length_swing]
accel_swing_times_plot = accel_swing_times[:min_length_swing]
plt.scatter(gyro_swing_times_plot, accel_swing_times_plot, color='red')
# annotate plot
plt.title('Gyroscope vs. Accelerometer - Swing')
plt.xlabel('Gyroscope Data')
plt.ylabel('Accelerometer Data')
plt.grid(True)
plt.show()

# swing correlation
swing_corr, swing_p_val = pearsonr(gyro_swing_times_plot, accel_swing_times_plot)
# correlation strength
if swing_corr <= 0.6 and swing_corr > 0.3:
    print(f'There is a moderate correlation for swing between accelerometer and gyroscope data, with a correlation coefficient of {swing_corr}.')
elif swing_corr > 0.6:
    print(f'There is a strong correlation for swing between accelerometer and gyroscope data, with a correlation coefficient of {swing_corr}.')
elif swing_corr <= 0.3:
    print(f'There is a weak correlation for swing between accelerometer and gyroscope data, with a correlation coefficient of {swing_corr}.')

# statistical signficance?
if swing_p_val <= 0.05:
    print(f'There is statistically significant correlation of swing times between gyroscope and accelerometer data (p-value = {swing_p_val}).')
else:
    print('There is no statistically significant correlation for swing.')

# stride plot
plt.figure(202, clear=True)
# get same length for plotting
min_length_stride = min(len(gyro_stride_times), len(accel_stride_times))
gyro_stride_times_plot = gyro_stride_times[:min_length_stride]
accel_stride_times_plot = accel_stride_times[:min_length_stride]
plt.scatter(gyro_stride_times_plot, accel_stride_times_plot, color='green')
# annotate plot
plt.title('Gyroscope vs. Accelerometer - Stride')
plt.xlabel('Gyroscope Data')
plt.ylabel('Accelerometer Data')
plt.grid(True)
plt.show()

# stride correlation
stride_corr, stride_p_val = pearsonr(gyro_stride_times_plot, accel_stride_times_plot)
# correlation strength
if stride_corr <= 0.6 and stride_corr > 0.3:
    print(f'There is a moderate correlation for stride between accelerometer and gyroscope data, with a correlation coefficient of {stride_corr}.')
elif swing_corr > 0.6:
    print(f'There is a strong correlation for stride between accelerometer and gyroscope data, with a correlation coefficient of {stride_corr}.')
elif swing_corr <= 0.3:
    print(f'There is a weak correlation for stride between accelerometer and gyroscope data, with a correlation coefficient of {stride_corr}.')
    
# statistical significance?
if stride_p_val <= 0.05:
    print(f'There is statistically significant correlation of stride times between gyroscope and accelerometer data (p-value = {stride_p_val}).')
else:
    print('There is no statistically significant correlation for stride.')



#%% Question 7


# Initiate the power analysis
power_analysis = TTestIndPower()

# Calculate sample size
sample_size = power_analysis.solve_power(effect_size = 0.8, alpha = 0.05, power = 0.8, alternative = 'two-sided')

# Print results
print('The sample size needed for each group is', round(sample_size))


#%% DIDN'T USE
    #     # tuple --> array
    #     HS_accel_stance = HS_inds_accel[0][i]
    #     TO_accel_stance = TO_inds_accel[0][i+1]
    #     HS_gyro_stance = HS_inds_gyro[0][i]
    #     TO_gyro_stance = TO_inds_gyro[0][i+1]

    #     # make sure same size in order to compute
    #     if HS_accel_stance.size == TO_accel_stance.size == HS_gyro_stance.size == TO_gyro_stance.size:
    #         # compute accel stance time and append to list
    #         stance_time_accel = TO_accel_stance - HS_accel_stance
    #         subject_stance_times_accel.append(stance_time_accel)
    #         # compute gyro stance time and append to list
    #         stance_time_gyro = TO_gyro_stance - HS_gyro_stance
    #         subject_stance_times_gyro.append(stance_time_gyro)
    #     else:
    #         print('Gait cycle lengths do not match in order to compute.')
    
    # # Append stance times for the subject to the list
    # all_data[f'subject{subject}']['gait_metrics'].append(subject_stance_times_accel)
    # all_data[f'subject{subject}']['gait_metrics'].append(subject_stance_times_gyro)
    
    # # get swing
    # for i in range(num_cycles-1):
    #     # tuple --> array
    #     HS_accel_swing = HS_inds_accel[0][i+1]
    #     TO_accel_swing = TO_inds_accel[0][i]
    #     HS_gyro_swing = HS_inds_gyro[0][i+1]
    #     TO_gyro_swing= TO_inds_gyro[0][i]
        
    #     # make sure same size in order to compute
    #     if HS_accel_stance.size == TO_accel_stance.size == HS_gyro_stance.size == TO_gyro_stance.size:
    #         # compute accel swing time and append to list
    #         swing_time_accel = HS_accel_swing - TO_accel_swing
    #         subject_swing_times_accel.append(swing_time_accel)
    #         # compute gyro swing time and append to list
    #         swing_time_gyro = HS_gyro_swing - TO_gyro_swing
    #         subject_swing_times_gyro.append(swing_time_gyro)
    #     else:
    #          print('Gait cycle lengths do not match in order to compute.') 
             
    # # Append swing times for the subject to the list
    # all_data[f'subject{subject}']['gait_metrics'].append(subject_swing_times_accel)
    # all_data[f'subject{subject}']['gait_metrics'].append(subject_swing_times_gyro)
    
    # # get stride
    # for i in range(num_cycles-1):
    #     # compute accel swing time and append to list
    #     stride_time_accel = HS_inds_accel[0][i+1] - HS_inds_accel[0][i]
    #     subject_stride_times_accel.append(stride_time_accel)
    #     # compute gyro swing time and append to list
    #     stride_time_gyro = HS_inds_gyro[0][i+1] - HS_inds_gyro[0][i]
    #     subject_stride_times_gyro.append(stride_time_gyro)
             
    # # Append stride times for the subject to the list
    # all_data[f'subject{subject}']['gait_metrics'].append(subject_stride_times_accel)
    # all_data[f'subject{subject}']['gait_metrics'].append(subject_stride_times_gyro)
