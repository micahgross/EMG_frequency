# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 08:53:07 2021

@author: Micah Gross

"""
# initiate app in Anaconda Navigator with
# cd "C:\Users\BASPO\.spyder-py3\EMG"
# streamlit run EMG_freq_app.py
# cd "C:\Users\BASPO\.spyder-py3\streamlit_apps\EMG_freq"
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy import stats
from scipy import integrate
# from scipy.signal import periodogram
# from scipy.integrate import cumtrapz
# import time
#%%
# '''
# with open(os.path.join(os.getcwd(),'saved_variables','sample_file_bytesIO.txt'), 'rb') as fh:
#     uploaded_file = BytesIO(fh.read())
# with open(os.path.join(os.getcwd(),'saved_variables','Options.json'), 'r') as fp:
#     Options = json.load(fp)
# with open(os.path.join(os.getcwd(),'saved_variables','sfreq.json'), 'r') as fp:
#     sfreq = json.load(fp)
# with open(os.path.join(os.getcwd(),'saved_variables','data_so_far.json'), 'r') as fp:
#     data_so_far = json.load(fp)
# del fp, fh
# '''
#%%
def user_input_options():
    Options = {}
    Options['show_processing_plots'] = st.sidebar.checkbox('show processing plots',
                                                           value=False
                                                           )
    Options['show_final_plots'] = st.sidebar.checkbox('show final plots',
                                                value=False
                                                )
    Options['show_table'] = st.sidebar.checkbox('show table',
                                                value=False
                                                )
    Options['export_style'] = st.sidebar.selectbox('export style', [
        'nexus',
        'proEMG',
        ],
        0)
    Options['trial_duration'] = st.sidebar.number_input('trial duration',
                                                        value=45,
                                                        )
    return Options
#%%
@st.cache
def import_raw_EMG(uploaded_file,export_style='nexus'):# uploaded_file=os.path.join(os.getcwd(),'sample_data', 'gooddata_1766_M01.csv') # uploaded_file=os.path.join(os.getcwd(),'sample_data', 'M1.csv')
    if export_style == 'nexus':
        sfreq = int(pd.read_csv(uploaded_file, sep=',', nrows=1).iloc[0,0])
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, sep=',', skiprows=3, low_memory=False).dropna()
        uploaded_file.seek(0)
        if all([type(x)!=str for x in df_raw.iloc[0]]):
            df_raw.columns = df_raw.iloc[0]
            df_raw = df_raw.drop(df_raw.index[0]).reset_index(drop=True)
        for c in df_raw.columns:
            if 'Frame' in c:
                df_raw[c] = df_raw[c].astype(int)
            else:
                df_raw[c] = df_raw[c].astype(float)
        time_raw = pd.Series(df_raw.index/sfreq)
    return df_raw, time_raw, sfreq#, fig_raw

@st.cache
def plot_raw(df_raw, time_raw):
    fig_raw = make_subplots(
        rows=len([c for c in df_raw.columns if 'Frame' not in c]),
        cols=1,
        shared_xaxes=True
        )
    for c,ch in enumerate([c for c in df_raw.columns if 'Frame' not in c]):# c,ch=0,[c for c in df_raw.columns if 'Frame' not in c][0]
        fig_raw.add_trace(
            go.Scatter(
                y=df_raw[ch],
                x=time_raw,
                name=ch,
                # fillcolor='blue',
                line={'color': 'blue'},
                ),
            row=c+1,
            col=1
            )
        fig_raw.update_yaxes(
            title_text='Hz',
            row=c+1,
            col=1)
    fig_raw.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=1)
    return fig_raw

@st.cache
def crop_EMG(df_raw, trial_duration, sfreq, Options):
    n_samples = int(trial_duration*sfreq)
    # for ch in Options['selected_channels']:# [c for c in df_raw.columns if 'Frame' not in c]:# ch=[c for c in df_raw.columns if 'Frame' not in c][0]
    #     if ch not in Options['selected_channels']:
    #         df_raw = df_raw.drop(ch, axis=1)
    if Options['crop_method']=='auto':
        off=[]
        for ch in Options['selected_channels']:# ch = Options['selected_channels'][0]
            sig = df_raw[ch]# sig.plot()
            rolling_sum = sig.abs().rolling(n_samples).sum()# (rolling_sum/rolling_sum.mean()).plot()
            off.append(rolling_sum.idxmax())
        off = int(np.mean(off))
        on = off-n_samples+1
    elif Options['crop_method']=='manual':
        [on, off] = [x*sfreq for x in Options['crop_window']]
        
    df_crop = df_raw[Options['selected_channels']].loc[on:off]#.reset_index(drop=True)# crop using 'on' and 'off'
    return df_crop, [on/sfreq, off/sfreq]#, time_crop

@st.cache
def plot_cropped(df_raw, df_crop, Options):
    time_raw = pd.Series(df_raw.index/sfreq)
    time_crop = pd.Series(df_crop.index/sfreq)
    fig_crop = make_subplots(
        rows=len(Options['selected_channels']),# len([c for c in df_raw.columns if 'Frame' not in c]),
        cols=1,
        shared_xaxes=True
        )
    for c,ch in enumerate(Options['selected_channels']):# enumerate([c for c in df_raw.columns if 'Frame' not in c]):# c,ch=0,[c for c in df_raw.columns if 'Frame' not in c][0]
        fig_crop.add_trace(
            go.Scatter(
                y=df_raw[ch],
                x=time_raw,
                name=ch+' raw',
                line={'color': 'blue',
                      'width': 1,
                      },
                ),
            row=c+1,
            col=1
            )
        fig_crop.update_yaxes(
            title_text='Hz',
            row=c+1,
            col=1)
        if ch in df_crop.columns:
            fig_crop.add_trace(
                go.Scatter(
                    y=df_crop[ch],
                    x=time_crop,
                    name=ch+ 'cropped',
                    line={'color': 'red'},
                    # line={'color': 'blue',
                    #       # 'width': 3,
                    #       },
                    ),
                row=c+1,
                col=1
                )
            
    fig_crop.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=1)
    return fig_crop

def filter_emg(df_crop, sfreq, Options, rectify=True, **kwargs):# [,sfreq,plots]=[,2000,True]
    '''
    # taken from https://scientificallysound.org/2016/08/22/python-analysing-emg-signals-part-4/
    # also helpful: https://www.cbcity.de/die-fft-mit-python-einfach-erklaert
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    sample input:
        sfreq = 2000
        band_pass = [10,500]
        high_pass = 20
        order = 3
        rectify = True
        plots = True
    '''
    low_pass = kwargs.get('low_pass')# low_pass = 450
    high_pass = kwargs.get('high_pass')# high_pass = 20
    band_pass = kwargs.get('band_pass')# band_pass = [10,450]
    order = kwargs.get('filter_order',3)# order = 3
    moving_avg_window = kwargs.get('moving_avg_window', 0.05)# moving_avg_window = 0.05
    if 'band_pass' in kwargs:
        # high_band = band_pass[0]/(sfreq/2)
        # low_band = band_pass[1]/(sfreq/2)
        band_pass = [x/(sfreq/2) for x in band_pass]
    if 'low_pass' in kwargs:
        low_pass = low_pass/sfreq
    if 'high_pass' in kwargs:
        high_pass = high_pass/sfreq

    df_filter = pd.DataFrame(columns=df_crop.columns)

    for c,ch in enumerate(Options['selected_channels']):# enumerate([c for c in df_crop.columns if 'Frame' not in c]):#enumerate([c for c in df_crop.columns if 'Frame' not in c]):# c,ch = 0, [c for c in df_crop.columns if 'Frame' not in c][0]
        emg_sig=df_crop[ch]
        # process EMG signal: remove mean
        emg_sig=emg_sig-emg_sig.mean()
    
        # normalise cut-off frequencies to sampling frequency
        if 'band_pass' in kwargs:
            # create band-pass filter for EMG
            # b1, a1 = signal.butter(order, [high_band,low_band], btype='bandpass')
            b1, a1 = signal.butter(order, band_pass, btype='bandpass')
            # process EMG signal: filter EMG
            emg_sig = signal.filtfilt(b1, a1, emg_sig)
            
        # create lowpass and/or highpass filter and apply to rectified signal to get EMG envelope
        if 'low_pass' in kwargs:
            b2, a2 = signal.butter(order, low_pass, btype='lowpass')
            emg_sig = signal.filtfilt(b2, a2, emg_sig)
        
        if 'high_pass' in kwargs:
            b2, a2 = signal.butter(order, high_pass, btype='highpass')
            emg_sig = signal.filtfilt(b2, a2, emg_sig)
        
        emg_sig = pd.Series(emg_sig)
        # process EMG signal: rectify
        if rectify:
            emg_sig = emg_sig.abs()

        if 'moving_avg_window' in kwargs:
            emg_sig = emg_sig.rolling(int(moving_avg_window*sfreq),center=True).mean()
            
        df_filter[ch]=emg_sig.values

    return df_filter

def plot_filter(df_raw, df_crop, df_filter, Options):
    time_raw = pd.Series(df_raw.index/sfreq)
    time_crop = pd.Series(df_crop.index/sfreq)
    fig_filter = make_subplots(
        rows=len(Options['selected_channels']),# len([c for c in df_raw.columns if 'Frame' not in c]),
        cols=1,
        shared_xaxes=True
        )
    for c,ch in enumerate(Options['selected_channels']):# enumerate([c for c in df_raw.columns if 'Frame' not in c]):# c,ch=0,[c for c in df_raw.columns if 'Frame' not in c][0]
        fig_filter.add_trace(
            go.Scatter(
                y=df_raw[ch],
                x=time_raw,
                name=ch+' raw',
                line={'color': 'blue',
                      'width': 1,
                      },
                ),
            row=c+1,
            col=1
            )
        fig_filter.update_yaxes(
            title_text='Hz',
            row=c+1,
            col=1)
        if ch in df_crop.columns:
            fig_filter.add_trace(
                go.Scatter(
                    y=df_crop[ch],
                    x=time_crop,
                    name=ch+ 'cropped',
                    line={'color': 'red'},
                    # line={'color': 'blue',
                    #       # 'width': 3,
                    #       },
                    ),
                row=c+1,
                col=1
                )
        if ch in df_filter.columns:
            fig_filter.add_trace(
                go.Scatter(
                    y=df_filter[ch],
                    x=time_crop,
                    name=ch+ 'filtered',
                    line={'color': 'green'},
                    # line={'color': 'blue',
                    #       # 'width': 3,
                    #       },
                    ),
                row=c+1,
                col=1
                )
            
    fig_filter.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=1)
    return fig_filter

#%%
def get_avg_inst_freq(df_filter, sfreq, Options, **kwargs):# [trial_duration,sfreq,epoch_duration]=[30,2000,1]
    '''
        # method according to Georgakis et al., Fatigue Analysis of the Surface EMG Signal in Isometric Constant Force Contractions Using the Averaged Instantaneous Frequency,
            IEEE Trasactions on Biomedical Engineering, vol. 50, no. 2, 2003, pp. 262-5
        # also described at https://pdfs.semanticscholar.org/68d4/9f25a71f5b9b6e6b1dc67fde538c15ee4872.pdf
        # also helpful to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

    sample input:
        sfreq=sfreq
        muscles=Muscles
    '''
    aif_parameters=pd.DataFrame()
    trial_duration=Options['trial_duration']
    epoch_duration=Options['epoch']

    if 'muscles' in kwargs:
        muscles=kwargs.get('muscles',None)
        labels=muscles
    else:
        labels=Options['selected_channels']# [c for c in df_filter.columns if 'Frame' not in c]

    fig_aif = make_subplots(
        rows=len(Options['selected_channels']),
        row_titles=['channel: '+x for x in Options['selected_channels']],
        cols=2,
        subplot_titles=sum([['AIF channel '+str(x), 'relative freq. change (linear)'] for x in Options['selected_channels']],[]),
        shared_xaxes=True
        )
    time = pd.Series(df_filter.index/sfreq)
    avg_inst_freq = {}
    t_epoch = {}
    for c,ch in enumerate(Options['selected_channels']):# c,ch = 0,Options['selected_channels'][0] # c,ch = c+1,Options['selected_channels'][c+1]
        emg_sig = np.array(df_filter[ch])
        analytic_signal = signal.hilbert(emg_sig)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                   (2.0*np.pi) * sfreq)
        avg_inst_freq[ch] = np.array([np.mean(instantaneous_frequency[int(i):int(i+(sfreq*epoch_duration))]) for i in np.arange(0,trial_duration*sfreq,epoch_duration*sfreq)])# i=0 # i=np.arange(0,trial_duration*sfreq,epoch_duration*sfreq)[-1]
        t_epoch[ch] = np.array([round(time.loc[int(i):int(i+(sfreq*epoch_duration))].mean(),3) for i in np.arange(0,trial_duration*sfreq,epoch_duration*sfreq)])# i=0
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_epoch[ch], avg_inst_freq[ch])# absolute change in AIF per second
        slope_norm = 100*(((t_epoch[ch][-1]*slope + intercept)-(t_epoch[ch][0]*slope + intercept))/abs(t_epoch[ch][0]*slope + intercept))/trial_duration# change in AIF/start value of AIF/trial duration in seconds, yielding percent change in AIF per second
        intercept_norm = 100
        fig_aif.add_trace(
            go.Scatter(
                y=avg_inst_freq[ch],
                x=t_epoch[ch],
                name='AIF '+labels[c],
                # line={'color': 'blue',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=1
            )
        fig_aif.add_trace(
            go.Scatter(
                y=t_epoch[ch]*slope + intercept,
                x=t_epoch[ch],
                name='linear '+labels[c],
                # line={'color': 'red',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=1
            )
        # fig_aif.add_trace(
        #     go.Scatter(
        #         y=avg_inst_freq[ch]/intercept,
        #         x=t_epoch[ch],
        #         name='AIF norm '+labels[c],
        #         # line={'color': 'blue',
        #         #       'width': 1,
        #         #       },
        #         ),
        #     row=c+1,
        #     col=2
        #     )
        fig_aif.add_trace(
            go.Scatter(
                y=t_epoch[ch]*slope_norm + intercept_norm,
                x=t_epoch[ch],
                name='linear norm'+labels[c],
                # line={'color': 'red',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=2
            )
        fig_aif['layout']['xaxis'+('' if c==0 else str(c+2))]['domain'] = (0.0, 0.4)
        fig_aif['layout']['xaxis'+('2' if c==0 else str(c+3))]['domain'] = (0.6, 1.0)
        
        annotations = list(fig_aif['layout']['annotations']) if fig_aif['layout']['annotations']!=() else []
        annotations.append(
            dict(
                x=2*Options['trial_duration']/3,
                # y=0,
                # y=intercept,
                y=max(avg_inst_freq[ch]),
                xref='x'+str(2*c+1),
                yref='y'+str(2*c+1),
                text='slope: '+str(round(slope,3))+' Hz/s',
                showarrow=False,
                )
            )
        annotations.append(
            dict(
                x=2*Options['trial_duration']/3,
                # y=0,
                y=intercept_norm,
                xref='x'+str(2*c+2),
                yref='y'+str(2*c+2),
                text='slope: '+str(round(slope_norm,1))+' %/s',
                showarrow=False,
                )
            )
        fig_aif['layout'].update(
            annotations=annotations
            )# fig_aif['layout']['annotations']
        fig_aif.update_yaxes(
            title_text='Hz',
            row=c+1,
            col=1)
        fig_aif.update_yaxes(
            title_text='%',
            row=c+1,
            col=2)
        # fig_aif.show()

        aif_parameters.loc[0,'slope_'+ch] = slope
        aif_parameters.loc[0,'slope_norm_'+ch] = slope_norm
        dFreq_pct = ((t_epoch[ch][-1]*slope_norm + intercept_norm)/(t_epoch[ch][0]*slope_norm + intercept_norm)-1)*100# total percent change in AIF over 'trial_duration'
        aif_parameters.loc[0,'dFreq_pct_'+ch] = dFreq_pct
    # for c,ch in enumerate(Options['selected_channels']):# c,ch = 0,Options['selected_channels'][0] # c,ch = c+1,Options['selected_channels'][c+1]
        # fig_aif['layout']['xaxis'+('' if c==0 else str(c+2))]['domain'] = (0.0, 0.4)
        # fig_aif['layout']['xaxis'+('2' if c==0 else str(c+3))]['domain'] = (0.6, 1.0)
        # fig_aif['layout'].update(
        #     annotations=[
        #         dict(
        #             ax=2*Options['trial_duration']/3,
        #             ay=intercept,
        #             xref='x'+str(c+1),
        #             yref='y'+str(c+1),
        #             text='my text 1',
        #             ),
        #         dict(
        #             ax=2*Options['trial_duration']/3,
        #             ay=100,
        #             xref='x'+str(c+2),
        #             yref='y'+str(c+2),
        #             text='my text 2',
        #             ),
        #         ]
        #     )# fig_aif['layout']['annotations']
        # fig_aif.update_yaxes(
        #     title_text='Hz',
        #     row=c+1,
        #     col=1)
        # fig_aif.update_yaxes(
        #     title_text='%',
        #     row=c+1,
        #     col=2)
    # fig_aif['layout']['annotations'] = [
    #     dict(
    #         x=2*Options['trial_duration']/3,
    #         y=90,
    #         xref='x1',
    #         yref='y1',
    #         text='text here',
    #         ax=10,
    #         ay=70,
    #         )
    #     ]
    # fig_aif['layout']['annotations'][0].update(
    #     x=2*Options['trial_duration']/3,
    #     y=90,
    #     # xref='x1',
    #     # yref='y1',
    #     text='text here',
    #     # ax=10,
    #     # ay=70,
    #     )
    fig_aif.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=1)
    fig_aif.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=2)
    # fig_aif.show()
    fig_aif['layout']['showlegend']=False

    if 'muscles' in kwargs:
        d=list(zip(df_filter.columns,muscles))
        cols=list(aif_parameters.columns)
        for c,col in enumerate(cols):# col=aif_parameters.columns[0] # c,col=1,aif_parameters.columns[1]
            if any([ch in col for ch,musc in d]):
                for ch,musc in d:# ch,musc = d[0]
                    col=col.replace(ch,musc)
                cols[c]=col
        aif_parameters.columns=cols
    if 'file_name' in kwargs:
        aif_parameters.insert(0, 'file_name', kwargs.get('file_name').split('.')[0])
        aif_parameters = aif_parameters.set_index('file_name')
    return aif_parameters, fig_aif

def plot_freq_spectrum(f, Pxx_den):
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
    
def get_median_freq(df_filter, sfreq, Options, **kwargs):
    '''
    Method according to:
        https://plux.info/signal-processing/454-fatigue-evaluation-evolution-of-median-power-frequency.html
    '''
    mdf_parameters=pd.DataFrame()
    trial_duration=Options['trial_duration']

    if 'muscles' in kwargs:
        muscles=kwargs.get('muscles',None)
        labels=muscles
    else:
        labels=Options['selected_channels']# [c for c in df_filter.columns if 'Frame' not in c]

    fig_mdf = make_subplots(
        rows=len(Options['selected_channels']),
        row_titles=['channel: '+x for x in Options['selected_channels']],
        cols=2,
        subplot_titles=sum([['MDF channel '+str(x), 'relative freq. change (linear)'] for x in Options['selected_channels']],[]),
        shared_xaxes=True
        )
    ons = range(0,
                df_filter.index.max(),
                int(Options['epoch']*sfreq),
                )
    median_freq = {}
    t_epoch = {}
    for c,ch in enumerate(Options['selected_channels']):# c,ch = 0,Options['selected_channels'][0] # c,ch = c+1,Options['selected_channels'][c+1]
        median_freq[ch] = np.empty((len(ons),))
        t_epoch[ch] = np.empty((len(ons),))
        emg_sig = df_filter[ch]
        for i,on in enumerate(ons):# i,on = 0,ons[0]
            off = ons[i+1]-1 if i+1<len(ons) else df_filter.index.max()
            # print([on,off])
            processing_window = emg_sig.loc[on:off]
            central_point = (on + off) / 2
            # t_epoch[ch] += [central_point / sfreq]
            t_epoch[ch][i] = central_point / sfreq
            # Processing window power spectrum (PSD) generation
            f, Pxx_den = signal.periodogram(np.array(processing_window), fs=float(sfreq))# plot_freq_spectrum(f, Pxx_den)
            # Median power frequency determination
            area_freq = integrate.cumtrapz(Pxx_den, f, initial=0)
            total_power = area_freq[-1]
            median_freq[ch][i] = f[np.where(area_freq >= total_power / 2)[0][0]]
            # The previous indexation [0][0] was specified in order to only the first sample that 
            # verifies the condition area_freq >= total_power / 2 be returned (all the subsequent 
            # samples will verify this condition, but, we only want the frequency that is nearest 
            # to the ideal frequency value that divides power spectrum into to regions with the 
            # same power - which is not achievable in a digital processing perspective).
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_epoch[ch],median_freq[ch])# absolute change in MDF per second
        slope_norm = 100*(((t_epoch[ch][-1]*slope + intercept)-(t_epoch[ch][0]*slope + intercept))/abs(t_epoch[ch][0]*slope + intercept))/trial_duration# change in MDF/start value of MDF/trial duration in seconds, yielding percent change in MDF per second
        intercept_norm = 100
        fig_mdf.add_trace(
            go.Scatter(
                y=median_freq[ch],
                x=t_epoch[ch],
                name='MDF '+labels[c],
                # line={'color': 'blue',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=1
            )
        fig_mdf.add_trace(
            go.Scatter(
                y=t_epoch[ch]*slope + intercept,
                x=t_epoch[ch],
                name='linear '+labels[c],
                # line={'color': 'red',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=1
            )
        # fig_mdf.add_trace(
        #     go.Scatter(
        #         y=median_freq[ch]/intercept,
        #         x=t_epoch[ch],
        #         name='MDF norm '+labels[c],
        #         # line={'color': 'blue',
        #         #       'width': 1,
        #         #       },
        #         ),
        #     row=c+1,
        #     col=2
        #     )
        fig_mdf.add_trace(
            go.Scatter(
                y=t_epoch[ch]*slope_norm + intercept_norm,
                x=t_epoch[ch],
                name='linear norm '+labels[c],
                # line={'color': 'red',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=2
            )
        fig_mdf['layout']['xaxis'+('' if c==0 else str(c+2))]['domain'] = (0.0, 0.4)
        fig_mdf['layout']['xaxis'+('2' if c==0 else str(c+3))]['domain'] = (0.6, 1.0)
        
        annotations = list(fig_mdf['layout']['annotations']) if fig_mdf['layout']['annotations']!=() else []
        annotations.append(
            dict(
                x=2*Options['trial_duration']/3,
                # y=0,
                # y=intercept,
                y=max(median_freq[ch]),
                xref='x'+str(2*c+1),
                yref='y'+str(2*c+1),
                text='slope: '+str(round(slope,3))+' Hz/s',
                showarrow=False,
                )
            )
        annotations.append(
            dict(
                x=2*Options['trial_duration']/3,
                # y=0,
                y=intercept_norm,
                xref='x'+str(2*c+2),
                yref='y'+str(2*c+2),
                text='slope: '+str(round(slope_norm,1))+' %/s',
                showarrow=False,
                )
            )
        fig_mdf['layout'].update(
            annotations=annotations
            )# fig_mdf['layout']['annotations']
        fig_mdf.update_yaxes(
            title_text='Hz',
            row=c+1,
            col=1)
        fig_mdf.update_yaxes(
            title_text='%',
            row=c+1,
            col=2)
        # fig_mdf.show()

        mdf_parameters.loc[0,'slope_'+ch]=slope
        mdf_parameters.loc[0,'slope_norm_'+ch]=slope_norm
        dFreq_pct=((t_epoch[ch][-1]*slope_norm + intercept_norm)/(t_epoch[ch][0]*slope_norm + intercept_norm)-1)*100# total percent change in MDF over 'trial_duration'
        mdf_parameters.loc[0,'dFreq_pct_'+ch]=dFreq_pct
    fig_mdf.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=1)
    fig_mdf.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=2)
    # fig_mdf.show()
    fig_mdf['layout']['showlegend']=False

    if 'muscles' in kwargs:
        d=list(zip(df_filter.columns,muscles))
        cols=list(mdf_parameters.columns)
        for c,col in enumerate(cols):# col=mdf_parameters.columns[0] # c,col=1,mdf_parameters.columns[1]
            if any([ch in col for ch,musc in d]):
                for ch,musc in d:# ch,musc = d[0]
                    col=col.replace(ch,musc)
                cols[c]=col
        mdf_parameters.columns=cols
    if 'file_name' in kwargs:
        mdf_parameters.insert(0, 'file_name', kwargs.get('file_name').split('.')[0])
        mdf_parameters = mdf_parameters.set_index('file_name')
    return mdf_parameters, fig_mdf
    
def get_mean_freq(df_filter, sfreq, Options, **kwargs):
    '''
    Method according to:
        https://stackoverflow.com/questions/37922928/difference-in-mean-frequency-in-python-and-matlab
    '''
    mnf_parameters=pd.DataFrame()
    trial_duration=Options['trial_duration']

    if 'muscles' in kwargs:
        muscles=kwargs.get('muscles',None)
        labels=muscles
    else:
        labels=Options['selected_channels']# [c for c in df_filter.columns if 'Frame' not in c]

    fig_mnf = make_subplots(
        rows=len(Options['selected_channels']),
        row_titles=['channel: '+x for x in Options['selected_channels']],
        cols=2,
        subplot_titles=sum([['MNF channel '+str(x), 'relative freq. change (linear)'] for x in Options['selected_channels']],[]),
        shared_xaxes=True
        )
    ons = range(0,
                df_filter.index.max(),
                int(Options['epoch']*sfreq),
                )
    ons = range(0,
                df_filter.index.max(),
                int(Options['epoch']*sfreq),
                )
    mean_freq = {}
    t_epoch = {}
    for c,ch in enumerate(Options['selected_channels']):# c,ch = 0,Options['selected_channels'][0] # c,ch = c+1,Options['selected_channels'][c+1]
        mean_freq[ch] = np.empty((len(ons),))
        t_epoch[ch] = np.empty((len(ons),))
        emg_sig = df_filter[ch]
        for i,on in enumerate(ons):# i,on = 0,ons[0]
            off = ons[i+1]-1 if i+1<len(ons) else df_filter.index.max()
            # print([on,off])
            processing_window = emg_sig.loc[on:off]
            central_point = (on + off) / 2
            # t_epoch[ch] += [central_point / sfreq]
            t_epoch[ch][i] = central_point / sfreq
            # Processing window power spectrum (PSD) generation
            f, Pxx_den = signal.periodogram(np.array(processing_window), fs=float(sfreq))# plot_freq_spectrum(f, Pxx_den)
            Pxx_den = np.reshape(Pxx_den, (1,-1))
            width = np.tile(f[1]-f[0], (1, Pxx_den.shape[1]))
            f = np.reshape(f, (1,-1))
            P = Pxx_den*width
            pwr = np.sum(P)
            mean_freq[ch][i] = np.dot(P, f.T)/pwr
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_epoch[ch],mean_freq[ch])# absolute change in MNF per second
        slope_norm = 100*(((t_epoch[ch][-1]*slope + intercept)-(t_epoch[ch][0]*slope + intercept))/abs(t_epoch[ch][0]*slope + intercept))/trial_duration# change in MNF/start value of MNF/trial duration in seconds, yielding percent change in MNF per second
        intercept_norm = 100
        fig_mnf.add_trace(
            go.Scatter(
                y=mean_freq[ch],
                x=t_epoch[ch],
                name='MNF '+labels[c],
                # line={'color': 'blue',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=1
            )
        fig_mnf.add_trace(
            go.Scatter(
                y=t_epoch[ch]*slope + intercept,
                x=t_epoch[ch],
                name='linear '+labels[c],
                # line={'color': 'red',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=1
            )
        # fig_mnf.add_trace(
        #     go.Scatter(
        #         y=mean_freq[ch]/intercept,
        #         x=t_epoch[ch],
        #         name='MNF norm '+labels[c],
        #         # line={'color': 'blue',
        #         #       'width': 1,
        #         #       },
        #         ),
        #     row=c+1,
        #     col=2
        #     )
        fig_mnf.add_trace(
            go.Scatter(
                y=t_epoch[ch]*slope_norm + intercept_norm,
                x=t_epoch[ch],
                name='linear '+labels[c],
                # line={'color': 'red',
                #       'width': 1,
                #       },
                ),
            row=c+1,
            col=2
            )
        fig_mnf['layout']['xaxis'+('' if c==0 else str(c+2))]['domain'] = (0.0, 0.4)
        fig_mnf['layout']['xaxis'+('2' if c==0 else str(c+3))]['domain'] = (0.6, 1.0)
        
        annotations = list(fig_mnf['layout']['annotations']) if fig_mnf['layout']['annotations']!=() else []
        annotations.append(
            dict(
                x=2*Options['trial_duration']/3,
                # y=0,
                # y=intercept,
                y=max(mean_freq[ch]),
                xref='x'+str(2*c+1),
                yref='y'+str(2*c+1),
                text='slope: '+str(round(slope,3))+' Hz/s',
                showarrow=False,
                )
            )
        annotations.append(
            dict(
                x=2*Options['trial_duration']/3,
                # y=0,
                y=intercept_norm,
                xref='x'+str(2*c+2),
                yref='y'+str(2*c+2),
                text='slope: '+str(round(slope_norm,1))+' %/s',
                showarrow=False,
                )
            )
        fig_mnf['layout'].update(
            annotations=annotations
            )# fig_mnf['layout']['annotations']
        fig_mnf.update_yaxes(
            title_text='Hz',
            row=c+1,
            col=1)
        fig_mnf.update_yaxes(
            title_text='%',
            row=c+1,
            col=2)
        # fig_mnf.show()

        mnf_parameters.loc[0,'slope_'+ch]=slope
        mnf_parameters.loc[0,'slope_norm_'+ch]=slope_norm
        dFreq_pct=((t_epoch[ch][-1]*slope_norm + intercept_norm)/(t_epoch[ch][0]*slope_norm + intercept_norm)-1)*100# total percent change in MNF over 'trial_duration'
        mnf_parameters.loc[0,'dFreq_pct_'+ch]=dFreq_pct
    fig_mnf.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=1)
    fig_mnf.update_xaxes(
        title_text='time (s)',
        row=c+1,
        col=2)
    # fig_mnf.show()
    fig_mnf['layout']['showlegend']=False

    if 'muscles' in kwargs:
        d=list(zip(df_filter.columns,muscles))
        cols=list(mnf_parameters.columns)
        for c,col in enumerate(cols):# col=mnf_parameters.columns[0] # c,col=1,mnf_parameters.columns[1]
            if any([ch in col for ch,musc in d]):
                for ch,musc in d:# ch,musc = d[0]
                    col=col.replace(ch,musc)
                cols[c]=col
        mnf_parameters.columns=cols
    if 'file_name' in kwargs:
        mnf_parameters.insert(0, 'file_name', kwargs.get('file_name').split('.')[0])
        mnf_parameters = mnf_parameters.set_index('file_name')
    return mnf_parameters, fig_mnf
    
#%%
@st.cache(allow_output_mutation=True)
def data_so_far():
    # return []
    return {
        'AIF': [],
        'MDF': [],
        'MNF': [],
        }
@st.cache(allow_output_mutation=True)
def file_names_so_far():
    return []

def generate_excel(Parameters, drop_duplicates=True):
    '''
    Parameters
    ----------
    Options : dict
        dict containing the keys 'parameters_to_excel', 'signals_to_excel'.

    Returns
    -------
    None.

    '''
    # thanks to https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
    writing_excel_container = st.empty()
    writing_excel_container.text('writing to excel')
    output = BytesIO()
    
    with pd.ExcelWriter(output) as writer:
        for k in Parameters.keys():# exercise = list(Results_parameters.keys())[0]
            if drop_duplicates:
                Parameters[k].drop_duplicates().to_excel(writer, sheet_name=k, index=True)
            else:
                Parameters[k].to_excel(writer, sheet_name=k, index=True)
            # workbook = writer.book
            worksheet = writer.sheets[k]
            # fmt_thousendth = workbook.add_format({'num_format': '#.###'})
            worksheet.set_column('A:A', 30)
            worksheet.set_column('B:Z', 18)
        # if Options['all_oneLiners']:
        #     for exercise in oneLiners.keys():# exercise = list(oneLiners.keys())[0]
        #         for entry_type in oneLiners[exercise].keys():# entry_type = list(oneLiners[exercise].keys())[0]
        #             if Options['transpose_oneLiner']:
        #                 oneLiners[exercise][entry_type].set_index('subject').T.to_excel(writer, sheet_name=exercise+'_'+entry_type)
        #             else:
        #                 oneLiners[exercise][entry_type].set_index('subject').to_excel(writer, sheet_name=exercise+'_'+entry_type)
        # else:
        #     exercise = 'all_exercises'# exercise = list(oneLiners.keys())[0]
        #     entry_type = 'full'# entry_type = list(oneLiners[exercise].keys())[0]
        #     if Options['transpose_oneLiner']:
        #         oneLiners[exercise][entry_type].set_index('subject').T.to_excel(writer, sheet_name=exercise+'_'+entry_type)
        #     else:
        #         oneLiners[exercise][entry_type].set_index('subject').to_excel(writer, sheet_name=exercise+'_'+entry_type)
        # if Options['signals_to_excel']:
        #     for exercise in Results_signals.keys():# exercise = list(Results_signals.keys())[0]
        #         for file in Results_signals[exercise].keys():# file = list(Results_signals[exercise].keys())[0]
        #             for rep in Results_signals[exercise][file].keys():# rep = list(Results_signals[exercise][file].keys())[0]
        #                 for direction in Results_signals[exercise][file][rep].keys():# direction = list(Results_signals[exercise][file][rep].keys())[0]
        #                     df = Results_signals[exercise][file][rep][direction]['combined']
        #                     for k in [x for x in Results_signals[exercise][file][rep][direction].keys() if x!='combined']:# k=[x for x in Results_signals[exercise][file][rep][direction].keys() if x!='combined'][0]
        #                         df = pd.concat((df, 
        #                                         pd.DataFrame(
        #                                             data=Results_signals[exercise][file][rep][direction][k].values,
        #                                             columns=['_'.join([str(k),c]) for c in Results_signals[exercise][file][rep][direction][k].columns]
        #                                             )
        #                                         ),
        #                                        axis=1)
        #                     df.to_excel(writer,
        #                                 # sheet_name='_'.join([exercise.replace('Syncro ',''),file,rep,direction]),
        #                                 sheet_name='_'.join([exercise.replace(' bilateral',''),file,rep,direction]),
        #                                 index=False)
        writer.save()
        processed_data = output.getvalue()
        
    b64 = base64.b64encode(processed_data)
    writing_excel_container.empty()
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Results.xlsx">Download Results as Excel File</a>' # decode b'abc' => abc

#%%
st.write("""

# EMG frequency analysis

""")
st.sidebar.header('Options')
Options = user_input_options()
options_container = st.empty()
uploaded_file = st.file_uploader("upload export file", accept_multiple_files=False)
plot_container_1 = st.empty()
# options_container.write(Options)
# Parameters = {}
# Parameters['AIF']=pd.DataFrame()
# Parameters['MDF']=pd.DataFrame()
# Parameters['MNF']=pd.DataFrame()
if uploaded_file is not None:
    file_name = uploaded_file.name# file_name = 'M1.csv'
    # st.write('running import_raw_EMG')
    df_raw, time_raw, sfreq = import_raw_EMG(uploaded_file,
                                             export_style=Options['export_style'],
                                             )
    if Options['show_processing_plots']:
        plot_container_1.plotly_chart(plot_raw(df_raw, time_raw))
    duration = time_raw.max()
    selected_channels = {}
    for c,ch in enumerate([c for c in df_raw.columns if 'Frame' not in c]):
        selected_channels[ch] = st.sidebar.checkbox(ch,
                                                    # value=True,
                                                    value=True if 'Knee' not in ch and c<2 else False,
                                                    key=ch,
                                                    )
    Options['selected_channels'] = [k for k in selected_channels.keys() if selected_channels[k]==True]
    Options['crop_method'] = st.sidebar.selectbox('crop method', [
        '',
        'auto',
        'manual',
        ],
        1,# Options['crop_method'] = 'auto'
        # 0,
        )
    # options_container.write(Options)
    if Options['crop_method'] == 'manual':
        Options['crop_window'] = [
            st.sidebar.number_input('crop window on (s)',
                                    min_value=0.0,
                                    max_value=time_raw.max()-Options['trial_duration'],
                                    step=1.0,
                                    value=0.0,
                                    ),
                                  ]
        Options['crop_window'].append(Options['crop_window'][0] + Options['trial_duration'])
    if Options['crop_method'] != '' and Options['trial_duration'] <= duration:
        # st.write('running crop_EMG')
        df_crop, Options['crop_window'] = crop_EMG(df_raw=df_raw,
                                                   trial_duration=Options['trial_duration'],
                                                   sfreq=sfreq,
                                                   Options=Options,
                                                   )
        # st.write('running plot_cropped')
        # fig_crop = plot_cropped(df_raw, df_crop, Options)
        if Options['show_processing_plots']:
            plot_container_1.plotly_chart(plot_cropped(df_raw, df_crop, Options))
        Options['filter_type'] = st.sidebar.selectbox('filter type', [
            '',
            'low pass',
            'high pass',
            'band pass',
            'moving average',
            ],
            # 0,
            3,# Options['filter_type']='band_pass'
            )
        # options_container.write(Options)
        if Options['filter_type'] != '':
            if Options['filter_type']=='low pass':
                Options['pass_value'] = st.sidebar.number_input('lowpass value',
                                                                min_value=1,
                                                                max_value=int(sfreq/2-1),
                                                                step=50,
                                                                value=450,
                                                                )
                df_filter = filter_emg(df_crop, sfreq, Options,
                                       rectify=True,
                                       low_pass=Options['pass_value'],
                                       )
            elif Options['filter_type']=='high pass':
                Options['pass_value'] = st.sidebar.number_input('highpass value',
                                                                min_value=1,
                                                                max_value=int(sfreq/2-1),
                                                                step=10,
                                                                value=10,
                                                                )
                df_filter = filter_emg(df_crop, sfreq, Options,
                                       rectify=True,
                                       high_pass=Options['pass_value'],
                                       )
            elif Options['filter_type']=='band pass':
                Options['pass_value'] = [
                    st.sidebar.number_input('bandpass value 1',
                                            min_value=1,
                                            max_value=int(sfreq/2-1),
                                            step=1,
                                            value=10,
                                            ),
                    st.sidebar.number_input('bandpass value 2',
                                            min_value=1,
                                            max_value=int(sfreq/2-1),
                                            step=1,
                                            value=450,
                                            )
                    ]# Options['pass_value']=[10,450] 
                df_filter = filter_emg(df_crop, sfreq, Options,
                                       rectify=True,
                                       band_pass=Options['pass_value'],
                                       )
            # options_container.write(Options)
            if Options['show_processing_plots']:
                plot_container_1.plotly_chart(plot_filter(df_raw, df_crop, df_filter, Options))
            # fig_filter = plot_filter(df_raw, df_crop, df_filter, Options)
            # fig_filter.write_image(os.path.join(os.getcwd(), 'plots', 'fig_filter.png'))
            if df_filter is not None:
                Options['epoch'] = st.sidebar.number_input('epoch (s)',
                                                           value=0.5)# Options['epoch']=0.5
                # options_container.write(Options)
                aif_parameters, fig_aif = get_avg_inst_freq(df_filter,
                                                            sfreq,
                                                            Options,
                                                            file_name=file_name,
                                                            )
                data_so_far()['AIF'].append(aif_parameters.to_dict(orient='records')[0])
                # st.write('AIF')
                # st.pyplot(fig_aif)
                mdf_parameters, fig_mdf = get_median_freq(df_filter,
                                                          sfreq,
                                                          Options,
                                                          file_name=file_name,
                                                          )
                data_so_far()['MDF'].append(mdf_parameters.to_dict(orient='records')[0])
                # st.write('MDF')
                # st.pyplot(fig_mdf)
                mnf_parameters, fig_mnf = get_mean_freq(df_filter,
                                                        sfreq,
                                                        Options,
                                                        file_name=file_name,
                                                        )
                data_so_far()['MNF'].append(mnf_parameters.to_dict(orient='records')[0])
                # st.write('MNF')
                # st.pyplot(fig_mnf)
                
                # st.write('dataframes in local variables')
                # st.write('Parameters' in locals())
                # st.write('dataframes in global variables')
                # st.write('Parameters' in globals())
                file_names_so_far().append(file_name)
                # st.write(file_names_so_far())
                # st.write(data_so_far())
                # st.write(pd.DataFrame.from_dict(data_so_far()['AIF']))
                Parameters = {
                    'AIF': pd.DataFrame.from_dict(data_so_far()['AIF']),
                    'MDF': pd.DataFrame.from_dict(data_so_far()['MDF']),
                    'MNF': pd.DataFrame.from_dict(data_so_far()['MNF']),
                    }
                for k in Parameters.keys():
                    Parameters[k].insert(0, 'file_name', file_names_so_far())
                    Parameters[k] = Parameters[k].set_index('file_name')
                    
                Parameters['all'] = pd.concat(
                    (
                        Parameters['AIF'].rename(
                            columns=dict(zip(Parameters['AIF'].columns,[c+'_AIF' for c in Parameters['AIF'].columns]))
                            ),
                        Parameters['MDF'].rename(
                            columns=dict(zip(Parameters['MDF'].columns,[c+'_MDF' for c in Parameters['MDF'].columns]))
                            ),
                        Parameters['MNF'].rename(
                            columns=dict(zip(Parameters['MNF'].columns,[c+'_MNF' for c in Parameters['MNF'].columns]))
                            ),
                        ),
                    axis=1
                    )
                # st.write('AIF')
                # st.write(Parameters['AIF'])
                # st.write('MDF')
                # st.write(Parameters['MDF'])
                # st.write('MNF')
                # st.write(Parameters['MNF'])
                st.markdown(generate_excel(Parameters), unsafe_allow_html=True)
                if Options['show_final_plots']:
                    st.write('AIF')
                    # st.pyplot(fig_aif)
                    st.plotly_chart(fig_aif)
                    st.write('MDF')
                    # st.pyplot(fig_mdf)
                    st.plotly_chart(fig_mdf)
                    st.write('MNF')
                    # st.pyplot(fig_mnf)
                    st.plotly_chart(fig_mnf)
                if Options['show_table']:
                    st.write(Parameters['all'])
                # with open(os.path.join(os.getcwd(),'saved_variables','data_so_far.json'), 'w') as fp:
                #     json.dump(data_so_far(), fp)
    # # with uploaded_file as f:#for f in uploaded_files:
    # #     # with open(os.path.join(os.getcwd(),'saved_variables',(f.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
    # #     with open(os.path.join(os.getcwd(),'saved_variables','sample_file_bytesIO.txt'), 'wb') as fp:
    # #         fp.write(f.getbuffer())
    # with open(os.path.join(os.getcwd(),'saved_variables','sample_file_bytesIO.txt'), 'wb') as fp:
    #     fp.write(uploaded_file.getbuffer())
    # with open(os.path.join(os.getcwd(),'saved_variables','Options.json'), 'w') as fp:
    #     json.dump(Options, fp)
    # with open(os.path.join(os.getcwd(),'saved_variables','sfreq.json'), 'w') as fp:
    #     json.dump(sfreq, fp)


