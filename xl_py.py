# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 13:20:31 2017

@author: 8ME-HW-171-L
"""

import openpyxl
from openpyxl import load_workbook
import numpy as np
import pandas as pd
import os
import operator

import matplotlib.pyplot as plt
import datetime
from datetime import datetime


os.getcwd()

## IMPORT RAW DATA
ie_sum_prod_p50 = pd.read_excel('./62SK_IE_Daily.xlsx', sheetname=0)
ie_sum_prod_p99 = pd.read_excel('./62SK_IE_Daily.xlsx', sheetname=1)
ie_sum_ghi = pd.read_excel('./62SK_IE_Daily.xlsx', sheetname=3)
ie_sum_poa = pd.read_excel('./62SK_IE_Daily.xlsx', sheetname=4)
ie_avg_temp = pd.read_excel('./62SK_IE_Daily.xlsx', sheetname=5)
ie_cell_temp = pd.read_excel('./62SK_IE_Daily.xlsx',sheetname=6)
ie_ws = pd.read_excel('./62SK_IE_Daily.xlsx', sheetname=7)
ie_pr = pd.read_excel('./62SK_IE_Daily.xlsx', sheetname=10)
ie_8760 = pd.read_excel('./ie_8760.xlsx', sheetname=0)
monthly_invoice = pd.read_csv('./Data_Extract_20170401-20170430_MonthlyInvoiceTotals_idd84f6722-c88.csv')
inverter = pd.read_csv('./Data_Extract_20170601-20170630_SB1WeekHourlyInverterEnergy_id9a7c158a-e9c.csv')

## Format inputs
monthly_invoice.columns = monthly_invoice.iloc[1]
monthly_invoice = monthly_invoice.drop(monthly_invoice.index[[0,1]])
monthly_invoice.rename(columns={'field':'Timestamp'}, inplace=True)
monthly_invoice['Timestamp'] = pd.to_datetime(monthly_invoice['Timestamp'])
monthly_invoice.index = monthly_invoice['Timestamp']

monthly_invoice['TotWhExp_max'] = monthly_invoice['TotWhExp_max'].apply(pd.to_numeric, errors='coerce')
#monthly_invoice = monthly_invoice.apply(pd.to_numeric, errors='coerce')

#monthly_invoice['NetWhExp'] = monthly_invoice['TotWhExp_max'].sub(monthly_invoice['TotWhExp_max'].shift(1), fill_value=0)
test1 = monthly_invoice['TotWhExp_max']
test2 = monthly_invoice['TotWhExp_max'].shift(1)
test3 = test1-test2

monthly_invoice['NetWhExp'] = test3.iloc[:,0]

from calendar import monthrange
#monthly_invoice.iloc[0, 0].month
date = (monthrange(monthly_invoice.iloc[0, 0].year, monthly_invoice.iloc[0, 0].month))
month = monthly_invoice.iloc[0,0].month
year = monthly_invoice.iloc[0,0].year
monthly_invoice = monthly_invoice.apply(pd.to_numeric, errors='coerce')

table_2_3 = pd.DataFrame(index=range(0,date[1]), dtype='float', columns=['Day','ENexp P50','ENexp P99','ENm','EPIM P50','EPIM P99'])

table_2_3['Day'] = range(1,date[1]+1)
table_2_3['ENexp P50'] = ie_sum_prod_p50.iloc[:,date[0]-1]
table_2_3['ENexp P99'] = ie_sum_prod_p99.iloc[:,date[0]-1]

test_month = monthly_invoice.resample('D').sum()['NetWhExp']

for i in range(0,len(table_2_3)):
    #print(test_month.iloc[i]/1000)
    #print(i)
    table_2_3.iloc[i,3] = test_month.iloc[i]/1000

table_2_3.iloc[:,4] = table_2_3.iloc[:,3]/table_2_3.iloc[:,1]*100
table_2_3.iloc[:,5] = table_2_3.iloc[:,3]/table_2_3.iloc[:,2]*100

#table_2_3
table_2_3.round(2)

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

width=0.2
fig, ax1 = plt.subplots()
t=table_2_3["Day"]

ax1.bar(t,table_2_3["ENexp P50"], width, label = 'ENexp P50', color='forestgreen')
ax1.bar(t+width,table_2_3["ENexp P99"], width, label = 'ENexp P99', color='green')
ax1.bar(t+width*2, table_2_3["ENm"], width, label = 'ENm', color='springgreen')

ax1.set_ylabel('Energy')
ax1.set_xlabel('Day')
ax1.tick_params('y')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles,labels)

ax2 = ax1.twinx()
ax2.plot(t,table_2_3["EPIM P50"], linewidth=2.0)
ax2.plot(t,table_2_3["EPIM P99"], linewidth=3.0)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles,labels)

ax2.set_ylabel('Percentage')

fig.tight_layout()
plt.show()


## Weather Data
os.getcwd()
weather_raw = pd.read_csv('./Data_Extract_20170101-20170430_HourlyWeatherAll_id5e7eadbf-f15.csv')
weather_raw.columns = weather_raw.iloc[0]
weather_raw.rename(columns={'field':'Timestamp'}, inplace=True)
weather_raw = weather_raw.drop(weather_raw.index[[0,1]])

weather_raw['Timestamp'] = pd.to_datetime(weather_raw['Timestamp'])
weather_raw.index = weather_raw['Timestamp']
date = (monthrange(weather_raw.iloc[0, 0].year, weather_raw.iloc[0, 0].month))
del weather_raw['Timestamp']

weather_raw = weather_raw.astype(float)

weather_raw = weather_raw.sort_index(axis=1)

ghi_avg = weather_raw.iloc[:,0:5]
ghi_sum = weather_raw.iloc[:,5:10]
poa_avg = weather_raw.iloc[:,10:15]
poa_sum = weather_raw.iloc[:,15:20]
ambtemp_avg = weather_raw.iloc[:,20:25]
bomtemp_avg = weather_raw.iloc[:,25:30]
bomtemp2_avg = weather_raw.iloc[:,30:35]
bomtemp3_avg = weather_raw.iloc[:,35:40]
maxminwnd_avg = weather_raw.iloc[:,40:45]
wndspd_avg = weather_raw.iloc[:,45:50]

ws_check = pd.DataFrame(index=weather_raw.index)
ws_check = pd.concat([ws_check,pd.DataFrame(columns=list('12345'))])
rows, cols = poa_avg.shape

for i in range(0,len(ws_check)):
    for j in range(0,5):
        if abs(((sum(poa_avg.iloc[i,:])-poa_avg.iloc[i,j])/(cols-1))-poa_avg.iloc[i,j])>poa_avg.iloc[i,:].std()*2:
            ws_check.iloc[i,j]=0
        else:
            ws_check.iloc[i,j]=1


weather_cols = ["GHIm","POAm","Tm","WSm","Precipm","cell_temp"]
weather_data = pd.DataFrame(index=weather_raw.index, columns=weather_cols)

for i in range(0,len(weather_data)):
    weather_data.iloc[i,0] = sumproduct(ws_check.iloc[i,:],ghi_sum.iloc[i,:])/sum(ws_check.iloc[i,:])
    weather_data.iloc[i,1] = sumproduct(ws_check.iloc[i,:],poa_sum.iloc[i,:])/sum(ws_check.iloc[i,:])
    weather_data.iloc[i,2] = np.mean(ambtemp_avg.iloc[i,:])
    weather_data.iloc[i,3] = np.mean(wndspd_avg.iloc[i,:])
    weather_data.iloc[i,4] = 0
    if weather_data.iloc[i,1] < 0:
        weather_data.iloc[i,5] = weather_data.iloc[i,2]
    else:
        weather_data.iloc[i,5] = weather_data.iloc[i,1]*np.exp(-3.47+0.0594*weather_data.iloc[i,3])+weather_data.iloc[i,2]+weather_data.iloc[i,1]/(1000*3)
    
    
table_3_2 = pd.DataFrame(index=range(0,date[1]), dtype='float', columns=['Day','G-GHI-exp','G-GHI-m','G-POA-exp','G-POA-m','Tamb-exp','Tamb-m','WS-exp','WS-m','Precip-m'])
temp_data = weather_data['Tm'].astype(int)
ws_data = weather_data['WSm'].astype(int)
table_3_2['Day'] = range(1,date[1]+1)


for i in range(0,len(table_3_2)):
    table_3_2.iloc[i,2] = weather_data.resample('D').sum()["GHIm"][i]/1000
    table_3_2.iloc[i,4] = weather_data.resample('D').sum()['POAm'][i]/1000
    table_3_2.iloc[i,6] = temp_data.resample('D').mean()[i]
    table_3_2.iloc[i,8] = ws_data.resample('D').mean()[i]
    table_3_2.iloc[i,9] = weather_data.resample('D').sum()["Precipm"][i]


table_3_2['G-GHI-exp'] = ie_sum_ghi.iloc[:,date[0]-1]/1000
table_3_2['G-POA-exp'] = ie_sum_poa.iloc[:,date[0]-1]/1000
table_3_2.iloc[:,5] = ie_avg_temp.iloc[:,date[0]-1]
table_3_2['WS-exp'] = ie_ws.iloc[:,date[0]-1]
table_3_2.round(2)


### Weather adjusted performance Analysis

ie_8760['Year'] = weather_data.index.year[1]
ie_8760.index = pd.to_datetime(ie_8760[['Year','Month','Day','Hour']])
avg_cell_temp = np.mean(ie_cell_temp.iloc[31,:])

## REgression coefficients
intercept = 0
ghi_coef = 257.6469548
ghi_cell_temp_coef = 0.339377076
ghi2_coef = -0.193910765

wa_raw = weather_data.join(ie_8760, how='inner')
del wa_raw['Tm']
del wa_raw['WSm']
del wa_raw['Precipm']

wa_raw['m/exp_ghi'] = 0
wa_raw['m/exp_temp'] = 0
wa_raw['m/exp_poa'] = 0

for i in range(0,len(wa_raw)):
    # m/exp GHI
    if wa_raw.iloc[i,8] != 0:
        wa_raw.iloc[i,10] = wa_raw.iloc[i,0]/wa_raw.iloc[i,8]
    else:
        wa_raw.iloc[i,10] = 0
    
    # m/exp Temp
    wa_raw.iloc[i,11] = 1-0.0041*(avg_cell_temp-weather_data.iloc[i,5])
    
    # m/exp POA
    if wa_raw.iloc[i,9] != 0:
        wa_raw.iloc[i,12] = weather_data.iloc[i,1]/wa_raw.iloc[i,9]
    else:
        wa_raw.iloc[i,12] = 0
        
wa_perf = pd.DataFrame(index=wa_raw.index, dtype='float',columns=['EN-exp-wa-ghi','EN-exp-wa-poa','EN-exp-war'])

for i in range(0, len(wa_raw)):
    #EN-exp-wa-ghi
    if wa_raw.iloc[i,7]*wa_raw.iloc[i,10]*wa_raw.iloc[i,11] > 108000 :
        wa_perf.iloc[i,0] = 108000
    else :
        wa_perf.iloc[i,0] = wa_raw.iloc[i,7]*wa_raw.iloc[i,10]*wa_raw.iloc[i,11]
    #EN-exp-wa-POA
    if wa_raw.iloc[i,7]*wa_raw.iloc[i,11]*wa_raw.iloc[i,12] > 108000 :
        wa_perf.iloc[i,1] = 108000
    else:
        wa_perf.iloc[i,1] = wa_raw.iloc[i,7]*wa_raw.iloc[i,11]*wa_raw.iloc[i,12]
    #EN-exp-WAR
    wa_perf.iloc[i,2] = intercept + weather_data.iloc[i,0]*ghi_coef + weather_data.iloc[i,0]*weather_data.iloc[i,5]*ghi_cell_temp_coef+(weather_data.iloc[i,0]**2)*ghi2_coef


    
table_4_1_3 = pd.DataFrame(index=range(0,date[1]),dtype='float',columns=['EN-exp-wa-GHI','EN-exp-wa-POA','EN-exp-war','EPIM-GHI','EPIM-POA','EPIM-R'])

for i in range(0,len(table_4_1_3)):
    table_4_1_3.iloc[i,0] = wa_perf.resample('D').sum()['EN-exp-wa-ghi'][i]
    table_4_1_3.iloc[i,1] = wa_perf.resample('D').sum()['EN-exp-wa-poa'][i]
    table_4_1_3.iloc[i,2] = wa_perf.resample('D').sum()['EN-exp-war'][i]
    
    #EPIM-GHI
    table_4_1_3.iloc[i,3] = table_2_3.iloc[i,3]*1000/table_4_1_3.iloc[i,0]

    #EPIM-POA
    table_4_1_3.iloc[i,4] = table_2_3.iloc[i,3]*1000/table_4_1_3.iloc[i,1]

    #EPIM-R
    table_4_1_3.iloc[i,5] = table_2_3.iloc[i,3]*1000/table_4_1_3.iloc[i,2]


## Performance Ratios

dc_cap_stc = 137080
ac_cap = 108000
ac_ppa_cap = 105000

ratios = pd.DataFrame(index=wa_raw.index, dtype='float', columns=['EY','EY-AC','EY-AC-PPA','PR','CPR'])

for i in range(0,len(ratios)):
    ratios.iloc[i,0] = monthly_invoice.iloc[i,13]/dc_cap_stc
    ratios.iloc[i,1] = monthly_invoice.iloc[i,13]/ac_cap
    ratios.iloc[i,2] = monthly_invoice.iloc[i,13]/ac_ppa_cap
    
    #PR
    if weather_data.iloc[i,1]<50 or monthly_invoice.iloc[i,13]<50:
        ratios.iloc[i,3] = np.nan
        ratios.iloc[i,4] = np.nan
    else:
        ratios.iloc[i,3] = ratios.iloc[i,0]/(weather_data.iloc[i,1]/1000)
        ratios.iloc[i,4] = ratios.iloc[i,0]/(weather_data.iloc[i,1]/1000*(1-0.0041*(avg_cell_temp-weather_data.iloc[i,5])))

table_4_2_2 = pd.DataFrame(index=range(0,date[1]), dtype='float', columns=['EY','EY-AC','EY-AC-PPA','PR','CPR','PR-exp'])

for i in range(0, len(table_4_2_2)):
    table_4_2_2.iloc[i,0] = ratios.resample('D').sum()['EY'][i]
    table_4_2_2.iloc[i,1] = ratios.resample('D').sum()['EY-AC'][i]
    table_4_2_2.iloc[i,2] = ratios.resample('D').sum()['EY-AC-PPA'][i]
    table_4_2_2.iloc[i,3] = ratios.resample('D').mean()['PR'][i]
    table_4_2_2.iloc[i,4] = ratios.resample('D').mean()['CPR'][i]
    table_4_2_2.iloc[i,5] = ie_pr.iloc[i,month]


## Reactive and Import Energy

test1 = monthly_invoice['TotWhExp_max']
test2 = monthly_invoice['TotWhExp_max'].shift(1)
test3 = test1-test2

reac = pd.DataFrame(index=wa_raw.index, dtype='float', columns=['Total Reactive Energy Exp','Total Reactive Energy Imp','Total Net Imp'])

reac.iloc[:,0] = monthly_invoice.iloc[:,1] - monthly_invoice.iloc[:,1].shift(1)
reac.iloc[:,1] = monthly_invoice.iloc[:,2] - monthly_invoice.iloc[:,2].shift(1)
reac.iloc[:,2] = monthly_invoice.iloc[:,4] - monthly_invoice.iloc[:,4].shift(1)

table_4_3_1 = pd.DataFrame(index=range(0,date[1]), dtype='float', columns=['Total Reactive Energy Exported','Total Reactive Energy Imported','Total Net Import'])

for i in range(0, len(table_4_3_1)):
    table_4_3_1.iloc[i,0] = reac.resample('D').sum()['Total Reactive Energy Exp'][i]
    table_4_3_1.iloc[i,1] = reac.resample('D').sum()['Total Reactive Energy Imp'][i]
    table_4_3_1.iloc[i,2] = reac.resample('D').sum()['Total Net Imp'][i]


## Availability and Clipping

inverter = pd.read_csv('./Data_Extract_20170601-20170630_SB1WeekHourlyInverterEnergy_id9a7c158a-e9c.csv')


inverter = inverter.rename(columns=inverter.iloc[2,:])
inverter = inverter.drop(inverter.index[[0,1,2]])
inverter.index = inverter['timestamp']

avail_clip = pd.DataFrame(index=wa_raw.index, dtype='float', columns=['Inv_Avail','Unavail_loss','Clip_loss_poa','Clip_loss_ghi'])



## Cumulative Monthly Table

master_monthly = pd.read_excel('./SB1_MASTER.xlsx', sheetname=0)

#master_monthly = pd.DataFrame(index=range(0,12), dtype='float', columns=['ENexp_P50','ENexp_P99','ENm','EPIM_P50','EPIM_P99',
#                              'G_ghi_exp','G_ghi_m','G_poa_exp','G_poa_m','Tcell_exp','Tcell_m','EY','EY_AC','EY_AC_PPA','PR',
#                              'CPR','EXP_PR','Exp_Soilig_loss','Soiling_Station_B502','Soiling_Station_B506'])
# Energy Values
master_monthly.iloc[month-1,1] = table_2_3.iloc[:,1].sum()
master_monthly.iloc[month-1,2] = table_2_3.iloc[:,2].sum()
master_monthly.iloc[month-1,3] = table_2_3.iloc[:,3].sum()
master_monthly.iloc[month-1,4] = table_2_3.iloc[:,4].mean()
master_monthly.iloc[month-1,5] = table_2_3.iloc[:,5].mean()
# Weather
master_monthly.iloc[month-1,6] = table_3_2.iloc[:,1].sum()
master_monthly.iloc[month-1,7] = table_3_2.iloc[:,2].sum()
master_monthly.iloc[month-1,8] = table_3_2.iloc[:,3].sum()
master_monthly.iloc[month-1,9] = table_3_2.iloc[:,4].sum()
master_monthly.iloc[month-1,10] = table_3_2.iloc[:,5].sum()
master_monthly.iloc[month-1,11] = table_3_2.iloc[:,6].sum()
# Performance Ratios
master_monthly.iloc[month-1,12] = table_4_2_2.iloc[:,0].sum()
master_monthly.iloc[month-1,13] = table_4_2_2.iloc[:,1].sum()
master_monthly.iloc[month-1,14] = table_4_2_2.iloc[:,2].sum()
master_monthly.iloc[month-1,15] = table_4_2_2.iloc[:,3].mean()
master_monthly.iloc[month-1,16] = table_4_2_2.iloc[:,4].mean()
master_monthly.iloc[month-1,17] = table_4_2_2.iloc[:,5].mean()
#Soiling

















