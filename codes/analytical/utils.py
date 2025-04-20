# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:24:23 2024

@author: shahriar
"""

import os
import pickle
import numpy as np


def save_data(data, filename, folder='data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {file_path}")
    
    
def load_data(filename, folder='data'):
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
    
def params_to_text(params, params_varied=[], string_varied='*', TREAT=False):
    #', '.join(["'%s' : %s" % (el,el) for el in a.replace('\n','').replace(' ','').split(',')])
    #print('\n'.join(["%s = params['%s']" % (el,el) for el in a.replace('\n','').replace(' ','').split(',')]))
    if type(string_varied)==str:
        string_varied = [string_varied] * len(params_varied)
    assert len(string_varied)==len(params_varied),"a string must be provided or a list of strings of equal length as params_varied"
    for i,parameter in enumerate(params_varied):
        params[parameter] = string_varied[i]
    r_s = params['r_s']
    r_r = params['r_r']
    c = params['c']
    d = params['d']
    K = params['K']
    t_s = params['t_s']
    t_r = params['t_r']
    # if TREAT==True:
    r_s_treatment = params['r_s_treatment']
    t_s_treatment = params['t_s_treatment']
    d_s = params['d_s']
    d_s_treatment = params['d_s_treatment']
    d_r = params['d_r']
    threshold_treatment_on = params['threshold_treatment_on']
    threshold_treatment_off = params['threshold_treatment_off']
    # else:
    #     r_s_treatment = np.nan
    #     t_s_treatment = np.nan
    #     d_s = np.nan
    #     d_s_treatment = np.nan
    #     d_r = np.nan
    #     threshold_treatment_on = np.nan
    #     threshold_treatment_off = np.nan

    dt = params['dt']
    text_left = []
    text_right = []
    if r_s == r_r and r_s==r_s_treatment:
        text_left.append(r'$r_s = r_r = r_s^{therapy} = $'+str(r_s))
    elif r_s == r_r:
        text_left.append(r'$r_s = r_r = $'+str(r_s)+r', $r_s^{therapy} = $' + str(r_s_treatment))
    else:
        text_left.append(r'$r_s = $'+str(r_s)+r', $r_r = $'+str(r_r)+r', $r_s^{therapy} = $' + str(r_s_treatment)) 
    if c == d:
        text_left.append(r'$c = d = $'+str(c)+r', $K = $' + str(K))
    else:
        text_left.append(r'$c = $'+str(c)+r', $d = $'+str(d)+r', $K = $' + str(K))
    if t_s == t_r and t_s==t_s_treatment:
        text_left.append(r'$t_s = t_r = t_s^{therapy} = $'+str(t_s))
    elif t_s == t_r:
        text_left.append(r'$t_s = t_r = $'+str(t_s)+r', $t_s^{therapy} = $' + str(t_s_treatment))        
    else:
        text_left.append(r'$t_s = $'+str(t_s)+r', $t_r = $'+str(t_r)+r', $t_s^{therapy} = $' + str(t_s_treatment))
                
    text_right.append('therapy start'+r'$ = $'+str(threshold_treatment_on)+'K')
    text_right.append('therapy stop'+r'$ = $'+str(threshold_treatment_off)+'K')
    if d_s == d_r and d_s==d_s_treatment:
        text_right.append(r'$d_s = d_r = d_s^{therapy} = $'+str(d_s))
    elif d_s == d_r:
        text_right.append(r'$d_s = d_r = $'+str(d_s)+r', $d_s^{therapy} = $' + str(d_s_treatment))        
    else:
        text_right.append(r'$d_s = $'+str(d_s)+r', $d_r = $'+str(d_r)+r', $d_s^{therapy} = $' + str(d_s_treatment))              

    return text_left,text_right

def get_suffix(text_left,text_right):
    suffix = ','.join(text_left)+','+','.join(text_right)
    suffix = suffix.replace(' ','').replace('_','').replace('{','').replace('}','').replace('^','').replace(',','_').replace('=','_').replace('therapy','th').replace('$','')
    return suffix




