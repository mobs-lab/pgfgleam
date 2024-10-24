#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions

Author: Guillaume St-Onge <g.st-onge@northeastern.edu>
"""

import numpy as np
import pandas as pd
from collections.abc import Iterable
from itertools import product
from datetime import date
from scipy.optimize import fsolve
from .cumulant import pdf_from_two_point_cumulants_NM

def calculate_growth_rate(R0, infectious_period, latency_period, nb_latent_state=1, nb_infectious_state=1):
    """See the following paper:
        https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020174
    """

    func = lambda r: R0 - r*(1 + r*latency_period/nb_latent_state)**nb_latent_state/\
            (infectious_period**(-1)*(1 - (1 + r*infectious_period/nb_infectious_state)**(-nb_infectious_state)))

    return fsolve(func, 0.1)[0]

def calculate_doubling_time(*args,**kwargs):
    return np.log(2)/calculate_growth_rate(*args,**kwargs)

def get_index_default(location_id, age_id, nb_location, nb_age_cat):
    """get_index_default provide a category index for metapopulation model with age structure.

    If location_id or age_id is None, all ids are used by default.

    Parameters
    ----------
    location_id : int or iterable
        Integer label(s) for a location.
    age_id : int or iterable
        Integer label(s) for age category.
    nb_age_cat" int
        Number of age categories
    """
    if age_id and location_id:
        idx = get_index(location_id, age_id, nb_age_cat)
    elif age_id and location_id is None:
        idx = get_index(np.arange(nb_location), age_id, nb_age_cat)
    elif location_id and age_id is None:
        idx = get_index(location_id, np.arange(nb_age_cat), nb_age_cat)
    else:
        idx = get_index(np.arange(nb_location), np.arange(nb_age_cat), nb_age_cat)
    return idx

def get_index(location_id, age_id, nb_age_cat):
    """get_index provide a category index for metapopulation model with age structure.

    Parameters
    ----------
    location_id : int or iterable
        Integer label(s) for a location.
    age_id : int or iterable
        Integer label(s) for age category.
    nb_age_cat" int
        Number of age categories
    """
    if isinstance(location_id,Iterable) and isinstance(age_id,Iterable):
        idx = [nb_age_cat*i + j for i,j in product(location_id,age_id)]
    elif isinstance(location_id,Iterable) and not isinstance(age_id,Iterable):
        idx = [nb_age_cat*i + age_id for i in location_id]
    elif not isinstance(location_id,Iterable) and isinstance(age_id,Iterable):
        idx = [nb_age_cat*location_id + j for j in age_id]
    else:
        idx = nb_age_cat*location_id+age_id
    return idx


def append_stats(distribution, quantiles, groupbycols, xcol='time'):
    #calculate cumulative
    dist = distribution.copy()
    dist['cum'] = dist.groupby(groupbycols)['probability'].cumsum()

    #get mean
    mean_df = dist.copy().drop(columns=['cum'])
    mean_df['mean'] = dist[xcol]*dist['probability']
    mean_df = mean_df.groupby(by=groupbycols).sum().reset_index().drop(columns=[xcol,'probability'])

    #get quantiles
    df_list = []
    for q in quantiles:
        df = dist[dist['cum'] >= q].copy()
        idx = df.groupby(by=groupbycols)['cum'].idxmin()
        df_ = dist.loc[idx].drop(columns=['probability','cum']).sort_values(
            by=groupbycols+[xcol]).copy()
        df_['quantile'] = q
        df_list.append(df_)

    quantile_df = pd.concat(df_list)
    quantile_df.rename(columns={xcol:'value'},inplace=True)
    quantile_df = quantile_df.pivot_table(values=['value'],index=groupbycols,
                                  columns=['quantile'])['value'].reset_index()
    quantile_df.columns = quantile_df.columns.astype(str)
    stats_df = pd.merge(mean_df,quantile_df, on=groupbycols)

    #merge stats
    dist = dist.merge(stats_df, on=groupbycols)

    return dist


#alias for backward compatibility
def get_time_to_detection(*args,**kwargs):
    return get_detection_time(*args,**kwargs)

def get_detection_time(distribution, D, quantiles=[0.5], groupbycols=['label']):
    """get_time_to_detection calculates the posterior distribution on the time to get D detections.

    Uses the distribution of # of detected cases

    Also calculates the mean detection time and desired ~quantiles, th first time such that the
    cumulative probability is bigger than a quantile q value, e.g., q = 0.5 for the median.

    Parameters
    ----------
    distribution : Pandas DataFrame
        Contains the result of a marginal distribution for a given quantity (detected)
    D : integer
        Fixed value for the number of detections
    quantiles: list of float
        q values for the desired quantiles to compute
    groupbycols: list of str
        Columns to groupby for the calculations.
    """
    cum = distribution.copy()[distribution['target'] < D].groupby(
        by=groupbycols+['time']).sum().reset_index().drop(columns=['target'])
    detection_time = cum #probability that target < D
    detection_time['probability_'] = cum.groupby(
        by=groupbycols)['probability'].shift()
    detection_time['probability'] = (detection_time['probability_'] - detection_time['probability']).abs()
    detection_time = detection_time.drop(columns={'probability_'}).dropna()

    #normalize
    norm = detection_time.groupby(by=groupbycols).sum().reset_index().drop(columns=['time']).rename(columns={'probability':'norm'})
    detection_time = detection_time.merge(norm, on=groupbycols)
    detection_time['probability'] /= detection_time['norm']
    detection_time = detection_time.drop(columns=['norm'])

    #append stats
    detection_time = append_stats(detection_time, quantiles, groupbycols)

    return detection_time


def get_pdf_given_detection(joint_cumulants, detection_time, D, target_values, quantiles=[0.5]):
    """get_pdf_given_detection_time calculates the posterior distribution on a quantity given a number of
    detections D. It combines two-point cumulants estimates with the detection time to get this distribution.

    Also calculates the mean and desired ~quantiles, i.e., the first time such that the
    cumulative probability is bigger than a quantile q value, e.g., q = 0.5 for the ~median.

    NOTE: Only the cumulants as a function of time and a single detection time distribution are allowed, i.e.,
    there must not be any other independent column, as opposed to the function get_detection_time.

    Parameters
    ----------
    joint_cumulants : Pandas DataFrame
        Contains the result for the two-point cumulants for a detected quantity and quantity of interest
        target1 corresponds to detected quantity and target2 corresponds to quantity of interest
    detection_time : Pandas DataFrame
        Contains the result of the marginal detection time for a detected quantity
    D : integer
        Fixed value for the number of detections
    quantiles: list of float
        q values for the desired quantiles to compute
    """
    det = detection_time.copy()
    det['probability'] /= det['probability'].sum() #ensure normalization
    tp_cum = joint_cumulants[joint_cumulants['time'].isin(det['time'])]
    times = tp_cum['time'].unique()
    dist_list = []
    for t_ in times:
        func = pdf_from_two_point_cumulants_NM(tp_cum, time=t_)
        dist = func(D,target_values)
        dist /= sum(dist)
        dist *= det[det['time'] == t_]['probability'].values[0]
        dist_list.append(dist)
    agg_dist = np.nansum(dist_list, axis=0)
    agg_dist /= np.sum(agg_dist)
    cumul_dist = np.cumsum(agg_dist)
    result = pd.DataFrame(
        {'target':target_values, 'probability':agg_dist,
         'cum':cumul_dist})
    #get quantiles
    for q in quantiles:
        val = np.searchsorted((cumul_dist-q),0)
        result[str(q)] = val
    return result
