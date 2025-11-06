from typing import Any, Dict, List, Union, Optional, Tuple
import pickle
import os
import pandas as pd
import geopandas as gpd
import libpysal as lps 
import numpy as np
import random 
import matplotlib.pyplot as plt

def time_since(group):
    g = group.copy()
    
    if g.iloc[0]== 0.0:
        g.iloc[0] = 999  
    
    count = list(range(len(g)))
    count = pd.Series(count, index = g.index)

    resets = count.where(g > 0).fillna(method='ffill')

    g = count - resets

    return g

def time_since_last_event(x: pd.Series) -> pd.Series:
    x = x.fillna(0).astype(bool)       
    idx = np.arange(len(x))
    last_idx = np.where(x, idx, np.nan)  
    last_idx = pd.Series(last_idx, index=x.index).ffill()
    s = pd.Series(idx, index=x.index) - last_idx
    return s.where(last_idx.notna(), np.inf) 

def decay(s: pd.Series, halflife: float) -> pd.Series:
    """Decay function

    See half-life formulation at
    https://en.wikipedia.org/wiki/Exponential_decay
    """

    return 2 ** ((-1 * s) / halflife)

def greater_or_equal(s: pd.Series, value: float) -> pd.Series:
    """ 1 if s >= value, else 0 """

    mask = s >= value
    y = mask.astype(int)

    return y

def moving_sum(
    s: pd.Series, 
    time: int
):
   
    # Groupby groupvar
    y = s.groupby(level=0)
    # Divide into rolling time window of size time
    # min_periods=0 lets the window grow with available data
    # and prevent the function from inducing missingness
    y = y.rolling(time, min_periods=0)
    # Compute the sum
    y = y.sum()
    # groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index()
    
    return y

def moving_avg(
    s: pd.Series, 
    time: int
):
   
    # Groupby groupvar
    y = s.groupby(level=0)
    # Divide into rolling time window of size time
    # min_periods=0 lets the window grow with available data
    # and prevent the function from inducing missingness
    y = y.rolling(time, min_periods=0)
    # Compute the sum
    y = y.mean()
    # groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index()
    
    return y

def moving_max(
    s: pd.Series, 
    time: int
):
   
    # Groupby groupvar
    y = s.groupby(level=0)
    # Divide into rolling time window of size time
    # min_periods=0 lets the window grow with available data
    # and prevent the function from inducing missingness
    y = y.rolling(time, min_periods=0)
    # Compute the sum
    y = y.max()
    # groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index()
    
    return y


def spatial_lag(
    gdf: gpd.GeoDataFrame, 
    gdf_geom: gpd.GeoDataFrame, 
    groupby: str,
    col: str,
):
    """ Compute spatial lag on col in gdf """

    def gdf_to_w_q(gdf_geom: gpd.GeoDataFrame):
        """Build queen weights from gdf.
        """
        # Compute first order spatial weight
        w = lps.weights.Queen.from_dataframe(gdf_geom, geom_col="geometry")

        return w

    def _splag(y: Any, w: Any):
        """ Flip argument order for transform """
        return lps.weights.lag_spatial(w, y)
    
    w = gdf_to_w_q(gdf_geom)
    s = gdf.groupby(groupby)[col].transform(_splag, w=w)
    return s

def interpolate(
    grouplevel,
    df: pd.DataFrame,
    limit_direction: str = "both",
    limit_area: Optional[str] = None,
    
) -> pd.DataFrame:
    """ Interpolate and extrapolate """
    return (
        df.sort_index()
        .groupby(level=grouplevel)
        .apply(
            lambda group: group.interpolate(
                method='linear',
                limit_direction=limit_direction, limit_area=limit_area
            )
        )
    )

def extrapolate(
    grouplevel,
    df: pd.DataFrame,
    limit_direction: str = "both",
    limit_area: Optional[str] = None,
    
) -> pd.DataFrame:
    """ Interpolate and extrapolate """
    return (
        df.sort_index()
        .groupby(level=grouplevel, group_keys=False)
        .apply(
            lambda group: group.interpolate(
                method='linear',
                limit_direction=limit_direction, limit_area=limit_area
            )
        )
    )


def ln(s: pd.Series) -> pd.Series:
    """ Natural log of s+1 """
    return np.log1p(s)

def ln_any(s, logn):
    return np.log(s+logn)

def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True