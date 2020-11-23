from tqdm import tqdm_notebook as tqdm
from pymove import trajutils, utils as ut
import numpy as np
import pandas as pd
import time

def compress_segment_to_point(df_, label_segment='segment'):
    try:
        print('Compression in segment')
        print("current_shape: {}  ---  compress_shape: {}".format(df_.shape, df_[label_segment].nunique()))
        
        if df_.index.name is None:
            df_.set_index([label_segment], inplace=True)
            print("...setting segment label to: {}".format(label_segment))
        
        segment_index = df_.index.unique()
          
        col_object = df_.select_dtypes(include=[np.object]).columns
        col_float = df_.select_dtypes(include=[np.float]).columns
        
        print("...Object colums: {}".format(col_object))
        dic = {}
        print("...creating array to each column: {} ".format(label_segment))
        
        for c in col_object:
            dic[c] = np.full(segment_index.shape[0], np.NAN, dtype=np.object)        
        for c in col_float:
            dic[c] = np.full(segment_index.shape[0], np.NAN, dtype=np.float32)

            
        print('...getting object with higher count to each segment')
        for i, s in enumerate(tqdm(segment_index)):
            for d in dic.keys():
                ##if segment have is a Series or a single point
                if isinstance(df_.loc[s, d], pd.Series):
                    # if float then a mean is calculed to each segment, else set object with higher count 
                    if df_.loc[s, d].dtype == np.dtype('float'):
                        dic[d][i] = df_.loc[s, d].mean()
                    else:
                        dic[d][i] = df_.loc[s, d].value_counts(sort=True).index[0]
                else:
                    dic[d][i] = df_.loc[s, d]

        df_compress = pd.DataFrame()
        print('...creating compress dataframe')
        
        df_compress[label_segment] = segment_index
        for c in col_object:
             for d in dic.keys():
                df_compress[d] = dic[d]

        df_.reset_index([label_segment], inplace=True)

        return df_compress
    except Exception as e:
        print("Error: segment: {} --- column: {} --- i: {}".format(s, d, i))
        raise e

def compress_segment_stop_to_point(df_, label_segment = 'segment_stop', label_stop = 'stop', point_mean = 'default', drop_moves=True):        
    
    """ compreess a segment to point setting lat_mean e lon_mean to each segment"""
    try:
            
        if (label_segment in df_) & (label_stop in df_):
            #start_time = time.time()

            print("...setting mean to lat and lon...")
            df_['lat_mean'] = -1.0
            df_['lon_mean'] = -1.0


            if drop_moves is False:
                df_.at[df_[df_[label_stop] == False].index, 'lat_mean'] = np.NaN
                df_.at[df_[df_[label_stop] == False].index, 'lon_mean'] = np.NaN
            else:
                print('...move segments will be dropped...')


            print("...get only segments stop...")
            segments = df_[df_[label_stop] == True][label_segment].unique()
            
            sum_size_id = 0
            df_size = df_[df_[label_stop] == True].shape[0]
            curr_perc_int = -1
            start_time = time.time()

            for idx in tqdm(segments):
                filter_ = (df_[label_segment] == idx)
                
                size_id = df_[filter_].shape[0]
                # veirify se o filter is None
                if(size_id > 1):
                    # get first and last point of each stop segment
                    ind_start = df_[filter_].iloc[[0]].index
                    ind_end = df_[filter_].iloc[[-1]].index
                    
                    # default: point  
                    if point_mean == 'default':
                        #print('...Lat and lon are defined based on point that repeats most within the segment')
                        p = df_[filter_].groupby(['lat', 'lon'], as_index=False).agg({'id':'count'}).sort_values(['id']).tail(1)                     
                        df_.at[ind_start, 'lat_mean'] = p.iloc[0,0]
                        df_.at[ind_start, 'lon_mean'] = p.iloc[0,1]    
                        df_.at[ind_end, 'lat_mean'] = p.iloc[0,0]
                        df_.at[ind_end, 'lon_mean'] = p.iloc[0,1] 
                    
                    elif point_mean == 'centroid':
                        #print('...Lat and lon are defined by centroid of the all points into segment')
                        # set lat and lon mean to first_point and last points to each segment
                        df_.at[ind_start, 'lat_mean'] = df_.loc[filter_]['lat'].mean()
                        df_.at[ind_start, 'lon_mean'] = df_.loc[filter_]['lon'].mean()     
                        df_.at[ind_end, 'lat_mean'] = df_.loc[filter_]['lat'].mean()
                        df_.at[ind_end, 'lon_mean'] = df_.loc[filter_]['lon'].mean()   
                else:
                    print('There are segments with only one point: {}'.format(idx))
                
                sum_size_id  += size_id
                curr_perc_int, est_time_str = ut.progress_update(sum_size_id, df_size, start_time, curr_perc_int, step_perc=5)
            
            shape_before = df_.shape[0]

            # filter points to drop
            filter_drop = (df_['lat_mean'] == -1.0) & (df_['lon_mean'] == -1.0)
            shape_drop = df_[filter_drop].shape[0]

            if shape_drop > 0:
                print("...Dropping {} points...".format(shape_drop))
                df_.drop(df_[filter_drop].index, inplace=True)

            print("...Shape_before: {}\n...Current shape: {}".format(shape_before,df_.shape[0]))
            print('...Compression time: {:.3f} seconds'.format((time.time() - start_time)))
            print('-----------------------------------------------------\n')
        else:
            print('{} or {} is not in dataframe'.format(label_stop, label_segment))
    except Exception as e:
        raise e

def compress_segment_stop_to_point_optimizer(df_, label_segment = 'segment_stop', label_stop = 'stop', point_mean = 'default', drop_moves=True):        
    
    """ compreess a segment to point setting lat_mean e lon_mean to each segment"""
    try:
            
        if (label_segment in df_) & (label_stop in df_):
            #start_time = time.time()

            print("...setting mean to lat and lon...")
            #df_['lat_mean'] = -1.0
            #df_['lon_mean'] = -1.0

            lat_mean = np.full(df_.shape[0], -1.0, dtype=np.float32)
            lon_mean = np.full(df_.shape[0], -1.0, dtype=np.float32)

            if drop_moves is False:
                lat_mean[df_[df_[label_stop] == False].index] = np.NaN
                lon_mean[df_[df_[label_stop] == False].index] = np.NaN
            else:
                print('...move segments will be dropped...')

            sum_size_id = 0
            df_size = df_[df_[label_stop] == True].shape[0]
            curr_perc_int = -1
            start_time = time.time()
            
            print("...get only segments stop...")
            segments = df_[df_[label_stop] == True][label_segment].unique()
            for idx in tqdm(segments):
                filter_ = (df_[label_segment] == idx)
                
                size_id = df_[filter_].shape[0]
                # veirify se o filter is None
                if(size_id > 1):
                    # get first and last point of each stop segment
                    ind_start = df_[filter_].iloc[[0]].index
                    ind_end = df_[filter_].iloc[[-1]].index

                    if point_mean == 'default':
                        #print('...Lat and lon are defined based on point that repeats most within the segment')
                        p = df_[filter_].groupby(['lat', 'lon'], as_index=False).agg({'id':'count'}).sort_values(['id']).tail(1)                     
                        lat_mean[ind_start] = p.iloc[0,0]
                        lon_mean[ind_start] = p.iloc[0,1] 
                        lat_mean[ind_end] = p.iloc[0,0]
                        lon_mean[ind_end] = p.iloc[0,1] 
                        
                    elif point_mean == 'centroid':
                        #print('...Lat and lon are defined by centroid of the all points into segment')
                        # set lat and lon mean to first_point and last points to each segment
                        lat_mean[ind_start] = df_.loc[filter_]['lat'].mean()
                        lon_mean[ind_start] = df_.loc[filter_]['lon'].mean() 
                        lat_mean[ind_end] = df_.loc[filter_]['lat'].mean()
                        lon_mean[ind_end] = df_.loc[filter_]['lon'].mean() 
                else:
                    print('There are segments with only one point: {}'.format(idx))
                
                sum_size_id  += size_id
                curr_perc_int, est_time_str = ut.progress_update(sum_size_id, df_size, start_time, curr_perc_int, step_perc=10)
            
            df_['lat_mean'] = lat_mean
            df_['lon_mean'] = lon_mean
            del lat_mean
            del lon_mean

            shape_before = df_.shape[0]
            # filter points to drop
            filter_drop = (df_['lat_mean'] == -1.0) & (df_['lon_mean'] == -1.0)
            shape_drop = df_[filter_drop].shape[0]

            if shape_drop > 0:
                print("...Dropping {} points...".format(shape_drop))
                df_.drop(df_[filter_drop].index, inplace=True)

            print("...Shape_before: {}\n...Current shape: {}".format(shape_before,df_.shape[0]))
            print('...Compression time: {:.3f} seconds'.format((time.time() - start_time)))
            print('-----------------------------------------------------\n')
        else:
            print('{} or {} is not in dataframe'.format(label_stop, label_segment))
    except Exception as e:
        raise e
