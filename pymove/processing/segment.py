from pymove.processing import trajutils
from pymove.core import  utils as ut
import numpy as np
import numpy as np
import pandas as pd
import time
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""main labels """
dic_labels = {"id" : 'id', 'lat' : 'lat', 'lon' : 'lon', 'datetime' : 'datetime'}

dic_features_label = {'tid' : 'tid', 'dist_to_prev' : 'dist_to_prev', "dist_to_next" : 'dist_to_next', 'dist_prev_to_next' : 'dist_prev_to_next', 
                    'time_to_prev' : 'time_to_prev', 'time_to_next' : 'time_to_next', 'speed_to_prev': 'speed_to_prev', 'speed_to_next': 'speed_to_next',
                    'period': 'period', 'day': 'day', 'index_grid_lat': 'index_grid_lat', 'index_grid_lon' : 'index_grid_lon',
                    'situation':'situation'}

def label_encode_df(df_, columns=None, first_encoding_number=0):
    df_output = pd.DataFrame()
    label_encode = {}
    if columns is not None:
         for col in columns:
            le = LabelEncoder()
            df_output[col] = le.fit_transform(df_[col]) 
            label_encode[col] = le
            if first_encoding_number != 0:
                df_output[col] = df_output[col] + first_encoding_number
    else:  
        for colname, col in df_.iteritems():
            le = LabelEncoder()
            df_output[colname] = le.fit_transform(col)
            label_encode[col] = le
            if first_encoding_number != 0:
                df_output[colname] = df_output[colname] + first_encoding_number
    
    return df_output

def segment_traj_by_max_dist_time_speed(df_, 
                                        label_id=dic_labels['id'], 
                                        max_dist_between_adj_points=3000, 
                                        max_time_between_adj_points=7200,
                                        max_speed_between_adj_points=50.0, 
                                        label_segment='tid_part'):
    """ segment trajectory based on threshold for each ID object"""
    """
    index_name is the current id.
    label_new_id is the new splitted id.
    time, dist, speeed features must be updated after split.
    """
        
    print('Split trajectories')
    print('...max_time_between_adj_points:', max_time_between_adj_points)
    print('...max_dist_between_adj_points:', max_dist_between_adj_points)
    print('...max_speed:', max_speed_between_adj_points)
    
    try:
        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_segment not in df_:
            df_[label_segment] = curr_tid

        ids = df_.index.unique()
        count = 0
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:
            curr_tid += 1
            
            filter_ = (df_.at[idx, dic_features_label['time_to_prev']] > max_time_between_adj_points) | \
                        (df_.at[idx, dic_features_label['dist_to_prev']] > max_dist_between_adj_points) | \
                        (df_.at[idx, dic_features_label['speed_to_prev']] > max_speed_between_adj_points)        

            """ check if object have only one point to be removed """
            if filter_.shape == ():
                print('id: {} has not point to split'.format(id))
                df_.at[idx, label_segment] = curr_tid
                count+=1
            else:
                tids = np.empty(filter_.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(filter_):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += len(tids)
                df_.at[idx, label_segment] = tids
            
            curr_perc_int, est_time_str = ut.progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_segment:
            df_.reset_index(drop=True, inplace=True)
            print('... label_id = label_segment, then reseting and drop index')
        else:
            df_.reset_index(inplace=True)
            print('... Reseting index')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')
        #if drop_single_points:
         #   shape_before_drop = df_.shape
         #   idx = df_[ df_[label_segment] == -1 ].index
          #  if idx.shape[0] > 0:
           #     print('...Drop Trajectory with a unique GPS point\n')
            #    ids_before_drop = df_[label_id].unique().shape[0]
             #   df_.drop(index=idx, inplace=True)
              #  print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, df_[label_id].unique().shape[0]))
               # print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
            #else:
             #   print('...No trajs with only one point.', df_.shape)
    except Exception as e:
        print('label_id:{}\nidx:{}\n'.format(label_id, idx))
        raise e

def segment_traj_by_max_dist(df_, 
                            label_id=dic_labels['id'],  
                            max_dist_between_adj_points=3000, 
                            label_segment='tid_dist'):
    """ Index_name is the current id.
    label_new_id is the new splitted id.
    Speed features must be updated after split.
    """     
    print('Split trajectories by max distance between adjacent points:', max_dist_between_adj_points) 
    try:
        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_segment not in df_:
            df_[label_segment] = curr_tid

        ids = df_.index.unique()
        count = 0
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:            
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter dist max"""
            dist = (df_.at[idx, dic_features_label['dist_to_prev']] > max_dist_between_adj_points)                      
            """ check if object have more than one point to split"""
            if dist.shape == ():   
                print('id: {} has not point to split'.format(idx))
                df_.at[idx, label_segment] = curr_tid
                count+=1
            else: 
                tids = np.empty(dist.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(dist):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += len(tids)
                df_.at[idx, label_segment] = tids

            curr_perc_int, est_time_str = ut.progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_segment:
            df_.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            df_.reset_index(inplace=True)
            print('... Reseting index')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n') 
    except Exception as e:
        print('label_id:{}\nidx:{}\n'.format(label_id, idx))
        raise e

def segment_traj_by_max_time(df_, 
                            label_id=dic_labels['id'], 
                            max_time_between_adj_points=900.0, 
                            label_segment='tid_time'):
    """
    index_name is the current id.
    label_new_id is the new splitted id.
    Speed features must be updated after split.
    """     
    print('Split trajectories by max_time_between_adj_points:', max_time_between_adj_points) 
    try:
        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_segment not in df_:
            df_[label_segment] = curr_tid

        ids = df_.index.unique()
        count = 0
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:            
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter time max"""
            times = (df_.at[idx, dic_features_label['time_to_prev']] > max_time_between_adj_points)        
                     
            """ check if object have only one point to be removed """
            if times.shape == ():
                print('id: {} has not point to split'.format(id))
                df_.at[idx, label_segment] = curr_tid
                count+=1
            else: 
                tids = np.empty(times.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(times):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += len(tids)
                df_.at[idx, label_segment] = tids

            curr_perc_int, est_time_str = ut.progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_segment:
            df_.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            df_.reset_index(inplace=True)
            print('... Reseting index')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')      
        #if drop_single_points:
         #   shape_before_drop = df_.shape
          #  idx = df_[ df_[label_segment] == -1 ].index
           # if idx.shape[0] > 0:
            #    print('...Drop Trajectory with a unique GPS point\n')
             #   ids_before_drop = df_[label_id].unique().shape[0]
              #  df_.drop(index=idx, inplace=True)
               # print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, df_[label_id].unique().shape[0]))
               # print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
            #else:
             #   print('...No trajs with only one point.', df_.shape)

    except Exception as e:
        print('label_id:{}\nidx:{}\n'.format(label_id, idx))
        raise e

def segment_traj_by_max_speed(df_, 
                                label_id=dic_labels['id'], 
                                max_speed_between_adj_points=50.0, 
                                label_segment='tid_speed'):
    """ Index_name is the current id.
    label_new_id is the new splitted id.
    Speed features must be updated after split.
    """     
    print('Split trajectories by max_speed_between_adj_points:', max_speed_between_adj_points) 
    try:
        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_segment not in df_:
            df_[label_segment] = curr_tid

        ids = df_.index.unique()
        count = 0
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:            
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter speed max"""
            speed = (df_.at[idx, dic_features_label['speed_to_prev']] > max_speed_between_adj_points)        
            """ check if object have only one point to be removed """
            if speed.shape == ():
                print('id: {} has not point to split'.format(id))
                df_.at[idx, label_segment] = curr_tid
                count+=1
            else: 
                tids = np.empty(speed.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(speed):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += len(tids)
                df_.at[idx, label_segment] = tids

            curr_perc_int, est_time_str = ut.progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_segment:
            df_.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            df_.reset_index(inplace=True)
            print('... Reseting index')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')
       
        #if drop_single_points:
         #   shape_before_drop = df_.shape
          #  idx = df_[df_[label_segment] == -1].index
           # if idx.shape[0] > 0:
            #    print('...Drop Trajectory with a unique GPS point\n')
             #   ids_before_drop = df_[label_id].unique().shape[0]
              #  df_.drop(index=idx, inplace=True)
               # print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, df_[label_id].unique().shape[0]))
               # print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
                #create_update_dist_time_speed_features(df_, label_segment, dic_labels)
            #else:
                #print('...No trajs with only one point.', df_.shape)

    except Exception as e:
        print('label_id:{}\nidx:{}\n'.format(label_id, idx))
        raise e

def segment_traj_by_radius_dist(df_,
                                    radius,
                                    sort=False,
                                    label_segment='segment',
                                    label_datetime=dic_labels['datetime'],
                                    label_id=dic_labels['id'],
                                    label_lat=dic_labels['lat'],
                                    label_lon=dic_labels['lon']):

    if sort:
        df_.sort_values([label_id, label_datetime], inplace=True)

    df_.reset_index(drop=True, inplace=True)

    segment_stop = 0

    seg_arr = np.full(df_.shape[0], -1, dtype=np.int32)
    start_time = time.time()
    for id_ in df_[label_id].unique():

        df_id = df_[df_[label_id] == id_]

        initial_index = df_id.index[0]

        i = 0

        lat_id = df_id[label_lat].to_numpy(dtype=np.float32)
        lon_id = df_id[label_lon].to_numpy(dtype=np.float32)

        lat_aux = np.empty(df_id.shape[0], dtype=np.float32)
        lon_aux = np.empty(df_id.shape[0], dtype=np.float32)

        while i < df_id.shape[0]:

            lat_aux[:] = lat_id[i]

            lon_aux[:] = lon_id[i]

            distances = trajutils.haversine(lat_aux,
                                            lon_aux,
                                            lat_id,
                                            lon_id)

            while i < df_id.shape[0] and distances[i] <= radius:
                lat_id[i] = np.NaN

                lon_id[i] = np.NaN

                seg_arr[i + initial_index] = segment_stop

                i += 1

            segment_stop += 1

    df_[label_segment] = seg_arr
    print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
    print('------------------------------------------\n')
    return df_

def segment_traj_by_label_optimizer(df_,
                          by_label='',
                          label_id=dic_labels['id'],
                          label_segment='segment',
                          sort=False):
    try:

        df_.reset_index(drop=True, inplace=True)

        start_time = time.time()
        diff_y = 'copy_y'
        diff_id = 'copy_id'
        
        #df_copy = df_[[by_label,label_id]].copy()
        df_copy = label_encode_df(df_, columns=[label_id, by_label], first_encoding_number=1) 
             
        df_copy[diff_y] = np.append([0], np.diff(df_copy[by_label].to_numpy()))
        df_copy[diff_id] = np.append([0], np.diff(df_copy[label_id].to_numpy()))
        
        index = df_copy[(df_copy[diff_y] != 0) | (df_copy[diff_id] != 0)].index
        index = index.insert(0,0)

        count = 1
        tids = np.full(df_.shape[0], -1, dtype=np.int64)

        for current_index in tqdm(index):
            tids[current_index:] = count           
            count+= 1

        df_[label_segment] = tids
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')

    except Exception as e:
        raise e

def segment_traj_by_label_and_time_optimizer(df_,
                                            by_label='',
                                            label_id=dic_labels['id'],
                                            label_segment='segment',
                                            slot_interval=15,
                                            sort=False):
    try:
        df_.reset_index(drop=True, inplace=True)

        start_time = time.time()
        diff_y = 'copy_y'
        diff_id = 'copy_id'
        diff_time = 'copy_time'

        # Transform the features em int
        df_copy = label_encode_df(df_, columns=[label_id, by_label], first_encoding_number=1) 

        label_time = 'time'    
        df_copy['datetime'] = df_['datetime']

        trajutils.create_time_slot_in_minute_from_datetime(df_copy, slot_interval=slot_interval, initial_slot=0, label_datetime='datetime',label_time_slot=label_time)
        # add two features using shift
        df_copy[diff_y] = np.append([0], np.diff(df_copy[by_label].to_numpy()))
        df_copy[diff_id] = np.append([0], np.diff(df_copy[label_id].to_numpy()))
        df_copy[diff_time] = np.append([0], np.diff(df_copy[label_time].to_numpy()))
        
        index = df_copy[(df_copy[diff_y] != 0) | (df_copy[diff_id] != 0) | (df_copy[diff_time] != 0)].index
        index = index.insert(0,0)

        count = 1
        tids = np.full(df_.shape[0], -1, dtype=np.int64)

        for current_index in tqdm(index):
            tids[current_index:] = count           
            count+= 1

        df_[label_segment] = tids
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')
    except Exception as e:
        raise e


# def label_encode_df(df_, columns=None, first_encoding_number=0):
#     df_output = pd.DataFrame()
#     label_encode = {}
#     if columns is not None:
#          for col in columns:
#             le = LabelEncoder()
#             df_output[col] = le.fit_transform(df_[col]) 
#             label_encode[col] = le
#             if first_encoding_number != 0:
#                 df_output[col] = df_output[col] + first_encoding_number
#     else:  
#         for colname, col in df_.iteritems():
#             le = LabelEncoder()
#             df_output[colname] = le.fit_transform(col)
#             label_encode[col] = le
#             if first_encoding_number != 0:
#                 df_output[colname] = df_output[colname] + first_encoding_number
    
#     return df_output