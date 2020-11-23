"""main labels """
dic_labels = {"id" : 'id', 'lat' : 'lat', 'lon' : 'lon', 'datetime' : 'datetime'}

dic_features_label = {'tid' : 'tid', 'dist_to_prev' : 'dist_to_prev', "dist_to_next" : 'dist_to_next', 'dist_prev_to_next' : 'dist_prev_to_next', 
                    'time_to_prev' : 'time_to_prev', 'time_to_next' : 'time_to_next', 'speed_to_prev': 'speed_to_prev', 'speed_to_next': 'speed_to_next',
                    'period': 'period', 'day': 'day', 'index_grid_lat': 'index_grid_lat', 'index_grid_lon' : 'index_grid_lon',
                    'situation':'situation'}

dic_event_labels = {'tnz_id':'id',
             'tnz_lat':'lat',
             'tnz_lon':'lon',
             'tnz_datetime':'datetime',
             
             'event_id':'id',
             'event_lat':'lat',
             'event_lon':'lon',
             'event_datetime':'datetime',

             'poi_id':'Nome do Local',
             'poi_lat':'Latitude',
             'poi_lon':'Longitude',
             
             'lat': 'lat',
             'lon': 'lon',
             'datetime': 'datetime',
             'id': 'id'}

dic_plot = {'radius': 150, 
            'event_point':'purple',
            'tnz_point':'orange', 
            'poi_point':'black',
            'line':'blue', 
            'start':'green', 
            'end':'red'}