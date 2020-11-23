from matplotlib import pyplot
from shapely.geometry import LineString
import geohash2 as gh
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_geohash = 'geohash'

base32 = ['0', '1', '2', '3', '4', '5', '6', '7',
          '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
          'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
          's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#binary = [np.asarray(list('{0:05b}'.format(x, 'b')), dtype=int) for x in range(0, len(base32))]
binary = [np.asarray(list('{0:05b}'.format(x)), dtype=int) for x in range(0, len(base32))]
base32toBin = dict(zip(base32, binary))

COLOR = {
    True:  '#6699cc',
    False: '#ffcc33'
    }

def v_color(ob):
    return COLOR[ob.is_simple]

def plot_coords(ax, ob, color='r'):
    x, y = ob.xy
    ax.plot(x, y, 'o', color=color, zorder=1)

def plot_bounds(ax, ob, color='b'):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, '-', color='g', zorder=1)

def plot_line(ax, ob, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2):
    x, y = ob.xy
    ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, solid_capstyle=solid_capstyle, zorder=2)


def _encode(lat, lon, precision=15):
    return gh.encode(lat, lon, precision)

def _decode(geohash):
    return gh.decode(geohash)

def _bin_geohash(lat, lon, precision=15):
    hashed = _encode(lat, lon, precision)
    return np.concatenate([base32toBin[x] for x in hashed])

def create_geohash_df(df_, label_lat='lat', label_lon='lon', precision=15, label_geohash=label_geohash):
    try:
        #print('... reseting index')
        #df_.reset_index(drop=True, inplace=True)
        df_size = df_.shape[0]
        vetor_geohash = np.full(df_size, 0, dtype=object)
        count = 0
        for index, row in tqdm(df_[[label_lat, label_lon]].iterrows(), total=df_size):
            vetor_geohash[count] = _encode(row[label_lat], row[label_lon], precision)       
            count+=1
        df_[label_geohash] = vetor_geohash
    except Exception as e:
        raise e

def create_bin_geohash_df(df_, label_lat='lat', label_lon='lon', precision=15, label_geohash=label_geohash):
    try:
        assert set([label_lat, label_lon]).issubset(set(df_.columns)), "ERRO: {} and {} don't exist in df".format(label_lat, label_lon)
        
        df_size = df_.shape[0]
        vetor_geohash = np.full(df_size, 0, dtype=object)
        count = 0
        for index, row in tqdm(df_[[label_lat, label_lon]].iterrows(), total=df_size):
            vetor_geohash[count] = _bin_geohash(row[label_lat], row[label_lon], precision)       
            count+=1
        df_[label_geohash] = vetor_geohash
    except Exception as e:
        raise e

def decode_geohash_to_latlon(df_, label_geohash=label_geohash):
    try:
        df_size = df_.shape[0]

        lat = np.full(df_size, np.NAN, dtype=np.float32)
        lon = np.full(df_size, np.NAN, dtype=np.float32)
        count = 0
        for i, row in tqdm(df_[[label_geohash]].iterrows(), total=df_size):
            print(row[label_geohash])
            lat_lon = _decode(row[label_geohash])   
            lat[count] = lat_lon[0]
            lon[count] = lat_lon[1]
            count+=1

        df_['lat_d'] = lat
        df_['lon_d'] = lon
    except Exception as e:
        raise e


