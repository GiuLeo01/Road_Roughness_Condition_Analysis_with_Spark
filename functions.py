from pyspark.sql.types import *
import pyspark.sql.functions as F
import numpy as np


def dataset_preparation(name:str, spark_session, data_path:str):
    """
    Funzione che prepara i dati di un driving process combinando i file cvs originali, restituendo un dataframe pyspark che rappresenta la time series multivariata.
    Nel dataset originale, per ogni PVS[n], sono presenti 3 file:
        - dataset_gps_mpu_left.csv: file contenente per ogni timestamp le misurazioni dei sensori posizionati sul lato sinistro del veicolo (e speed, longitude, latitude).
        - dataset_gps_mpu_right.csv: file contenente per ogni timestamp le misurazioni dei sensori posizionati sul lato destro del veicolo (e speed, longitude, latitude).
        - dataset_labels.csv: file contenente per ogni timestamp le label.
    """

    # leggo i 3 file originali
    pvs_df_left = spark_session.read.csv(f'{data_path}/{name}/dataset_gps_mpu_left.csv', header=True)
    pvs_df_right = spark_session.read.csv(f'{data_path}/{name}/dataset_gps_mpu_right.csv', header=True)
    pvs_labels = spark_session.read.csv(f'{data_path}/{name}/dataset_labels.csv', header=True)

    # visto che il timestamp originale ha una differenza di tempo fissa (0.01s), per comodità uso un timestep discreto
    pvs_df_left = pvs_df_left.withColumn('timestep', F.monotonically_increasing_id())
    pvs_df_right = pvs_df_right.withColumn('timestep', F.monotonically_increasing_id())
    pvs_labels = pvs_labels.withColumn('timestep', F.monotonically_increasing_id())

    # nomi degli attributi, per distinguerli uso _L per i sensori a sinistra, e _R per i sensori a destra
    left_colnames = [f'{colname}_L' if colname not in ['timestep', 'latitude', 'longitude', 'speed'] else colname for colname in pvs_df_left.columns]
    right_colnames = [f'{colname}_R' if colname not in ['timestep', 'latitude', 'longitude', 'speed'] else colname for colname in pvs_df_right.columns]

    # La label originale della Road Roughness Condition distingue tra la condizione della strada a sx e a dx, per la mia analisi non è interessante, quindi combino queste informazioni:
    #   - se almeno uno dei due lati ha label 'bad', allora assegno 'bad' all'intera osservazione.
    #   - altrimenti, se almeno uno dei due lati ha label 'regulare', assegno 'regular' all'intera osservazione.
    #   - altrimenti, assegno 'good'
    pvs_labels = pvs_labels.select(['timestep','good_road_left', 'good_road_right', 'regular_road_left', 'regular_road_right','bad_road_left','bad_road_right'])
    cond = F.when((F.col('bad_road_left') == 1) | (F.col('bad_road_right') == 1), 'bad')\
        .when((F.col('regular_road_left') == 1) | (F.col('regular_road_right') == 1), 'regular')\
        .otherwise('good')

    pvs_labels = pvs_labels.withColumn('road_condition', cond).select(['timestep', 'road_condition'])

    # creo dei nuovi df con i nomi delle colonne che ho modificato prima (per distinguere left e right)
    pvs_df_left = pvs_df_left.toDF(*left_colnames)
    pvs_df_right = pvs_df_right.toDF(*right_colnames)

    # faccio una join dei dati di sinistra, quelli di destra, e le label. In realtà sarebbe bastata una concatenazione orizzontale dei dataframe, ma non ho trovato un metodo per farlo :(
    # tra i dati di destra faccio la join non solo sul timestep, ma anche su latitude, longitude e speed, che per costruzione dei dataset originali sono per forza di cose uguali tra i due file (il GPS sul veicolo è unico) 
    pvs_df = pvs_df_left.join(pvs_df_right, on=['timestep','latitude','longitude','speed'], how='left').join(pvs_labels, on='timestep', how='left').orderBy('timestep')
    pvs_df = pvs_df.drop('timestamp_L', 'timestamp_R','timestamp_gps_L', 'timestamp_gps_R')

    return pvs_df


def approximate_ts(df, features:list, timesteps:int=200, label=True):
    """
    Funzione che approssima una timeseries, dividendola in segmenti lunghi quanto il parametro timesteps, e calcola le delle statistiche aggregative delle feature come valori rappresentati dei segmenti.
    Se le feature vengono tutte aggregate solo con la media, dovrebbe essere equivalente al metodo Piecewice Aggregate Approximation.
    Nota: il segmento finale potrebbe essere composto da meno di 'timesteps' osservazioni.
    """
    # aggiungo una colonna con la divisione intera del timestep orginale, con il parametro 'timesteps', questo forma dei gruppi (non sovrapposti), con quello che sarà il timestep della time series approssimata.
    df = df.withColumn('group_by', (F.col('timestep') / timesteps).cast('integer'))

    # se tenere o meno la label nella time series approssimata
    if label:
        res = df.groupBy('group_by').agg(F.max('timestep').alias('timestep'),F.mode('road_condition').alias('road_condition'),\
                                                F.avg('latitude').alias('latitude'), F.avg('longitude').alias('longitude'),*features).orderBy('timestep')
    else:
        res = df.groupBy('group_by').agg(F.max('timestep').alias('timestep'),F.avg('latitude').alias('latitude'), F.avg('longitude').alias('longitude'),*features).orderBy('timestep')
    return res


def moving_average(arr, window_size):
    """Funzione per fare la media mobile di una sequenza"""
    padded_arr = np.pad(arr, (window_size // 2, window_size // 2), mode='reflect')
    moving_avg = np.convolve(padded_arr, np.ones(window_size) / window_size, mode='valid')
    return moving_avg[:-1]