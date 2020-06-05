# loading modules
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import datetime, os
import importlib
import plotly.graph_objects as go
import geopandas as gp
import shapely

SHP_PATH = 'Shp'

# loading data
INPUTFN ='CERCACOVID_questionari_clean_2020-05-04.csv'
assert os.path.exists(INPUTFN)


def pprint_pivot(table, tot=None, perc_col_name = None):
    assert len(table.columns) == 1, table.columns
    c = table.columns[0]
    if tot is None:
        tot_ = table[c].sum()
    else:
        tot_ = tot
        
    if not perc_col_name:
        perc_col_name = 'Percentage'
    table[perc_col_name] = 100.*table[c] / tot_
    if tot is None:
        assert np.isclose(table[perc_col_name].sum(), 100)
    fmt = {perc_col_name: "{:.1f}%"}
    return table.style.format(fmt).bar(subset=[perc_col_name], color='#d65f5f', vmin=0)


def sintomi_matrix(df):
    sintomi_df = {'FEBBRE' : np.where(df['FEBBRE'] >= 37., 1, 0)}
    for s in SINTOMI:
        if s == 'FEBBRE':
            continue
        sintomi_df[s] = np.where(df[s] != 'No', 1, 0)
    sintomi_df = pd.DataFrame(sintomi_df, index=df.index)        
    sintomi_df['NUMERO_SINTOMI'] = np.sum(sintomi_df, axis=1)        
    return sintomi_df

def sintomi_gravi_matrix(df):
    sintomi_df = {
        'FEBBRE' : np.where(df['FEBBRE'] >= 37.5, 1, 0),
        'GUSTO_OLFATTO' : np.where(df['GUSTO_OLFATTO'] != 'No', 1, 0), 
        'TOSSE' : np.where(np.logical_and(df['TOSSE'] != 'No', df['TOSSE'] != 'Si, ma non persistente'), 1, 0), 
        'DOLORI_MUSCOLARI' : np.where(df['DOLORI_MUSCOLARI'] == 'Si, nelle ultime due settimane ho iniziato ad accusare dolori', 1, 0), 
        'STANCHEZZA' : np.where(np.logical_and(df['STANCHEZZA'] != 'No', df['STANCHEZZA'] != 'Si, ma nella norma', ), 1, 0), 
        'CONGIUNTIVITE' : np.where(df['CONGIUNTIVITE'] != 'No', 1, 0), 
        'DIARREA' : np.where(df['DIARREA'] != 'No', 1, 0), 
        'RAFFREDDORE' : np.where(df['RAFFREDDORE'] == 'Si, abbastanza forte', 1, 0)
    }
    sintomi_df = pd.DataFrame(sintomi_df, index=df.index)
    sintomi_df['NUMERO_SINTOMI'] = np.sum(sintomi_df, axis=1)    
    return sintomi_df


def load_shapes():
    shp_df = gp.read_file(os.path.join(SHP_PATH, "Com2016_ED50_g", "Com2016_ED50_g.shp"))
    shp_df['PRO_COM'] = shp_df['PRO_COM'].astype(str).str.zfill(6)
    return shp_df

def load_prov_shapes():
    shp_df = gp.read_file(os.path.join(SHP_PATH, "CMProv2016_ED50_g", "CMProv2016_ED50_g.shp"))
    return shp_df

def prepare_generic_plot_data(shp_df, mov_comune, use_all_prov_comuni=True):
    data_merge_key = 'DOMICILIO_COMUNE_ISTAT'
    shp_merge_key = 'PRO_COM'

    # get data
    filtered_data = mov_comune.copy()
    filtered_data['DOMICILIO_COMUNE_ISTAT'] = filtered_data['DOMICILIO_COMUNE_ISTAT'].astype(str).str[2:]
    # get shapes
    if use_all_prov_comuni:
        list_comuni = shp_df[shp_df[shp_merge_key].isin(filtered_data[data_merge_key])]['COD_PRO'].unique()

        filtered_shapes = shp_df[shp_df['COD_PRO'].isin(list(list_comuni))]
        total_df=filtered_shapes.merge(filtered_data, left_on=shp_merge_key, right_on=data_merge_key, how='outer')
        total_df = total_df.dropna(subset=['geometry'])
    else:
        filtered_shapes = shp_df[shp_df[shp_merge_key].isin(filtered_data[data_merge_key])]
        total_df=filtered_shapes.merge(filtered_data, left_on=shp_merge_key, right_on=data_merge_key)
    
    return total_df


def create_prov_data(shp_df_prov, list_province, fill_value = np.nan):
    shp_merge_key='SIGLA'
    total_df = shp_df_prov.loc[shp_df_prov[shp_merge_key].isin(list_province)].copy()
    
    total_df['value'] = fill_value
    
    return total_df


def discretizza_sintomi(sintomi_all_gravi, qs):
    sintomi_all_gravi['interval'] = np.nan
    sintomi_all_gravi['% discreta'] = np.nan
    
    for interval in list(qs.keys()):
        curr =sintomi_all_gravi.loc[np.logical_and(sintomi_all_gravi['%'] > qs[interval]['min'],
                                                   sintomi_all_gravi['%'] <= qs[interval]['max'])]
        sintomi_all_gravi.loc[curr.index, '% discreta'] = qs[interval]['val']
        sintomi_all_gravi.loc[curr.index, 'interval'] = interval
    
    return sintomi_all_gravi


def plot_map(score_map, col, title, qs, threshold_utenti=50):
    score_map1 = score_map.copy().reset_index()
    score_map1.loc[score_map1['UTENTI'] < threshold_utenti, col] = np.nan
    score_map1 = score_map1.rename(columns={col: '%'})

    score_map1_q = discretizza_sintomi(score_map1, qs)

    # plot
    plot_data = prepare_generic_plot_data(shape_df, score_map1_q)

    plot_key = '% discreta'
    
    fig = pu.map_plot_v2(plot_data, 
                      plot_key, 
                      "COMUNE",
                      cmap = plt.cm.RdYlGn_r,
                      plot_showLegend = True,
                      lower_bound = np.min([qs[x]['val'] for x in list(qs.keys())]),
                      upper_bound = np.max([qs[x]['val'] for x in list(qs.keys())]),
                      plot_keyLegendNames = 'interval',
                      plot_dictBoundaries = plot_dictBoundaries_com,
                      plot_title = title
                    )

    # add boundaries
    plot_prov = create_prov_data(shp_prov, list(PROVINCIE.index), fill_value = np.nan)
    fig = pu.append_prov_borders(fig, plot_prov, plot_dictBoundaries = plot_dictBoundaries_prov)
    return fig


import plot_utils as pu
importlib.reload(pu)
shape_df = load_shapes()
shp_prov = load_prov_shapes()

plot_dictBoundaries_com = dict(color = 'rgb(130, 130, 130)', width = 1 )
plot_dictBoundaries_prov = dict(color = 'rgb(89, 89, 89)', width = 2 )


# list of symptoms
SINTOMI = ['FEBBRE', 'GUSTO_OLFATTO', 'TOSSE', 'DOLORI_MUSCOLARI', 'STANCHEZZA', 'CONGIUNTIVITE', 'DIARREA', 'RAFFREDDORE']

# df with population data
PROVINCIE = pd.DataFrame({
    'SIGLA': ['BG', 'VA', 'CR', 'PV', 'BS', 'LC', 'MN', 'MI', 'LO', 'MB', 'SO', 'CO'],
    'CAPOLUOGO': ['Bergamo', 'Varese', 'Cremona', 'Pavia', 'Brescia', 'Lecco', 'Mantova', 'Milano', 'Lodi', 'Monza', 'Sondrio', 'Como'],
    'POPOLAZIONE': [1112464, 888080, 354969, 544800, 1265954, 336537, 404780, 3244510, 228700, 873935, 181095, 591000]
}).set_index(keys='SIGLA')


def i2str(i):
    assert type(i) is int
    return format(i, ',d').replace(',', '.')


# score computation
score_map_ = {
    'RAFFREDDORE': {
        'No': 0,
        'Si, leggera': 1,
        'Si, abbastanza forte': 3
    },
    'TOSSE': {
        'No': 0,
        'Si, ma non persistente': 4,
        'Si, persistente e grassa': 5,        
        'Si, persistente e secca': 6        
    }
}

def score_predittivo(df):
    score_df = {
        'FEBBRE' : np.where(df['FEBBRE'] >= 37.5, 5, 0),
        'GUSTO_OLFATTO' : np.where(df['GUSTO_OLFATTO'] != 'No', 8, 0), 
        'TOSSE' : df['TOSSE'].apply(lambda x: score_map_['TOSSE'][x]),
        'DOLORI_MUSCOLARI' : np.where(df['DOLORI_MUSCOLARI'] != 'No', 3, 0), 
        'STANCHEZZA' : np.where(np.logical_and(df['STANCHEZZA'] != 'No', df['STANCHEZZA'] != 'Si, ma nella norma', ), 3, 0),         
        'CONGIUNTIVITE' : np.where(df['CONGIUNTIVITE'] != 'No', 2, 0), 
        'DIARREA' : np.where(df['DIARREA'] != 'No', 3, 0), 
        'RAFFREDDORE' : df['RAFFREDDORE'].apply(lambda x: score_map_['RAFFREDDORE'][x])
    }
    score_df = pd.DataFrame(score_df, index=df.index)
    score_df['SCORE'] = np.sum(score_df, axis=1)    
    return score_df


# load data
dtype_UC = {'DOMICILIO_COMUNE_ISTAT': 'object', 
            'DOMICILIO_CAP': 'object', 
            'LAVORO_COMUNE_ISTAT': 'object', 
            'LAVORO_CAP': 'object', 
            'ID_QUESTIONARIO': 'object', 
           }
for c in ['DOMICILIO_PROVINCIA', 'PERSONE_COVID19', 'LUOGHI_COVID19', 'CLASSE_ETA', 'CLASSE_ETA_10', 'SESSO',
          'GUSTO_OLFATTO', 'TOSSE', 'DOLORI_MUSCOLARI', 'STANCHEZZA', 'CONGIUNTIVITE', 'DIARREA', 'RAFFREDDORE', 
          'PATOLOGIA_SOVRAPPESO', 'PATOLOGIA_DIABETE', 'PATOLOGIA_IPERTENSIONE', 'PATOLOGIA_NESSUNA', 
          'SPOSTAMENTI_LAVORO', 'PROFILO', 'TAMPONE', 'TAMPONE_ESITO']:
    dtype_UC[c] = 'category'

df_clean = pd.read_csv(INPUTFN, dtype=dtype_UC)

# remove last day to drop incomplete data
df_clean = df_clean.loc[df_clean['DT_CREATION_DATA'] < df_clean['DT_CREATION_DATA'].max()]


assert df_clean['ID_QUESTIONARIO'].nunique() == len(df_clean) 
df_clean['DEVICE_UUID'] = df_clean['ID_SURVEY_DETAIL_ANAG']

# date column
df_clean['DT_CREATION_DATA'] = pd.to_datetime(df_clean['DT_CREATION_DATA'], format="%Y-%m-%d")
last_day = df_clean['DT_CREATION_DATA'].max().date()


assert len(df_clean) == len(df_clean['ID_QUESTIONARIO'].unique())
Nq = len(df_clean)
Nu = len(df_clean['DEVICE_UUID'].unique())

df_last = df_clean.sort_values(by='DT_CREATION_DATA', inplace=False).drop_duplicates(subset=['DEVICE_UUID'], inplace=False, keep='last')
assert len(df_last) == Nu

# add column with number of questionnaries
count_questionari = df_clean.groupby(by='DEVICE_UUID').DT_CREATION_DATA.count()
df_last = df_last.set_index('DEVICE_UUID')
df_last['COUNT_QUESTIONARI'] = count_questionari
df_last = df_last.reset_index()


# score computation
score_df = score_predittivo(df_clean)

score_df = pd.merge(score_df, df_clean[['DEVICE_UUID', 'DT_CREATION_DATA', 
                                    'DOMICILIO_PROVINCIA', 'DOMICILIO_COMUNE', 
                                    'DOMICILIO_COMUNE_ISTAT', 'DOMICILIO_POPOLAZIONE_TOT']], 
                                    left_index=True, right_index=True)
score_df = score_df.sort_values(by='SCORE', ascending=False).drop_duplicates(subset='DEVICE_UUID', keep='first')
# for each user keep only the maximum score 
score = score_df['SCORE']
assert len(score) == Nu


# write score csv
score_csv = score_df[['SCORE', 'DEVICE_UUID']].rename(columns={'DEVICE_UUID': 'ID_SURVEY_DETAIL_ANAG'}).reset_index(drop=True)
score_csv.to_csv('CERCACOVID_questionari_score.csv', index=False)


# ### Numerical values


# compute values
num_values = pd.DataFrame(columns = ['value'])
num_values.loc['users that downloaded the app', 'value'] = df_clean['ID_SURVEY_DETAIL_ANAG'].nunique()
num_values.loc['completed questionnaires', 'value'] = len(df_clean)
num_values.loc['municipalities', 'value'] = df_clean['DOMICILIO_COMUNE'].nunique()
num_values.loc['% of female users', 'value'] = df_last.loc[df_last['SESSO'] == 'Femmina'].shape[0] / df_last.shape[0] * 100
num_values.loc['% of male users', 'value'] = df_last.loc[df_last['SESSO'] == 'Maschio'].shape[0] / df_last.shape[0] * 100
num_values.loc['users median age', 'value'] = df_last['ETA'].median()
num_values.loc['users between 20 and 60', 'value'] = df_last.loc[np.logical_and(df_last['ETA'] >= 20, df_last['ETA'] <= 60)].shape[0]
num_values.loc['putative COVID - users', 'value'] = score_df.loc[score_df['SCORE'] <= 3].shape[0]
num_values.loc['putative COVID + users', 'value'] = score_df.loc[score_df['SCORE'] >= 8].shape[0]

curr = df_clean.pivot_table(index='DT_CREATION_DATA', values='ID_QUESTIONARIO', aggfunc='count').sort_index().reset_index()
curr1 = curr.loc[curr['DT_CREATION_DATA'] < pd.to_datetime('2020-04-16', format='%Y-%m-%d')]['ID_QUESTIONARIO'].median()
num_values.loc['median users per day before April 16th', 'value'] = curr1

curr2 = curr.loc[curr['DT_CREATION_DATA'] >= pd.to_datetime('2020-04-16', format='%Y-%m-%d')]['ID_QUESTIONARIO'].median()
num_values.loc['median users per day after April 16th', 'value'] = curr2

num_values.to_excel('Numerical_values.xlsx')


# ### Extended Data Figure 3. Distribution of users in Lombardy.

# map
popolazione = df_last[['DEVICE_UUID', 'DOMICILIO_COMUNE', 'DOMICILIO_COMUNE_ISTAT', 'DOMICILIO_POPOLAZIONE_TOT']].copy()
popolazione = popolazione.groupby(['DOMICILIO_COMUNE', 'DOMICILIO_COMUNE_ISTAT', 'DOMICILIO_POPOLAZIONE_TOT']).count()
popolazione = popolazione.reset_index(level = 2)
popolazione['%'] = 100. * popolazione['DEVICE_UUID'] / popolazione['DOMICILIO_POPOLAZIONE_TOT']

qs = {
    '<4%': {'min': -1, 'max': 4., 'val': 4.},
    '4-7%': {'min': 4., 'max': 7., 'val': 3},
    '7-10%': {'min': 7., 'max': 10., 'val': 2.},
    '10-13%': {'min': 10., 'max': 13., 'val': 1.5},
    '>13%': {'min': 13., 'max': 1000, 'val': 1.},
}
popolazione = discretizza_sintomi(popolazione, qs)


# create data ----------
data_merge_key= 'DOMICILIO_COMUNE_ISTAT'
shp_merge_key='PRO_COM'

filtered_data = popolazione.reset_index()
filtered_data['DOMICILIO_COMUNE_ISTAT'] = filtered_data['DOMICILIO_COMUNE_ISTAT'].astype(str).str[2:]

filtered_shapes = shape_df[shape_df[shp_merge_key].isin(filtered_data[data_merge_key])]
total_df=filtered_shapes.merge(filtered_data, left_on=shp_merge_key, right_on=data_merge_key)

plot_key = '% discreta' 
fig = pu.map_plot_v2(total_df, 
                  plot_key, 
                  "COMUNE",
                  cmap = plt.cm.summer,
                  lower_bound = 1.0,
                  upper_bound = 4.0,
                  plot_showLegend=True,
                  plot_keyLegendNames = 'interval',
                  plot_dictBoundaries = plot_dictBoundaries_com,
                  plot_title = 'Percentage of CercaCovid users'
                )
plot_prov = create_prov_data(shp_prov, list(PROVINCIE.index), fill_value = np.nan)
fig = pu.append_prov_borders(fig, plot_prov, plot_dictBoundaries = plot_dictBoundaries_prov)

go.FigureWidget(fig).write_image("Extended Data Figure 3.png", width=2500, height=2500)


# ### Extended Data Figure 4. Frequencies of symptoms among CercaCovid users.

# table
sintomi_all_gravi = sintomi_gravi_matrix(df_last)

pvt_gravi = pd.DataFrame(np.sum(sintomi_all_gravi.drop(columns='NUMERO_SINTOMI').T, axis=1))
pvt_gravi = pvt_gravi.sort_values(by=0, ascending=False)
SINTOMI_DESC = {'GUSTO_OLFATTO': 'Disgeusia/Ageusia', 
                'DOLORI_MUSCOLARI': 'Muscle pain', 
                'DIARREA': 'Diarrhoea', 
                'CONGIUNTIVITE': 'Conjunctivitis',
                'STANCHEZZA': 'Fatigue',
                'TOSSE': 'Cough',
                'RAFFREDDORE': 'Nasal obstruction',
                'FEBBRE': 'Fever'}
pvt_gravi.columns = ['Num. questionnaries']
pvt_gravi = pvt_gravi.reset_index()
pvt_gravi['index'] = [SINTOMI_DESC[x] for x in pvt_gravi['index']]
pvt_gravi = pvt_gravi.set_index('index')
pvt_gravi.index.name = None
table = pprint_pivot(pvt_gravi, tot=Nu, perc_col_name='% questionnaries')
table.to_excel('Extended Data Figure 4_table.xlsx')


# ### Extended Data Figure 5 . Distribution of COVID score among users.

# plot
plt.figure(figsize=(70,40))
ax = sns.distplot(score, kde=False, color='#2F7126')
plt.ylabel('Number of questionnaries')
plt.xlabel('Covid-score value')

figure = plt.gcf()
figure.set_size_inches(14, 8)
figure.savefig("Extended Data Figure 5.png", dpi=1500)


# ### Extended Data Figure 6 Distribution of COVID score values in Lombardy.

# get latest data
assert len(df_last) == len(score_df)

# add municipalities to score df
grpcols = ['DOMICILIO_COMUNE', 'DOMICILIO_COMUNE_ISTAT']
score_map = score_df[grpcols + ['DOMICILIO_POPOLAZIONE_TOT']].groupby(by=grpcols).max()
score_map['UTENTI'] = score_df[grpcols + ['DEVICE_UUID']].groupby(by=grpcols).count()
score_map['POSITIVI >= 8'] = score_df[score_df['SCORE'] >= 8][grpcols + ['DEVICE_UUID']].groupby(by=grpcols).count()
score_map['% POSITIVI >= 8'] = 100. * score_map['POSITIVI >= 8'] / score_map['UTENTI']
score_map['SCORE MEDIO'] = score_df[grpcols + ['SCORE']].groupby(by=grpcols).mean()

# mean score map
qs = {
    '<= 2': {'min': 0, 'max': 2., 'val': 1},
    '2-3': {'min': 2., 'max': 3., 'val': 2},
    '3-4': {'min': 3., 'max': 4., 'val': 3},
    '>= 4': {'min': 4., 'max': 1000, 'val': 4},
}
title = 'Average COVID-score'
fig = plot_map(score_map, 'SCORE MEDIO', title, qs)
#go.FigureWidget(fig).show(renderer='png', width=800, height=800)
go.FigureWidget(fig).write_image("Extended Data Figure 6a.png", width=2500, height=2500)

# putative + map
qs = {
    '<= 10%': {'min': 0., 'max': 10., 'val': 1},
    '10%-15%': {'min': 10., 'max': 15., 'val': 2},
    '15%-20%': {'min': 15., 'max': 20., 'val': 3},
    '20%-25%': {'min': 20., 'max': 25., 'val': 4},
    '>= 25%': {'min': 25., 'max': 1000, 'val': 5},
}
title = '% Putative CODIV+ users'
fig = plot_map(score_map, '% POSITIVI >= 8', title, qs)
#go.FigureWidget(fig).show(renderer='png', width=800, height=800)
go.FigureWidget(fig).write_image("Extended Data Figure 6b.png", width=2500, height=2500)
