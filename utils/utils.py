import pandas as pd 
from typing import Callable , List, Tuple, TypeAlias 


def get_serie_type(serie:pd.Series) -> str: 
  is_num = lambda serie: pd.api.types.is_numeric_dtype(serie.dropna()) 
  is_str = lambda serie: pd.api.types.is_string_dtype(serie.dropna()) 
  is_date = lambda serie: pd.api.types.is_datetime64_any_dtype(serie.dropna()) 
  if is_date(serie): 
    return 'date' 
  if is_num(serie): 
    return 'numerical' 
  if is_str(serie): 
    return 'string' 
  return str(serie.dtype) 

def get_df_types(df:pd.DataFrame) -> pd.Series: 
  return pd.Series({ c:get_serie_type(df[c]) for c in df.columns}, name='types' ) 


def is_castable(seriea:pd.Series, serieb:pd.Series): 
  if get_serie_type(seriea) == get_serie_type(serieb): 
    return True 
  try:
    serieb.astype(seriea.dtype) 
    return True 
  except (ValueError, TypeError): 
    return False 


# ! Distributional summary ---------------- 
def distributional_summary(df:pd.DataFrame) -> pd.DataFrame: 
  '''
  Takes a dataframe, returns a distributional summary (similar to describe, but more) 
  
  - type, (numerical, string, date, other) 
  - dtype, (specific dtype) 
  - count, (non-missing values) 
  - missing%, (percentage of missing values) 
  - unique%, (percentage of unique values) 
  - most_freq, (most frequent value) 
  - least_freq, (least frequent value) 
  - min, max, quartiles (for numerical values) 
  '''
  num_cols = list(df.select_dtypes('number').columns) 
  df_num_describe = df[num_cols].describe().drop('count') 
  df_num_describe.loc['sknewness'] = {c:df[c].skew() for c in df[num_cols].columns} 
  df_num_describe.loc['kurtosis'] = {c:df[c].kurt() for c in df[num_cols].columns} 

  N = df.shape[0]
  count = df.count()
  df_describe = pd.DataFrame( {'type': get_df_types(df)} ).T 
  df_describe.loc['dtype'] = df.dtypes
  df_describe.loc['N'] = N
  df_describe.loc['count'] = count
  df_describe.loc['missing_p'] = (N-count)/N
  df_describe.loc['cardinality'] = df.nunique()
  df_describe.loc['unique_p'] = { c:(df[c].value_counts() == 1).sum() for c in df.columns }
  df_describe.loc['unique_p'] = df_describe.loc['unique_p']/df_describe.loc['count'] 
  df_describe.loc['most_freq'] = { c:df[c].value_counts().idxmax() for c in df.columns } 
  df_describe.loc['least_freq'] = { c:df[c].value_counts().idxmin() for c in df.columns } 
  

  return pd.concat([df_describe, df_num_describe])



# ! Validity --------------- 
#SerieValidityFunc: TypeAlias = Callable[[pd.DataFrame, str], List[bool]] 
SerieValidityMapper: TypeAlias = dict[str, Callable[[pd.DataFrame, str], List[bool]]] 

#def degrees_validity(extracted:pd.DataFrame, validity_map:dict[str, SerieValidityFunc]): 
def degree_validity(extracted:pd.DataFrame, mapper:SerieValidityMapper): 
  N = extracted.shape[0] 
  df_validity = pd.DataFrame( { c:func(extracted, c) for c, func in mapper.items() }).sum() 
  df_validity = df_validity/N 
  return df_validity 


# ! Completeness ----------- 
def degree_completeness(df:pd.DataFrame, axis=0)-> pd.DataFrame: 
  N = df.shape[axis] 
  df_completeness = pd.DataFrame( ((N - pd.isnull(df).astype(int).sum(axis=axis)) / N), columns=['completeness'] ) 
  return df_completeness 


