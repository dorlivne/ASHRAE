from data_visualization import *
from sklearn.preprocessing import LabelEncoder
import gc
import math


def make_is_bad_zero(Xy_subset, min_interval=48, summer_start=3000, summer_end=7500):
    """Helper routine for 'find_bad_zeros'.

    This operates upon a single dataframe produced by 'groupby'. We expect an
    additional column 'meter_id' which is a duplicate of 'meter' because groupby
    eliminates the original one."""
    meter = Xy_subset.meter_id.iloc[0]
    is_zero = Xy_subset.meter_reading == 0
    if meter == 0:  # electricity
        # Electrical meters should never be zero. Keep all zero-readings in this table so that
        # they will all be dropped in the train set.
        return is_zero
    # the meter reading changes from zero to not zero from the i obs to the i+1 obs
    transitions = (is_zero != is_zero.shift(1))
    all_sequence_ids = transitions.cumsum()  # cumulative sum, we now know whats the size of each sequence
    ids = all_sequence_ids[is_zero].rename("ids")  # we care only about a sequence which is zero in the meter readings
    if meter in [2, 3]:
        # It's normal for steam and hotwater to be turned off during the summer
        not_summer = set(ids[(Xy_subset.timestamp_h < summer_start) |
                       (Xy_subset.timestamp_h > summer_end)].unique())
        is_bad = ids.isin(not_summer) & (ids.map(ids.value_counts()) >= min_interval)
        # each id with sequence bigger than 48 is bad news, thus we delete it
    elif meter == 1:  # chilledwater
        time_ids = ids.to_frame().join(Xy_subset.timestamp_h).set_index("timestamp_h").ids
        is_bad = ids.map(ids.value_counts()) >= min_interval

        # Cold water may be turned off during the winter
        jan_id = time_ids.get(0, False)
        dec_id = time_ids.get(8283, False)
        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and
                dec_id == time_ids.get(8783, False)):
            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))
    else:
        raise Exception(f"Unexpected meter type: {meter}")

    result = is_zero.copy()
    result.update(is_bad)
    return result

def find_bad_sitezero(X):
    """Returns indices of bad rows from the early days of Site 0 (UCF)."""
    return X[(X['timestamp'] < "2016-05-21 00:00:00") & (X.site_id == 0) & (X.meter == 0)].index

def find_bad_building1099(X, y):
    """Returns indices of bad rows (with absurdly high readings) from building 1099."""
    return X[(X.building_id == 1099) & (X.meter == 2) & (y > 3e4)].index

def find_bad_zeros(X, y):
    """Returns an Index object containing only the rows which should be deleted."""
    Xy = X.assign(meter_reading=y, meter_id=X.meter)
    is_bad_zero = Xy.groupby(["building_id", "meter"]).apply(make_is_bad_zero)
    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])


def find_bad_rows(X:pd.DataFrame):
    y = X['meter_reading']
    return find_bad_zeros(X, y).union(find_bad_sitezero(X)).union(find_bad_building1099(X, y))


def print_missing_in_df(df: pd.DataFrame):
    for col in df.columns:
        if not col in ['row_id', 'meter_reading']:
            missing_data = len(df) - df[col].count()
            if (missing_data > 0 or missing_data == 'NaN'):
                precentage_missing = round(100 * (missing_data / len(df)), 3)
                print(col, ':', missing_data, 'missing values is', str(precentage_missing), '% of total')


def fill_missing_data(df:pd.DataFrame):
    print_missing_in_df(df)
    df['month'] = df['timestamp'].dt.month.astype(np.int8)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(np.int8)
    df['day_of_month'] = df['timestamp'].dt.day.astype(np.int8)
    df['hour'] = df['timestamp'].dt.hour
    if cfg.verbose:
        print("filling missing values for air_temperature feature according to mean by hour according to site")
    df['air_temperature'].fillna(df.groupby(['site_id', 'day_of_month', 'month'])['air_temperature'].transform('mean'), inplace=True)
    df['air_temperature'].fillna(df.groupby(['site_id', 'month'])['air_temperature'].transform('mean'), inplace=True)
    if cfg.verbose:
        # print("filling missing values for floor_count feature according to MAP estimation according to site")
        print("dropping floor_count feature more than 80% missing")
    df.drop('floor_count', inplace=True, axis='columns')
    if cfg.verbose:
        print("filling missing values for year_built feature according to mean by site")
    df['year_built'] = df['year_built'].fillna(1969)
    if cfg.verbose:
        print("filling missing values for cloud coverage feature according to mean by site and by hour and month")
    # df['cloud_coverage'] = df['cloud_coverage'].fillna(df.groupby(['site_id', 'day_of_month', 'month'])['cloud_coverage'].transform('mean'))
    df['cloud_coverage'] = df['cloud_coverage'].fillna(round(df.groupby(['site_id', 'day_of_month', 'month'])['cloud_coverage'].transform('mean'), 0))
    df['cloud_coverage'] = df['cloud_coverage'].fillna(round(df['cloud_coverage'].mean(), 0))
    if cfg.verbose:
        print("filling missing values for dew_temperature feature according to mean by site and month")
    df['dew_temperature'] = df['dew_temperature'].fillna(df.groupby(['site_id', 'day_of_month', 'month'])['dew_temperature'].transform('mean'))
    df['dew_temperature'] = df['dew_temperature'].fillna(df.groupby(['site_id', 'month'])['dew_temperature'].transform('mean'))
    # df['dew_temperature'] = df['dew_temperature'].fillna(df.groupby(['site_id'])['dew_temperature'].transform('mean'))
    if cfg.verbose:
        print("filling missing values for precip_depth_1_hr feature according to mean by site and month")
    df['precip_depth_1_hr'] = df['precip_depth_1_hr'].fillna(df.groupby(['site_id', 'day_of_month', 'month'])['precip_depth_1_hr'].transform('mean'))
    df['precip_depth_1_hr'] = df['precip_depth_1_hr'].fillna(round(df['precip_depth_1_hr'].mean(), 0))
    if cfg.verbose:
        print("filling missing values for sea_level_pressure feature according to mean by site and month")
    df['sea_level_pressure'] = df['sea_level_pressure'].fillna(
        df.groupby(['site_id', 'day_of_month', 'month'])['sea_level_pressure'].transform('mean'))
    df['sea_level_pressure'] = df['sea_level_pressure'].fillna(round(df['sea_level_pressure'].mean(), 2))
    if cfg.verbose:
        print("filling missing values for wind_direction feature according to mean by site and month")
    df['wind_direction'] = df['wind_direction'].fillna(
        df.groupby(['site_id', 'day_of_month', 'month'])['wind_direction'].transform('mean'))
    df['wind_direction'] = df['wind_direction'].fillna(
        df.groupby(['site_id', 'month'])['wind_direction'].transform('mean'))
    if cfg.verbose:
        print("filling missing values for wind_speed feature according to mean by site and month")
    df['wind_speed'] = df['wind_speed'].fillna(
        df.groupby(['site_id', 'day_of_month', 'month'])['wind_speed'].transform('mean'))
    df['wind_speed'] = df['wind_speed'].fillna(
        df.groupby(['site_id', 'month'])['wind_speed'].transform('mean'))
    df.drop('hour', inplace=True, axis='columns')
    df.drop('day_of_week', inplace=True, axis='columns')
    df.drop('day_of_month', inplace=True, axis='columns')
    df.drop('month', inplace=True, axis='columns')
    print_missing_in_df(df)
    # most_common_floor_number = df['floor_count'].mode(dropna=True)
    # df['floor_count'] = df['floor_count'].fillna(most_common_floor_number)
    return df

def convert_month_to_season(month):
    if (month <= 2) | (month == 12):
        return 0  # winter
    elif month <= 5:
        return 1  # spring
    elif month <= 8:
        return 2  # summer
    elif month <= 11:
        return 3  # fall

def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize. """
    # if categorical_colunms are not given than treat object as categorical features
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df

def feature_extraction(df:pd.DataFrame):
    print("combining primary_use categories with low number of samples")
    df['primary_use'].replace({"Healthcare": "Other", "Parking": "Other", "Warehouse/storage": "Other", "Manufacturing/industrial": "Other",
                                "Retail": "Other", "Services": "Other", "Technology/science": "Other", "Food sales and service": "Other",
                                "Utility": "Other", "Religious worship": "Other"}, inplace=True)
    print("skewing square feet variable with log(p)")
    df['square_feet'] = np.log1p(df['square_feet'])
    print("adding age feature")
    df['age'] = df['year_built'].max() - df['year_built'] + 1
    print("removing year_built bias")
    df['year_built'] = df['year_built'] - 1900
    print("adding time features")
    df['month'] = df['timestamp'].dt.month.astype(np.int8)
    # df['weekofyear_datetime'] = df['timestamp'].dt.weekofyear.astype(np.int8)
    # df['dayofyear_datetime'] = df['timestamp'].dt.dayofyear.astype(np.int16)
    df['season'] = df.month.apply(convert_month_to_season)
    df['hour'] = df['timestamp'].dt.hour.astype(np.int8)
    df['day_week'] = df['timestamp'].dt.dayofweek.astype(np.int8)
    df['day_month_datetime'] = df['timestamp'].dt.day.astype(np.int8)
    # df['week_month_datetime'] = df['timestamp'].dt.day / 7
    # df['week_month_datetime'] = df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(
    #     np.int8)
    # df.drop('year_built', inplace=True, axis='columns'
    return df


def remove_outliers(df: pd.DataFrame):
    df['timestamp_h'] = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    print("removing discrepancy in meter reading for site id 10 before May 2016")
    print("removing anomality values of building 1099")
    print("removing zero electricity meter reading ")
    print("removing sequences of +48 hours of zero reading of steam and hotwater except during the summer")
    print("removing sequences of +48 hours of zero reading of chilled water except during the winter")
    # criteria = (df['site_id'] == 0) & (df['timestamp'] < "2016-05-21 00:00:00")
    idx_to_drop = find_bad_rows(df)
    df.drop(idx_to_drop, axis='rows', inplace=True)
    df.drop(['timestamp_h'], inplace=True, axis='columns')
    return df




def drop_features(df:pd.DataFrame):
    drop_columns = ['timestamp', 'sea_level_pressure', 'wind_speed']
    print("features dropped: \t {}".format(drop_columns))
    df.drop(drop_columns, inplace=True, axis='columns')
    return df


if __name__ == '__main__':
    train_df, test_df = load_combined()
    print("train data frame shape : \t {}".format(train_df.shape))
    print("test data frame shape : \t {}".format(test_df.shape))
    print(" ----------------- Removing Outliers -----------------")
    train_df = remove_outliers(train_df)
    print(" ----------------- Filling Missing values for train -----------------")
    train_df = fill_missing_data(train_df)
    print(" ----------------- Filling Missing values for test -----------------")
    test_df = fill_missing_data(test_df)
    # test_df.insert(test_df.shape[1] - 1, 'meter_reading', np.nan)  # need to be predicted
    train_df['DataType'], test_df['DataType'] = 'train', 'test'
    total_df = pd.concat([train_df, test_df], ignore_index=True, axis=0, sort=False)
    # total_df['meter'] = pd.Categorical(total_df['meter']).rename_categories(METER_DIC)
    print(" ----------------- Features Extraction -----------------")
    total_df = feature_extraction(total_df)
    print(" ----------------- Encode Categorical columns -----------------")
    # total_df = pd.get_dummies(total_df, columns=None)
    le = LabelEncoder()
    total_df['primary_use'] = total_df['primary_use'].astype(str)
    total_df['primary_use'] = le.fit_transform(total_df['primary_use']).astype(np.int8)
    total_df['meter'] = le.fit_transform(total_df['meter']).astype(np.int8)
    print(" ----------------- Drop features -----------------")
    total_df = drop_features(total_df)
    print(" ----------------- Divide to Test and Train -----------------")
    train_df = total_df[total_df['DataType'] == 'train']
    test_df = total_df[total_df['DataType'] == 'test']
    train_df.drop(['DataType'], axis='columns', inplace=True)
    test_df.drop(['DataType'], axis='columns', inplace=True)
    train_y = np.log1p(train_df['meter_reading'])
    train_x = train_df.drop(['meter_reading', 'row_id'], axis='columns')
    print("skewing meter_reading variable with log(1+p)")
    test_df = test_df.drop(['meter_reading'], axis='columns')
    print("train data frame shape : \t {}".format(train_df.shape))
    print("test data frame shape : \t {}".format(test_df.shape))
    train_x.to_pickle(path=cfg.ready_dir + "train_X.pkl")
    train_y.to_pickle(path=cfg.ready_dir + "train_y.pkl")
    test_df.to_pickle(path=cfg.ready_dir + "test_X.pkl")
