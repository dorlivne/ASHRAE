from data_manipulation import *
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from tqdm import tqdm


def build_result_line(title, dataset, model_name, rmse):
    return {'title': title, 'dataset': dataset, 'model_name': model_name, 'rmse': rmse}


if __name__ == '__main__':
    if not os.path.isfile(cfg.results_df):
        results_df = pd.DataFrame()
        # results_df.set_index('title')
        results_df.to_pickle(path=cfg.results_df)
    else:
        results_df = pd.read_pickle(path=cfg.results_df)
    # test = pd.read_pickle(path=cfg.ready_dir +"/test.pkl")
    print("------------- Reading Train from path -------------")
    train_X_total = pd.read_pickle(path=cfg.ready_dir + 'train_X.pkl')
    # train_X_total.drop(['row_id'], axis='columns', inplace=True)
    train_Y_total = pd.read_pickle(path=cfg.ready_dir + 'train_y.pkl')
    print("------------- Predicting with K-Fold -------------")
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'rmse'},
        'subsample_freq': 1,
        'learning_rate': 0.3,
        'bagging_freq': 5,
        'num_leaves': 330,
        'feature_fraction': 0.9,
        'lambda_l1': 1,
        'lambda_l2': 1
    }

    folds = 5
    seed = 666
    shuffle = False
    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
    models = []
    categoricals = ["site_id", "building_id", "primary_use", "meter", "wind_direction"]
    numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
                  "dew_temperature", 'precip_depth_1_hr']
    feat_cols = categoricals + numericals
    for train_index, val_index in kf.split(train_X_total, train_X_total['building_id']):
        train_X = train_X_total.iloc[train_index]
        val_X = train_X_total.iloc[val_index]
        train_y = train_Y_total.iloc[train_index]
        val_y = train_Y_total.iloc[val_index]
        lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
        lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=500,
                        valid_sets=(lgb_train, lgb_eval),
                        early_stopping_rounds=50,
                        verbose_eval=50)
        models.append(gbm)

    # %%see which variables are the most relevant
    feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importance(), gbm.feature_name()), reverse=True),
                               columns=['Value', 'Feature'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM FEATURES')
    plt.tight_layout()
    plt.show()
    print("")
    test_X = pd.read_pickle(path=cfg.ready_dir + 'test_X.pkl')
    test_total = test_X.drop('row_id', axis='columns')
    i = 0
    res = []
    step_size = 50000
    for j in tqdm(range(int(np.ceil(test_total.shape[0] / 50000)))):
        res.append(np.expm1(sum([model.predict(test_total.iloc[i:i + step_size]) for model in models]) / folds))
        i += step_size

        # %% remove the columns that cannot be calculated in the test data and rerun the last step. DataFrame.dtypes for data must be int, float or bool.

    res = np.concatenate(res)

    # %% submission. all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 100000 and the array at index 416 has size 97600
    sample_submission = pd.read_csv(cfg.data_dir + "/sample_submission.csv")
    sample_submission['meter_reading'] = res
    sample_submission.loc[sample_submission['meter_reading'] < 0, 'meter_reading'] = 0
    sample_submission.to_csv('submission.csv', index=False)