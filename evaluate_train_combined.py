## Data management tools
import pandas as pd
import numpy as np
import geopandas as gpd
import pyreadstat 
import random 
import os
from collections import Counter
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix,average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from random import randrange
import shap
import matplotlib as mp
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime
from datetime import timedelta

# Functions
import functions_transforms as fun
import operator

# Randomly select household indexes
def resampled_index(
    df,
    cols,
    share_positives,
    share_negatives,
    threshold,
    random_state,
):
    """ Resample a dataframe with respect to cols

    Resampling is a technique for changing the positive/negative balance
    of a dataframe. Positives are rows where any of the specified cols
    are greater than the threshold. Useful for highly unbalanced
    datasets where positive outcomes are rare.

    """

    # Negatives are rows where all cols are close to zero
    mask_negatives = np.isclose(df[cols], threshold).max(axis=1)
    # Positives are all the others
    mask_positives = ~mask_negatives

    df_positives = df.loc[mask_positives]
    df_negatives = df.loc[mask_negatives]

    len_positives = len(df_positives)
    len_negatives = len(df_negatives)

    n_positives_wanted = int(share_positives * len_positives)
    n_negatives_wanted = int(share_negatives * len_negatives)

    replacement_pos = share_positives > 1
    replacement_neg = share_negatives > 1
    df = pd.concat(
        [
            df_positives.sample(n=n_positives_wanted, replace=replacement_pos, random_state=random_state),
            df_negatives.sample(n=n_negatives_wanted, replace=replacement_neg, random_state=random_state),
        ]
    )
    
    
    return df

def cross_validation_split(df, set_seed=np.random.randint(0, 1000), n_rep=3):
    random.seed(set_seed)
    indxs_split_list = []
    
    for _ in range(n_rep):
        indxs = df.reset_index().drop_duplicates(subset='hh_n').hh_n.unique()

        indxs_split = list()
        indxs_copy = list(indxs)
        indxs_split = random.sample(indxs_copy, k=round(len(indxs_copy) * 0.2))
        indxs_split_list.append(indxs_split)
        
    return indxs_split_list

def get_train_indx(df, indxs_split):
    indxs = df.reset_index().drop_duplicates(subset='hh_n').hh_n.unique()
    indxs_train = [x for x in indxs if x not in indxs_split]
    return indxs_train


def apply_imputation(
        imp_estimator,
        df,
        miss_vars,
        pred_vars,
        max_value,
        randomstate,

):
    # Initialise the imputer
    imputer_f = IterativeImputer(
        estimator=imp_estimator,
        min_value=0,
        max_value=max_value,
        skip_complete=True,
        max_iter=250,
        random_state=randomstate,
        verbose=1)

    #
    df_out = pd.DataFrame(
        imputer_f.fit_transform(df[miss_vars + pred_vars]),
        columns=miss_vars + pred_vars,
        index=df[miss_vars + pred_vars].index
    )
    return df_out[miss_vars]

def split_train_test_eval(
    test_indx,
    df,
    cols_X,
    col_Y,
    modelname,
    estimator,
    esttype,
    fold,
    model_to_impute,
    sampling,
    osampling_strategy,
    calibration=False,
    imp_estimator=linear_model.BayesianRidge(), # Default
    imp_iter=20,
    imp_random_state=1308,
    method_calib=None,
    shap_val=False,
    shap_val_inter=False,
    imputation=True,
    imputation_old=False,
    perm_importance = True, 
    share_neg=None,
    imp_numerical_missing =None,
    imp_numerical_predictors =None,
    imp_ordinal_missing =None,
    imp_ordinal_predictors =None,
    imp_categorical_missing =None,
    imp_categorical_predictors =None,

):
    
    # Creat training index.
    print('Splitting data')
    train_indx = get_train_indx(df, test_indx)

    train_X, test_X = df.loc[train_indx][cols_X], df.loc[test_indx][cols_X]
    train_y, test_y = df.loc[train_indx][col_Y], df.loc[test_indx][col_Y]

    # Impute
    if imputation_old:
        print('Impute training set')
        # Linear imputation
        imp_train = IterativeImputer(estimator=imp_estimator,min_value=0,skip_complete=True,max_iter=imp_iter, random_state=imp_random_state, verbose=1)
        imputed_np_train = imp_train.fit(train_X[model_to_impute['cols_features']].to_numpy())
        f_X_train_dropped = train_X.drop(model_to_impute['cols_features'], axis=1, inplace = False)
        imputed_cols_train = pd.DataFrame(imputed_np_train.transform(train_X[model_to_impute['cols_features']].to_numpy()),
                                          columns=train_X[model_to_impute['cols_features']].columns,
                                          index=train_X[model_to_impute['cols_features']].index
                                         )
        fold_X_train = pd.concat([f_X_train_dropped,imputed_cols_train],axis=1)
        
        print('Impute test set')
        imp_test = IterativeImputer(estimator=imp_estimator,min_value=0,skip_complete=True,max_iter=imp_iter, random_state=imp_random_state, verbose=1)
        imputed_np_test = imp_test.fit(test_X[model_to_impute['cols_features']].to_numpy())
        f_X_test_dropped = test_X.drop(model_to_impute['cols_features'], axis=1, inplace = False)
        imputed_cols_test = pd.DataFrame(imputed_np_test.transform(test_X[model_to_impute['cols_features']].to_numpy()),
                                          columns=test_X[model_to_impute['cols_features']].columns,
                                          index=test_X[model_to_impute['cols_features']].index
                                         )
        
        fold_X_test = pd.concat([f_X_test_dropped,imputed_cols_test],axis=1)

    if 'numerical' in imputation:
        # Train set
        print('imputing numerical vars for training')
        fold_X_train = train_X.copy()

        imp_num_df_train = apply_imputation(
            imp_estimator=linear_model.BayesianRidge(),
            df=train_X,
            miss_vars=imp_numerical_missing,
            pred_vars=imp_numerical_predictors,
            randomstate=1308,
            max_value=None,
        )

        fold_X_train[imp_numerical_missing] = imp_num_df_train

        # Test set
        print('imputing numerical vars for testing')
        fold_X_test = test_X.copy()

        imp_num_df_test = apply_imputation(
            imp_estimator=linear_model.BayesianRidge(),
            df=test_X,
            miss_vars=imp_numerical_missing,
            pred_vars=imp_numerical_predictors,
            randomstate=1308,
            max_value=None,
        )

        fold_X_test[imp_numerical_missing] = imp_num_df_test

    if 'categorical' in imputation:
        # Train set
        print('imputing categorical vars for training')

        imp_cat_df_train = apply_imputation(
            imp_estimator=linear_model.LogisticRegression(max_iter=1000),
            df=fold_X_train,
            miss_vars=imp_categorical_missing,
            pred_vars=imp_categorical_predictors,
            randomstate=1308,
            #max_iter=250,
            max_value=1,
        )

        fold_X_train[imp_categorical_missing] = imp_cat_df_train

        # Test set
        print('imputing categorical vars for testing')

        imp_cat_df_test = apply_imputation(
            imp_estimator=linear_model.LogisticRegression(max_iter=1000),
            df=fold_X_test,
            miss_vars=imp_categorical_missing,
            pred_vars=imp_categorical_predictors,
            randomstate=1308,
            #max_iter=250,
            max_value=1,
        )

        fold_X_test[imp_categorical_missing] = imp_cat_df_test

    if 'ordinal' in imputation:
        # Train set
        print('imputing ordinal vars for training')

        imp_ord_df_train = apply_imputation(
            imp_estimator=linear_model.BayesianRidge(),
            df=fold_X_train,
            miss_vars=imp_ordinal_missing,
            pred_vars=imp_ordinal_predictors,
            randomstate=1308,
            max_value=10,
        )

        fold_X_train[imp_ordinal_missing] = imp_ord_df_train

        # Test set
        print('imputing ordinal vars for testing')

        imp_ord_df_test = apply_imputation(
            imp_estimator=linear_model.BayesianRidge(),
            df=fold_X_test,
            miss_vars=imp_ordinal_missing,
            pred_vars=imp_ordinal_predictors,
            randomstate=1308,
            max_value=10,
        )

        fold_X_test[imp_ordinal_missing] = imp_ord_df_test
        
    if sampling == 'over':
        print('Implement oversampling')
        oversample = RandomOverSampler(sampling_strategy=osampling_strategy,random_state=imp_random_state)

        X_train_over, y_train_over = oversample.fit_resample(
            fold_X_train[cols_X],
            train_y
        )
    
    if sampling == 'under':
        sampled = resampled_index(
        df=pd.concat([fold_X_train,train_y],axis=1),
        cols=[col_Y],
        share_positives=1,
        share_negatives=share_neg,
        threshold=0,
        random_state=1308
        ).sort_index(axis=0,level=[0,1],inplace=False) # which level
        
        X_train_over,y_train_over = sampled[cols_X], sampled[col_Y]
        
    if sampling == 'none':
        
        X_train_over,y_train_over = fold_X_train.copy(), train_y.copy()
        

    # Fit model.
    print('fitting models')
    fitted_model = estimator.fit(X_train_over,y_train_over)

    # Predict. 
    print('predicting')
    predicted_y = fitted_model.predict_proba(fold_X_test)[:,1]

    # Evaluations.
    print('Compute auc and ap')
    auc = roc_auc_score(
        y_true = test_y, 
        y_score = predicted_y
    )
    ap = average_precision_score(
        y_true = test_y, 
        y_score = predicted_y
    )

    fpr_roc, tpr_roc, thresh_roc = roc_curve(test_y, predicted_y)
    precision, recall, thresh_pr = precision_recall_curve(test_y, predicted_y)

    if shap_val:
        print('Computing shapley values')
        print('Outcome in probabilities')

        explainer = shap.TreeExplainer(model=fitted_model,data=X_train_over,feature_perturbation="interventional",model_output='probability')
        shap_values = explainer(X=fold_X_test) #
        x_train_out = X_train_over

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model, shap_values, x_train_out

    if shap_val_inter:
        print('Computing shap interaction values')
        explainer = shap.TreeExplainer(model=fitted_model,
                                       feature_perturbation="tree_path_dependent")  # data=X_train_over
        shap_values = explainer(X=fold_X_test)
        shap_interactionvals = explainer.shap_interaction_values(X=fold_X_test)
        x_train_out = X_train_over

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model, shap_values, shap_interactionvals, x_train_out

    if perm_importance:
        print('Computing permutation based fi')

        perm_feats = permutation_importance(fitted_model, fold_X_test, test_y)

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model, perm_feats

    else:
        print('Return preds')

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model

    # Fit model.
    print('fitting models')
    fitted_model = estimator.fit(X_train_over,y_train_over)

    # Predict. 
    print('predicting')
    predicted_y = fitted_model.predict_proba(fold_X_test)[:,1]

    # Evaluations.
    print('Compute auc and ap')
    auc = roc_auc_score(
        y_true = test_y, 
        y_score = predicted_y
    )
    ap = average_precision_score(
        y_true = test_y, 
        y_score = predicted_y
    )

    fpr_roc, tpr_roc, thresh_roc = roc_curve(test_y, predicted_y)
    precision, recall, thresh_pr = precision_recall_curve(test_y, predicted_y)

    if shap_val:
        print('Computing shapley values')
        print('Outcome in probabilities')

        explainer = shap.TreeExplainer(model=fitted_model,data=X_train_over,feature_perturbation="interventional",model_output='probability')
        shap_values = explainer(X=fold_X_test) #
        x_train_out = X_train_over

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model, shap_values, x_train_out

    if shap_val_inter:
        print('Computing shap interaction values')
        explainer = shap.TreeExplainer(model=fitted_model,
                                       feature_perturbation="tree_path_dependent")  # data=X_train_over
        shap_values = explainer(X=fold_X_test)
        shap_interactionvals = explainer.shap_interaction_values(X=fold_X_test)
        x_train_out = X_train_over

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model, shap_values, shap_interactionvals, x_train_out

    if perm_importance:
        print('Computing permutation based fi')

        perm_feats = permutation_importance(fitted_model, fold_X_test, test_y)

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model, perm_feats

    else:
        print('Return preds')

        return predicted_y, test_y, auc, ap, fpr_roc, tpr_roc, thresh_roc, precision, recall, thresh_pr, fitted_model

def tuning_rf(df, 
          model_early_stopping, 
          cv_folds, 
          estimator, 
          early_stopping_rounds, 
          evalmetricstr, 
          new_imp, 
          model_to_impute,
          #imp_estimator, 
          imp_max_iter, 
          oversampling, 
          evalmetric,
          random_state,
          o_sampling_strategy,
          output_paths,
          #undersampling, 
          verbose=1):
    """Roll our own CV 
    train each kfold with early stopping
    return average metric, sd over kfolds, average best round"""
    
    metrics_ap, metrics_auroc = [],[]
    metrics_ap_train, metrics_auroc_train = [],[]


    for i,cv_index in enumerate(cv_folds):
        print(f'Fold {i}: Splitting df')
        # Training set. 
        train_fold = get_train_indx(df[model_early_stopping['col_outcome']], cv_index)
        f_X_train = df.loc[train_fold][model_early_stopping['cols_features']]
        f_y_train = df.loc[train_fold][model_early_stopping['col_outcome']]
        # Test set.
        f_X_test=df.loc[cv_index][model_early_stopping['cols_features']]
        fold_y_test=df.loc[cv_index][model_early_stopping['col_outcome']]
        
        # Impute.
        if new_imp: 
            print('Impute training set')
            imp_train = IterativeImputer(min_value=0, skip_complete=True, max_iter=imp_max_iter, random_state=random_state, verbose=2)
            imputed_np_train = imp_train.fit(f_X_train[model_to_impute['cols_features']].to_numpy())
            f_X_train_dropped = f_X_train.drop(model_to_impute['cols_features'], axis=1)
            imputed_cols_df_train = pd.DataFrame(imputed_np_train.transform(
    f_X_train[model_to_impute['cols_features']].to_numpy()),
                columns=f_X_train[model_to_impute['cols_features']].columns,
                index=f_X_train[model_to_impute['cols_features']].index)
            fold_X_train = pd.concat([f_X_train_dropped,imputed_cols_df_train],axis=1)
            
            print('Impute test set')
            imp_test = IterativeImputer(min_value=0, skip_complete = True, max_iter=imp_max_iter, random_state=random_state, verbose=2)
            imputed_np_test = imp_test.fit(f_X_test[model_to_impute['cols_features']].to_numpy())
            f_X_test_dropped = f_X_test.drop(model_to_impute['cols_features'], axis=1)
            imputed_cols_df_test = pd.DataFrame(imputed_np_test.transform(
                f_X_test[model_to_impute['cols_features']].to_numpy()),
                columns=f_X_test[model_to_impute['cols_features']].columns,
                index=f_X_test[model_to_impute['cols_features']].index)
            fold_X_test = pd.concat([f_X_test_dropped,imputed_cols_df_test],axis=1)
            
        # Oversampling/Undersampling.
        if oversampling:
            print('Implement oversampling')
            oversample = RandomOverSampler(sampling_strategy=o_sampling_strategy,random_state=random_state)
            
            X_train_over, y_train_over = oversample.fit_resample(
                fold_X_train.reset_index()[model_early_stopping['cols_features']+['hh_n','yrmo']], 
                f_y_train.reset_index()[model_early_stopping['col_outcome']]#+['hh_n','yrmo']]
            )
            
            X_train_over = X_train_over.set_index(['hh_n','yrmo'])
            #y_train_over = y_train_over.set_index(['hh_n','yrmo'])
        else:
            X_train_over = fold_X_train.copy()
            y_train_over = f_y_train.copy()

        fitted_model = estimator.fit(X_train_over, y_train_over,
                          #eval_set=[(X_train_over, y_train_over),(fold_X_test, fold_y_test)],
                          #eval_metric=evalmetricstr,
                          #verbose=verbose,
                     )
        
        y_pred_train = fitted_model.predict_proba(fold_X_train)[:,1]
        
        y_pred_test = fitted_model.predict_proba(fold_X_test)[:,1]
        
        
        # Performance
        
        mname = model_early_stopping['model_name']
        
        metrics_ap.append(average_precision_score(fold_y_test, y_pred_test)) #roc_auc_score
        metrics_auroc.append(roc_auc_score(fold_y_test, y_pred_test))
        metrics_ap_train.append(average_precision_score(y_train_over, y_pred_train)) #roc_auc_score
        metrics_auroc_train.append(roc_auc_score(y_train_over, y_pred_train))
        
    return np.mean(metrics_ap), np.mean(metrics_auroc), np.mean(metrics_ap_train), np.mean(metrics_auroc_train) 
        
def my_cv(df, 
          model_early_stopping, 
          cv_folds, 
          estimator, 
          parm_1,
          parm_2,
          early_stopping_rounds, 
          evalmetricstr, 
          new_imp, 
          model_to_impute,
          #imp_estimator, 
          imp_max_iter, 
          oversampling, 
          evalmetric,
          random_state,
          o_sampling_strategy,
          output_paths,
          #undersampling, 
          verbose=1):
    """Roll our own CV 
    train each kfold with early stopping
    return average metric, sd over kfolds, average best round"""
    metrics = []
    best_iterations = []
    best_ntree = []
    val_diff = []

    for i,cv_index in enumerate(cv_folds):
        print(f'Fold {i}: Splitting df')
        # Training set. 
        train_fold = get_train_indx(df[model_early_stopping['col_outcome']], cv_index)
        f_X_train = df.loc[train_fold][model_early_stopping['cols_features']]
        f_y_train = df.loc[train_fold][model_early_stopping['col_outcome']]
        # Test set.
        f_X_test=df.loc[cv_index][model_early_stopping['cols_features']]
        fold_y_test=df.loc[cv_index][model_early_stopping['col_outcome']]
        
        # Impute.
        if new_imp: 
            print('Impute training set')
            imp_train = IterativeImputer(min_value=0, skip_complete=True, max_iter=imp_max_iter, random_state=random_state, verbose=2)
            imputed_np_train = imp_train.fit(f_X_train[model_to_impute['cols_features']].to_numpy())
            f_X_train_dropped = f_X_train.drop(model_to_impute['cols_features'], axis=1)
            imputed_cols_df_train = pd.DataFrame(imputed_np_train.transform(
    f_X_train[model_to_impute['cols_features']].to_numpy()),
                columns=f_X_train[model_to_impute['cols_features']].columns,
                index=f_X_train[model_to_impute['cols_features']].index)
            fold_X_train = pd.concat([f_X_train_dropped,imputed_cols_df_train],axis=1)
            
            print('Impute test set')
            imp_test = IterativeImputer(min_value=0, skip_complete = True, max_iter=imp_max_iter, random_state=random_state, verbose=2)
            imputed_np_test = imp_test.fit(f_X_test[model_to_impute['cols_features']].to_numpy())
            f_X_test_dropped = f_X_test.drop(model_to_impute['cols_features'], axis=1)
            imputed_cols_df_test = pd.DataFrame(imputed_np_test.transform(
                f_X_test[model_to_impute['cols_features']].to_numpy()),
                columns=f_X_test[model_to_impute['cols_features']].columns,
                index=f_X_test[model_to_impute['cols_features']].index)
            fold_X_test = pd.concat([f_X_test_dropped,imputed_cols_df_test],axis=1)
            
        # Oversampling/Undersampling.
        if oversampling:
            print('Implement oversampling')
            oversample = RandomOverSampler(sampling_strategy=o_sampling_strategy,random_state=random_state)
            
            X_train_over, y_train_over = oversample.fit_resample(
                fold_X_train.reset_index()[model_early_stopping['cols_features']+['hh_n','yrmo']], 
                f_y_train.reset_index()[model_early_stopping['col_outcome']]#+['hh_n','yrmo']]
            )
            
            X_train_over = X_train_over.set_index(['hh_n','yrmo'])
            #y_train_over = y_train_over.set_index(['hh_n','yrmo'])
        else:
            X_train_over = fold_X_train.copy()
            y_train_over = f_y_train.copy()

        fitted_model = estimator.fit(X_train_over, y_train_over,
                          early_stopping_rounds=early_stopping_rounds,
                          #callbacks=[
                          #EarlyStopping(rounds=early_stopping_rounds)
                          #EvaluationMonitor(metric=evalmetricstr)
                          #],
                          eval_set=[(X_train_over, y_train_over),(fold_X_test, fold_y_test)],
                          eval_metric=evalmetricstr,
                          verbose=verbose,
                     )
        
        y_pred_test=fitted_model.predict_proba(fold_X_test)[:,1]
        print('AP',evalmetric(fold_y_test, y_pred_test))
        
        # Save and show performance as plots
        
        results_fit = estimator.evals_result()
        epochs = len(results_fit['validation_0']['aucpr'])
        x_axis = range(0, epochs)
        
        best_tree_index = estimator.best_ntree_limit-1
        val_diff.append(results_fit['validation_0']['aucpr'][best_tree_index]-results_fit['validation_1']['aucpr'][best_tree_index])
        
        plt.figure(figsize=(10,7))
        plt.plot(results_fit["validation_0"]["aucpr"], label="Training aucpr")
        plt.plot(results_fit["validation_1"]["aucpr"], label="Validation aucpr")
        plt.axvline(estimator.best_ntree_limit, color="gray", label=f"Optimal tree number ({estimator.best_ntree_limit})")
        #plt.axvline(estimator.best_ntree_limit, color="blue", label=f"Optimal tree number ({estimator.best_ntree_limit})")
        plt.xlabel("Number of trees")
        plt.ylabel("AUCPR")
        plt.legend()
        
        mname = model_early_stopping['model_name']
        #plt.savefig(os.path.join(output_paths['tuning'],f'early_stopping_{mname}_imp_kfolds_{i}_{parm_1}_{parm_2}.png'),
                    #bbox_inches="tight", dpi=400, transparent=False)
        
        metrics.append(average_precision_score(fold_y_test, y_pred_test)) #roc_auc_score
        #best_iterations.append(estimator.best_iteration)
        best_ntree.append(estimator.best_ntree_limit)
        
    return np.max(val_diff), np.mean(val_diff), np.max(metrics), np.mean(metrics), np.min(metrics), np.min(best_ntree), np.max(best_ntree)

def cv_over_param_dict(
    df, 
    param_dict, 
    parm_1,
    parm_2,
    model_early_stopping, 
    cv_folds, 
    rounds_boosting, 
    early_stopping_rounds,
    random_state,
    new_imp,
    model_to_impute,
    evalmetricstr,
    evalmetric,
    output_paths,
    imp_max_iter,
    oversampling,
    o_sampling_strategy,
):
    """given a list of dictionaries of xgb params
    run my_cv on params, store result in array
    return updated param_dict, results dataframe
    """
    start_time = datetime.now()
    print("%-20s %s" % ("Start Time", start_time))

    results = []
    results_val,results_test = [],[]
    
    dict_vals,dict_results = {},{}

    for i, d in enumerate(param_dict):
        print('Parameters', d)
        xgb = XGBClassifier(
            objective='binary:logistic',
            n_estimators=rounds_boosting,
            random_state=random_state,    
            verbosity=1,
            n_jobs=14,
            **d
        ) 
        
        parm1 = d[parm_1]
        parm2 = d[parm_2]
       
        val_diff_max, val_diff_mean, metric_out_max, metric_out_mean, metric_out_min, best_ntree_min, best_ntree_max = my_cv(
            df = df,
            parm_1 = parm1,
            parm_2 = parm2,
            model_early_stopping = model_early_stopping,
            cv_folds=cv_folds,
            estimator = xgb,
            early_stopping_rounds = early_stopping_rounds,
            new_imp=new_imp,
            model_to_impute = model_to_impute,
            imp_max_iter = imp_max_iter,
            oversampling = oversampling,
            random_state = random_state,
            o_sampling_strategy = o_sampling_strategy,
            output_paths=output_paths,
            evalmetricstr = evalmetricstr,
            evalmetric = evalmetric#'aucpr',#'auc',
        ) 
        results.append([val_diff_max, val_diff_mean, metric_out_max, metric_out_mean, metric_out_min, best_ntree_min, best_ntree_max, d])
        
    end_time = datetime.now()
    print("%-20s %s" % ("Start Time", start_time))
    print("%-20s %s" % ("End Time", end_time))
    print(str(timedelta(seconds=(end_time-start_time).seconds)))
    
    results_df = pd.DataFrame(results, columns=['val_diff_max', 'val_diff_mean', 'metric_out_max', 'metric_out_mean', 'metric_out_min', 'best_ntree_min', 'best_ntree_max','param_dict']).sort_values('val_diff_max',ascending=True)
    display(results_df.head())
    
    best_params = results_df.iloc[0]['param_dict']
    return best_params, results_df