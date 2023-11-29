"""

A utility module that provides convenient access to commonly used
modules.

"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#TODO: identify a nice interface / nice interfaces to unify this!

N_JOBS = 4

def standard_rf(random_state=None,class_weight=None):
    if class_weight is None:
        class_weight = "balanced"
    elif class_weight == "equal":
        class_weight = None 
    else:
        raise ValueError
    return RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt',
                               max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                               n_jobs=N_JOBS, random_state=random_state, verbose=0, warm_start=False, 
                               class_weight=class_weight,
                               ccp_alpha=0.0, max_samples=None)

def standard_rf_proba_calib(cv="prefit"):
    clf = standard_rf()
    cc = CalibratedClassifierCV(base_estimator=clf,cv=cv,method="isotonic",)
    return cc

def standard_log_reg(random_state=None, normal_scaling=False):
    
    lr = LogisticRegression(penalty='l2', 
        dual=False, tol=0.0001, C=1.0, fit_intercept=True,
        intercept_scaling=1, class_weight=None, random_state=random_state,
        solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0,
        warm_start=False, n_jobs=N_JOBS, l1_ratio=None)

    if normal_scaling:
        return Pipeline(steps=[('scaler', StandardScaler()), ('clf', lr)])
    else:
        return lr


def standard_svc_rbf(random_state=None,):
    return SVC(C=1.0, kernel='rbf', degree=3,
        gamma='scale', coef0=0.0, shrinking=True,
        probability=True, tol=0.001, cache_size=200,
        class_weight=None, verbose=False, max_iter=- 1,
        decision_function_shape='ovr', break_ties=False,
        random_state=random_state)

def standard_svc_linear(random_state=None,):
    return SVC(C=1.0, kernel='linear', degree=3,
        gamma='scale', coef0=0.0, shrinking=True,
        probability=True, tol=0.001, cache_size=200,
        class_weight=None, verbose=False, max_iter=- 1,
        decision_function_shape='ovr', break_ties=False,
        random_state=random_state)