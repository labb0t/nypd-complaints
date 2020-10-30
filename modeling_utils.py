import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict 
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

def get_classifier_scores(X, y, models, preprocessor):
    '''
    Fits the specified classification models to provided data, runs cross validation, 
    and returns the average accuracy score for each model.
    
    Parameters
    ----------
    X : feature data
    y : target data
    models : list of classification models to fit and score
    preprocessor: Predefined sklearn Pipeline steps  for processing the X and y data as needed

    Returns
    -------
    Average accuracy score for each model from cross validation
    '''   
    for clf_model in models:
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', clf_model)])
        clf_score =  np.mean(cross_val_score(estimator=clf,X=X,y=y)) #consider changing to cross-val and printing more metrics
        print(clf_model)
        print(clf_score)


def get_classifier_performance(X,y,model,preprocessor):
    '''
    Fits the specified classification model to provided data, runs cross validation, 
    and returns classification report, confusion matrix, and roc curve.
    
    Parameters
    ----------
    X : feature data
    y : target data
    model : classification model
    preprocessor: Predefined sklearn Pipeline steps for processing the X and y data as needed

    Returns
    -------
    Average accuracy score for each model from cross validation
    ''' 
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', model)])
    
    # generate cross-validation predictions
    y_cv_pred = cross_val_predict(clf, X, y, cv=5)
    y_cv_pred_proba = cross_val_predict(clf, X, y, cv=5,method='predict_proba')

    # generate classification report
    print(classification_report(y, y_cv_pred))    

    # generate and plot confusion matrix
    fig, axes = plt.subplots(nrows=1, ncols=1)

    cf_matrix = confusion_matrix(y, y_cv_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.0%', cmap='YlGnBu', annot_kws={"fontsize":14}, cbar = False)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title('Confusion Matrix')
    
    # plot roc curve
    clf_roc_auc = roc_auc_score(y, y_cv_pred)
    fpr, tpr, thresholds = roc_curve(y, y_cv_pred_proba[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % clf_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Ruling: Conduct Occurred')
    plt.legend(loc="lower right")
    plt.show()

def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names