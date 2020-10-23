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

    fig, axes = plt.subplots(nrows=1, ncols=1)

    # generate confusion matrix
    cf_matrix = confusion_matrix(y, y_cv_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.0%', cmap='Purples')
    
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