#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('./loan_data.csv')
df


# In[3]:


numerical = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
categorical = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


for col in numerical:
    plt.figure(figsize=(6, 4))

    sns.histplot(df[col], bins=50, color='black', alpha=1)

    plt.ticklabel_format(style='plain', axis='both')

    plt.title(f'Histogram of {col}')
    plt.xlabel(col)

    plt.show()


# In[8]:


df = df[df['person_age'] < 72]


# In[9]:


df = df[df['person_income'] < 1000000]


# In[125]:


df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].replace(['No', 'Yes'], [0, 1]).astype(int)


# In[126]:


pd.set_option('display.float_format', '{:.2f}'.format)


# In[127]:


df.describe()


# In[13]:


pd.reset_option('display.float_format')


# In[14]:


df = df.reset_index(drop=True)


# In[15]:


numerical = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"]
categorical = ["person_gender", "person_education", "person_home_ownership", "loan_intent"]


# In[16]:


correlation_matrix = df[numerical].corr()


# In[17]:


plt.figure(figsize=(5, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[18]:


df.describe()


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=21)


# In[21]:


df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=21)


# In[22]:


df_train


# In[23]:


y_train = df_train.loan_status.values
y_val = df_val.loan_status.values
y_train_full = df_train_full.loan_status.values
y_test = df_test.loan_status.values


# In[24]:


del df_train['loan_status']
del df_val['loan_status']
del df_train_full['loan_status']
del df_test['loan_status']


# In[25]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score


# In[76]:


def train_model_and_get_metrics(df_train, y_train, df_val, y_val, features_list, c, class_weight='balanced'):
    print(f"Training final model on full train set with C={c}")

    # Train on training data
    train_dict = df_train[features_list].to_dict(orient='records')
    dv_train = DictVectorizer(sparse=False)
    dv_train.fit(train_dict)
    X_train = dv_train.transform(train_dict)

    # Train model on training set
    model_final = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=21, class_weight=class_weight)
    model_final.fit(X_train, y_train)

    # Evaluate on TEST set
    val_dict = df_val[features_list].to_dict(orient='records')
    X_val = dv_train.transform(val_dict)
    y_pred_val = model_final.predict_proba(X_val)[:, 1]

    # Try different thresholds
    thresholds = np.linspace(0.1, 0.8, 81)  # Test 81 thresholds from 0.1 to 0.9
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_val >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)

    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Apply best threshold
    y_pred_val_final = (y_pred_val >= best_threshold).astype(int)

    # Calculate final TEST metrics
    print(f"\nBest treshold: {best_threshold:.4f}")
    print(f"\nBest F1 Score: {f1_score(y_val, y_pred_val_final):.4f}")
    print(f"Test Set ROC-AUC Score: {roc_auc_score(y_val, y_pred_val):.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_val, y_pred_val_final))
    return model_final


# In[91]:


def train_model_and_get_metrics_dynamic(df_train, y_train, df_val, y_val, features_list, 
                                c_values=None, class_weight='balanced', 
                                optimize_for='f1'):
    if c_values is None:
        c_values = [0.001, 0.01, 0.1, 1, 10, 100]

    print(f"Searching for best C from {c_values}")
    print(f"Optimizing for: {optimize_for}")

    # Prepare training data
    train_dict = df_train[features_list].to_dict(orient='records')
    dv_train = DictVectorizer(sparse=False)
    dv_train.fit(train_dict)
    X_train = dv_train.transform(train_dict)

    # Prepare validation data
    val_dict = df_val[features_list].to_dict(orient='records')
    X_val = dv_train.transform(val_dict)

    # Find best C
    best_c = None
    best_score = -1
    best_threshold = 0.5
    results = []

    for c in c_values:
        # Train model
        model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, 
                                  random_state=21, class_weight=class_weight)
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        if optimize_for == 'f1':
            # Find best threshold for F1
            thresholds = np.linspace(0.1, 0.8, 81)
            f1_scores = []

            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                f1_scores.append(f1)

            best_idx = np.argmax(f1_scores)
            threshold = thresholds[best_idx]
            score = f1_scores[best_idx]
        else:  # roc_auc
            score = roc_auc_score(y_val, y_pred_proba)
            threshold = 0.5

        results.append({'C': c, 'score': score, 'threshold': threshold})
        print(f"C={c:7.4f} | {optimize_for.upper()}={score:.4f} | threshold={threshold:.4f}")

        if score > best_score:
            best_score = score
            best_c = c
            best_threshold = threshold

    # Train final model with best C
    print(f"\n{'='*60}")
    print(f"Best C: {best_c}")
    print(f"Best {optimize_for.upper()}: {best_score:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"{'='*60}\n")

    model_final = LogisticRegression(solver='liblinear', C=best_c, max_iter=1000, 
                                    random_state=21, class_weight=class_weight)
    model_final.fit(X_train, y_train)

    # Final evaluation
    y_pred_proba = model_final.predict_proba(X_val)[:, 1]
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)

    print(f"Final Model Performance:")
    print(f"F1 Score: {f1_score(y_val, y_pred_final):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_final))

    return model_final, dv_train, best_threshold


# In[92]:


numerical_for_regression = ["person_age", "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "credit_score", "previous_loan_defaults_on_file"]
#numerical_for_regression = ["person_emp_exp", "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "credit_score", "previous_loan_defaults_on_file"]
#numerical_for_regression = ["cb_person_cred_hist_length", "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "credit_score", "previous_loan_defaults_on_file"]


# In[93]:


all_features = categorical + numerical
all_features


# In[94]:


all_features = categorical + numerical
all_features_for_regression = categorical + numerical_for_regression
train_model_and_get_metrics_dynamic(df_train, y_train, df_val, y_val, all_features, class_weight='balanced')


# In[96]:


train_model_and_get_metrics_dynamic(df_train, y_train, df_val, y_val, all_features, class_weight=None)


# In[97]:


train_model_and_get_metrics_dynamic(df_train, y_train, df_val, y_val, all_features_for_regression, class_weight='balanced')


# In[100]:


train_model_and_get_metrics_dynamic(df_train, y_train, df_val, y_val, all_features_for_regression, class_weight=None)


# In[41]:


#!jupyter nbconvert --to script Midterm-project.ipynb


# In[158]:


def train_model_and_get_metrics(df_train, y_train, df_val, y_val, features_list, 
                                c_values=None, class_weights=None, 
                                optimize_for='f1'):
    if c_values is None:
        c_values = [0.001, 0.01, 0.1, 1, 10, 100]

    if class_weights is None:
        class_weights = [None, 'balanced']

    print(f"Searching for best C from {c_values}")
    print(f"Searching for best class_weight from {class_weights}")
    print(f"Optimizing for: {optimize_for}")
    print(f"Total combinations to test: {len(c_values) * len(class_weights)}\n")

    # Prepare training data
    train_dict = df_train[features_list].to_dict(orient='records')
    dv_train = DictVectorizer(sparse=False)
    dv_train.fit(train_dict)
    X_train = dv_train.transform(train_dict)

    # Prepare validation data
    val_dict = df_val[features_list].to_dict(orient='records')
    X_val = dv_train.transform(val_dict)

    # Find best combination of C and class_weight
    best_c = None
    best_class_weight = None
    best_score = -1
    best_threshold = 0.5
    results = []

    for class_weight in class_weights:
        for c in c_values:
            # Train model
            model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, 
                                      random_state=21, class_weight=class_weight)
            model.fit(X_train, y_train)

            # Get predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            if optimize_for == 'f1':
                # Find best threshold for F1
                thresholds = np.linspace(0.1, 0.8, 81)
                f1_scores = []

                for threshold in thresholds:
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append(f1)

                best_idx = np.argmax(f1_scores)
                threshold = thresholds[best_idx]
                score = f1_scores[best_idx]
            else:  # roc_auc
                score = roc_auc_score(y_val, y_pred_proba)
                threshold = 0.5

            results.append({
                'C': c, 
                'class_weight': class_weight, 
                'score': score, 
                'threshold': threshold
            })

            cw_str = str(class_weight).ljust(10)
            print(f"class_weight={cw_str} | C={c:7.4f} | {optimize_for.upper()}={score:.4f} | threshold={threshold:.4f}")

            if score > best_score:
                best_score = score
                best_c = c
                best_class_weight = class_weight
                best_threshold = threshold

    # Train final model with best parameters
    print(f"\n{'='*60}")
    print(f"Best class_weight: {best_class_weight}")
    print(f"Best C: {best_c}")
    print(f"Best {optimize_for.upper()}: {best_score:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"{'='*60}\n")

    model_final = LogisticRegression(solver='liblinear', C=best_c, max_iter=1000, 
                                    random_state=21, class_weight=best_class_weight)
    model_final.fit(X_train, y_train)

    # Final evaluation
    y_pred_proba = model_final.predict_proba(X_val)[:, 1]
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)

    print(f"Final Model Performance:")
    print(f"F1 Score: {f1_score(y_val, y_pred_final):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_final))

    return model_final, dv_train, best_threshold


# In[149]:


log_regression_model, log_regression_dv, log_regression_threshold = train_model_and_get_metrics(df_train, y_train, df_val, y_val, all_features)


# In[104]:


train_model_and_get_metrics(df_train, y_train, df_val, y_val, all_features_for_regression)


# In[106]:


train_model_and_get_metrics(df_train, y_train, df_val, y_val, all_features_for_regression, optimize_for='roc_auc')


# In[107]:


train_model_and_get_metrics(df_train_full, y_train_full, df_test, y_test, all_features_for_regression)


# In[108]:


train_model_and_get_metrics(df_train_full, y_train_full, df_test, y_test, all_features_for_regression, optimize_for='roc_auc')


# In[110]:


from sklearn.ensemble import RandomForestClassifier


# In[113]:


def train_random_forest_and_get_metrics(df_train, y_train, df_val, y_val, features_list,
                                       n_estimators_values=None, max_depth_values=None,
                                       min_samples_split_values=None, min_samples_leaf_values=None,
                                       max_features_values=None, class_weights=None,
                                       optimize_for='f1', n_jobs=-1):
    if n_estimators_values is None:
        n_estimators_values = [50, 100, 200]
    if max_depth_values is None:
        max_depth_values = [3, 5, 10, 15, None]
    if min_samples_split_values is None:
        min_samples_split_values = [2, 5, 10]
    if min_samples_leaf_values is None:
        min_samples_leaf_values = [1, 2, 4]
    if max_features_values is None:
        max_features_values = ['sqrt', 'log2', None]
    if class_weights is None:
        class_weights = [None, 'balanced']

    total_combinations = (len(n_estimators_values) * len(max_depth_values) * 
                         len(min_samples_split_values) * len(min_samples_leaf_values) *
                         len(max_features_values) * len(class_weights))

    print(f"Random Forest Hyperparameter Search")
    print(f"{'='*70}")
    print(f"n_estimators: {n_estimators_values}")
    print(f"max_depth: {max_depth_values}")
    print(f"min_samples_split: {min_samples_split_values}")
    print(f"min_samples_leaf: {min_samples_leaf_values}")
    print(f"max_features: {max_features_values}")
    print(f"class_weight: {class_weights}")
    print(f"Optimizing for: {optimize_for}")
    print(f"Total combinations to test: {total_combinations}")
    print(f"Note: This may take a while...\n")

    # Prepare training data
    train_dict = df_train[features_list].to_dict(orient='records')
    dv_train = DictVectorizer(sparse=False)
    dv_train.fit(train_dict)
    X_train = dv_train.transform(train_dict)

    # Prepare validation data
    val_dict = df_val[features_list].to_dict(orient='records')
    X_val = dv_train.transform(val_dict)

    # Grid search
    best_params = {}
    best_score = -1
    best_threshold = 0.5
    results = []
    iteration = 0

    for class_weight in class_weights:
        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                for min_samples_split in min_samples_split_values:
                    for min_samples_leaf in min_samples_leaf_values:
                        for max_features in max_features_values:
                            iteration += 1

                            # Train model
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                class_weight=class_weight,
                                random_state=21,
                                n_jobs=n_jobs
                            )
                            model.fit(X_train, y_train)

                            # Get predictions
                            y_pred_proba = model.predict_proba(X_val)[:, 1]

                            if optimize_for == 'f1':
                                # Find best threshold for F1
                                thresholds = np.linspace(0.1, 0.8, 81)
                                f1_scores = [f1_score(y_val, (y_pred_proba >= t).astype(int)) 
                                            for t in thresholds]
                                best_idx = np.argmax(f1_scores)
                                threshold = thresholds[best_idx]
                                score = f1_scores[best_idx]
                            else:  # roc_auc
                                score = roc_auc_score(y_val, y_pred_proba)
                                threshold = 0.5

                            results.append({
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'max_features': max_features,
                                'class_weight': class_weight,
                                'score': score,
                                'threshold': threshold
                            })

                            if iteration % 10 == 0:
                                print(f"Progress: {iteration}/{total_combinations} combinations tested...")

                            if score > best_score:
                                best_score = score
                                best_threshold = threshold
                                best_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features,
                                    'class_weight': class_weight
                                }
                                print(f"  New best score: {best_score:.4f} with params: {best_params}")

    # Train final model with best parameters
    print(f"\n{'='*70}")
    print(f"Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best {optimize_for.upper()}: {best_score:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"{'='*70}\n")

    model_final = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        class_weight=best_params['class_weight'],
        random_state=21,
        n_jobs=n_jobs
    )
    model_final.fit(X_train, y_train)

    # Final evaluation
    y_pred_proba = model_final.predict_proba(X_val)[:, 1]
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)

    print(f"Final Model Performance:")
    print(f"F1 Score: {f1_score(y_val, y_pred_final):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_final))

    return model_final, dv_train, best_threshold


# In[123]:


random_forest_model, random_forest_dv, random_forest_threshold = train_random_forest_and_get_metrics(df_train, y_train, df_val, y_val, all_features)


# In[115]:


from sklearn.tree import DecisionTreeClassifier


# In[120]:


def train_decision_tree_and_get_metrics(df_train, y_train, df_val, y_val, features_list,
                                       max_depth_values=None, min_samples_split_values=None,
                                       min_samples_leaf_values=None, class_weights=None,
                                       optimize_for='f1'):

    if max_depth_values is None:
        max_depth_values = [2, 3, 5, 7, 10, 12, 15, None]
    if min_samples_split_values is None:
        min_samples_split_values = [2, 5, 7, 10]
    if min_samples_leaf_values is None:
        min_samples_leaf_values = [1, 2, 4]
    if class_weights is None:
        class_weights = [None, 'balanced']

    total_combinations = (len(max_depth_values) * len(min_samples_split_values) * 
                         len(min_samples_leaf_values) * len(class_weights))

    print(f"Decision Tree Hyperparameter Search")
    print(f"{'='*70}")
    print(f"max_depth: {max_depth_values}")
    print(f"min_samples_split: {min_samples_split_values}")
    print(f"min_samples_leaf: {min_samples_leaf_values}")
    print(f"class_weight: {class_weights}")
    print(f"Optimizing for: {optimize_for}")
    print(f"Total combinations to test: {total_combinations}\n")

    # Prepare training data
    train_dict = df_train[features_list].to_dict(orient='records')
    dv_train = DictVectorizer(sparse=False)
    dv_train.fit(train_dict)
    X_train = dv_train.transform(train_dict)

    # Prepare validation data
    val_dict = df_val[features_list].to_dict(orient='records')
    X_val = dv_train.transform(val_dict)

    # Grid search
    best_params = {}
    best_score = -1
    best_threshold = 0.5
    results = []

    for class_weight in class_weights:
        for max_depth in max_depth_values:
            for min_samples_split in min_samples_split_values:
                for min_samples_leaf in min_samples_leaf_values:
                    # Train model
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        class_weight=class_weight,
                        random_state=21
                    )
                    model.fit(X_train, y_train)

                    # Get predictions
                    y_pred_proba = model.predict_proba(X_val)[:, 1]

                    if optimize_for == 'f1':
                        # Find best threshold for F1
                        thresholds = np.linspace(0.1, 0.8, 81)
                        f1_scores = [f1_score(y_val, (y_pred_proba >= t).astype(int)) 
                                    for t in thresholds]
                        best_idx = np.argmax(f1_scores)
                        threshold = thresholds[best_idx]
                        score = f1_scores[best_idx]
                    else:  # roc_auc
                        score = roc_auc_score(y_val, y_pred_proba)
                        threshold = 0.5

                    results.append({
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'class_weight': class_weight,
                        'score': score,
                        'threshold': threshold
                    })

                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'class_weight': class_weight
                        }

    # Train final model with best parameters
    print(f"\n{'='*70}")
    print(f"Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best {optimize_for.upper()}: {best_score:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"{'='*70}\n")

    model_final = DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        class_weight=best_params['class_weight'],
        random_state=21
    )
    model_final.fit(X_train, y_train)

    # Final evaluation
    y_pred_proba = model_final.predict_proba(X_val)[:, 1]
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)

    print(f"Final Model Performance:")
    print(f"F1 Score: {f1_score(y_val, y_pred_final):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_final))

    return model_final, dv_train, best_threshold


# In[124]:


decision_tree_model, decision_tree_dv, decision_tree_treshold = train_decision_tree_and_get_metrics(df_train, y_train, df_val, y_val, all_features)


# In[132]:


X_test = decision_tree_dv.transform(df_test.to_dict(orient='records'))
y_test_pred_proba = decision_tree_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_test_pred_proba >= decision_tree_treshold).astype(int)


# In[143]:


print(f"F1 Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_test_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))


# In[144]:


from sklearn.metrics import accuracy_score


# In[145]:


accuracy_score(y_test, y_pred_final)


# In[146]:


X_test = random_forest_dv.transform(df_test.to_dict(orient='records'))
y_test_pred_proba = random_forest_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_test_pred_proba >= random_forest_threshold).astype(int)


# In[147]:


print(f"F1 Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_test_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))


# In[148]:


accuracy_score(y_test, y_pred_final)


# In[150]:


X_test = log_regression_dv.transform(df_test.to_dict(orient='records'))
y_test_pred_proba = log_regression_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_test_pred_proba >= log_regression_threshold).astype(int)


# In[151]:


print(f"F1 Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_test_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))


# In[152]:


accuracy_score(y_test, y_pred_final)


# Final model training

# In[823]:


random_forest_model, random_forest_dv, random_forest_threshold = train_random_forest_and_get_metrics(df_train_full, y_train_full, df_test, y_test, all_features)


# In[155]:


X_test = random_forest_dv.transform(df_test.to_dict(orient='records'))
y_test_pred_proba = random_forest_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_test_pred_proba >= random_forest_threshold).astype(int)


# In[156]:


print(f"F1 Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_test_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))


# In[157]:


accuracy_score(y_test, y_pred_final)


# In[821]:


import pickle


# In[822]:


with open('midterm_model.bin', 'wb') as f_out:
    pickle.dump((random_forest_model, random_forest_dv, random_forest_threshold), f_out)


# In[ ]:




