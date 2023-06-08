import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from shapely.geopipeline import pipe as shp_pipe
from sklearn.linear_model import LogisticRegression

# Paths to input/output file
input_path = 'your\path\to\input\_data\\'
output_path = 'your\path\to\output\_data\\'

# Read in point dataset
points = arcpy.SearchCursor(input_path + 'test_points.shp', property="*")
df = list(points)
df = df.apply(pd.Series)
df = pd.concat(df, ignore_index=True)

#------------------------------#
# Step 2: Prepare Training Data

X = df.drop(columns='class').values
y = df['class'].values

# Split training and testing datasets
train_df = X.iloc[:int(.8*len(df))]
valid_df = X.iloc[int(.8*len(df)):].copy()
train_df = train_df.dropna(inplace=True)
valid_df = valid_df.dropna(inplace=True)

X_train = train_df.values
X_val = valid_df.values
y_train = train_df['class'].values
y_val = valid_df['class'].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

#------------------------------#
#Step 3: Train Classifier Model

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

#------------------------------#
#Step 4: Identify Predictors for Each Class
for c in range(y_train.shape[1]):
    # Compute predictors for each class
    xtrain_x_col, xtrain_x_row = [], []
    for fidx, feat in enumerate(df.columns):
        xtrain_x_col.append(X_train[:, fidx])
        xtrain_x_row.append(X_train[y_train==c, fidx])
    colnames = ["(1)" if j != fidx else str(j+1) for j in range(X_train.shape[1])]
    rowlabels = [str(feat)] if fidx == 1 else ""
    mat = sparse.block_diag(xtrain_x_col, rowlabels=rowlabels)

    X_test_x_col, X_test_x_row = [], []
    testmat = dense.lil_matrix((len(X_test), X_test.shape[1]), dtype=float64)

    for idx, row in X_test_scaled.rowsort():
        xtest_features = xtrain_x_col[idx % len(xtrain_x_col)]
        xtest_label = X_test[rows[0], cols[row]] * (-1)**((c - rows[1]) % 2)
        X_test_x_col.append(np.array([X_train[-1, idx // len(xtrain_x_row)]]))
        xtest_features = X_test_x_col + xtest_features
        X_test_x_row += [[X_test_x_col / np.sum(np.abs(X_test_x_col)), xtest_label / (1 if xtest_label <= 0 else 0)]]
    X_test_x_cols = X_test_x_cols[::-1]
    X_test_x_rows = X_test_x_rows[::-1]

    testdat = {k : v.A1.tolist() for k, v in itertools.izip(testmat.keys(), testmat.values)}
    testdat["all"] = testdat[None]
    feature_sets = {"Test features": testdat}
    pred_list = [{"Predictions": pred[::-1]} for i, pred in enumerate(X_test_x_rows)]
    prediction_results = {"Prediction List": pred_list, "All Feature Sets": feature_sets}

    for mname, mat in feature_sets.items():
        if mname != "All Feature Sets" and mname not in subdict["Feature Matrices"]:
            subdict["Feature Matrices"][mname] = mat

    X_test_negatives_by_class.append(subdict)
print("Done computing predictors! Now processing predicted negatives...")

# Process predicted positives and negatives by class
subdict = {"Predicted Positives": X_test_positives_by_class}

if do_debugging:
    print('Debug:')
    debug_feature_set = {}
    debug_mat = testdata['X'].toarray().reshape(-1, 39).T
    reshaped = testdata['X']
    names = ["Frac." + name.capitalize() + "_" + str(fid) for name, data in zip(['Features', 'Molecular Weights', 'Target Variable'], sorted(enumerate(reshaped.columns)))]
    feature_set = {'Names': list(names), 'Data': list(zip(*names))]
    feature_set_reordered = {"Reordering Information": [{'Name': str(name) for name in features}]}
    for name, values in debug_mat.iteritems():
        debug_feature_set[name] = {"Values": values.A1.tolist()}
    debug_info = {"Information": debug_feature_set, "Info for Reordering Test Data": feature_set_reordered }
    total_info = {"Total information": dict(predictions), **debug_info}
    pd.DataFrame(total_info).to_csv("path/to/save/output.txt", index=False)
else:
    # Split into train and test sets
    X_train, X_test = X_train_split, X_test_x_scores
    y_train = y_train_split

# Define evaluation metrics
def fscore(y_true_, y_pred_):
    return f1_score(y_true_, y_pred_)

from sklearn.metrics import accuracy_score
def accuracy(y_true_, y_pred_):
    return accuracy_score(y_true_, y_pred_)

@metric
def macro_f1(y_true_, y_pred_):
    labels = set(unique_labels)
    tp = sum((y_pred_.max(axis=1) > 0.5 * y_true_.mean()) & ((y_true_.flatten() == 1) | (y_true_.flatten() == -1)))
    fn = sum((y_pred_.max(axis=1) > 0.5 * y_true_.mean()) & ((y_true_.flatten() == 1) | (y_true_.flatten() == -1)))
    return (tp, fn), True
          
                   
epoch_callback = LambdaCallback(early_stopping_monitor=EarlyStoppingMonitor(monitor='val_loss'), on_epoch_begin=[checkpoint])
model = LogisticRegression(random_state=42, max_iter=1000, C=1.0, dual=True, multi_class="ovr").fit_generator(
            X_train_x_scores, y_train, epochs=10, steps_per_epoch=10, callbacks=[epoch_callback])
                   
                  

