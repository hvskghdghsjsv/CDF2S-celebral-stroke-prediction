import os

import math
import numpy as np
import pandas as pd
import shap as shap
from sklearn.svm import SVC
import xgboost as xgb
from deepforest import CascadeForestClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, auc, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from ib3 import ib3_test


def CBUC(data, K, target, random_state):
    # --------------------------------------------CBUC算法----------------------------------------------------------------------------------
    # step 1
    X_0 = data[data[target] == 0]
    X_1 = data[data[target] == 1]
    X_train_0, X_test_0 = train_test_split(X_0, test_size=0.1)
    # step 2 对多数类训练集使用cbu算法
    selected_ind = cbu(X_train_maj=X_train_0.drop([target], axis=1), y_train_maj=X_train_0['stroke'], K=4, Ks=124)
    X_00 = X_0.iloc[selected_ind]

    print('x_test_0', len(X_test_0))
    print('x_train_0', len(X_00))

    # step 3 对少数类使用Kmeans划分
    kmeans = KMeans(n_clusters=K, random_state=42).fit(X_1.drop([target], axis=1))
    C = []
    for i in range(K):
        C.append(X_1[kmeans.labels_ == i])
    C_train = []
    C_test = []
    for c in C:
        c_train, c_test = np.split(c.sample(frac=1, random_state=42), [int(0.9 * len(c))])
        C_train.append(c_train)
        C_test.append(c_test)

    # step 4 整合训练集和测试集
    X_pretrain = pd.concat([X_00] + C_train, ignore_index=True)
    X_test = pd.concat([X_test_0] + C_test, ignore_index=True)
    y_train = X_pretrain[target]
    X_train = X_pretrain.drop(target, axis=1)
    y_test = X_test.stroke
    X_test = X_test.drop(['stroke'], axis=1)

    return X_train, y_train, X_test, y_test


def cbu(X_train_maj, y_train_maj, K, Ks):
    # 对数据进行聚类
    kmeans = KMeans(n_clusters=K, random_state=42).fit(X_train_maj)
    # 得到聚类中心
    center = kmeans.cluster_centers_
    knn = KNeighborsClassifier(n_neighbors=Ks, metric='euclidean')
    knn.fit(X_train_maj, y_train_maj)
    distances, indices = knn.kneighbors(center)
    indices1 = np.array(indices)
    indices_final = indices1.flatten()
    return indices_final


def data_proprecesing():
    df = pd.read_csv('dataset/dataset.csv')
    df = df.drop(['id'], axis=1)
    df = df.dropna()
    # 将离散文字的数据使用固定数字替代
    df.gender = df.gender.map({'Male': 0, 'Female': 1, 'Other': 2})
    df['ever_married'] = df['ever_married'].map({"Yes": 1, "No": 0})
    df['Residence_type'] = df['Residence_type'].map({"Urban": 1, "Rural": 0})
    df.work_type = df.work_type.map({'Private': 0, 'Self-employed': 1,
                                     'children': 2, 'Govt_job': 3,
                                     'Never_worked': 4})
    df.smoking_status = df.smoking_status.map({"never smoked": 0, "formerly smoked": 1, "smokes": 2})
    df = df[df['age'] >= 10]
    df = df[df['bmi'] <= 60]
    # 对数据进行归一化处理

    scaler = StandardScaler()
    scaler_label = ['age', 'avg_glucose_level', 'bmi']
    scaler_col = df[scaler_label]
    scalered_col = scaler.fit_transform(scaler_col)
    for i in range(len(scaler_label)):
        df[scaler_label[i]] = scalered_col[:, i]

    return df


if __name__ == '__main__':
    # %%

    # 使用Gini不纯度计算 特征重要性
    prepro_data = data_proprecesing()
    X_train, y_train, X_test, y_test = CBUC(prepro_data, K=5, target='stroke', random_state=1)
    model = CascadeForestClassifier(backend='sklearn')
    model.fit(X_train, y_train)
    importances = model.get_layer_feature_importances(layer_idx=0)

    feature_names = X_train.columns
    sorted_indices = importances.argsort()[::-1]  # 降序排序后的索引
    sorted_feature_importance = importances[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    plt.figure(figsize=(11.1,6))

    # plt.pie(sorted_feature_importance, labels=sorted_feature_names, startangle=140,textprops={'fontsize': 9})
    plt.barh(sorted_feature_names[::-1], sorted_feature_importance[::-1])
    print(sorted_feature_importance[::-1])
    plt.title("mean(Feature importance)")
    plt.show()

    # %%
    '''
    prepro_data = data_proprecesing()
    performance = []
    for i in range(0, 1):
        p = []

        X_train, y_train, X_test, y_test = UCO(prepro_data, K=5, target='stroke', random_state=i + 1)
        # gcforest
        model = CascadeForestClassifier(backend='sklearn', n_trees=50)

        # RF模型；注意：使用时需要注释import中从deepforest引用的RF
        # model = RandomForestClassifier(criterion='gini', min_samples_split=5, min_samples_leaf=2, n_estimators=10)

        # KNN
        # model = KNeighborsClassifier(algorithm='auto', leaf_size=1, n_neighbors=17, weights='uniform')

        # MLP
        # model = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(5, 10, 20), learning_rate='constant',
        #                     solver='sgd')

        # SVM
        # model = SVC(C=100, gamma=0.01, kernel='sigmoid')

        # XGBoost
        # model = xgb.XGBClassifier(learning_rate=0.001, max_depth=3, n_estimators=100, min_child_weight=5, subsample=1,
        #                           colsample_bytree=0.8)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 计算acc
        acc = accuracy_score(y_test, y_pred)
        p.append(acc)

        # 计算 SPC 和 SEN 值
        y_pred_binary = [1 if proba >= 0.5 else 0 for proba in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        p.append(specificity)
        p.append(sensitivity)

        # 计算Gmean
        G_mean = math.sqrt(sensitivity * specificity)
        p.append(G_mean)

        # 计算auc
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        aucValue = auc(fpr, tpr)
        p.append(aucValue)

        print("\n acc: {:.3f} spc: {:.3f} sen: {:.3f}  Gmean: {:.3f} auc: {:.3f}".format(acc, specificity, sensitivity,
                                                                                         G_mean, aucValue))
        performance.append(p)

    final_p = np.array(performance)
    print('均值结果：', final_p.mean(axis=0))
    # %%
    # 10倍交叉验证:使用分层k折交叉验证对象
    '''
    '''
    X_train, y_train, X_test, y_test = UCO(prepro_data, K=5, target='stroke', random_state=1)

    # 10倍交叉验证:使用分层k折交叉验证对象
    fold_size = len(X_train) // 9
    accs = []
    spe = []
    sen = []
    Gmean = []
    aucs = []
    print(len(X_train),len(X_test))
    for i in range(0, 9):
        start = i * fold_size
        end = (i+1) * fold_size
        X_train_t = X_train.iloc[start:end]
        print(len(X_train_t))


        model = CascadeForestClassifier(backend='sklearn', n_trees=50)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 计算acc
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)

        # 计算 SPC 和 SEN 值
        y_pred_binary = [1 if proba >= 0.5 else 0 for proba in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        spe.append(specificity)
        sen.append(sensitivity)

        # 计算Gmean
        G_mean = math.sqrt(sensitivity * specificity)
        Gmean.append(G_mean)

        # 计算auc
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        aucValue = auc(fpr, tpr)
        aucs.append(aucValue)
    print("结束")
    print(np.array(accs).mean(axis=0))
    print(np.array(spe).mean(axis=0))
    print(np.array(sen).mean(axis=0))
    print(np.array(Gmean).mean(axis=0))
    print(np.array(aucs).mean(axis=0))
    '''
