#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from abc import abstractmethod, ABC


class MethodCreator:
    def __init__(self, random_state=42, problem='classification'):
        self.rs = random_state
        self.problem = problem
        
    def create_dict(self):
        if self.problem == 'classification':
            names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
                 "Decision Tree", "Random Forest", "Naive Bayes"]

            classifiers = [
                KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                GaussianNB(),
                ]
            clf_dict = dict(zip(names, classifiers))
            return clf_dict

class DSCreator:
    def __init__(self, random_state=42, data_type='data_for_classification'):
        self.rs = random_state
        self.data_type = data_type
        
    def create_ds(self):
        if self.data_type == 'data_for_classification':
            X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=self.rs, n_clusters_per_class=1)
            rng = np.random.RandomState(self.rs)
            X +=  2 * rng.uniform(size=X.shape)
            linearly_separable = (X, y)

            datasets = [make_moons(noise=0.3, random_state=self.rs),
                        make_circles(noise=0.2, factor=0.5, random_state=self.rs),
                        linearly_separable
                        ]
        return datasets

class Visualiser(ABC):
    def __init__(self,  ds_list, methods_list, names):
        self.ds_list = ds_list
        self.methods_list = methods_list
        self.names = names
        self.ds_number = len(ds_list)
        self.methods_number = len(methods_list)
        
#         self.figure = plt.figure(figsize=(len(classifiers) * 3 + 3, len(datasets) * 3))
        self.pics_width = 21
        self.pics_high = 15
    
    @abstractmethod
    def plot_input_data(self):
        pass
    
    @abstractmethod
    def plot_comparison(self):
        pass  
        


# In[10]:


class ClfVisualiser(Visualiser):
    def __init__(self, ds_list, methods_list, names, h=.02, cm=plt.cm.viridis):
        super().__init__(ds_list, methods_list, names)
        self.cm = cm
#         self.cm_bright = ListedColormap(['#0000FF', '#FFFF00'])
        self.h = h
    
    @staticmethod
    def preplotting(ax, X, y, splitting=True, test_size=.4, random_state=42, cm=plt.cm.viridis, h=.02):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        if splitting:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k')
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6, edgecolors='k')
        else:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=self.cm, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        return xx, yy

    def plot_input_data(self):
        figure = plt.figure(figsize=(min(self.pics_high/self.ds_number, 6), 
                                          self.ds_number * min(self.pics_high/self.ds_number, 6)))
        for ds_cnt, ds in enumerate(self.ds_list):
            X, y = ds
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
            ax = plt.subplot(self.ds_number, 1, ds_cnt + 1)
            if ds_cnt == 0:
                ax.set_title("Input data", fontsize = 20)
            xx, yy = self.preplotting(ax=ax, X=X, y=y, cm=self.cm, h=self.h)
        nm = str(np.random.randint(10**10))
        pic_link = './static/images/' + nm + '.png'        
        plt.tight_layout()
        plt.savefig(pic_link, dpi=50)
        # plt.show()
        return 'images/' + nm + '.png'
        
    def plot_comparison(self):
        if self.methods_number == 0:
            filename = self.plot_input_data()
            return filename
        else:
            figure = plt.figure(
                 figsize=((self.methods_number + 1) * min(self.pics_high/self.ds_number, self.pics_width/(self.methods_number + 1), 6), 
                          self.ds_number * min(self.pics_high/self.ds_number, self.pics_width/(self.methods_number + 1), 6))
             )
            i = 1
            for ds_cnt, ds in enumerate(self.ds_list):
                X, y = ds
                X = StandardScaler().fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
                ax = plt.subplot(self.ds_number,self.methods_number + 1, i)
                if ds_cnt == 0:
                    ax.set_title("Input data", fontsize = 20)
                self.preplotting(ax=ax, X=X, y=y, cm=self.cm)
                i += 1
                for name, clf in zip(self.names, self.methods_list):
                    ax = plt.subplot(self.ds_number, self.methods_number + 1, i)
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                    
                    xx, yy = self.preplotting(ax=ax, X=X, y=y, cm=self.cm, h=self.h)

                    if hasattr(clf, "decision_function"):
                        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                    else:
                        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                    Z = Z.reshape(xx.shape)
                    ax.contourf(xx, yy, Z, cmap=self.cm, alpha=.8)

                    
                    if ds_cnt == 0:
                        ax.set_title(name, fontsize = 20)
                    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                            size=15, horizontalalignment='right')
                    i += 1
            nm = str(np.random.randint(10**10))
            pic_link = './static/images/' + nm + '.png'        
            plt.tight_layout()
            plt.savefig(pic_link, dpi=50)
            # plt.show()
            return 'images/' + nm + '.png'


            
if __name__ == "__main__":
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
         "Decision Tree", "Random Forest", "Naive Bayes"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        GaussianNB(),
        ]

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X +=  2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]
    
    clf_vis = ClfVisualiser(ds_list=datasets[:2], methods_list=classifiers[:2], names=names[:2])
    clf_vis.plot_comparison()


# In[ ]:




