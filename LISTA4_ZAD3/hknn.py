import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from itertools import zip_longest
from collections import defaultdict

from numpy.typing import NDArray


KNNTree = tuple[KNeighborsClassifier, dict]


class HierarchicalKNN:
    def __init__(
        self, 
        labels: list[list], 
        k_neigh: int = 5, 
        metric = 'cosine',
        min_samples_to_train: int = 20,
    ) -> None:
        self.ltree = self._build_ltree(labels)
        
        self.k_neigh = k_neigh
        self.metric = metric
        self.min_samples_to_train = max(min_samples_to_train, self.k_neigh)
        self.max_depth = max(map(len, labels))
    
    def _build_ltree(self, y: list) -> dict:
        subcats = defaultdict(list)
        
        for cats in filter(None, y):
            subcats[cats[0]].append(None if len(cats) < 2 else cats[1:])

        tree = {p: (i+1, self._build_ltree(s)) 
                for i, (p, s) in enumerate(subcats.items())}
        return tree
    
    def _build_itree(self, y: list, ltree: dict) -> dict: 
        subidx = dict()
        indices = defaultdict(list)
        labels = []
        
        for i, cats in enumerate(y):
            if not cats:
                labels.append(0)
            else:
                f, lsubtree = ltree[cats[0]]
                labels.append(f)
                
                if f not in subidx: 
                    subidx[f] = ([], lsubtree)
                    
                subidx[f][0].append(cats[1:])
                indices[f].append(i)
            
        tree = {f: (*self._build_itree(c, t), indices[f]) 
                for f, (c, t) in subidx.items()}
        return labels, tree
        
    def _fit(
        self, 
        itree: dict, 
        X: NDArray, 
        y: NDArray
    ) -> KNNTree:
        if not itree or X.shape[0] < self.min_samples_to_train:
            return None
        
        knn = KNeighborsClassifier(
            n_neighbors=self.k_neigh, 
            metric=self.metric
        ).fit(X, y)      
         
        tree = {i: self._fit(subtree, X[idx], l)
                for i, (l, subtree, idx) in itree.items()}
        return (knn, tree)
        
    def fit(self, X, y):
        labels, self.itree = self._build_itree(y, self.ltree)
        self.knn_tree = self._fit(self.itree, X, labels)
        return self
    
    def _predict(
        self, 
        knn_tree: KNNTree, 
        X: NDArray, 
        depth: int
    ) -> NDArray:
        y = np.zeros((X.shape[0], 1))
        
        if not knn_tree or X.shape[0] == 0:
            return y
        
        knn, tree = knn_tree
        pred = knn.predict(X)
        
        next_p = np.zeros((X.shape[0], self.max_depth - depth))
        
        for p, subtree in tree.items():
            y[pred == p] = p
            new_p = self._predict(subtree, X[pred == p], depth+1)
            next_p[pred == p] = new_p
        
        return np.concat([y, next_p], axis=1)
    
    def predict(self, X):
        return self._predict(self.knn_tree, X, depth=0)[:, :self.max_depth]

    def _to_indices(self, y: list, ltree):    
        if not y: 
            return []
        
        new_idx, subtree = ltree[y[0]]
        return [new_idx] + self._to_indices(y[1:], subtree) if y else []

    def to_indices(self, y: list):
        idx = [self._to_indices(c, self.ltree) for c in y]
        return np.array(list(zip_longest(*idx, fillvalue=0))).T