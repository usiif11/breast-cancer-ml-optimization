"""Reproducible research pipeline for the Wisconsin diagnostic dataset."""
from dataclasses import dataclass
from functools import lru_cache
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score)

SEED = 42
FEATURES = ('mean texture', 'mean concave points', 'mean symmetry',
            'mean fractal dimension', 'area error', 'worst texture',
            'worst area', 'worst smoothness', 'worst concavity')
PARAMS = dict(n_estimators=474, max_depth=24, min_samples_split=3,
              min_samples_leaf=4, max_features=0.155)
BOUNDS = ((50,500),(1,30),(2,20),(1,20),(.1,1.))
KEYS = tuple(PARAMS)

@dataclass(frozen=True)
class GAConfig:
    population: int = 50
    generations: int = 20
    mutation_rate: float = .05
    feature_mutation: float = .02
    seed: int = SEED
    jobs: int = 1


def dataset(seed=SEED):
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target,
        test_size=.2, random_state=seed, stratify=data.target)
    return Xtr, Xte, ytr, yte, data.feature_names


def decode(ind, n):
    mask = np.asarray(ind[:n], dtype=bool)
    params = dict(zip(KEYS, [int(v) for v in ind[n:n+4]] + [float(ind[n+4])]))
    return mask, params


def genetic_search(X_train, y_train, config=GAConfig()):
    """Search on training data only. Cache fitness; score both metrics per fit."""
    if config.population < 4 or config.generations < 1:
        raise ValueError('Need population >= 4 and generations >= 1')
    if not 0 <= config.mutation_rate <= 1 or not 0 <= config.feature_mutation <= 1:
        raise ValueError('Mutation probabilities must be in [0, 1]')
    n = X_train.shape[1]
    rng = random.Random(config.seed)
    nrng = np.random.RandomState(config.seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.seed)
    @lru_cache(maxsize=None)
    def fitness(ind):
        mask, params = decode(ind, n)
        if not mask.any():
            return 0.
        model = RandomForestClassifier(random_state=config.seed, n_jobs=1, **params)
        scores = cross_validate(model, X_train[:,mask], y_train, cv=cv,
            scoring={'auc':'roc_auc','acc':'accuracy'}, n_jobs=config.jobs, error_score='raise')
        return float(.7*scores['test_auc'].mean()+.3*scores['test_acc'].mean()-.02*mask.mean())
    def create():
        return tuple(nrng.randint(2,size=n).tolist()+[rng.randint(*b) for b in BOUNDS[:4]]+[round(rng.uniform(*BOUNDS[4]),3)])
    def mutate(ind):
        a = list(ind)
        for i in range(n):
            if rng.random() < config.feature_mutation:
                a[i] = 1-a[i]
        for j,step in enumerate((50,3,2,2)):
            if rng.random() < .1:
                a[n+j] = int(np.clip(a[n+j]+rng.randint(-step,step),*BOUNDS[j]))
        if rng.random() < .15:
            a[n+4] = float(np.clip(a[n+4]+rng.uniform(-.1,.1),*BOUNDS[4]))
        return tuple(a)
    pop = [create() for _ in range(config.population)]
    scores = np.array([fitness(i) for i in pop]); history=[]
    for gen in range(config.generations):
        def tournament():
            ids=nrng.choice(len(pop),3,replace=False)
            return pop[max(ids,key=lambda i:scores[i])]
        elite=[pop[i] for i in np.argsort(scores)[-2:]]
        children=[]
        while len(children)<config.population-2:
            a,b=tournament(),tournament()
            if rng.random()<.9:
                lo,hi=sorted(rng.sample(range(len(a)),2))
                a,b=a[:lo]+b[lo:hi]+a[hi:], b[:lo]+a[lo:hi]+b[hi:]
            for child in (a,b):
                if len(children)<config.population-2:
                    children.append(mutate(child) if rng.random()<config.mutation_rate else child)
        pop=children+elite
        scores=np.array([fitness(i) for i in pop])
        best=pop[int(np.argmax(scores))]
        history.append({'generation':gen+1,'best_fitness':float(scores.max()),
            'mean_fitness':float(scores.mean()),'features':int(sum(best[:n]))})
    mask,params=decode(best,n)
    if not mask.any():
        raise RuntimeError('Search returned no features; increase budget')
    info=fitness.cache_info()
    return mask,params,history,{'cache_hits':info.hits,'unique_evaluations':info.misses}


def evaluate(Xtr,Xte,ytr,yte,mask,params=None):
    model=RandomForestClassifier(random_state=SEED,n_jobs=1,**(params or {}))
    model.fit(Xtr[:,mask],ytr)
    pred=model.predict(Xte[:,mask]); prob=model.predict_proba(Xte[:,mask])
    # sklearn encodes malignant=0, benign=1; use malignancy as positive for clinical-style metrics.
    y_m=(yte==0).astype(int); p_m=(pred==0).astype(int)
    metrics=dict(features=int(mask.sum()),accuracy=float(accuracy_score(yte,pred)),
        roc_auc=float(roc_auc_score(y_m,prob[:,0])),
        precision_malignant=float(precision_score(y_m,p_m,zero_division=0)),
        recall_malignant=float(recall_score(y_m,p_m,zero_division=0)),
        f1_malignant=float(f1_score(y_m,p_m,zero_division=0)),
        confusion_matrix=confusion_matrix(yte,pred,labels=[0,1]).tolist())
    return model,metrics,prob[:,0]
