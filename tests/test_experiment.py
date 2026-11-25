import numpy as np
from sklearn.datasets import load_breast_cancer
from src.experiment import dataset,decode,evaluate,genetic_search,GAConfig,FEATURES,PARAMS,BOUNDS

def test_dataset():
    Xtr,Xte,ytr,yte,names=dataset()
    assert Xtr.shape==(455,30) and Xte.shape==(114,30)
    assert list(load_breast_cancer().target_names)==['malignant','benign']
    assert np.isin(names,FEATURES).sum()==9

def test_decode():
    mask,params=decode((1,0,1,474,24,3,4,.155),3)
    assert mask.tolist()==[True,False,True] and params==PARAMS

def test_replay():
    Xtr,Xte,ytr,yte,names=dataset()
    _,b,_=evaluate(Xtr,Xte,ytr,yte,np.ones(30,dtype=bool))
    _,m,_=evaluate(Xtr,Xte,ytr,yte,np.isin(names,FEATURES),PARAMS)
    assert b['confusion_matrix']==[[39,3],[2,70]]
    assert m['confusion_matrix']==[[40,2],[0,72]]
    assert np.isclose(b['accuracy'],109/114) and np.isclose(m['accuracy'],112/114)
    assert np.isclose(m['roc_auc'],b['roc_auc'])

def test_search_smoke():
    Xtr,_,ytr,_,_=dataset()
    mask,params,h,cache=genetic_search(Xtr,ytr,GAConfig(population=4,generations=2))
    assert mask.shape==(30,) and mask.any()
    assert len(h)==2 and h[1]['best_fitness']>=h[0]['best_fitness']-1e-12
    assert cache['cache_hits']>0
    assert cache['cache_hits']+cache['unique_evaluations']==12
    assert all(lo<=v<=hi for v,(lo,hi) in zip(params.values(),BOUNDS))
