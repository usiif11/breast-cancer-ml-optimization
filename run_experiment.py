"""Run the saved configuration replay, or a fresh GA search with --search."""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay
from src.experiment import dataset, evaluate, genetic_search, GAConfig, FEATURES, PARAMS


def run(search=False,population=50,generations=20,jobs=1,out='results'):
    start=time.perf_counter(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    Xtr,Xte,ytr,yte,names=dataset()
    history=[]; cache=None
    if search:
        mask,params,history,cache=genetic_search(Xtr,ytr,GAConfig(population=population,generations=generations,jobs=jobs))
    else:
        mask=np.isin(names,FEATURES); params=PARAMS.copy()
    baseline,b,pb=evaluate(Xtr,Xte,ytr,yte,np.ones(len(names),dtype=bool))
    optimized,m,pm=evaluate(Xtr,Xte,ytr,yte,mask,params)
    report={'mode':'fresh_search' if search else 'saved_configuration_replay',
        'seed':42,'train_samples':len(ytr),'test_samples':len(yte),
        'label_order':['malignant','benign'],'positive_class':'malignant',
        'baseline':b,'optimized':m,'selected_features':names[mask].tolist(),
        'hyperparameters':params,'search_config':{'population':population,'generations':generations} if search else None,
        'cache':cache,'history':history,'python':platform.python_version(),'scikit_learn':sklearn.__version__,
        'elapsed_seconds':round(time.perf_counter()-start,3)}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6,5))
    for label,p,metric in [('Baseline',pb,b),('Selected features',pm,m)]:
        fpr,tpr,_=roc_curve(yte==0,p);ax.plot(fpr,tpr,label=f"{label} (AUC={metric['roc_auc']:.4f})")
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Test ROC (malignant positive)');ax.legend();fig.tight_layout();fig.savefig(out/'roc.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,metric,title in zip(axes,[b,m],['Baseline','Selected features']):
        ConfusionMatrixDisplay(np.array(metric['confusion_matrix']),display_labels=['Malignant','Benign']).plot(ax=ax,colorbar=False,cmap='Blues');ax.set_title(title)
    fig.tight_layout();fig.savefig(out/'confusion_matrices.png',dpi=160);plt.close(fig)
    order=np.argsort(optimized.feature_importances_)
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(names[mask][order],optimized.feature_importances_[order]);ax.set(xlabel='Impurity-based importance',title='Selected feature importances');fig.tight_layout();fig.savefig(out/'feature_importances.png',dpi=160);plt.close(fig)
    if history:
        fig,ax=plt.subplots();ax.plot([h['generation'] for h in history],[h['best_fitness'] for h in history],label='Best');ax.plot([h['generation'] for h in history],[h['mean_fitness'] for h in history],label='Mean');ax.set(xlabel='Generation',ylabel='Fitness',title='Fresh GA search');ax.legend();fig.tight_layout();fig.savefig(out/'fitness_history.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--search',action='store_true');p.add_argument('--population',type=int,default=50);p.add_argument('--generations',type=int,default=20);p.add_argument('--jobs',type=int,default=1);p.add_argument('--out',default=None);a=p.parse_args();run(a.search,a.population,a.generations,a.jobs,a.out or ('results/search' if a.search else 'results'))
