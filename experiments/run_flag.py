"""Run FLAG (label-adaptive aggregation) on all multi-label cells, both backbones,
appending to the results files so it joins the B/C tables."""
import json, os
from fedprior.algorithms import METHODS, federated
from experiments.full_benchmark import build_task
from experiments.pretrained_gate import _cfg as pt_cfg, _build as pt_build, ROUNDS as PT_R
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"full")
def append(path, runner):
    per=json.load(open(path)) if os.path.exists(path) else {}
    hs=per.get("FLAG",[])
    for si,seed in enumerate((0,1,2)):
        if si<len(hs): continue
        h=runner(seed); hs.append(h); per["FLAG"]=hs; json.dump(per,open(path,"w"))
        print(f"[{os.path.basename(path)} s{seed}] FLAG mAP={h['test_acc'][-1]:.4f}",flush=True)
for task in ("voc","coco"):
    for a in (0.1,0.5):
        # from-scratch
        def fs(seed,task=task,a=a):
            cfg,cl,gc,tc,rounds,_=build_task(task,a,seed); return federated(METHODS["FLAG"],rounds,cl,gc,tc,cfg)[1]
        append(os.path.join(OUT,f"results_{task}_a{a}.json"), fs)
        # pretrained
        def pt(seed,task=task,a=a):
            cfg=pt_cfg(task,seed); cl,gc,tc=pt_build(task,a,seed); return federated(METHODS["FLAG"],PT_R,cl,gc,tc,cfg)[1]
        append(os.path.join(OUT,f"results_pt_{task}_a{a}.json"), pt)
print("RUN_FLAG_DONE",flush=True)
