"""From-scratch 2x2 disentangling (GN ResNet) on VOC/COCO a0.1: BalBCE + FedND-std
appended to the main results files (which hold FedAvg + FedNTD-BCE), to attribute
the from-scratch FedNTD-BCE gain (balancing vs distillation) for the paper."""
import json, os, statistics
from fedprior.algorithms import METHODS, federated
from experiments.full_benchmark import build_task
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"full")
for task,alpha in [("voc",0.1),("coco",0.1)]:
    tag=f"{task}_a{alpha}"; path=os.path.join(OUT,f"results_{tag}.json")
    per=json.load(open(path)) if os.path.exists(path) else {}
    for name in ["BalBCE","FedND-std"]:
        done=per.get(name,[])
        if len(done)>=3: continue
        for si,seed in enumerate((0,1,2)):
            if si<len(done): continue
            cfg,cl,gc,tc,rounds,_=build_task(task,alpha,seed)
            _,h=federated(METHODS[name],rounds,cl,gc,tc,cfg)
            done.append(h); per[name]=done; json.dump(per,open(path,"w"))
            print(f"[{tag} s{seed}] {name} mAP={h['test_acc'][-1]:.4f}",flush=True)
print("=== FROM-SCRATCH 2x2 (mAP%) ===")
for task,alpha in [("voc",0.1),("coco",0.1)]:
    d=json.load(open(os.path.join(OUT,f"results_{task}_a{alpha}.json")))
    def m(k): return statistics.mean(statistics.mean(h["test_acc"][-10:])*100 for h in d[k]) if k in d else float('nan')
    print(f"{task}_a{alpha}: stdBCE {m('FedAvg'):.1f}/{m('FedND-std'):.1f}  balBCE {m('BalBCE'):.1f}/{m('FedNTD-BCE'):.1f}  (no-distill/+distill)")
print("FS_DISENTANGLE_DONE",flush=True)
