import json,os
from fedprior.algorithms import METHODS, federated
from experiments.pretrained_gate import _cfg,_build,ROUNDS
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"full")
for task in ("voc","coco"):
    for alpha in (0.1,0.5):
        tag=f"pt_{task}_a{alpha}"; p=os.path.join(OUT,f"results_{tag}.json")
        per=json.load(open(p)) if os.path.exists(p) else {}
        hs=per.get("FedAwS",[])
        for si,seed in enumerate((0,1,2)):
            if si<len(hs): continue
            cfg=_cfg(task,seed); cl,gc,tc=_build(task,alpha,seed)
            _,h=federated(METHODS["FedAwS"],ROUNDS,cl,gc,tc,cfg)
            hs.append(h); per["FedAwS"]=hs; json.dump(per,open(p,"w"))
            print(f"[{tag} s{seed}] FedAwS mAP={h['test_acc'][-1]:.4f}",flush=True)
print("FEDAWS_PT_DONE",flush=True)
