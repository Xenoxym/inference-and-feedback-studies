"""Audit saved arrays and explain delayed expert stability. Does not train or tune."""
from pathlib import Path
import argparse, csv, hashlib, json, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import run_feedback as experiment

def expert_matrix(delay):
    h=experiment.V*experiment.DT
    a=np.zeros((2+delay,2+delay))
    a[0,0]=1.; a[0,1]=h; a[1,1]=1.
    gains=np.array([-.25,-.8])
    if delay:
        a[1,2]=h/experiment.L
        if delay>1: a[2:1+delay,3:2+delay]=np.eye(delay-1)
        a[-1,:2]=gains
    else:
        a[1,:2]+=h/experiment.L*gains
    return a

def main():
    p=argparse.ArgumentParser();p.add_argument("--results",required=True);a=p.parse_args()
    folder=Path(a.results)
    with (folder/"trajectory_metrics.csv").open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    with (folder/"summary.csv").open(encoding="utf-8",newline="") as f: summary=list(csv.DictReader(f))
    assert len(rows)==7200 and len(summary)==120
    by_file={}
    for row in rows: by_file.setdefault(row["raw_file"],[]).append(row)
    env_fingerprints=set()
    comparisons=0
    for rel, rr in by_file.items():
        with np.load(folder/rel,allow_pickle=False) as dat:
            result=experiment.metrics(dat,rr[0]["scenario"])
            assert len(result)==len(rr)==12
            env_hash=hashlib.sha256()
            for k in ["curvature","initial","disturbance","kick","delay"]:
                env_hash.update(dat[k].tobytes())
            env_fingerprints.add(env_hash.hexdigest())
            delay=int(dat["delay"])
            if delay:
                assert np.all(dat["executed_action"][:,:delay]==0)
                assert np.allclose(dat["executed_action"][:,delay:],dat["proposed_action"][:,:-delay])
            else: assert np.array_equal(dat["executed_action"],dat["proposed_action"])
            for saved,recalc in zip(sorted(rr,key=lambda r:int(r["episode"])),result):
                for k,v in recalc.items():
                    if v is None: assert saved[k]==""
                    else: assert np.isclose(float(saved[k]),float(v),rtol=1e-12,atol=1e-12),(rel,k,saved[k],v)
                    comparisons+=1
    for s in summary:
        rr=[r for r in rows if all(r[k]==s[k] for k in ["coverage","label_noise_std","method","scenario"])]
        assert len(rr)==60
        for sk,rk in [("failure_rate","failed"),("rmse_y_full","rmse_y_full"),("mean_recovery_capped_s","recovery_capped_s"),("recovery_rate","recovered"),("dev_clean_mse","dev_clean_mse")]:
            assert np.isclose(float(s[sk]),np.mean([float(r[rk]) for r in rr]),rtol=1e-12,atol=1e-12)
    train_files=list((folder/"training").glob("*.npz"))
    assert len(train_files)==90
    for file in train_files:
        with np.load(file,allow_pickle=False) as dat:
            assert dat["features"].shape==(6000,4)
            assert dat["training_labels"].shape==(6000,)
            assert dat["validation_features"].shape==(2400,4)
    assert len(by_file)==600 and len(env_fingerprints)==25
    stability=[]
    for d in [0,2,6]:
        matrix=expert_matrix(d)
        eigen=np.linalg.eigvals(matrix)
        stability.append({"delay_steps":d,"delay_ms":d*50,"matrix":matrix.tolist(),
            "spectral_radius":float(np.max(np.abs(eigen))),
            "eigenvalues":[{"real":float(x.real),"imag":float(x.imag)} for x in eigen]})
    result={"status":"PASS","rows":len(rows),"summary_rows":len(summary),"test_npz_files":len(by_file),
        "training_npz_files":len(train_files),"metric_values_recomputed":comparisons,
        "distinct_environment_batches":len(env_fingerprints),"distinct_test_episodes":12*len(env_fingerprints),
        "pairing":"Environments are identical across methods, training coverage and label-noise conditions for a seed/scenario. Different scenarios use different seeds, so delay contrasts are not paired road-by-road.",
        "uncertainty":"Five training/test seed groups per cell, twelve episodes each; no statistical-significance claim or independence across conditions.",
        "expert_local_stability":stability,
        "stability_scope":"Unsaturated, zero-curvature, disturbance-free local dynamics; spectral radius is not a global or real-vehicle safety certificate.",
        "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    methods={"bc":"BC","augmentation":"状态扰动增广","dagger":"DAgger","expert":"解析专家"}
    table=["| 训练覆盖 | 标签噪声(rad) | 方法 | 开环 MSE(rad²) | 常规 | 初始偏离 | 中途扰动 | 100ms延迟 | 300ms延迟 |",
           "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for coverage in experiment.CONFIG["coverage"]:
        for noise in experiment.CONFIG["label_noise_std_rad"]:
            for method in experiment.CONFIG["methods"]:
                ss=[s for s in summary if s["coverage"]==coverage and float(s["label_noise_std"])==noise and s["method"]==method]
                counts=[str(round(float(s["failure_rate"])*60))+"/60" for s in ss]
                table.append("| "+" | ".join([coverage,str(noise),methods[method],format(float(ss[0]["dev_clean_mse"]),".3g"),*counts])+" |")
    (folder/"all_failure_results.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status","rows","test_npz_files","training_npz_files","distinct_test_episodes"]}))
    print(json.dumps([{"delay_ms":s["delay_ms"],"spectral_radius":s["spectral_radius"]} for s in stability]))
if __name__=="__main__":main()
