"""Restricted feedback experiment, NOT a visual or real-driving benchmark.
Run: python run_feedback.py --output ../../results/driving_feedback
All grids fixed before the first run; no condition selection based on results.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np

CONFIG = {
    "experiment": "linear_lane_feedback_v1",
    "seeds": [11, 22, 33, 44, 55],
    "coverage": ["centered", "narrow", "broad"],
    "label_noise_std_rad": [0.0, 0.01],
    "methods": ["bc", "augmentation", "dagger", "expert"],
    "test_scenarios": ["nominal", "initial_offset", "kick", "delay_100ms", "delay_300ms"],
    "train_episodes": 30, "train_steps": 200, "dev_episodes": 12,
    "test_episodes": 12, "test_steps": 300,
    "dt_s": 0.05, "speed_mps": 8.0, "wheelbase_m": 2.7,
    "steering_limit_rad": 0.5, "lane_half_width_m": 1.75,
    "expert_lateral_gain": 0.25, "expert_heading_gain": 0.8,
    "feature_scales": [1.0, 1.0, 0.2, 0.03],
    "ridge_grid": [0.0, 1e-6, 1e-4, 1e-2],
    "dagger_initial_episodes": 10, "dagger_round_episodes": [10, 10],
    "dagger_teacher_probability": [0.5, 0.0],
    "augmentation_extra_labels": 4000,
    "augmentation_uniform_half_width": [0.8, 0.12],
    "recovery_threshold": [0.15, 0.03], "recovery_consecutive_steps": 20,
    "split_note": "Separate entire episodes and disjoint seed namespaces for train/dev/test; no test tuning.",
    "label_note": "Oracle controls are clean; demonstration target noise is added only for training labels.",
    "failure_note": "Crossing |lateral error|>1.75 m is recorded. Mathematical rollouts continue without reset; this is not a physical crash simulation.",
}
DT, V, L = CONFIG["dt_s"], CONFIG["speed_mps"], CONFIG["wheelbase_m"]
LIMIT = CONFIG["steering_limit_rad"]
SCALE = np.array(CONFIG["feature_scales"])
TRUE_W = np.array([0., -.25, -.8*.2, L*.03])

def features(state, curvature):
    return np.column_stack((np.ones(len(state)), state[:,0], state[:,1]/.2, curvature/.03))

def expert_action(x):
    return np.clip(x @ TRUE_W, -LIMIT, LIMIT)

def make_env(seed, n, steps, coverage, scenario="training"):
    rng = np.random.default_rng(seed)
    times = np.arange(steps)*DT
    phase = rng.uniform(-np.pi, np.pi, (n,1))
    freq = rng.uniform(.15, .5, (n,1))
    amp = rng.uniform(.005, .025, (n,1))
    curvature = amp*np.sin(freq*times+phase) + .003*np.sin(.9*times+2*phase)
    init = np.zeros((n,2))
    disturbance = np.zeros((n,steps,2))
    kick = np.zeros((n,steps,2))
    delay = 0
    if scenario == "training":
        if coverage == "narrow":
            init = rng.normal(size=(n,2))*[.02,.002]
            disturbance = rng.normal(size=(n,steps,2))*[.0005,.00005]
        elif coverage == "broad":
            init = rng.normal(size=(n,2))*[.7,.08]
            disturbance = rng.normal(size=(n,steps,2))*[.005,.0005]
    elif scenario == "nominal":
        disturbance = rng.normal(size=(n,steps,2))*[.0005,.00005]
    elif scenario in ("initial_offset", "delay_100ms", "delay_300ms"):
        init = rng.uniform(-1,1,(n,2))*[1.2,.12]
        disturbance = rng.normal(size=(n,steps,2))*[.005,.0005]
        delay = {"initial_offset":0,"delay_100ms":2,"delay_300ms":6}[scenario]
    elif scenario == "kick":
        disturbance = rng.normal(size=(n,steps,2))*[.005,.0005]
        sign = rng.choice([-1.,1.],size=(n,))
        kick[:,100,:] = sign[:,None]*[.8,.05]
    else:
        raise ValueError(scenario)
    return dict(curvature=curvature, initial=init, disturbance=disturbance, kick=kick, delay=np.array(delay))

def subset_env(env, start, end):
    return {k: (v[start:end].copy() if v.ndim else v.copy()) for k,v in env.items()}

def rollout(env, weights=None, teacher_probability=0., mix_seed=0):
    curvature = env["curvature"]
    n, steps = curvature.shape
    state = env["initial"].copy()
    states = np.zeros((n,steps,2))
    xall = np.zeros((n,steps,4))
    proposed = np.zeros((n,steps))
    executed = np.zeros((n,steps))
    labels = np.zeros((n,steps))
    teacher_mask = np.zeros((n,steps),dtype=bool)
    rng = np.random.default_rng(mix_seed)
    delay = int(env["delay"])
    for t in range(steps):
        state = state + env["kick"][:,t,:]
        x = features(state,curvature[:,t])
        label = expert_action(x)
        action = label.copy() if weights is None else np.clip(x@weights,-LIMIT,LIMIT)
        if teacher_probability:
            use_teacher = rng.random(n) < teacher_probability
            action = np.where(use_teacher,label,action)
            teacher_mask[:,t] = use_teacher
        states[:,t] = state
        xall[:,t] = x
        labels[:,t] = label
        proposed[:,t] = action
        actual = proposed[:,t-delay] if t>=delay else np.zeros(n)
        executed[:,t] = actual
        old_heading = state[:,1].copy()
        state[:,0] += V*DT*old_heading
        state[:,1] += (V*DT/L)*actual - V*DT*curvature[:,t]
        state += env["disturbance"][:,t,:]
    if not all(np.isfinite(a).all() for a in (states,xall,proposed,executed)):
        raise FloatingPointError("Non-finite rollout; retain and diagnose instead of dropping.")
    return dict(states=states, features=xall, oracle_action=labels, proposed_action=proposed,
                executed_action=executed, teacher_mask=teacher_mask)

def flatten_data(roll):
    return roll["features"].reshape(-1,4), roll["oracle_action"].reshape(-1)

def fit_policy(x,y,dx,dy):
    candidates=[]
    for ridge in CONFIG["ridge_grid"]:
        if ridge==0:
            w=np.linalg.lstsq(x,y,rcond=1e-12)[0]
        else:
            regularizer=np.diag([0.,1.,1.,1.])
            w=np.linalg.solve(x.T@x/len(x)+ridge*regularizer,x.T@y/len(x))
        score=float(np.mean((np.clip(dx@w,-LIMIT,LIMIT)-dy)**2))
        candidates.append((score,ridge,w))
    score,ridge,w=min(candidates,key=lambda z:(z[0],z[1]))
    singular=np.linalg.svd(x,compute_uv=False)
    rank=int(np.linalg.matrix_rank(x,tol=singular[0]*1e-12))
    # Unsaturated zero-delay local linearization; not a global stability certificate.
    ky=w[1]; kp=w[2]/.2
    a=np.array([[1,V*DT],[V*DT/L*ky,1+V*DT/L*kp]])
    meta=dict(weights=w.tolist(),ridge=ridge,dev_clean_mse=score,rank=rank,
              singular_values=singular.tolist(),local_zero_delay_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(a)))),
              candidate_dev_mse=[dict(ridge=r,mse=s) for s,r,_ in candidates],labels=len(x))
    return w,meta

def metrics(roll, scenario):
    n,steps=roll["proposed_action"].shape
    results=[]
    start=100 if scenario=="kick" else 0
    for ep in range(n):
        e=roll["states"][ep,:,0]; h=roll["states"][ep,:,1]
        exits=np.flatnonzero(np.abs(e)>CONFIG["lane_half_width_m"])
        stable=(np.abs(e)<.15)&(np.abs(h)<.03)
        hit=np.flatnonzero(np.convolve(stable[start:].astype(int),np.ones(20,dtype=int),mode="valid")==20)
        recovery=(float(hit[0]*DT) if len(hit) else None)
        valid_until=int(exits[0])+1 if len(exits) else steps
        results.append(dict(episode=ep,failed=int(len(exits)>0),first_exit_s=float(exits[0]*DT) if len(exits) else None,
                            rmse_y_full=float(np.sqrt(np.mean(e*e))),rmse_y_before_exit=float(np.sqrt(np.mean(e[:valid_until]**2))),
                            max_abs_y=float(np.max(np.abs(e))),recovery_s=recovery,
                            recovery_capped_s=recovery if recovery is not None else (steps-start)*DT,
                            recovered=int(recovery is not None),
                            action_variation=float(np.mean(np.abs(np.diff(roll["executed_action"][ep]))))))
    return results

def selfcheck():
    rng=np.random.default_rng(345)
    state=rng.uniform(-1,1,(100,2))*[.4,.04]
    k=rng.uniform(-.02,.02,100)
    x=features(state,k); y=expert_action(x)
    w,_=fit_policy(x,y,x,y)
    assert np.max(np.abs(w-TRUE_W))<1e-10
    env=make_env(99,3,100,"centered")
    roll=rollout(env)
    assert np.max(np.abs(roll["states"]))<1e-12
    delayed={**env,"delay":np.array(2)}
    rr=rollout(delayed)
    assert np.all(rr["executed_action"][:,:2]==0.)
    assert np.allclose(rr["executed_action"][:,2:],rr["proposed_action"][:,:-2])
    assert np.linalg.matrix_rank(flatten_data(roll)[0],tol=1e-9)==2
    return "PASS: coefficient recovery, centered oracle invariance, FIFO time alignment, rank deficiency."

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    (out/"raw").mkdir(exist_ok=True); (out/"training").mkdir(exist_ok=True)
    (out/"config.json").write_text(json.dumps(CONFIG,indent=2),encoding="utf-8")
    t0=time.perf_counter()
    checks=selfcheck()
    rows=[]; models=[]
    for seed in CONFIG["seeds"]:
      for ci,coverage in enumerate(CONFIG["coverage"]):
        # Same test environments across coverage and label-noise regimes.
        train_env=make_env(100000+seed,30,200,coverage)
        train_oracle=rollout(train_env)
        train_x,train_clean_y=flatten_data(train_oracle)
        dev=rollout(make_env(200000+seed,12,200,coverage))
        dx,dy=flatten_data(dev)
        for noise in CONFIG["label_noise_std_rad"]:
          key=f"seed{seed}_{coverage}_noise{noise:g}"
          nrng=np.random.default_rng(300000+seed)
          yn=train_clean_y+nrng.normal(0,noise,len(train_clean_y))
          all_data={"bc":(train_x,yn)}
          x0=train_oracle["features"][:10].reshape(-1,4)
          y0=train_oracle["oracle_action"][:10].reshape(-1)
          # The same initial noisy labels are shared across methods.
          y0n=yn[:len(y0)]
          arng=np.random.default_rng(400000+seed)
          indices=arng.integers(len(x0),size=4000)
          ax=x0[indices].copy()
          ax[:,1]+=arng.uniform(-.8,.8,4000)
          ax[:,2]+=arng.uniform(-.12,.12,4000)/.2
          ay=expert_action(ax)+arng.normal(0,noise,4000)
          all_data["augmentation"]=(np.concatenate((x0,ax)),np.concatenate((y0n,ay)))
          dagger_x=x0.copy(); dagger_y=y0n.copy()
          dagger_rounds=[]
          for ri,beta in enumerate(CONFIG["dagger_teacher_probability"]):
              interim,_=fit_policy(dagger_x,dagger_y,dx,dy)
              env_part=subset_env(train_env,10+ri*10,20+ri*10)
              generated=rollout(env_part,interim,beta,mix_seed=500000+seed+ri)
              gx,gy=flatten_data(generated)
              gy+=np.random.default_rng(600000+seed+ri).normal(0,noise,len(gy))
              dagger_x=np.concatenate((dagger_x,gx)); dagger_y=np.concatenate((dagger_y,gy))
              dagger_rounds.append(generated)
          all_data["dagger"]=(dagger_x,dagger_y)
          for method in CONFIG["methods"]:
            if method=="expert":
                weights=None
                meta=dict(weights=TRUE_W.tolist(),ridge=None,dev_clean_mse=0.,rank=None,labels=0,
                          local_zero_delay_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(
                           [[1,V*DT],[-V*DT/L*.25,1-V*DT/L*.8]])))))
            else:
                x,y=all_data[method]
                assert len(y)==6000
                weights,meta=fit_policy(x,y,dx,dy)
                np.savez_compressed(out/"training"/f"{key}_{method}.npz",features=x,training_labels=y,
                                    validation_features=dx,validation_labels=dy)
            meta.update(seed=seed,coverage=coverage,label_noise_std=noise,method=method)
            models.append(meta)
            for scenario in CONFIG["test_scenarios"]:
              # The same seed regenerates identical initial state/noise/roads for every method.
              si=CONFIG["test_scenarios"].index(scenario)
              env=make_env(700000+seed+si*1000,12,300,coverage,scenario)
              roll=rollout(env,weights)
              filename=f"{key}_{method}_{scenario}.npz"
              np.savez_compressed(out/"raw"/filename,**roll,**env)
              for row in metrics(roll,scenario):
                  row.update(seed=seed,coverage=coverage,label_noise_std=noise,method=method,scenario=scenario,
                             dev_clean_mse=meta["dev_clean_mse"],raw_file="raw/"+filename)
                  rows.append(row)
          print(f"completed {key}",flush=True)
    with (out/"trajectory_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/"models.json").write_text(json.dumps(models,indent=2),encoding="utf-8")
    grouped=[]
    for coverage in CONFIG["coverage"]:
      for noise in CONFIG["label_noise_std_rad"]:
       for method in CONFIG["methods"]:
        for scenario in CONFIG["test_scenarios"]:
          rr=[r for r in rows if r["coverage"]==coverage and r["label_noise_std"]==noise and r["method"]==method and r["scenario"]==scenario]
          per_seed=[]
          for seed in CONFIG["seeds"]:
              ss=[r for r in rr if r["seed"]==seed]
              per_seed.append(np.mean([r["failed"] for r in ss]))
          grouped.append(dict(coverage=coverage,label_noise_std=noise,method=method,scenario=scenario,
                              n_trajectories=len(rr),failure_rate=float(np.mean([r["failed"] for r in rr])),
                              failure_seed_std=float(np.std(per_seed,ddof=1)),
                              rmse_y_full=float(np.mean([r["rmse_y_full"] for r in rr])),
                              mean_recovery_capped_s=float(np.mean([r["recovery_capped_s"] for r in rr])),
                              recovery_rate=float(np.mean([r["recovered"] for r in rr])),
                              dev_clean_mse=float(np.mean([r["dev_clean_mse"] for r in rr]))))
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(grouped[0]));writer.writeheader();writer.writerows(grouped)
    manifest=dict(config=CONFIG,checks=checks,python=platform.python_version(),numpy=np.__version__,
                  platform=platform.platform(),run_seconds=time.perf_counter()-t0,
                  total_trajectories=len(rows),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation="Synthetic low-dimensional feedback and data-coverage validation only. No real sensor, CARLA, or real vehicle results.")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("checks","run_seconds","total_trajectories","script_sha256")}),flush=True)

if __name__=="__main__":
    main()
