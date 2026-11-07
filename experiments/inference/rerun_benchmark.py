"""Run the unchanged measured benchmark in a separate result directory."""
import argparse, importlib.util, pathlib, sys
root=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--output',default='results/inference_rerun')
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--tokens',type=int,default=64)
a=p.parse_args()
out=pathlib.Path(a.output)
if not out.is_absolute():out=root/out
if (out/'records.jsonl').exists():raise FileExistsError('Choose a new output directory to preserve prior records: '+str(out))
out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('measured_benchmark',root/'experiments/inference/benchmark.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=out
sys.argv=[str(root/'experiments/inference/benchmark.py'),'--repeats',str(a.repeats),'--tokens',str(a.tokens)]
m.main()
