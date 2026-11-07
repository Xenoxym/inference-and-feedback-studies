"""Exercise the pinned author's actual nucleus verification function on a two-branch toy tree."""
import sys,types,importlib.util,pathlib,json,math
import torch
root=pathlib.Path(__file__).resolve().parents[2]
sys.modules['draftretriever']=types.ModuleType('draftretriever')
path=root/'sources/Adaptix/aded/model/utils.py'
spec=importlib.util.spec_from_file_location('adaptix_utils',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1);torch.manual_seed(20261106)
results=[];n=30000
# Both branch logits at their shared root represent exactly the same target p.
# If neither draft is accepted, the published update keeps root logits; the next
# outer iteration samples the next root token anew. We include that emitted token.
for order in [(0,1),(1,0)]:
 counts=[0,0];accepted=[0,0];reject=0
 logits=torch.tensor([[[math.log(.8),math.log(.2)],[0.,0.]],[[math.log(.8),math.log(.2)],[0.,0.]]])
 candidates=torch.tensor([[0,order[0]],[0,order[1]]])
 for _ in range(n):
  best,a=m.evaluate_posterior(logits.clone(),candidates,temperature=1.,top_p=.999)
  if int(a):
   x=int(candidates[int(best),1]);accepted[x]+=1
  else:
   reject+=1;x=int(torch.multinomial(torch.tensor([.8,.2]),1))
  counts[x]+=1
 expected_A=.928 if order==(0,1) else .768
 results.append({'candidate_order':order,'n':n,'counts':counts,'empirical_pA':counts[0]/n,'target_pA':.8,'derived_pA':expected_A,'standard_error':math.sqrt(expected_A*(1-expected_A)/n),'accepted_counts':accepted,'rejected_both':reject})
# Exact mathematical calculation: accept first matching independently sampled row;
# a miss triggers a new target sample, not a residual correction.
# This unit test does not execute full model/tree masking and says nothing about greedy branch speed.
report={'code_commit':'241dad836fd71aa295a3967f7efe4cfcd2dd9609','torch':torch.__version__,'seed':20261106,'function':'evaluate_posterior -> get_nucleus_posterior_mask','fallback_entry':'llm_judge/gen_model_answer_aded.py aded_forward passes top_p and temperature','results':results,'limitation':'A constructed two-branch case exercising actual published verifier; not full-model corpus distribution test and not proof about every version.'}
p=root/'results/inference/sampling_audit.json';p.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
