"""Auditable CPU greedy decoding benchmark; not an Adaptix reproduction.
All methods share one target, float32 SDPA attention, DynamicCache, batch=1.
Proposal: deterministic trigram frequency table; only committed histories.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, collections, copy, hashlib, json, pathlib, platform, random, statistics, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'results/inference';OUT.mkdir(parents=True,exist_ok=True)

def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def add(table,seq,start=2):
    for i in range(max(2,start),len(seq)):
        table.setdefault((seq[i-2],seq[i-1]),collections.Counter())[seq[i]]+=1

def propose(table,seq,budget):
    out=[];a,b=seq[-2:]
    for _ in range(budget):
        c=table.get((a,b))
        if not c: break
        t=min(c,key=lambda t:(-c[t],t));out.append(t);a,b=b,t
    return out

def synthetic_replay():
    rng=random.Random(20261106)
    train=[[1,2,3,4]*32,[10,11,12,13]*32]
    cases={'repetitive':[1,2,3,4]*40,'shift':[1,2,3,4]*20+[1,2,8,9]*20,'novel':[rng.randrange(100,140) for _ in range(160)]}
    rows=[];rounds=[]
    for scope in ['prompt_only','request','cross']:
      for name,trace in cases.items():
       for policy in ['ar','fixed4','fixed8','adaptive']:
        table={}
        if scope=='cross':
          for s in train:add(table,s)
        seq=trace[:8];add(table,seq);pos=8;budget=4
        while pos<len(trace):
          b=0 if policy=='ar' else (budget if policy=='adaptive' else int(policy[5:]))
          d=propose(table,seq+[trace[pos]],min(b,len(trace)-pos-1))
          a=0
          for x,y in zip(d,trace[pos+1:]):
            if x!=y:break
            a+=1
          committed=trace[pos:pos+a+1];old=len(seq);seq+=committed
          if scope!='prompt_only' and policy!='ar':add(table,seq,old)
          # Illustrative cost units ONLY. Not seconds, GPU time or measured speedup.
          cost=1+0.12*len(d)+(0.03*len(d) if b else 0)
          rounds.append(dict(scope=scope,case=name,policy=policy,position=pos,budget=b,draft=d,accepted=a,committed=committed,cost_units=cost))
          pos+=a+1
          if policy=='adaptive':budget=min(8,budget+2) if len(d)==b and a==len(d) else max(1,budget-1)
        rr=[x for x in rounds if x['scope']==scope and x['case']==name and x['policy']==policy]
        assert seq==trace
        rows.append(dict(scope=scope,case=name,policy=policy,rounds=len(rr),tokens=152,accepted=sum(x['accepted'] for x in rr),cost_units=sum(x['cost_units'] for x in rr)))
    save('synthetic_replay.json',{'seed':20261106,'traces':cases,'prior':train,'rows':rows,'rounds':rounds,'warning':'Cost units are assumptions; no speed claim.'})

PROMPTS=[
('prose','The library opens at nine each morning. Visitors come to read books, study, and use the computers. The librarian explained that'),
('science','A simple experiment compares two metal rods of the same length. One rod is heated while the other remains at room temperature. The result shows'),
('code','def factorial(n):\n    """Return the factorial of a nonnegative integer."""\n    if n == 0:\n        return 1\n    else:\n'),
('dialogue','Question: Why do leaves change color in autumn?\nAnswer:'),
('structured','Record 1: status=ready, action=continue.\nRecord 2: status=ready, action=continue.\nRecord 3: status=ready, action=continue.\nRecord 4:'),
('repetition','red blue green red blue green red blue green red blue green red blue green')]
PRIOR_PROMPTS=['The museum has an exhibition about ancient pottery. A visitor asked the guide','def square(x):\n    """Return the square of x."""\n','Item A: checked.\nItem B: checked.\nItem C:']

def main():
 p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=5);p.add_argument('--tokens',type=int,default=96);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 synthetic_replay();print('Replay complete',flush=True)
 ti=time.perf_counter();import torch;import transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261106)
 import_seconds=time.perf_counter()-ti;print('Imports',import_seconds,flush=True)
 model_dir=ROOT/'sources/SmolLM2-135M';ti=time.perf_counter()
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').eval()
 load_seconds=time.perf_counter()-ti;print('Model loaded',load_seconds,flush=True)
 meta={'date':'2026-11-06','python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'platform':platform.platform(),'processor':platform.processor(),'torch_threads':4,'interop_threads':1,'device':'cpu','dtype':'float32','attention':'sdpa','batch':1,'kv':'DynamicCache; crop rejected suffix; reset each request','sampling':'argmax, no penalties; stop EOS or max_new_tokens','model':'HuggingFaceTB/SmolLM2-135M','revision':(model_dir/'revision.txt').read_text(),'weights_sha256':hashlib.sha256((model_dir/'model.safetensors').read_bytes()).hexdigest(),'import_seconds':import_seconds,'model_load_seconds':load_seconds,'repeats':args.repeats,'max_new_tokens':args.tokens,'prior_prompts':PRIOR_PROMPTS,'prompts':PROMPTS,'time_definition':'warm_request_s: table initialization + prefill + decode; excludes tokenizer, imports, load, serialization. decode_s includes all Python control. Stages measured using perf_counter; no accelerator async on CPU.'}
 @torch.inference_mode()
 def decode(ids,policy,scope,base,max_tokens):
    tr=time.perf_counter();table=copy.deepcopy(base) if scope=='cross' and policy!='ar' else {}
    if policy!='ar':add(table,ids)
    setup=time.perf_counter()-tr;seq=list(ids);initial=len(seq);cache=DynamicCache(config=model.config)
    tp=time.perf_counter();out=model(torch.tensor([seq]),past_key_values=cache,use_cache=True);logits=out.logits[:,-1,:];prefill=time.perf_counter()-tp
    td=time.perf_counter();rr=[];budget=4
    while len(seq)-initial<max_tokens:
      begin=time.perf_counter();old=len(seq);remain=max_tokens-(old-initial)
      t=time.perf_counter();anchor=int(logits.argmax(-1));b=0 if policy=='ar' else (budget if policy=='adaptive' else int(policy[5:]));draft=propose(table,seq+[anchor],min(b,remain-1));draft_s=time.perf_counter()-t
      # EOS/final singleton needs no unused target forward; all policies use this rule.
      if anchor==tok.eos_token_id or remain==1:
        commit=[anchor];seq.extend(commit)
        rr.append({'budget':b,'draft':draft,'accepted':0,'committed':commit,'draft_s':draft_s,'verify_s':0.,'commit_s':time.perf_counter()-t-draft_s,'rollback_s':0.,'round_s':time.perf_counter()-begin,'discarded':0,'failed':False,'final_shortcut':True});break
      t=time.perf_counter();out=model(torch.tensor([[anchor]+draft]),past_key_values=cache,use_cache=True);verify_s=time.perf_counter()-t
      t=time.perf_counter();pred=out.logits[0,:len(draft),:].argmax(-1).tolist();a=0
      for x,y in zip(draft,pred):
        if x!=y:break
        a+=1
        if x==tok.eos_token_id:break
      commit=[anchor]+draft[:a];seq.extend(commit);logits=out.logits[:,a,:]
      if scope!='prompt_only' and policy!='ar':add(table,seq,old)
      if policy=='adaptive':budget=min(8,budget+2) if len(draft)==b and a==len(draft) else max(1,budget-1)
      commit_s=time.perf_counter()-t;t=time.perf_counter();cache.crop(len(seq));rollback_s=time.perf_counter()-t
      rr.append({'budget':b,'draft':draft,'accepted':a,'committed':commit,'draft_s':draft_s,'verify_s':verify_s,'commit_s':commit_s,'rollback_s':rollback_s,'round_s':time.perf_counter()-begin,'discarded':len(draft)-a,'failed':a<len(draft),'final_shortcut':False})
      if tok.eos_token_id in commit:break
    decode_s=time.perf_counter()-td;total=time.perf_counter()-tr
    return {'output_ids':seq[initial:],'output_text':tok.decode(seq[initial:]),'setup_s':setup,'prefill_s':prefill,'decode_s':decode_s,'warm_request_s':total,'rounds':rr}
 ids=tok('A short warmup sentence about science.',add_special_tokens=True).input_ids
 cold=decode(ids,'ar','request',{},16);save('first_request.json',cold)
 for _ in range(2):decode(ids,'fixed4','request',{},16)
 # External disjoint prior, fixed before evaluation; no test answers stored in draft table.
 prior={};prior_records=[]
 for text in PRIOR_PROMPTS:
    a=tok(text,add_special_tokens=True).input_ids;r=decode(a,'ar','request',{},64);add(prior,a+r['output_ids']);prior_records.append({'prompt':text,**r})
 save('prior_records.json',prior_records);meta['prior_entries']=len(prior)
 save('metadata.json',meta)
 prompts=PROMPTS[:1] if args.smoke else PROMPTS
 methods=['ar','fixed4','fixed8','adaptive'];scopes=['prompt_only','request','cross'];records=[];rng=random.Random(731)
 # A separate per-prompt AR reference and HF generation check; never given to proposal.
 references={}
 for name,text in prompts:
    a=tok(text,add_special_tokens=True).input_ids;r=decode(a,'ar','request',{},args.tokens);references[name]=r['output_ids']
    g=model.generate(torch.tensor([a]),attention_mask=torch.ones((1,len(a)),dtype=torch.long),max_new_tokens=args.tokens,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id).tolist()[0][len(a):]
    if g!=r['output_ids']:raise AssertionError('Reference differs from HF generate: '+name)
 save('greedy_references.json',references)
 log=OUT/('smoke_records.jsonl' if args.smoke else 'records.jsonl')
 with log.open('w',encoding='utf-8') as f:
  for rep in range(args.repeats):
   jobs=[(name,text,scope,method) for name,text in prompts for scope in scopes for method in methods];rng.shuffle(jobs)
   for name,text,scope,method in jobs:
    a=tok(text,add_special_tokens=True).input_ids;r=decode(a,method,scope,prior,args.tokens)
    r.update(prompt_id=name,scope=scope,method=method,repeat=rep,consistent=r['output_ids']==references[name],input_tokens=len(a),output_tokens=len(r['output_ids']))
    f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();records.append(r)
    if not r['consistent']:raise AssertionError(f'Greedy mismatch {name} {scope} {method}')
   print('Repeat',rep+1,'complete',len(records),'requests',flush=True)
 summary=[]
 for scope in scopes:
  for method in methods:
   subset=[r for r in records if r['scope']==scope and r['method']==method];byrep=[sum(r['warm_request_s'] for r in subset if r['repeat']==i) for i in range(args.repeats)]
   tokens=sum(r['output_tokens'] for r in subset);rounds=[x for r in subset for x in r['rounds']];base=[sum(r['warm_request_s'] for r in records if r['scope']==scope and r['method']=='ar' and r['repeat']==i) for i in range(args.repeats)]
   summary.append({'scope':scope,'method':method,'requests':len(subset),'output_tokens':tokens,'median_total_s_per_repeat':statistics.median(byrep),'min_total_s':min(byrep),'max_total_s':max(byrep),'paired_speedups':[a/b for a,b in zip(base,byrep)],'median_speedup':statistics.median(a/b for a,b in zip(base,byrep)),'accepted_per_round':sum(x['accepted'] for x in rounds)/len(rounds),'committed_per_round':tokens/len(rounds),'draft_tokens':sum(len(x['draft']) for x in rounds),'accepted_draft_tokens':sum(x['accepted'] for x in rounds),'discarded_draft_tokens':sum(x['discarded'] for x in rounds),'failed_rounds':sum(x['failed'] for x in rounds),'verify_s':sum(x['verify_s'] for x in rounds),'failed_round_verify_s':sum(x['verify_s'] for x in rounds if x['failed']),'draft_s':sum(x['draft_s'] for x in rounds),'commit_s':sum(x['commit_s'] for x in rounds),'rollback_s':sum(x['rollback_s'] for x in rounds),'all_consistent':all(r['consistent'] for r in subset)})
 save('smoke_summary.json' if args.smoke else 'summary.json',summary)
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
