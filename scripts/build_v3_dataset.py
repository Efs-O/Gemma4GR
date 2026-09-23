from __future__ import annotations
import argparse, hashlib, json, random, re, unicodedata, subprocess
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]; SEED=3407; MAX=2048
TOKENS={'km','kg','cm','mm','ml','mg','DNA','UNESCO','Wi-Fi','USB','AI','ΕΕ','ΗΠΑ','ΟΗΕ','ΦΠΑ','BBC','NASA','LED','PDF','GPS','SMS','CPU','URL','°C','%'}
GARB=re.compile(r'[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώΐΰ][+)(\[\]{}<>|\\^~=_*#@$]+[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώΐΰ]')
LAT=re.compile('[A-Za-z]'); OTHER=re.compile('[\u0400-\u052f\u0590-\u05ff\u0600-\u06ff\u0e00-\u0e7f\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
MOJI=('Ã','Â','Î','Ï','â€','ðŸ','�'); TERMINAL='.;!…»)'
PROBES=['Τι είναι το μελτέμι;','Πώς επηρεάζει το μελτέμι τα καλοκαιρινά ταξίδια;','Πες μου μια πρόταση με τη λέξη Μελτέμι.','Τι σημαίνει η λέξη τσούχτρα;','Πώς χρησιμοποιούμε τη λέξη τζαμπατζής;','Περιέγραψε ένα τσουρέκι που μόλις βγήκε από τον φούρνο.','Τι είναι το κυκλοφοριακό;','Πώς λειτουργεί ένας ανεμόμυλος;','Πες μου για το παλιό γεφύρι της Άρτας.','Τι γνωρίζεις για την Καστοριά;','Πώς είναι ένα πρωινό στην ορεινή Ναυπακτία;','Τι σημαίνει η λέξη μουρμούρα;','Πώς φτιάχνεται η σκορδαλιά;','Τι είναι το γαϊδουράγκαθο;','Πες μου για τη γειτονιά των Εξαρχείων.','Πώς περιγράφεις έναν καλοκαιρινό καύσωνα;','Τι σημαίνει η λέξη παραθαλάσσιος;','Πώς περνά μια οικογένεια την Καθαρά Δευτέρα;','Τι είναι το χρυσοπράσινο φύλλο;','Πες μου για την πλατεία της Αγίας Παρασκευής.','Πώς λέγεται το μικρό χταπόδι;','Τι σημαίνει η λέξη τζιτζίκι;','Πώς θα περιέγραφες ένα παλιό καλντερίμι;','Τι είναι το καραβάκι των Χριστουγέννων;','Πες μου για ένα πανηγύρι στην Ήπειρο.','Πώς φτιάχνουμε παραδοσιακό γαλακτομπούρεκο;','Τι σημαίνει η λέξη αστροφεγγιά;','Πώς είναι μια βόλτα στο Ναύπλιο;','Τι είναι το γλυκοχάραμα;','Πες μου μια πρόταση με τη λέξη τσουρουφλίζω.']
def H(b): return hashlib.sha256(b).hexdigest()
def norm(s): return ' '.join(unicodedata.normalize('NFC',s).split())
def parse(s):
 m=re.fullmatch(r'<bos><\|turn>system\n(.*?)<turn\|>\n<\|turn>user\n(.*?)<turn\|>\n<\|turn>model\n(.*?)<turn\|>\n?',s,re.S)
 return m.groups() if m else None
def row(line,i):
 try: x=json.loads(line); t=parse(x['text'])
 except Exception:return None
 return {'id':i,'system':t[0],'user':t[1],'answer':t[2],'text':x['text']} if t else None
@lru_cache(maxsize=30000)
def shingles(s): return {s[i:i+5] for i in range(max(0,len(s)-4))}
def near(a,b):
 sa=norm(a['user']+' '+a['answer']); sb=norm(b['user']+' '+b['answer'])
 x=shingles(sa);y=shingles(sb)
 if not x or not y:return False
 if min(len(x),len(y))/max(len(x),len(y)) < .85:return False
 return len(x&y)/len(x|y)>=.85
def signature(r): return shingles(norm(r['user']+' '+r['answer']))
def indexed_near(r,candidates,index):
 sx=signature(r)
 if not sx:return False
 # With Jaccard >= .85, and the exact |A|/|B| >= .85 bound, fewer
 # than 15% of either set can be absent from the intersection. Probe
 # slightly more than that many rare shingles to retain every true match.
 probe=sorted(sx,key=lambda sh:(len(index.get(sh,())),sh))[:max(1,int(.16*len(sx))+1)]
 ids=set().union(*(index.get(sh,set()) for sh in probe))
 return any(near(r,candidates[i]) for i in sorted(ids))
def index_add(r,i,index):
 for sh in signature(r):index[sh].add(i)
def fp(r): return norm(r['user'])+'\0'+norm(r['answer'])
def read(p): return p.read_bytes().decode('utf-8-sig').splitlines()
def write(p,rs):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8',newline='\n')
def info(r):return {'text':r['text']}
def flags(u,a,whitelist):
 s=u+' '+a; f=[]
 if OTHER.search(s):f.append('non_greek_script')
 rem=s
 for t in TOKENS:
  if re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)',s,re.I):whitelist[t]+=1
  rem=re.sub(r'(?<!\w)'+re.escape(t)+r'(?!\w)','',rem,flags=re.I)
 letters=[c for c in rem if c.isalpha()]; ratio=sum(bool(LAT.fullmatch(c)) for c in letters)/len(letters) if letters else 0
 if ratio>.2:f.append('latin_heavy')
 if GARB.search(s):f.append('garbage_in_word')
 if any(z in s for z in MOJI) or any((ord(c)<32 and c not in '\n\t') or 0xd800<=ord(c)<=0xdfff for c in s):f.append('mojibake')
 if not a.strip() or len(a.strip())<15:f.append('empty_or_truncated')
 return f,ratio
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=SEED);ap.add_argument('--train',default='data/train_qa.jsonl');ap.add_argument('--val',default='data/val_qa.jsonl');ap.add_argument('--combined',default='data/train_qa_combined.jsonl');ap.add_argument('--out',default='data/v3');args=ap.parse_args()
 out=(ROOT/args.out).resolve(); out.mkdir(parents=True,exist_ok=True);(out/'eval').mkdir(exist_ok=True)
 paths={k:(ROOT/v).resolve() for k,v in [('train',args.train),('val',args.val),('combined',args.combined)]}
 tp=Path('N:/.cache/huggingface/hub/gemma-4-E4B-it/tokenizer.json'); tok=Tokenizer.from_file(str(tp)); drops=defaultdict(lambda:defaultdict(list)); mods=Counter(); whitelist=Counter({t:0 for t in TOKENS}); mid=[]; stats={}
 def process(src,path):
  allr=[]
  for i,line in enumerate(read(path),1):
   r=row(line,i)
   if not r:drops[src]['template'].append(i);continue
   f,ratio=flags(r['user'],r['answer'],whitelist)
   if .1<=ratio<=.2:mid.append({'source':src,'id':i,'ratio':round(ratio,4),'user':r['user'][:160],'answer':r['answer'][:160]})
   u=unicodedata.normalize('NFC',r['user']);a=unicodedata.normalize('NFC',r['answer'])
   if (u,a)!=(r['user'],r['answer']):
    mods[src]+=1
    r['text']=f"<bos><|turn>system\n{r['system']}<turn|>\n<|turn>user\n{u}<turn|>\n<|turn>model\n{a}<turn|>\n"
   r['user'],r['answer']=u,a
   if f:drops[src][f[0]].append(i);continue
   allr.append(r)
  lens=[len(tok.encode(r['text']).ids) for r in allr]; sl=sorted(lens)
  stats[src]={'count':len(sl),'min':min(sl,default=0),'p50':sl[round((len(sl)-1)*.5)] if sl else 0,'p95':sl[round((len(sl)-1)*.95)] if sl else 0,'p99':sl[round((len(sl)-1)*.99)] if sl else 0,'max':max(sl,default=0),'over_2048':sum(x>MAX for x in sl)}
  pool=[]
  for r in allr:
   n=len(tok.encode(r['text']).ids)
   if n>MAX:drops[src]['too_long'].append(r['id']);continue
   pool.append(r)
  cutoff=sorted(len(tok.encode(r['text']).ids) for r in pool)[int(.9*(len(pool)-1))] if pool else 0; keep=[]
  for r in pool:
   if r['answer'].rstrip() and r['answer'].rstrip()[-1] not in TERMINAL and len(tok.encode(r['text']).ids)>=cutoff:drops[src]['empty_or_truncated'].append(r['id']);continue
   keep.append(r)
  seen=set();ded=[];near_index=defaultdict(set)
  for r in keep:
   if fp(r) in seen:drops[src]['exact_dup'].append(r['id']);continue
   if indexed_near(r,ded,near_index):drops[src]['near_dup'].append(r['id']);continue
   seen.add(fp(r));ded.append(r);index_add(r,len(ded)-1,near_index)
  return ded
 train=process('train',paths['train']); vp=process('val',paths['val']); combined=process('combined',paths['combined'])
 def overlap(a,b):
  out=[]
  for x in a:
   if any(fp(x)==fp(y) or norm(x['user'])==norm(y['user']) or near(x,y) for y in b):out.append(x)
  return out
 rng=random.Random(args.seed); rng.shuffle(vp); ev=vp[:150]; ids={x['id'] for x in ev}; val=[x for x in vp if x['id'] not in ids]
 removed={}
 bad=overlap(ev,train);removed['eval_train']=[x['id'] for x in bad];ev=[x for x in ev if x not in bad]
 bad=overlap(ev,val);removed['eval_val']=[x['id'] for x in bad];ev=[x for x in ev if x not in bad]
 bad=overlap(val,train);removed['val_train']=[x['id'] for x in bad];val=[x for x in val if x not in bad]
 write(out/'train.jsonl',[info(x) for x in train]);write(out/'val.jsonl',[info(x) for x in val]);write(out/'eval/text_eval.jsonl',[info(x) for x in ev]);write(out/'eval/garbage_probe.jsonl',[{'id':f'garbage_{i+1:02d}','prompt':p} for i,p in enumerate(PROBES)])
 def reread(p):return [row(x,i) for i,x in enumerate(read(p),1)]
 proof={'train_val':len(overlap(reread(out/'train.jsonl'),reread(out/'val.jsonl'))),'train_eval':len(overlap(reread(out/'train.jsonl'),reread(out/'eval/text_eval.jsonl'))),'val_eval':len(overlap(reread(out/'val.jsonl'),reread(out/'eval/text_eval.jsonl')))}
 trainfp={fp(x) for x in train}; voice=[x for x in combined if fp(x) not in trainfp];pc=Counter(norm(x['user']) for x in voice);used=Counter();cand=[]
 reference=train+val+ev;ref_index=defaultdict(set)
 for i,y in enumerate(reference):index_add(y,i,ref_index)
 cand_index=defaultdict(set)
 for x in voice:
  p=norm(x['user'])
  if used[p]>=10 or indexed_near(x,reference,ref_index) or indexed_near(x,[{'user':z['user'],'answer':z['answer']} for z in cand],cand_index):continue
  used[p]+=1;cand.append(x);index_add(x,len(cand)-1,cand_index)
  if len(cand)==300:break
 write(out/'candidates_voice_dedup.jsonl',[info(x) for x in cand])
 # Audio fallback with available files and train overlap check.
 apath=ROOT/'data/val_stt_final.jsonl'; stpath=ROOT/'data/train_stt_final.jsonl'; raw=read(apath);tr=read(stpath); trainaudio=set();traintext=set()
 trainbases=set()
 for line in tr:
  try:
   for m in json.loads(line).get('messages',[]):
    for c in m.get('content',[]) if isinstance(m.get('content'),list) else []:
     if c.get('type')=='audio':
      trainbases.add(Path(c.get('audio','')).name.casefold())
     if c.get('type')=='text' and m.get('role')=='assistant':traintext.add(norm(c.get('text','')))
  except Exception:pass
 aud=[];missing=[]
 for i,line in enumerate(raw,1):
  try:x=json.loads(line)
  except Exception:continue
  cont=[c for m in x.get('messages',[]) if m.get('role')=='user' for c in m.get('content',[])]; ans=[c.get('text','') for m in x.get('messages',[]) if m.get('role')=='assistant' for c in m.get('content',[]) if c.get('type')=='text']; p=next((c.get('audio') for c in cont if c.get('type')=='audio'),None);prompt=next((c.get('text') for c in cont if c.get('type')=='text'),'')
  if not p:continue
  pp=Path(p); basename=pp.name
  actual=next((q for q in [pp,ROOT/pp,ROOT/basename,ROOT/'data/human_voice_resampled'/basename,ROOT/'data/human_voice_dataset/wavs'/basename] if q.is_file()),None)
  if actual is None:missing.append(i);continue
  transcript=ans[-1] if ans else ''
  import wave
  with wave.open(str(actual),'rb') as wf:
   rate=wf.getframerate();channels=wf.getnchannels();duration=wf.getnframes()/rate
  aud.append({'id':i,'audio_path':actual.relative_to(ROOT).as_posix(),'audio_sha256':H(actual.read_bytes()),'sample_rate':rate,'channels':channels,'duration_s':round(duration,6),'reference':transcript,'prompt':prompt,'category':x.get('category')})
 rng_audio=random.Random(args.seed); cats=defaultdict(list)
 for x in aud:cats[x.get('category')].append(x)
 if any(k is not None for k in cats):
  chosen=[];keys=sorted(cats,key=lambda k:str(k))
  for k in keys:rng_audio.shuffle(cats[k])
  while len(chosen)<min(40,len(aud)):
   progressed=False
   for k in keys:
    if cats[k] and len(chosen)<40:chosen.append(cats[k].pop());progressed=True
   if not progressed:break
  aud=chosen
 else:
  rng_audio.shuffle(aud);aud=aud[:40]
 write(out/'eval/audio_eval.jsonl',aud);audio_over=[x['id'] for x in aud if Path(x['audio_path']).name.casefold() in trainbases or norm(x['reference']) in traintext]
 proof['audio_train_overlap']=len(audio_over)
 output_files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='MANIFEST.json')
 def hashrec(p):
  b=p.read_bytes();return {'sha256_raw':H(b),'sha256_lf':H(b.replace(b'\r\n',b'\n').replace(b'\r',b'\n')),'rows':len(p.read_text(encoding='utf-8').splitlines())}
 promptcnt=Counter(norm(x['user']) for x in train); answercnt=defaultdict(set)
 for x in train:answercnt[norm(x['user'])].add(norm(x['answer']))
 dup=[(k,v,len(answercnt[k])) for k,v in promptcnt.items() if len(answercnt[k])>1]
 manifest={'seed':args.seed,'thresholds':{'latin_gt':.2,'latin_report_min':.1,'near_jaccard':.85,'max_tokens':MAX,'text_eval_target':150,'audio_target':40,'voice_max':300,'voice_per_prompt':10},'inputs':{**{k:{'path':str(p),'sha256_raw':H(p.read_bytes()),'sha256_lf':H(p.read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n')),'rows':len(read(p))} for k,p in paths.items()}, **{k:{'path':str(p),'sha256_raw':H(p.read_bytes()),'sha256_lf':H(p.read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n')),'rows':len(read(p))} for k,p in {'audio_val':ROOT/'data/val_stt_final.jsonl','audio_train_overlap_check':ROOT/'data/train_stt_final.jsonl'}.items()}},'outputs':{str(p.relative_to(out)):hashrec(p) for p in output_files},'drop_indices':{s:{r:sorted(set(ix)) for r,ix in d.items()} for s,d in drops.items()},'drop_counts':{s:{r:len(set(ix)) for r,ix in d.items()} for s,d in drops.items()},'nfc_modifications':dict(mods),'length_stats':stats,'latin_10_20_examples':mid,'whitelist_hits':dict(whitelist),'train_prompt_duplicate_count':len(dup),'train_prompt_duplicate_top20':sorted(dup,key=lambda x:-x[1])[:20],'combined_voice_rows':len(voice),'combined_voice_distinct_prompts':len(pc),'combined_voice_top20':pc.most_common(20),'audio_missing_indices':missing,'audio_train_overlap_indices':audio_over,'zero_overlap_proof':proof,'tokenizer':{'path':str(tp),'sha256':H(tp.read_bytes())},'script_git_blob_hash':subprocess.check_output(['git','hash-object','scripts/build_v3_dataset.py'],cwd=ROOT,text=True).strip(),'unmodified_rows_byte_preserved':'Original text strings retained for rows without NFC modification; output JSON is canonical JSONL.'}
 (out/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
if __name__=='__main__':main()
