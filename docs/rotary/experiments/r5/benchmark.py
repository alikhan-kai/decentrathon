import hashlib,importlib.util,json,pathlib,statistics,sys,time
import torch
root=pathlib.Path(__file__).parent
variants={}
for name,file,block in [('baseline','baseline.py',256),('baseline512','baseline512.py',512),('packed','solution/rotary.py',256),('packed512','solution/rotary.py',512)]:
 spec=importlib.util.spec_from_file_location(name,root/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 if name.startswith('packed'):m.BLOCK_SIZE=block
 variants[name]=m
assert torch.cuda.is_available() and torch.cuda.get_device_name()=='Tesla T4'
torch.manual_seed(729)
started=time.monotonic();results=[]
def run(m,q,k,c,s,gq,gk):
 out=m.apply_rotary_pos_emb(q,k,c,s)
 return (*out,*torch.autograd.grad(out,(q,k),(gq,gk)))
for dtype in (torch.float32,torch.float16):
 for t in (64,256,512):
  q=torch.randn(4,t,12,128,device='cuda',dtype=dtype).transpose(1,2).requires_grad_()
  k=torch.randn(4,t,2,128,device='cuda',dtype=dtype).transpose(1,2).requires_grad_()
  c,s=[torch.randn(4,t,128,device='cuda',dtype=dtype) for _ in range(2)]
  for layout in ('packed','contiguous'):
   gq,gk=[torch.randn_like(x) if layout=='packed' else torch.randn(x.shape,device='cuda',dtype=dtype) for x in (q,k)]
   eager=variants['baseline']._eager_rope(q,k,c,s)
   oracle=(*eager,*torch.autograd.grad(eager,(q,k),(gq,gk)))
   compiles={}
   for name,m in variants.items():
    start=time.monotonic();got=run(m,q,k,c,s,gq,gk);torch.cuda.synchronize();compiles[name]=time.monotonic()-start
    for a,b in zip(got,oracle):torch.testing.assert_close(a,b,atol=0,rtol=0,equal_nan=True)
   del oracle,got,eager
   for _ in range(3):
    for m in variants.values():run(m,q,k,c,s,gq,gk)
   torch.cuda.synchronize();timings={n:[] for n in variants};wall={n:[] for n in variants}
   for rep in range(7):
    names=list(variants);names=names[rep%len(names):]+names[:rep%len(names)]
    for name in names:
     begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     st=time.perf_counter();begin.record()
     for _ in range(5):run(variants[name],q,k,c,s,gq,gk)
     end.record();end.synchronize();wall[name].append((time.perf_counter()-st)*1000/5);timings[name].append(begin.elapsed_time(end)/5)
   med={name:statistics.median(v) for name,v in timings.items()}
   item={'dtype':str(dtype),'t':t,'gradient_layout':layout,'first_call_s':compiles,'event_ms':timings,'wall_ms':wall,'ratios':{n:med[n]/med['baseline'] for n in variants}}
   results.append(item)
   assert time.monotonic()-started<180,'cooperative benchmark budget exceeded'
resources={}
for name,m in variants.items():
 cache=m._triton_kernel().device_caches[0][0]
 resources[name]=[{'n_regs':v.n_regs,'n_spills':v.n_spills,'shared':v.metadata.shared} for v in cache.values()]
print(json.dumps({'device':torch.cuda.get_device_name(),'torch':torch.__version__,'elapsed_s':time.monotonic()-started,'cases':results,'resources':resources,'hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('baseline.py','baseline512.py','solution/rotary.py')}},indent=2))
