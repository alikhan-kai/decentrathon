import ast, hashlib, json, pathlib, subprocess, types
import torch

torch.set_num_threads(1)
ROOT='/home/homelab/.t3/worktrees/decentrathon/feat-rotary'
SHA='25855208695fd113fa1897385abccc73f71a19cd'
SITE=pathlib.Path('/home/homelab/.t3/projects/decentrathon/.venv/lib/python3.13/site-packages')
def blob(path): return subprocess.check_output(['git','show',f'{SHA}:{path}'],cwd=ROOT)
sources={p:blob(p) for p in ['solution/lora.py','solution/nf4.py','solution/swiglu_pack.py','task/task.yaml']}
installed={p:(SITE/p).read_bytes() for p in ['peft/tuners/lora/bnb.py','peft/tuners/lora/layer.py','bitsandbytes/nn/modules.py']}
print('HASHES',json.dumps({p:hashlib.sha256(b).hexdigest() for p,b in (sources|installed).items()},sort_keys=True))
class CpuTorch:
    # PEFT's no-argument autocast query addresses CUDA. Adapt ONLY the query
    # to CPU so real CPU AMP takes the corresponding autocast-enabled branch.
    def __getattr__(self,key): return getattr(torch,key)
    def is_autocast_enabled(self): return torch.is_autocast_enabled('cpu')
def compile_fn(node,ns):
    tree=ast.Module(body=[node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),'<source AST>','exec'),ns)
    return ns[node.name]
pt=ast.parse(installed['peft/tuners/lora/bnb.py'])
cls=next(n for n in ast.walk(pt) if isinstance(n,ast.ClassDef) and n.name=='Linear4bit')
ns={'torch':CpuTorch(),'VARIANT_KWARG_KEYS':['alora_offsets']}
orig=compile_fn(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'),ns)
ns2={'torch':torch,'_original_forward':orig}
fast=compile_fn(next(n for n in ast.walk(ast.parse(sources['solution/lora.py'])) if isinstance(n,ast.FunctionDef) and n.name=='fast_forward'),ns2)
class Drop(torch.nn.Module):
    def forward(self,x):
        self.input=x.detach().clone()
        y=torch.nn.functional.dropout(x,.05,True)
        self.output=y.detach().clone()
        return y
class Base(torch.nn.Module):
    def forward(self,x): return x*.25
class Fake:
    def __init__(self,a,b):
        self.disable_adapters=False; self.merged=False
        self.base_layer=Base(); self.active_adapters=['default']
        self.lora_A={'default':torch.nn.Linear(64,16,bias=False,dtype=torch.bfloat16)}
        self.lora_B={'default':torch.nn.Linear(16,64,bias=False,dtype=torch.bfloat16)}
        with torch.no_grad():
            self.lora_A['default'].weight.copy_(a); self.lora_B['default'].weight.copy_(b)
        self.lora_dropout={'default':Drop()}; self.scaling={'default':2.}; self.use_dora={}; self.lora_variant={}
    def _check_forward_args(self,*args,**kwargs): pass
    def _cast_input_dtype(self,x,dtype): return x.to(dtype)
    def unmerge(self): self.merged=False; self.unmerged=True

def metric(a,b):
    a=a.float(); b=b.float(); d=(a-b).abs()
    return {'max_abs':d.max().item(),'relative_l2':(d.norm()/a.norm().clamp_min(1e-30)).item(),'different':int((a!=b).sum()),'numel':a.numel(),'fails_fp32_tol':int((d>1e-4+1e-4*a.abs()).sum()),'fails_half_tol':int((d>2e-3+2e-3*a.abs()).sum())}
torch.manual_seed(19)
x0=torch.randn(8,64)*1.3+.0137
a=torch.randn(16,64)*.1
b=torch.randn(64,16)*.1
g=torch.randn(8,64)*.1
for dtype in [torch.float16,torch.bfloat16]:
    for zero_b in [True,False]:
        records=[]
        for fn in [orig,fast]:
            m=Fake(a,b*0 if zero_b else b); x=x0.clone().requires_grad_()
            torch.manual_seed(419)
            with torch.autocast('cpu',dtype=dtype): y=fn(m,x)
            rng=torch.random.get_rng_state().clone()
            y.backward(g)
            records.append((y.detach(),x.grad,m.lora_A['default'].weight.grad,m.lora_B['default'].weight.grad,m.lora_dropout['default'],rng))
        u,v=records
        print('AMP',json.dumps({'dtype':str(dtype),'zero_b':zero_b,'output_dtype':str(u[0].dtype),'dropout_input_dtypes':[str(u[4].input.dtype),str(v[4].input.dtype)],'rng_equal':torch.equal(u[5],v[5]),'mask_equal':torch.equal(u[4].output==0,v[4].output==0),'dropout':metric(u[4].output,v[4].output),'output':metric(u[0],v[0]),'dx':metric(u[1],v[1]),'dA':metric(u[2],v[2]),'dB':metric(u[3],v[3])}))
# Scalar example using real FP16 tensor operations. This isolates exactly the
# old scaling-before-FP32-add vs new FP32-cast-before-add(alpha=2) boundaries.
for grad in [2.**-25,2.**-24,1.]:
    vals=[]
    for fused in [False,True]:
        o=torch.tensor([.25],dtype=torch.float16,requires_grad=True)
        r=torch.tensor([.125],dtype=torch.float32,requires_grad=True)
        y=torch.add(r,o.float(),alpha=2.) if fused else r+o*2.
        y.backward(torch.tensor([grad],dtype=torch.float32))
        vals.append({'y':y.item(),'d_lora_out':o.grad.item(),'d_base':r.grad.item()})
    print('SCALE_BOUNDARY',json.dumps({'upstream':grad,'original':vals[0],'new':vals[1]}))
o=torch.tensor([40000.],dtype=torch.float16); r=torch.tensor([-80000.],dtype=torch.float32)
print('SCALE_OVERFLOW',json.dumps({'original':(r+o*2).item(),'new':torch.add(r,o.float(),alpha=2.).item(),'scope':'synthetic boundary, not observed model activation'}))
# Exact dispatch bodies, tiny stand-ins for unmerge and variants.
for fn in [orig,fast]:
    m=Fake(a,b);m.disable_adapters=True;m.merged=True
    fn(m,x0)
    print('DISABLED_MERGED',fn.__name__,json.dumps({'unmerged':getattr(m,'unmerged',False),'merged_after':m.merged}))
    try:
        m=Fake(a,b);m._mixed_batch_forward=lambda *a,**k:torch.tensor([987.])
        print('ADAPTER_NAMES',fn.__name__,fn(m,x0,adapter_names=['__base__']*8).flatten()[0].item())
    except Exception as e: print('ADAPTER_NAMES',fn.__name__,type(e).__name__,str(e))
    m=Fake(a,b);m.lora_variant={'default':types.SimpleNamespace(forward=lambda *a,**k: torch.tensor([987.]))}
    with torch.autocast('cpu',dtype=torch.float16): print('VARIANT_EMPTY_LEGACY_MAP',fn.__name__,fn(m,x0).flatten()[0].item())
for contiguous in [True,False]:
    t=torch.arange(32).reshape(4,8)
    if not contiguous:t=t.t()
    old=t.contiguous();new=t if t.is_contiguous() else t.contiguous()
    print('CONTIGUOUS',json.dumps({'contiguous':contiguous,'same_values':torch.equal(old,new),'old_is_input':old is t,'new_is_input':new is t}))
assert sources=={p:blob(p) for p in sources}
assert installed=={p:(SITE/p).read_bytes() for p in installed}
print('SOURCE_HASHES_UNCHANGED True')
class CountBase(Base):
    def __init__(self): super().__init__();self.calls=0
    def forward(self,x): self.calls+=1;return super().forward(x)
for prior_vanilla in [False,True]:
    states=[]
    for fn in [orig,fast]:
        m=Fake(a,b);m.base_layer=CountBase()
        m.lora_A['dora']=m.lora_A['default'];m.lora_B['dora']=m.lora_B['default']
        m.lora_dropout['dora']=Drop();m.scaling['dora']=2.;m.use_dora={'dora':True}
        m.active_adapters=(['default'] if prior_vanilla else [])+['dora']
        m.lora_variant={'dora':types.SimpleNamespace(forward=lambda self,**kw: kw['result']+1.)}
        torch.manual_seed(419)
        with torch.autocast('cpu',dtype=torch.float16): y=fn(m,x0)
        states.append(torch.random.get_rng_state().clone())
        print('DORA_DISPATCH',json.dumps({'function':fn.__name__,'prior_vanilla':prior_vanilla,'base_calls':m.base_layer.calls,'scope':'control-flow stand-in, variant arithmetic not tested'}))
    print('DORA_RNG_EQUAL',prior_vanilla,torch.equal(*states))
