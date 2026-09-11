"""Check full-state continuation and exact resume over a short window beyond 500k."""
import json
import pickle
import run as common
from run_task5_continue_1m import RUNS


def read(folder,name):return json.loads((RUNS/folder/name).read_text())


def fingerprint(key,value):
    if key=='replay':
        value={k:common.np.asarray([r[k] for r in value]) for k in value[0]}
    return common.tree_hash(value)


def main():
    for folder in ('smoke_warm','smoke_random','smoke_resume'):
        assert read(folder,'CHECKS_PASSED.json')['status']=='passed'
        assert read(folder,'DONE.json')['updates']==120030
        origin=read(folder,'origin_state_verified.json')
        assert origin['origin_steps']==500000 and origin['origin_updates']==120000
        assert origin['optimizer_and_alpha_states_preserved'] and origin['latest_agent_equals_final_agent']
    states=[]
    for folder in ('smoke_warm','smoke_resume'):
        with (RUNS/folder/'latest.pkl').open('rb') as f:states.append(pickle.load(f))
    fields=('agent','step','replay','episode','trace','ob','keys','numpy_rng')
    hashes={k:fingerprint(k,states[0][k]) for k in fields}
    for k in fields:assert hashes[k]==fingerprint(k,states[1][k]),'Resume mismatch: '+k
    for filename in ('eval_500120_002.json','eval_500120_002_mean_residual.json'):
        assert read('smoke_warm',filename)['records']==read('smoke_resume',filename)['records']
    common.atomic_json(RUNS/'PREFLIGHT_PASSED.json',dict(status='passed',
        origin_steps=500000,origin_updates=120000,additional_smoke_steps=120,additional_smoke_updates=30,
        both_arms_full_state_restored=True,no_new_warmup=True,exact_resume_fields=list(fields),hashes=hashes))


if __name__=='__main__':main()
