import time
from check_host1_status import run_remote

host = '39.106.118.206'
pw = 'Gen@song@2026622'

print('Waiting for preds.json to appear or SWE-bench to finish (Polling for up to 3 minutes)...')
for i in range(18):
    out, err = run_remote(host, pw, 'ls -la /root/flashkv0516/SWE-agent/trajectories/root/default__openai--deepseek-v4-flash__t-0.00__p-1.00__c-0.00___swe_bench_verified_test__30000_smoke_test_litellm_fixed/preds.json 2>/dev/null || echo "not_yet"')
    if 'not_yet' not in out:
        print('preds.json FOUND!')
        break
    time.sleep(10)
else:
    print('Still running inference...')
