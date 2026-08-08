# Debug Session: localhost-8000-liveness

- Status: OPEN
- Scope: host2 local model provider liveness for `http://localhost:8000/v1`
- Symptom: `run-batch` reaches model query stage, but direct probes to `localhost:8000/v1` return `connection refused`.
- Current frontier: identify which host2 local provider should bind `:8000`, whether it never starts, starts then exits, or is cleared by another flow.

## Hypotheses

1. The expected host2 local provider for `:8000` is never started during the sample window.
2. The provider starts briefly but exits before or during the sample window.
3. Another process or script clears the provider process during the sample window.
4. The provider is healthy on host2, but `run-batch` points to the wrong local endpoint/port.

## Evidence Log

- `run_cgc_cloud_openai.py` is a long-lived host2 process and has stayed up across samples with stable start time (`Sun Jun 28 12:49:28 2026`).
- `sglang.launch_server` is also a long-lived host2 process and has stayed up across samples with stable start time (`Sun Jun 28 23:07:00 2026`).
- Host2 listeners observed during the sample window:
  - `127.0.0.1:30000` served by `sglang.launch_server`
  - `0.0.0.0:8001` served by `ray::ProxyActor`
  - No listener on `:8000`
- Direct endpoint probes during the same window:
  - `http://localhost:8000/v1/models` => `ConnectionRefusedError(111, 'Connection refused')`
  - `http://localhost:8001/v1/models` => `200 OK`
  - `http://127.0.0.1:30000/v1/models` => `200 OK`
- `run_cgc_cloud_openai.py` environment indicates:
  - `CGC_CLOUD_HTTP_PORT=50052`
  - `CGC_REAL_CLOUD_BASE_URL=http://127.0.0.1:30000`
- Current `run-batch` path still targets `CGC_SWEBENCH_API_BASE=http://localhost:8000/v1`.

## Hypothesis Status

1. The expected host2 local provider for `:8000` is never started during the sample window.
   - Partially supported: no listener is ever observed on `:8000`, but healthy local providers do exist on `:8001` and `:30000`.
2. The provider starts briefly but exits before or during the sample window.
   - Rejected for the observed provider chain: the candidate provider processes on `:8001` and `:30000` remain stable across samples.
3. Another process or script clears the provider process during the sample window.
   - Rejected for the observed provider chain: no lifecycle churn is seen for `run_cgc_cloud_openai.py` or `sglang.launch_server`.
4. The provider is healthy on host2, but `run-batch` points to the wrong local endpoint/port.
   - Strongly supported: healthy endpoints are `:8001` and `:30000`, while `run-batch` still points to `:8000`.

## Next Action

- Keep business logic unchanged for now.
- If the user wants to proceed, move to the next evidence-backed step: minimally align `CGC_SWEBENCH_API_BASE` with the healthy local provider path and immediately rerun `astropy__astropy-13453` to verify the first non-empty agent action/observation.
