# CUDA Project 2 — Progress and Milestones

작성일: 2026-10-25
상태: 핵심 실험 종료 및 최종 인계 완료. M3–M5 완료; 추가 필수 실험·구현 작업 없음.
기준 명세: `docs/project2/PROJECT_SPEC.md`
기준 repo commit: `4ab640f994c489254ce28c384a2d7f415e5061e4`

## 현재 상태

- Current milestone: M3–M5 및 최종 인계 완료. 핵심 실험 종료.
- Last completed: 최종 12조건 중앙값·IQR·speedup·처리시간 감소율 통합 표와 Project 2 FINAL_HANDOFF 작성.
- Current blocker: 없음. 새 Conda 환경 설치 및 캐시 준비 완료.
- Next action: 관리 대화에서 [FINAL_HANDOFF.md](FINAL_HANDOFF.md) 검토·공유. 핵심 실험의 추가 필수 작업은 없음.
- Active session_id: `2026-10-25T113152.111380_0000_43aecc1c` (completed/audited full); trace `aaaea78b1d90460c91df9e18816974a5_trace`.
- Latest validated measurements: full 384/384 correctness 통과, max abs logits error 0.0. A/C speedup B=1/8/64: 1.1020×/1.0323×/1.0036×; A/D: 1.0993×/1.0342×/1.0048×. D의 짧은 B=8,N=19 trace에서 kernel overlap 1.408 μs 관찰.
- Canonical repository: `/home/jekim/cuda/cuda-ai-inference-benchmark`. 인계 문서는 원래 workspace 상위 `docs/project2/`에 있었고 Git 저장소 안으로 복사함.
- Terminal commands/cache/output: [RUNBOOK.md](RUNBOOK.md).

## Milestones

### M0 — workspace와 실제 환경 확인

작업:
- 루트 AGENTS와 Project 2 scope 블록을 확인/추가한다.
- Project 1 문서·핵심 source·필요 evidence를 읽고 git 상태와 기준 commit을 기록한다.
- README에 Project 1/2 문서 링크를 작게 추가한다. 필요하면 루트 PLAN에 Project 2 진행표 링크만 추가한다.
- 기존 tracked Project 1 evidence의 hash manifest를 만든다.
- 현재 Python, GPU, driver/runtime, package, 접근 가능한 cache를 짧게 확인한다.
- 새 환경 snapshot을 별도 경로에 기록한다.

Gate:
- 올바른 GPU에서 작은 FP32 연산이 성공하고, model/dataset revision과 사용할 cache 위치가 확인됨.
- 파일 충돌·cache 누락·GPU 부재는 구체적으로 보고하고, 가능한 문서/정적 구현 작업을 계속함.
- 설치, 대형 다운로드, GPU가 없는 환경에서의 PyTorch 재설치로 문제를 우회하지 않음.

### M1 — A/B/C 구현과 작은 검증

작업:
- 공통 input/output 계약, 두 input 슬롯, full/tail shape, runner CLI를 구현한다.
- A/B/C를 동일한 완료 상태로 만든다.
- timing 밖에서 output coverage와 reference logits를 비교한다.
- B=1/N=5 및 B=8/N=19에서 1–2회 짧은 smoke를 실행한다.

Gate:
- 원래 row 순서의 모든 logits가 보관되고 prediction/logit 검사가 통과함.
- 사전할당·pinning이 timer 밖이고 non-blocking 호출 반환을 완료로 오해하지 않음.

### M2 — D 구현과 동기화·timing smoke

작업:
- Copy/Compute Stream, upload-ready와 input-consumed Event, 두 input 슬롯을 구현한다.
- slot 재사용과 cross-stream 수명을 코드/짧은 검증으로 확인한다.
- 네 동기화 API의 대기 대상 차이를 짧은 동작 검증으로 확인한다.
- A–D의 작은 smoke를 통과시키고 B=64/N=137도 bounded 확인한다.
- raw schema, protocol hash, 완료 gate, full-run 명령을 준비한다.

Gate:
- A–D의 correctness, tail, slot 재사용, 공통 timer 경계 통과.
- full 모드가 stale/missing smoke gate를 감지함.
- 이 시점에서 Codex는 전체 실험을 실행하지 않고 사용자 터미널 명령을 전달함.

### M3 — 전체 correctness와 짧은 trace

상태: **완료** — 전체 correctness 12/12 및 별도 A/C/D trace 감사 통과.

실행자: 사용자. Codex는 명령 준비, 결과 검사 및 작은 수정 담당.

작업:
- 실제 872 rows의 모든 B×variant를 확인한다.
- 모든 output/label/order와 actual tail shape 검사를 수행한다.
- 짧은 A/C/D timeline을 별도 진단으로 수집한다.
- profiling 미지원이면 blocker와 대안을 기록한다.

Gate:
- 전체 output correctness 통과. 실제 관찰된 overlap 여부와 trace 경로를 기록함.
- trace의 성능 수치를 주 benchmark 결과로 사용하지 않음.
- routine bug fix는 scope 안에서 진행하되 바뀐 source에 맞춰 smoke와 gate를 갱신함.

### M4 — 전체 성능 실험과 독립 감사

상태: **완료** — 384/384 raw·정확성·균형 순서·독립 집계 감사 통과.

실행자: 사용자. Codex는 실행 명령과 감사 담당.

작업:
- 32회 균형 순서로 B=1/8/64 × A/B/C/D의 총 384개 raw 측정을 보존한다.
- 각 timed run의 CPU 결과를 timer 밖에서 검증한다.
- metadata, source/protocol hash, 실행 순서, sample 수와 산식을 검사한다.
- Project 1 evidence 변경 여부도 확인한다.

Gate:
- complete session의 모든 raw가 존재하고 유효함.
- 변경된 source나 다른 protocol/session을 묵시적으로 합치지 않음.
- 실패 session과 느린 측정값을 삭제하지 않음.

### M5 — 해석과 보고 근거 정리

상태: **완료** — 상세 분석·그래프·최종 통합 표·인계 완료.

작업:
- raw에서 중앙값·IQR·throughput·A 대비 speedup을 계산/재검산한다.
- A→B, B→C, C→D 변화와 trace를 함께 해석한다.
- setup/pinning 비용, 작은 전송량, Python dispatch, Event overhead 등 증거가 지지하는 설명만 사용한다.
- 최소한의 비교 그래프, analysis.md, 재현 명령과 한계를 남긴다.

Gate:
- speedup과 overlap 주장이 실제 증거와 일치함.
- 개선 폭이 작거나 악화된 경우도 그대로 기록함.
- 사용자에게 이 관리 대화에 공유할 짧은 milestone 보고를 제공함.
- 발표자료 구성·새 최적화 축 추가는 별도 후속 작업.

## 인계 시 확정한 초기 선택

- workload/FP32/S=64 유지.
- GPU baseline 대비 A/B/C/D 순차적 비교.
- batch 1/8/64, 전체 872 rows.
- CPU-visible full logits를 공통 완료 계약으로 채택.
- D는 copy/compute 두 Stream과 두 input 슬롯, forward끼리는 순차.
- prepared-buffer warm E2E가 주 지표.
- 기본 반복 수 32, 4개 실행 순서를 8회 반복.
- Codex는 작은 smoke, 사용자는 full 실행.
- 이 선택의 구체적인 정의는 PROJECT_SPEC만 authoritative source로 사용.

## Decision log

새 결정은 날짜, 변경 내용, 이유, 영향받는 protocol/session만 간단히 추가한다. 과거 결과 해석을 plan에 장문 복제하지 않는다.

| 날짜 | 결정 | 이유/영향 |
| --- | --- | --- |
| 2026-10-25 | 초기 명세와 단계 구성 인계 | 구현 전, 새 측정값 없음 |
| 2026-10-25 | 실제 Git 저장소에 인계 문서 복사, `project2-async-pipeline` branch 생성 | 깨끗한 main/기준 commit 확인; 원본 문서 보존 |
| 2026-10-25 | 사용자 지시: 재사용 탐색 종료, 사용자 로컬 Miniforge + `cuda-project2` | 새 서버 RTX 5070 Ti/driver 595.99.02. 설치·다운로드는 사용자가 실행 |
| 2026-10-25 | Python 3.10.20 / torch 2.12.0+cu130 / transformers 5.16.1 / datasets 5.0.1 유지 | 공식 cp310 Linux wheel 및 배포·의존성 확인. tokenizers 0.23.2 / hub 1.32.0도 유지. 실제 runtime 검증 대기 |
| 2026-10-25 | D의 각 batch에 미리 만든 ready/consumed Event 사용 | Event 세대 혼동 방지; CUDA handle 초기화도 timer 밖. 두 input 슬롯/두 Stream 계약 유지 |
| 2026-10-25 | smoke/correctness gate에 source·protocol·환경·cache/입력 identity 및 evidence hash 연결 | stale/missing gate는 full 실행 차단. 실패 세션 보존 |
| 2026-10-25 | 초기 강의 PDF 추가·열람 요청 철회 | 사용자 정정: 검토·최적화 선정은 관리 대화에서 완료. PDF는 Codex 작업이나 gate/blocker가 아님. 확정 A–D 정의와 진행 상태 유지 |

## Codex 진행 보고 형식

```text
Completed:
Validation:
Issues:
Files changed:
Next step:
Session / protocol:
```

현재 상태와 이 log만 최신화하고 완료 milestone 기록을 삭제하지 않는다. 매 턴마다 전체 문서를 사용자에게 재출력하지 않는다.

## 2026-10-25 실행 기록

### M0 — 환경 조사 / 설치 인계 (gate pending)

Completed: 기준 commit `4ab640f994c489254ce28c384a2d7f415e5061e4`, 시작 git 상태 clean. scope 블록 1회 추가. Project 1 보호 대상 85개 문서/source/evidence의 SHA256 manifest 생성. 실제 GPU/driver/Python 및 패키지·Conda·cache 부재 확인. 공식 배포 확인 및 단계별 설치 CLI 준비.
Validation: RTX 5070 Ti, SM 12.0, driver 595.99.02. system Python 3.12.3. 공식 cu130 cp310 wheel, conda-forge Python 3.10.20, Miniforge 26.9.4-0 installer checksum 확인. 이전 Project 1 환경은 그대로 보존.
Issues: CUDA FP32 및 모델 forward 미실행. 필수 패키지/캐시 없음.
Files changed: AGENTS.md, README.md, .gitignore; docs/project2/ 문서 복사; scripts/project2_setup.sh; docs/project2/requirements-hf.txt; results/project2/environment.json, environments/, project1_manifest.json, distribution_check.json, conda_distribution_check.json.
Next step: 사용자 설치·다운로드 후 `--mode env`로 실제 GPU 연산과 1-row forward 검증. 성공 시 새 snapshot 생성; 이전 blocked snapshot 보존.
Session / protocol: Project 2 유효 GPU session 없음. 현재 environment 상태 blocked.

### M1 — A/B/C 구현 / 정적 검증 (GPU gate pending)

Completed: 별도 Project 2 policy, 두 input 슬롯, CPU full logits, full/tail shape, A pageable blocking / B pinned blocking / C pinned single-stream non-blocking 구현. 별도 A pass 기준 prediction/logit 비교, 모든 timed run 결과를 timer 밖에서 검사.
Validation: Python 3.10 문법 및 CLI 검사. 실제 enqueue 코드를 표준 라이브러리 stream DAG에서 4 variants × 3 smoke cases × 12 interleavings 검증; slot 재사용 wait를 제거한 대조군에서 손상을 탐지. 이는 GPU/model correctness 증거가 아님.
Issues: 실제 B=1/N=5, B=8/N=19 GPU smoke는 설치 후 실행 필요.
Files changed: src/project2_pipeline.py, src/project2_protocol.py, tests/test_project2_schedule.py.
Next step: env 통과 후 `--mode smoke --variants ABC`.
Session / protocol: 실패 smoke는 아래 M2 기록. 유효 raw 측정 없음.

### M2 — D / gate / bounded CLI 구현 (GPU gate pending)

Completed: Copy/Compute Stream, upload-ready/input-consumed 의존성, 두 슬롯 수명, 네 동기화 API의 작은 동작 검증 CLI. 모든 variant에 공통 perf_counter_ns + 최종 device sync. 균형 32-round 순서, raw fsync/실패 보존, 해시 gate, 독립 summary 및 짧은 trace CLI 준비.
Validation: 표준 라이브러리 검사 10/10 통과. 384-row 전체 순서에서 variant별 각 position 8회, full B=64 tail 40 정적 확인. raw 누락/중복/순서/산식/실패 status와 missing/stale/tampered gate 탐지. 셸 구문/--help 및 Python 3.10 문법 통과. Project 1 85개 보호 파일 일치.
Issues: 실제 smoke 진입은 `ModuleNotFoundError: No module named 'torch'`로 실패. 게이트/summary/raw를 생성하지 않았고 실패 metadata/error는 보존. GPU timing/correctness/overlap 또는 speedup 주장을 하지 않음.
Files changed: src/project2_analysis.py; tests/test_project2_protocol.py; results/project2/static_validation.json; 실패 smoke 디렉터리; docs/project2/RUNBOOK.md, PLAN.md.
Next step: 설치 후 env → ABC smoke → ABCD smoke. 통과 후에만 사용자에게 M3 correctness/trace와 M4 full 명령 실행 인계. 전체 correctness/384 sweep은 Codex가 실행하지 않음.
Session / protocol: 실행 시점의 protocol/source hash는 `results/project2/static_validation.json` 및 각 session의 metadata.json이 기준. 실패 session ID와 최신 hash는 아래 검증 상태에 기록.

### 최신 검증 상태

- Current protocol SHA-256: `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.
- Failed smoke: `2026-10-25T104647.110253_0000_67d5d263`; protocol `57b1ea11cd034fe554af032a173da92363eccb57c2966d2ffcd80fabcab65357`; `ModuleNotFoundError: No module named 'torch'`.
- Validation evidence: `results/project2/development_validation.json`, `static_validation.json`.
- Project 1 protected file hashes: 85/85 unchanged. Scope marker: 1 block. Full benchmark and full correctness: not run.
- Follow-up required: user setup completion → env/ABC smoke/ABCD smoke → update this log with actual GPU evidence.

### M0 설치 오류 보완 — 2026-10-25

Completed: 사용자가 보고한 `curl: command not found`와 이후 Conda 경로 오류를 확인. Downloader를 curl → wget → Python 순으로 선택하도록 수정. `all` 단계는 첫 실패에서 중단. Conda/환경 선행 조건을 확인하며 기존 호환 환경은 재사용하고 다른 Python의 환경은 보존.
Validation: 외부 네트워크·설치·HOME 변경 없이 로컬 fixture 회귀검사 8/8 통과. curl/wget 부재의 Python 다운로드, wget fallback, 실패 다운로드의 기존 파일 보존, 선행 단계 부재, all 중단, 기존 환경 재사용/보존 검증. bash 문법과 diff 검사 통과.
Issues: 실제 Conda 설치·패키지 설치·캐시 다운로드는 아직 완료 확인 전. GPU gate 상태는 그대로 pending.
Files changed: scripts/project2_setup.sh, tests/test_project2_setup.py, docs/project2/RUNBOOK.md, docs/project2/PLAN.md, results/project2/setup_repair_validation.json.
Next step: 사용자 터미널에서 `bash scripts/project2_setup.sh all` 재실행 후 완료 출력 또는 첫 실패 오류 공유.
Session / protocol: benchmark source·protocol 변경 없음. 새 GPU session 없음.

### M0 runtime gate 통과 — 2026-10-25

Completed: 사용자 설치 완료 확인, cuda-project2에서 환경 검증 수행.
Validation: 32×32 FP32 CUDA 연산과 pinned-revision 모델 1-row forward 통과; environment status ready.
Issues: 없음. 전체 correctness/benchmark는 미실행.
Files changed: results/project2/environment.json, environments/, m0_runtime_check.txt, docs/project2/PLAN.md.
Next step: ABC smoke → ABCD smoke.
Session / protocol: environment snapshot `717460e9743d7848dfbddf5f4de6fa9baec03b95cb3d86e11ee2a0e73d8fe38c`.

### M1 GPU smoke 통과 — 2026-10-25

Completed: ABC smoke의 세 조건(B/N=1/5,8/19,64/137) 실행 완료.
Validation: 18 timed rows; 모든 logits finite, prediction/allclose 통과; full/tail shape·slot 재사용·원래 순서 coverage 통과. 최대 logits 절대오차 0.0.
Issues: 전체 872-row correctness와 성능 sweep은 미실행.
Files changed: results/project2/smoke/2026-10-25T112446.134301_0000_737f153b/ 및 m1_abc_smoke.txt, docs/project2/PLAN.md.
Next step: ABCD smoke 및 동기화 API probe.
Session / protocol: 2026-10-25T112446.134301_0000_737f153b / af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78.

### M2 GPU gate 및 독립 감사 통과 — 2026-10-25

Completed: 새 cuda-project2 환경에서 ABCD bounded smoke 완료. 네 동기화 API probe 통과. CPU-visible full logits와 공통 최종 device sync 계약 확인.
Validation: B/N=1/5,8/19,64/137; 24 raw rows, 12 variant별 observed coverage 검사 통과. 모든 timed output의 prediction/logit 비교 통과, 최대 절대오차 0.0. 두 슬롯 재사용·full/tail shape·출력 행 순서·주소 검사 통과. stale smoke 및 missing correctness gate가 차단됨을 확인. Project 1 보호 대상 85/85 hash 일치.
Issues: 전체 872-row correctness, 384 raw sweep, CUDA trace는 아직 실행하지 않음. Overlap/speedup은 미확립.
Files changed: results/project2/m2_abcd_smoke.txt, m2_runtime_validation.json, smoke/<session_id>/, smoke_gate.json, docs/project2/PLAN.md, RUNBOOK.md.
Next step: 사용자 터미널에서 --mode correctness 실행 후 metadata.json/correctness.json/raw.csv를 감사. 이후 별도 trace 및 full 명령 사용.
Session / protocol: `2026-10-25T112508.818449_0000_40f9eefb` / `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

### M3 전체 correctness 및 독립 감사 통과 — 2026-10-25

Completed: 사용자가 실행한 correctness 세션을 감사. B=1/8/64 × A–D의 12개 조건에서 매 실행 872 rows 처리 완료.
Validation: raw 12/12, observed coverage 12/12, 실제 CPU FP32 [872,2] reference artifact 3개 확인. prediction/logits/finite/row order·labels·slot·주소·full/tail shape 통과, B=64 실제 tail 40, max abs logits error 0.0. raw/CPU reference/correctness/source/환경 SHA256 및 smoke/correctness gate 일치. summary의 total/IQR/throughput 재검산. Project 1 보호 파일 85/85 일치.
Issues: 별도 GPU timeline 및 384-run full benchmark 미실행. 실제 overlap과 성능 개선 여부는 아직 주장하지 않음.
Files changed: 해당 session의 correctness_audit.json; docs/project2/PLAN.md, RUNBOOK.md.
Next step: 사용자 터미널에서 trace와 full을 별도 명령으로 실행. trace CUDA activity/실제 중첩을 확인하고 full raw 384개를 감사.
Session / protocol: `2026-10-25T112729.784641_0000_35de3e8e` / `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

### 작업 지침 정정 — 2026-10-25

Completed: 강의 PDF 추가·열람 요구와 관련 blocker를 제거. 검토·최적화 선정은 관리 대화에서 완료된 것으로 명시. Codex 역할은 확정 A–D 구현, correctness·timing 검증, 실행 명령 준비와 결과 분석.
Validation: 현재 milestone/session/next action 유지. 실행 코드와 해시 대상 PROJECT_SPEC의 실험 정의는 수정하지 않아 기존 protocol 및 smoke/correctness gate가 유효함. 명세의 강의·쪽수 인용은 배경 설명으로만 취급.
Issues: 없음. trace 및 full benchmark의 실제 결과 확인은 계속 필요.
Files changed: AGENTS.md, docs/project2/AGENTS_APPEND.md, CODEX_START.md, RUNBOOK.md, PLAN.md 및 상위 workspace의 인계 지침 사본.
Next step: 기존 trace/full 실행 결과를 확인하고 분석.
Session / protocol: 전체 correctness session과 protocol 유지.

### M3 timeline 검증 완료 — 2026-10-25

Completed: 사용자가 수집한 A/C/D B=8,N=19 trace를 CUDA correlation으로 CPU batch marker와 연결해 해석.
Validation: 각 trace의 9개 batch marker, kernel 366개, H2D 6개, logits D2H 3개 확인. D Copy Stream 25 / Compute Stream 21. D에서 batch 1/2 H2D와 이전 forward의 실제 kernel overlap 각각 0.832/0.576 μs, 총 1.408 μs. A/C는 0. Input upload 완료·slot 0 재사용 순서 통과. Kernel 공백을 overlap으로 세지 않는 interval union 검사 3개 통과.
Issues: 한 profiled B=8,N=19 trace로 다른 shape/전체 workload의 중첩량을 추정하지 않음. 모델 내부 1-byte D2H + stream sync가 C/D에도 3회 남음(aten::item 계열).
Files changed: src/project2_trace_audit.py, tests/test_project2_trace_audit.py; trace session의 trace_audit.json, gpu_intervals.csv.
Next step: full 통계와 결합한 해석은 아래 M5에 기록.
Session / protocol: `aaaea78b1d90460c91df9e18816974a5_trace` / `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

### M4 full benchmark 독립 감사 완료 — 2026-10-25

Completed: 사용자 full session의 384 raw 및 CPU reference artifacts 감사.
Validation: B=1/8/64 × A–D ×32 =384/384. 모든 timed output finite/prediction/allclose 통과, max abs error 0.0. 각 variant가 각 position에 8회. Coverage 12/12, 실제 CPU reference 3개·row/label/input metadata·B64 tail 40·source/환경/결과 hash와 gates 확인. 중앙값·type-7 Q1/Q3·IQR·throughput·speedup 12조건 재검산. Project 1 85개 보호 파일 일치. 누락·중복·제외된 정상 raw 없음.
Issues: 단일 session의 기술 통계. Clock/온도 연속 측정 및 외부 GPU 작업 부재 확인 자료는 없음.
Files changed: src/project2_results_audit.py; full session의 audit.json. 원본 raw/metadata/correctness/summary는 변경하지 않음.
Next step: M5 분석/최소 그래프 완료.
Session / protocol: `2026-10-25T113152.111380_0000_43aecc1c` / `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

### M5 분석·근거 정리 완료 — 2026-10-25

Completed: A→B, B→C, C→D 비교와 trace 해석, setup 비용·측정 경계·한계 기록. Standalone SVG 3개와 오프라인 HTML 보고서 생성.
Validation: C의 A 대비 speedup B=1/8/64: 1.1020×/1.0323×/1.0036×. D는 1.0993×/1.0342×/1.0048×. D가 C보다 B1에서 0.251% 느린 결과도 보존. D의 실제 overlap은 관찰됐지만 추가 성능 이득은 작음. SVG XML 및 산출물 hash 확인. 그래프는 고정 A 중앙값으로 변환한 run-time IQR을 표시하며 CI로 해석하지 않음.
Issues: 결과는 준비한 입력/출력의 warm E2E이며 application 입력 공급/시작 비용이나 요청 latency가 아님. 일반적 속도 우위·포화·대역폭 병목/성능 상한은 단정하지 않음.
Files changed: scripts/project2_report.py, scripts/project2_render.cjs; full session의 analysis.md, analysis_generated.md(기존 자동 문서 보존), comparisons.csv, setup_costs.csv, plots/, report.html, report_metadata.json; docs/project2/RUNBOOK.md, PLAN.md, README.md.
Next step: 분석 검토 및 관리 대화 공유. 발표자료 생성이나 새 최적화 축은 별도 요청 범위.
Session / protocol: `2026-10-25T113152.111380_0000_43aecc1c` / `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`. 실행 코드는 그대로 두고 후처리 source와 renderer hash를 별도 기록. 실험을 재실행하거나 Python 환경을 변경하지 않음.

## 핵심 실험 종료 및 최종 인계 — 2026-10-25

Completed: M3–M5 완료 상태 확정. Batch 1/8/64 × A–D의 중앙값·IQR·A 대비 speedup·처리시간 감소율을 한 최종 표로 통합. docs/project2/FINAL_HANDOFF.md에 환경·재현 명령·유효 session/protocol·핵심 해석·한계·그래프/trace 경로 정리.
Validation: 384 raw에서 12조건 표를 독립 재산출하고 기존 summary와 일치 확인. 감소율은 반올림 전 중앙값으로 계산하며 B64 B의 음수 감소율도 보존. 기존 source/protocol·게이트 및 모든 기존 Project 2 evidence와 Project 1 85개 보호 파일을 hash로 확인.
Issues: 핵심 실험의 미완료 작업 없음. 단일 session·준비 비용 제외·짧은 trace 범위 등 해석 한계는 인계에 명시.
Files changed: docs/project2/FINAL_HANDOFF.md, docs/project2/PLAN.md; full session의 final_summary.csv, final_handoff_validation.json. 기존 루트 FINAL_HANDOFF.md는 Project 1 기록으로 보존.
Next step: 관리 대화에서 최종 인계 검토·공유. 발표자료, 추가 benchmark/최적화는 별도 요청 범위.
Session / protocol: `2026-10-25T113152.111380_0000_43aecc1c` / `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.

### 남은 작업

- 핵심 실험·정확성·감사·분석·최종 인계: 완료. 추가 필수 작업 없음.
- 사용자 후속: 최종 인계 문서 검토 및 관리 대화 공유.
- 발표자료·추가 실험/최적화: 현재 요청 범위 밖이며 별도 지시 전까지 진행하지 않음.
