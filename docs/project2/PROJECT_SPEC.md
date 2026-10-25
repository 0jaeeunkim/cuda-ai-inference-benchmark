# CUDA Project 2 — Experiment Specification

작성일: 2026-10-25
상태: 구현 전 초기 명세. 아래 프로토콜로 시작하며 실제 환경·API 호환성은 M0에서 확인한다.
기준 repo commit: `4ab640f994c489254ce28c384a2d7f415e5061e4`

## 1. 목표와 범위

주제: DistilBERT 추론의 비동기 파이프라인 및 동기화 최적화.

질문: 같은 workload에서 Host 대기를 줄이고 다음 batch의 H2D와 현재 batch의 계산을 중첩하면 전체 처리시간이 줄어드는가? 효과는 batch size에 따라 어떻게 달라지는가?

기존 모델의 FP32 forward는 유지한다. 이번 비교는 GPU A/B/C/D 사이의 비교다. CPU 성능 비교나 process-cold crossover 재측정은 핵심 범위가 아니다. 과제는 표지 포함 10쪽 이내이며 슬라이드는 결과 분석 이후에 구성한다.

## 2. 기존 증거와 해석 제한

- 기존 warm GPU E2E와 inference-only 중앙값의 차이는 약 0.03–0.05 ms였다.
- 상대 차이는 batch 1에서 2.51%, batch 64에서 0.34%였다.
- 이 차이는 순수 PCIe 전송시간이 아니며 모든 최적화의 이론적 상한으로 사용하지 않는다.
- 기존 application runner는 입력 H2D와 logits D2H에 `non_blocking=False`를 쓴다.
- 버퍼 사전할당, eval mode, inference_mode, warm-up은 이미 적용되어 있었다.
- 전송 대역폭 병목, GPU 포화, occupancy 개선은 아직 검증되지 않았다.
- Batch 64에서 H2D 입력은 2 × 64 × 64 × 8 = 65,536 B, D2H logits는 64 × 2 × 4 = 512 B이다. 작은 전송에서의 호출·대기 비용과 전체 실행 순서도 확인한다.

근거: 기존 `results/analysis.md`, `results/audit.md`, `src/application_validation.py`, `src/timing.py`.

## 3. 고정 workload와 환경

| 항목 | 정책 |
| --- | --- |
| Model | `distilbert-base-uncased-finetuned-sst-2-english` |
| Model revision | `714eb0fa89d2f80546fda750413ed43d93601a13` |
| Dataset | `nyu-mll/glue`, `sst2`, `validation` |
| Dataset revision | `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` |
| Rows | 공개 row 순서의 872개 전체, 매 실행 정확히 한 번 처리 |
| 기존 fingerprint | `0e2b00033b48693f`; 다른 library version으로 달라지면 원인을 확인하고 기록 |
| Input | 동일한 `input_ids`, `attention_mask`; 각각 int64 |
| Sequence length | 64 고정, max-length padding/truncation, 기존 tokenizer revision 고정 |
| Model precision | FP32, eval(), torch.inference_mode() |
| Matmul | float32 precision highest, CUDA matmul TF32 off, cuDNN TF32 off |
| Batch sizes | 1, 8, 64 |
| 기존 장비 | RTX 5080 / Ryzen 7 9700X |
| 기존 주요 software | Python 3.10.20, PyTorch 2.12.0+cu130, Transformers 5.16.1 |

장비·software는 실제 현재 값을 한 번 기록한다. 과거 metadata를 현재 환경 확인 없이 복사하지 않는다. 기존 환경과 cache가 적합하면 그대로 재사용하고 불필요한 upgrade/reinstall을 하지 않는다. Tokenizer의 구현/backend 선택도 기록한다.

기존 `results/environment.json`은 보존하고 현재 환경은 `results/project2/environment.json`에 저장한다. 환경 변경 시 새 snapshot을 만들고 각 session이 사용하는 snapshot/hash를 명시한다.

## 4. 모든 버전의 공통 출력 계약

각 실행은 CPU 입력에서 시작해 `[872, 2]` FP32 logits 전체가 CPU output 배열에 도착한 상태에서 끝난다. 원래 sample 순서대로 모든 행을 보관한다.

기존 runner는 같은 batch output 버퍼를 반복 재사용한다. Project 2는 전체 결과를 보관하는 계약으로 맞추므로 기존 0.203 s 등의 수치를 새 speedup의 분모로 쓰지 않는다. A도 새 계약으로 같은 session에서 다시 측정한다.

- 같은 token 값, model weights, forward 호출 수, batch 구성, 출력 내용과 완료 상태를 유지한다.
- Partial batch는 유효 행만 처리하고 보관한다. 전체 크기 B로 계산한 뒤 일부를 버리는 방식으로 바꾸지 않는다.
- A–D 모두 두 device input 슬롯을 준비한다. A/B/C의 슬롯은 동일 Stream의 순서로 안전하게 재사용한다.
- 각 버전의 memory 종류를 제외한 불필요한 buffer/queue 차이는 최소화한다.
- 최종 output 배열은 모든 버전에 preallocate하며 각 batch가 자기 row 범위에 쓴다.

| B | Forward 호출 수 | 마지막 batch |
| --- | --- | --- |
| 1 | 872 | 1 |
| 8 | 109 | 8 |
| 64 | 14 | 40 |

## 5. 비교 버전

| ID | Host 입력/출력 | 제출·실행 방식 | 비교 목적 |
| --- | --- | --- | --- |
| A | Pageable | 단일 Stream, blocking H2D와 D2H | 새 계약의 baseline |
| B | Pinned | A와 같은 blocking copy와 순차 실행 | A→B: memory 종류 |
| C | Pinned | 단일 Stream, non-blocking copies, per-batch Host completion wait 제거 | B→C: Host 대기와 제출 방식 |
| D | Pinned | 별도 Copy Stream + Compute Stream, Event 의존성, 두 입력 슬롯 | C→D: 전송·계산 중첩 파이프라인 |

- A/B/C는 `H2D_i → forward_i → D2H_i` 순서를 같은 Stream에 둔다.
- C는 같은 Stream 내부의 복사·계산이 서로 겹치는 버전이 아니다. Host가 다음 작업을 먼저 제출할 수 있는 버전이다.
- D의 Copy Stream은 H2D를, Compute Stream은 forward와 뒤따르는 D2H를 수행한다. Forward끼리는 하나의 Compute Stream에서 순차 실행한다.
- D에서만 H2D_(i+1)와 forward_i가 겹칠 수 있다. 실제 overlap은 trace로 확인한다.
- 3번째 D2H Stream, 다수 forward 동시 실행, stream 수 sweep은 초기 범위에 포함하지 않는다.
- Pinned host tensors는 반복 루프 밖에서 한 번 준비해 유지한다. 매 batch `pin_memory()`를 호출하지 않는다.
- 전체 CPU token 배열을 준비해 유지하므로, timed loop에 별도 pageable→pinned CPU staging을 추가하지 않는다. 실제 입력 공급이 필요한 다른 사용 시나리오와 구분한다.

## 6. Stream, Event, buffer 수명

7주차 28쪽의 네 기능을 적절한 위치에서 사용·설명한다. 모두를 모든 버전에 삽입하는 것을 목표로 하지 않는다.

| CUDA 기능 | PyTorch 대응 | 역할 |
| --- | --- | --- |
| cudaDeviceSynchronize | torch.cuda.synchronize() | 시작 전 잔여 GPU 작업 정리, 측정 종료의 공통 전체 완료 |
| cudaStreamSynchronize | stream.synchronize() | 작은 동작 검증에서 특정 Stream의 Host 대기 확인; blocking 경로와 비교 설명 |
| cudaEventSynchronize | event.synchronize() | 작은 동작 검증의 CPU 결과 사용 전 완료 대기 |
| cudaStreamWaitEvent | stream.wait_event(event) | D의 Copy/Compute 사이 GPU 의존성 |

- 네 기능의 비교 검증은 짧게 수행하고 추가 full sweep 축으로 만들지 않는다.
- 모든 A–D의 주 측정은 동일한 최종 `torch.cuda.synchronize()`로 전체 완료를 보장한다. 좁은 대기 API로 바꾸는 것 자체를 speedup으로 주장하지 않는다.
- Compute가 input 슬롯을 읽기 전에 해당 H2D 완료를 Event로 기다린다.
- Copy가 같은 device input 슬롯에 새 입력을 쓰기 전에 이전 forward의 마지막 input 사용 완료를 기다린다.
- Input 마지막 사용 Event는 해당 forward 뒤에 기록한다. CPU 반환 시점을 GPU 완료로 취급하지 않는다.
- Pinned CPU 입력은 H2D 완료 전 변경·해제하지 않는다.
- D2H 목적지의 행 범위는 서로 겹치지 않고, CPU는 완료 뒤에만 읽는다.
- 모델 준비·default Stream에서의 초기 할당·warm-up 완료를 측정 시작 전에 연결한다.
- Cross-stream tensor 수명은 persistent 참조, Event 의존성, 필요한 allocator 수명 관리로 보장한다. 임시 GPU logits가 D2H 완료 전에 재활용되지 않게 한다.
- Event는 기록할 작업을 제출한 뒤 기다린다. 재사용하는 Event는 이전 세대에 대한 wait 제출이 끝나기 전에 새 record로 혼동하지 않도록 한다.
- Event·Stream·buffer 생성은 측정 밖에서 한다. Timed loop에서 필요한 record/wait 호출 비용은 포함한다.
- 코드에는 각 buffer의 writer, reader, 재사용 가능 시점을 짧게 적는다.

## 7. 측정 계약

주 지표의 이름은 `prepared-buffer warm E2E total time`이다.

시작 전: model, tokenized input, Host/device buffers, Stream/Event 준비 및 warm-up을 완료하고 GPU 잔여 작업을 동기화한다.

측정: `perf_counter_ns` 시작 → 전체 batch loop의 H2D/forward/D2H 및 제출·의존성 비용 → 최종 GPU 전체 완료 대기 → timer 종료.

포함: Python loop/dispatch, copies, forward, 필요한 Event record/wait와 최종 대기.
제외: import, tokenization, model loading, context 초기화, allocation/pinning/Stream/Event 준비, correctness 비교, file I/O, trace 수집.

- Pinned 준비 비용과 준비한 byte 수는 별도 setup 항목으로 기록한다. 이 주 지표를 입력 준비까지 포함하는 application E2E라고 부르지 않는다.
- 모든 버전은 동일한 start/end 및 CPU-visible output 계약을 따른다.
- 한 Stream의 종료 Event만으로 여러 Stream 전체 시간을 대신 재지 않는다.
- 보조 CUDA Event 측정은 필요할 때만 진단용으로 분리한다. 서로 다른 timer 경계의 차이를 순수 copy 비용으로 쓰지 않는다.
- 주 지표: 중앙값 total ms, IQR, samples/s, A 대비 speedup. 요청별 latency는 이 실험으로 추정하지 않는다.
- Throughput = 872 × 1000 / total_ms.
- Speedup_X = median(T_A) / median(T_X), 같은 batch와 같은 session끼리 비교.
- B→C 및 C→D 비율도 원인 해석용으로 계산할 수 있다. 중앙값 비율과 paired-run speedup을 혼용하지 않는다.

## 8. 반복과 실행 순서

전체 실행은 batch 1/8/64 × A/B/C/D × 32 runs = 384개 raw 측정이다. 32는 4개 순서의 균형을 맞추면서 기존 30회 수준을 유지하기 위한 값이다.

각 B에서 아래 네 순서를 8번 반복한다. 시작 행은 batch별로 회전해 기록한다.

```text
A B D C
B C A D
C D B A
D A C B
```

각 버전이 각 position에 8번 나타난다. Row order, round, position을 raw에 기록한다. Batch 내부 작업을 재배치하는 것이 아니라 버전 실행 순서를 바꾸는 것이다.

- 측정 전 각 버전에서 최소 5회의 짧은 warm-up sequence를 수행한다. full/partial shape와 D의 slot 재사용이 포함되게 한다.
- Smoke에서는 warm-up과 반복을 1–2회로 제한한다.
- 형식적으로 정상인 느린 측정값을 임의로 제거하지 않는다.
- 실패·중단 session의 raw는 보존하고 completed/failed 상태를 기록한다. 불완전 session을 정상 summary로 합치지 않는다.
- 다른 GPU 작업, 온도/clock 상태, 실행 시간대를 제한된 metadata로 기록한다. 실험 중 GPU 관리 설정은 임의로 변경하지 않는다.

## 9. Correctness와 bounded smoke

성능 해석 전에 blocking A의 reference logits를 기준으로 B/C/D를 비교한다. 고정한 동일 model·input을 사용한다.

- 모든 logits는 finite이고 shape는 `[N, 2]`이다.
- 모든 sample이 정확히 한 번, 원래 순서대로 보관된다.
- Prediction은 reference와 정확히 일치해야 한다.
- Logits 비교 기준은 처음부터 `atol=1e-5, rtol=1e-4`로 고정하고 최대 절대오차도 기록한다. 실패를 숨기려고 기준을 완화하지 않는다.
- 동일 Stream A의 별도 reference pass와 필요하면 작은 CPU sanity check를 사용한다.
- 전체 실행의 각 timed run 직후 결과를 CPU에서 검증한다. 이 비교는 timer 밖에 둔다.
- Tail, slot 재사용, 출력 순서, input/output 주소 중복을 명시적으로 검사한다.

권장 smoke:
1. B=1, N=5: 여러 slot 재사용.
2. B=8, N=19: batch [8,8,3], 두 슬롯 이후 재사용과 tail.
3. 앞의 두 조건 통과 후 B=64, N=137: batch [64,64,9]로 큰 shape와 tail의 짧은 확인.
4. B=64의 실제 872-row split [64×13,40]은 정적 coverage 검사 및 full-run correctness에서 확인.

Smoke throughput은 sample_count=N으로 계산하고 full 872-sample 결과와 섞지 않는다. 전체 872-sample correctness/benchmark는 사용자가 실행한다.

## 10. 짧은 timeline 검증

처음부터 여러 profiler를 설치하지 않는다. 기존 환경에서 CUDA activity를 지원하는 PyTorch profiler를 우선 확인하고, 이미 사용 가능한 Nsight Systems가 더 적합하면 하나를 선택해 기록한다.

- 원래 workload의 B=1 또는 8, A/C/D, 소수의 batch만 수집한다.
- H2D, forward, D2H, Copy/Compute Stream, Host 대기와 GPU 공백을 식별한다.
- D에서 실제 copy/compute overlap이 관찰되는지 확인한다. 관찰되지 않으면 그대로 보고한다.
- Trace를 켠 시간은 주 성능 결과에 넣지 않는다.
- 도구 사용이 불가능하면 blocker와 필요한 최소 조치를 기록한다. Trace 없이 overlap 달성을 단정하지 않는다.
- 대형 profiling은 사용자가 실행하며 전체 benchmark 뒤 별도 명령으로 제공한다.

## 11. 결과·metadata·재현성

권장 경로:

```text
results/project2/environment.json
results/project2/smoke/<smoke_id>/
results/project2/<session_id>/raw.csv
results/project2/<session_id>/metadata.json
results/project2/<session_id>/correctness.json
results/project2/<session_id>/summary.csv
results/project2/<session_id>/analysis.md
results/project2/<session_id>/plots/
results/project2/<session_id>/traces/
```

Raw 최소 필드: session_id, protocol_id/hash, source_commit, timestamp, batch_size, sample_count, variant, round, execution_order, position, forward_count, final_batch_size, total_ms, throughput, correctness_pass.

Metadata: 실제 실행 명령/작업경로, 실행한 source 파일 hash와 git 상태, 환경 snapshot hash, model/dataset/tokenizer revision, input 및 reference checksum, dtype/TF32 설정, memory 종류/byte 수, buffer 수, Stream 역할, Event 정책, timing 경계, warm-up/repeat/order 정책, setup 비용, session 상태.

- 새 session별 디렉터리를 만들고 기존 raw를 덮어쓰지 않는다.
- Raw는 개별 측정이 끝날 때마다 보존한다. 재집계로 raw를 수정하지 않는다.
- Summary는 raw에서 독립 재구성하고 expected row 수·순서·산식을 검사한다.
- Project 1의 tracked evidence 파일은 시작 시 manifest/hash를 남기고 작업 뒤 변경 여부를 확인한다.
- 설명 문서는 정상 결과와 느려진 결과를 모두 보고한다.
- 그래프는 결과 분석 이후 필요한 최소 수만 만든다. 발표용 10쪽 구성을 지금 작성하지 않는다.

## 12. 비목표와 변경 원칙

초기 범위에서 제외: 다른 모델/dataset, FP16/BF16/TF32 on, quantization, dynamic padding/length bucketing, sequence-length sweep, custom CUDA kernel, kernel fusion/torch.compile/CUDA Graphs, Unified Memory, 다중 GPU, 여러 forward 동시 실행, process-cold 재실험, 본격 CPU sweep.

Routine 구현 선택은 Codex가 처리하고 기록한다. Workload·precision·출력 계약·주 측정 경계·비교군을 바꾸는 변경은 별도 protocol version과 변경 사유를 남긴다. 결과 개선이 작다는 이유로 범위를 자동 확장하지 않는다.

## 13. 완료 기준

1. A–D의 모든 correctness와 coverage 검사가 통과한다.
2. Full 384개 raw와 충분한 재현 metadata가 보존된다.
3. 결과를 raw에서 재구성할 수 있다.
4. 최종 완료 조건과 setup 제외 범위가 일치한다.
5. Trace의 관찰 범위 안에서만 overlap을 설명한다.
6. Project 1 원본 결과와 동작을 보존한다.
7. 개선·동률·악화 모두 데이터에 근거해 해석한다.
8. 코드/실험 한계와 발표에 사용할 수 있는 결론이 정리된다.

## 14. 참고 자료

- 강의: `references/project2/`의 4–7주차 PDF. 7주차 28쪽은 네 동기화 기능, 뒷부분은 chunk·double buffer·전체 완료·pipeline 해석.
- 기존 source/evidence: 기준 commit의 `PROJECT_SPEC.md`, `PLAN.md`, `results/analysis.md`, `results/audit.md`, `results/environment.json`, `src/application_validation.py`, `src/timing.py`, `src/benchmark_harness.py`.
- PyTorch pinned/non-blocking tutorial: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- CUDA Stream API: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- CUDA Event API: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html

공식 문서의 최신 버전과 실제 설치 버전을 구분한다. 새 API가 필요하면 해당 설치 버전에서 지원되는지 먼저 확인한다.
