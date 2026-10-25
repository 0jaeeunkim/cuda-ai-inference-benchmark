# CUDA Project 2 — 최종 인계

기준일: 2026-10-25 (Asia/Seoul). **핵심 실험 종료, M3–M5 완료.** 추가 필수 실험·구현 작업은 없다.

## 최종 결과표

주 지표: **prepared-buffer warm E2E total time**. 매 실행 872 rows의 원래 순서 CPU-visible FP32 logits `[872,2]` 전체 완료, 각 조건 32회. 아래 값은 raw에서 재산출했다.

| Batch | Variant | 중앙값 ms | IQR ms | A 대비 speedup | 처리시간 감소율 |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 1 | A | 926.597 | 3.841 | 1.000000× | 0.0000% |
| 1 | B | 925.341 | 3.538 | 1.001357× | 0.1355% |
| 1 | C | 840.821 | 2.999 | 1.102014× | 9.2571% |
| 1 | D | 842.931 | 2.263 | 1.099256× | 9.0294% |
| 8 | A | 334.065 | 0.438 | 1.000000× | 0.0000% |
| 8 | B | 333.977 | 0.446 | 1.000263× | 0.0263% |
| 8 | C | 323.624 | 0.199 | 1.032263× | 3.1255% |
| 8 | D | 323.014 | 0.580 | 1.034212× | 3.3081% |
| 64 | A | 234.661 | 0.439 | 1.000000× | 0.0000% |
| 64 | B | 234.665 | 0.813 | 0.999982× | -0.0018% |
| 64 | C | 233.815 | 0.626 | 1.003620× | 0.3607% |
| 64 | D | 233.550 | 0.570 | 1.004758× | 0.4736% |

IQR = Q3−Q1 (type-7/inclusive quartiles). Speedup = median(A)/median(X).
처리시간 감소율 = 100×[1−median(X)/median(A)]. 음수는 시간 증가다.
표의 반올림 전 수치는 [final_summary.csv](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/final_summary.csv)에 보존했다. IQR은 신뢰구간이 아니다.

A: pageable/blocking, B: pinned/blocking, C: pinned/단일 Stream·non-blocking, D: pinned/Copy+Compute Stream·Event·두 input 슬롯. 모두 같은 출력·완료 계약이다.

## 유효 session과 검증

| 용도 | Session ID | 상태 |
| --- | --- | --- |
| Full 성능 | `2026-10-25T113152.111380_0000_43aecc1c` | 384/384 raw 및 correctness·독립 감사 통과 |
| 전체 correctness | `2026-10-25T112729.784641_0000_35de3e8e` | 12/12 조건 ×872 rows 통과 |
| A–D bounded smoke | `2026-10-25T112508.818449_0000_40f9eefb` | 24/24, smoke gate 통과 |
| 별도 A/C/D trace | `aaaea78b1d90460c91df9e18816974a5_trace` | CUDA 활동·batch correlation·실제 overlap 감사 완료 |

Protocol SHA256: `af20bd21bbe26d2bc71d9e3bf89ca6fd52695b08a8f130d12bfa91e10754fd78`.
모든 성능 raw의 최대 logits 절대오차 **0.0**, prediction 일치. B=64 tail은 40이다.
기준 commit은 `4ab640f994c489254ce28c384a2d7f415e5061e4`; 실제 실행 source 해시와 git 상태는 [metadata.json](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/metadata.json), 감사는 [audit.json](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/audit.json)에 있다. Smoke/correctness/trace 시간은 full 통계에 섞지 않았다.

## 실행 환경과 명령

작업 디렉터리: `/home/jekim/cuda/cuda-ai-inference-benchmark`.
Conda `cuda-project2`: Python 3.10.20, PyTorch 2.12.0+cu130 / CUDA runtime 13.0, Transformers 5.16.1, Datasets 5.0.1, Tokenizers 0.23.2, HF Hub 1.32.0.
RTX 5070 Ti (16 GB, SM 12.0), driver 595.99.02, Ryzen 7 9700X, CPU threads 8. FP32/highest, TF32 off, eval/inference_mode, S=64.
Model/tokenizer revision `714eb0fa89d2f80546fda750413ed43d93601a13`; SST-2 validation revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`, 공개 row 순서 872개 유지.
환경: [environment.json](../../results/project2/environment.json), SHA256 `910146957dbf7b1b5000d900063643b819208df900cc609f13e76b417a0878fd`.
Cache: 작업 디렉터리의 `.cache/huggingface/` (`hub/`, `datasets/`); 아래 측정 명령은 offline이다.

아래는 **재현용 기록**이다. 이번 최종 정리에서는 재실행하지 않았다. 세부 설치·감사·그림 재현 명령은 [RUNBOOK.md](RUNBOOK.md)에 있다.

```bash
cd /home/jekim/cuda/cuda-ai-inference-benchmark
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode env
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode smoke
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode correctness
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode trace
/home/jekim/miniforge3/bin/conda run --no-capture-output -n cuda-project2 python src/project2_pipeline.py --mode full
```

측정은 매번 새 `results/project2/sessions/<session_id>/`에 raw/metadata/correctness/summary를, trace는 `results/project2/diagnostics/<id>_trace/`에 저장한다. Full은 동일 source/protocol/환경/입력의 smoke·correctness gate를 요구한다.

## 핵심 해석과 한계

- Pinned memory만 적용한 A→B 변화는 작았다. C는 A 대비 batch 1에서 **1.1020×**, 처리시간 **9.2571% 감소**였다. 작은 batch의 명시적 Host 대기 감소가 주요 관측 결과다.
- D는 C 대비 batch 1에서 약 **0.251% 느렸고**, batch 8/64에서는 시간 감소가 약 **0.188%/0.113%**였다. D의 추가 이득은 작다.
- D의 실제 H2D/kernel overlap **1.408 μs**는 별도 **B=8,N=19** trace에서 관찰했다. A/C는 0. C/D에도 forward 내부 1-byte D2H와 Host stream sync가 남아 있다. Overlap 관찰이 큰 E2E 이득을 뜻하지 않는다.
- 모델 로드·토큰화·pinning/버퍼/Stream/Event 준비·warm-up·검증·파일 I/O·trace는 주 timer 밖이다. [setup_costs.csv](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/setup_costs.csv)에 별도 준비 비용이 있다. 입력 준비/공급까지 포함한 application E2E나 요청 latency가 아니다.
- 단일 장비·단일 session의 기술 통계다. 작은 차이의 유의성·범용 우위·GPU 포화·순수 PCIe 비용·대역폭 병목/성능 상한은 주장하지 않는다. Trace instrumentation 시간과 full 성능을 합치지 않으며 다른 batch/872 rows로 overlap 양을 일반화하지 않는다.

## 산출물과 남은 작업

- [상세 분석](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/analysis.md), [오프라인 그래프 보고서](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/report.html), [후처리 provenance](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/report_metadata.json).
- SVG: [speedup](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/plots/speedup.svg), [timeline](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/plots/timeline.svg), [overlap 상세](../../results/project2/sessions/2026-10-25T113152.111380_0000_43aecc1c/plots/overlap_detail.svg).
- Trace 원본: `results/project2/diagnostics/aaaea78b1d90460c91df9e18816974a5_trace/traces/`의 `A_b8_n19.json`, `C_b8_n19.json`, `D_b8_n19.json`; [trace 감사](../../results/project2/diagnostics/aaaea78b1d90460c91df9e18816974a5_trace/trace_audit.json), [GPU interval CSV](../../results/project2/diagnostics/aaaea78b1d90460c91df9e18816974a5_trace/gpu_intervals.csv). 원본 trace는 로컬 gitignore 대상이므로 공유 시 별도 보관한다.
- 실행 코드, 기존 raw·metadata·correctness·summary 및 Project 1 문서/코드/결과를 보존했다. Project 1 보호 파일 85개 hash가 일치한다.
- 남은 작업은 관리 대화에서 인계 문서를 검토하는 것이다. 추가 benchmark/최적화와 발표자료는 별도 요청 범위이며 이번에는 진행하지 않았다. 강의자료 검토·최적화 선정은 관리 대화에서 완료된 것으로 따른다.
