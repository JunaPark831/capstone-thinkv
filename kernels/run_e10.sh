#!/usr/bin/env bash
# E10 서버 실행 (스펙 6절 3~5순서). 저장소 루트에서:  bash kernels/run_e10.sh
# 순서: 환경 출력 -> pytest(Mac 테스트 + GPU 3순서 -> 4순서 -> 5순서, 첫 실패에서 멈춤) -> 벤치 quick -> 벤치 본 측정
# 로그는 results/e10_*.log. 멈추면 그 로그의 마지막 60줄을 전달한다.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))" | tee results/e10_env.log
python -m pip install -q pytest numpy
python -m pytest tests/kernels -x -v -s -p no:cacheprovider 2>&1 | tee results/e10_pytest.log
python -m kernels.bench --quick 2>&1 | tee results/e10_bench_quick.log
python -m kernels.bench 2>&1 | tee results/e10_bench.log
echo "끝. results/e10_bench.log 의 마지막 표를 전달한다."
