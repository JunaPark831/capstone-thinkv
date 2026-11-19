"""win256 레이어 0의 낮은 언덕(0.05~0.25)과 높은 언덕(0.40~0.55)에 속한 스텝의 실제 토큰을 뽑아 비교한다.
사전 기록: docs/e3_paper_calibration_spec.md "같이 하는 확인(레이어 0 토큰)". 판정 기준 없음, 사실만 보고.

사용: HF_HUB_OFFLINE=1 python -I layer0_tokens.py --dir DATA_DIR [--prompts 0002,0004,0012] [--n 15]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keyword_signal import label_tokens, token_text  # noqa: E402

LOW, HIGH = (0.05, 0.25), (0.40, 0.55)


def kind(piece):
    s = piece.strip(" ")
    if "\n" in piece:
        return "줄바꿈"
    if s == "":
        return "공백"
    if re.fullmatch(r"[0-9.,]+", s):
        return "숫자"
    if s.startswith("\\") or s in {"{", "}", "^", "_", "$"}:
        return "LaTeX"
    if re.fullmatch(r"[^\w\s]+", s):
        return "구두점/기호"
    if re.fullmatch(r"[A-Za-z]+", s):
        return "영단어"
    return "기타"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prompts", default="0002,0004,0012")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    rng = np.random.default_rng(0)
    for nm in a.prompts.split(","):
        r = torch.load(Path(a.dir) / f"{nm}.pt", weights_only=True)
        ids = r["token_ids"]
        sp = r["sparsity"][r["variants"].index("win256"), 0].float().numpy()
        text, starts, pieces = token_text(tok, ids)
        lab, _, _ = label_tokens(text, starts, len(ids), 64, "A")
        lab = lab[:-1]
        print(f"\n######## 문제 {nm} (스텝 {len(sp)}), win256 레이어 0")
        for tag, (lo, hi) in (("낮은 언덕", LOW), ("높은 언덕", HIGH)):
            idx = np.where((sp >= lo) & (sp < hi))[0]
            if len(idx) == 0:
                print(f"  [{tag} {lo}~{hi}] 스텝 없음")
                continue
            kinds = [kind(pieces[s]) for s in idx]
            kc = {k: kinds.count(k) for k in sorted(set(kinds), key=lambda k: -kinds.count(k))}
            lc = {n: int((lab[idx] == v).sum()) for n, v in (("E", 0), ("R", 1), ("T", 2), ("미라벨", -1))}
            pos = idx / len(sp)
            print(f"  [{tag} {lo}~{hi}] 스텝 {len(idx)}개 ({len(idx) / len(sp):.1%}), 위치(0~1) 중앙값 {np.median(pos):.2f} 사분위 {np.percentile(pos, 25):.2f}~{np.percentile(pos, 75):.2f}")
            print(f"      쿼리 토큰 종류: {kc}")
            print(f"      키워드 라벨: {lc}")
            for s in sorted(rng.choice(idx, size=min(a.n, len(idx)), replace=False)):
                ctx = "".join(pieces[max(0, s - 8):s]).replace("\n", "⏎")
                cur = pieces[s].replace("\n", "⏎")
                print(f"      s={s:6d} sp={sp[s]:.2f} …{ctx!r} ▶ {cur!r}")


if __name__ == "__main__":
    main()
