# press 비교 시각화: results/baseline_*.jsonl → results/compare.png + 요약표 출력
import glob, os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # press 순서(none 먼저)대로 고정. 4개 넘으면 팔레트 추가
TARGET, INTERVAL = 1024, 512  # baseline.py make_press("knorm") 설정과 같아야 한다


def load() -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "baseline_*.jsonl")))
    df = pd.concat([pd.read_json(p, lines=True) for p in paths if os.path.getsize(p) > 0], ignore_index=True)
    df["kv_keep"] = df["cache_len"] / (df["input_len"] + df["gen_len"] - 1)  # KV 유지율, none 은 항상 1
    df["problem"] = df["id"].str.removeprefix("test/").str.removesuffix(".json")
    return df


def press_order(df: pd.DataFrame) -> list:
    return sorted(df["press"].unique(), key=lambda p: (p != "none", p))  # none 먼저 → 색이 press 를 따라간다


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("press").agg(
        n=("id", "size"),
        acc=("correct", "mean"),
        truncated=("truncated", "sum"),
        gen_len=("gen_len", "mean"),
        tok_per_s=("tok_per_s", "mean"),
        compressed=("kv_keep", lambda s: (s < 1).sum()),
        kv_keep_compressed=("kv_keep", lambda s: s[s < 1].mean()),  # 압축이 실제로 걸린 문제만
    ).reindex(press_order(df))


def plot(df: pd.DataFrame, out: str):
    presses = press_order(df)
    # ponytail: 시드별 평균. 시드가 여러 개면 ✗ 대신 문제별 정답률을 적는 게 낫다
    g = df.groupby(["press", "problem"]).agg(
        gen_len=("gen_len", "mean"), cache_len=("cache_len", "mean"), tok_per_s=("tok_per_s", "mean"),
        correct=("correct", "mean"), truncated=("truncated", "any"))
    order = g.loc[presses[0]].sort_values("gen_len").index  # none 생성 길이 순 → 압축이 걸리는 긴 문제가 오른쪽
    x = range(len(order))
    w = 0.8 / len(presses)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    panels = [("gen_len", "Generated tokens"),
              ("cache_len", f"KV cache length (tokens) · shaded = compressed range {TARGET}–{TARGET + INTERVAL - 1}"),
              ("tok_per_s", "Decode speed (tok/s)")]
    for ax, (col, title) in zip(axes, panels):
        for k, p in enumerate(presses):
            v = g.loc[p].reindex(order)
            xs = [i + (k - (len(presses) - 1) / 2) * w for i in x]
            ax.bar(xs, v[col], w * 0.9, color=COLORS[k], label=p)
            if col == "gen_len":  # 틀린 문제·잘린 문제만 직접 표시
                for xi, (_, r) in zip(xs, v.iterrows()):
                    tag = "\n".join(t for t, on in [("cut", r.truncated), ("✗", r.correct < 1)] if on)
                    if tag:
                        ax.text(xi, r.gen_len, tag, ha="center", va="bottom", fontsize=9, color="#52514e")
                ax.margins(y=0.15)
        if col == "cache_len":
            ax.axhspan(TARGET, TARGET + INTERVAL - 1, color="#f0efec", zorder=0)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(list(x), order, rotation=30, ha="right")
    axes[0].legend(frameon=False, ncol=len(presses), loc="upper left")
    fig.suptitle("MATH-500 first 10 · press comparison  (✗ wrong, cut = hit max_new_tokens)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    df = load()
    print(summary(df).to_string(float_format=lambda v: f"{v:.3f}"))
    out = os.path.join(RESULTS_DIR, "compare.png")
    plot(df, out)
    print(f"saved {out}")
