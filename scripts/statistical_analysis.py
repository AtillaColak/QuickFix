import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re, ast 

# parameters
CSV_PATH = 'query_metrics.csv'  
ALPHA = 0.05                 
BOOT_N = 10_000          
SEED = 42    

np.random.seed(SEED)

# 1. load data
df = pd.read_csv(CSV_PATH)

metrics  = ['dale_chall', 'fkgl', 'coleman_liau',
            'toxicity', 'profanity', 'threat', 'insult']
variants = ['orig', 'full', 'r1', 'r2', 'r3']

wide = {m: df.pivot(index='index', columns='variant', values=m) for m in metrics}

rows = []
for m, w in wide.items():
    for v in ['full', 'r1', 'r2', 'r3']:
        paired = w[['orig', v]].dropna()
        if len(paired) < 2:
            rows.append([m, f'{v} vs orig', 'NA', np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        diff = paired[v] - paired['orig']
        normal = stats.shapiro(diff)[1] >= ALPHA

        if normal:
            stat, p = stats.ttest_rel(paired[v], paired['orig'])
            test = 'paired-t'
            point = diff.mean()
            boots  = np.random.choice(diff, (BOOT_N, diff.size),
                replace=True).mean(axis=1)
        else:
            stat, p = stats.wilcoxon(diff)
            test = 'Wilcoxon'
            point = np.median(diff)
            boots = np.median(np.random.choice(diff, (BOOT_N, diff.size), replace=True), axis=1)

        # bootstrap CI for mean diff
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

        rows.append([m, f'{v} vs orig', test, point, p])

results = pd.DataFrame(rows,
                       columns=['metric', 'comparison', 'test', 'point_estimate',
                                'p_value'])

# 2. print tidy results
pd.set_option('display.max_rows', None, 'display.max_columns', None)
print('\n=== Paired-test results (n=301 queries) ===\n')
print(results.round(6).to_string(index=False))

# 3. violin plots for readability metrics
for m in ['dale_chall', 'fkgl', 'coleman_liau']:
    plt.figure(figsize=(6, 4))
    # Set seaborn style with grey-ish background
    sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f0"})
    data = [wide[m][v].dropna() for v in variants]
    sns.violinplot(data)
    plt.xticks(range(0, len(variants)), variants)
    metric_titles = {
        'dale_chall': 'Dale-Chall',
        'fkgl': 'Flesch-Kincaid Grade Level',
        'coleman_liau': 'Coleman-Liau',
    }
    title_m = metric_titles.get(m, m)
    plt.title(f'Distribution of {title_m}')
    plt.tight_layout()

# 4. violin plots for toxicity metrics
for m in ['toxicity', 'profanity', 'threat', 'insult']:
    plt.figure(figsize=(6, 4))
    # Set seaborn style with grey-ish background
    sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f0"})
    data = [wide[m][v].dropna() for v in variants]
    sns.violinplot(data)
    plt.xticks(range(0, len(variants)), variants)
    metric_titles = {
        'toxicity': 'Toxicity',
        'profanity': 'Profanity',
        'threat': 'Threat',
        'insult': 'Insult',
    }
    title_m = metric_titles.get(m, m)
    plt.title(f'Distribution of {title_m}')
    plt.tight_layout()

# 5. distributions (visual normality) 
# ------------------------------------------------

def qq_grid(csv_path,
            metrics=('dale_chall', 'fkgl', 'coleman_liau',
                     'toxicity', 'profanity', 'threat', 'insult'),
            variants=('full', 'r1', 'r2', 'r3'),
            figsize=(14, 16),
            title="Q–Q plots (variant − orig)"):
    """
    Draw a 7×4 grid of Q–Q plots: rows = metrics, cols = variant-orig differences.
    """

    sns.set_theme(style="whitegrid")          # prettier background
    sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f0"})

    # ---------- load & wide-form ----------
    df   = pd.read_csv(csv_path)
    wide = {m: df.pivot(index='index', columns='variant', values=m)
            for m in metrics}

    fig, axes = plt.subplots(len(metrics), len(variants),
                             figsize=figsize, sharex=False, sharey=False)
    fig.suptitle(title, fontsize=16, y=0.93)

    for r, metric in enumerate(metrics):
        for c, variant in enumerate(variants):
            diff = (wide[metric][variant] - wide[metric]['orig']).dropna()
            ax   = axes[r, c]

            # Q–Q plot
            stats.probplot(diff, dist="norm", plot=ax)
            line = ax.get_lines()[1]                # reference line
            line.set_color('tab:orange')
            ax.get_lines()[0].set_markersize(3)     # sample points
            ax.get_lines()[0].set_markerfacecolor(sns.color_palette()[0])

            # cosmetics
            if r == 0:
                ax.set_title(f"{variant} – orig", fontsize=10)
            else:
                ax.set_title("")

            if c == 0:
                ax.set_ylabel(metric, rotation=0, labelpad=35, fontsize=9)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# ------------------ usage ------------------
qq_grid("query_metrics.csv")


def hist_grid(csv_path,
              metrics=('dale_chall', 'fkgl', 'coleman_liau',
                       'toxicity', 'profanity', 'threat', 'insult'),
              variants=('full', 'r1', 'r2', 'r3'),
              figsize=(14, 16),
              title="Histograms of differences (variant − orig)"):
    """
    Draw a 7 × 4 grid of histograms for (variant − orig) differences.
    A thin orange line shows the fitted normal density for reference.
    """

    plt.style.use('seaborn-v0_8-whitegrid')   # light grey background

    df   = pd.read_csv(csv_path)
    wide = {m: df.pivot(index='index', columns='variant', values=m)
            for m in metrics}

    fig, axes = plt.subplots(len(metrics), len(variants),
                             figsize=figsize, sharex=False, sharey=False)
    fig.suptitle(title, fontsize=16, y=0.93)

    for r, metric in enumerate(metrics):
        for c, variant in enumerate(variants):
            diff = (wide[metric][variant] - wide[metric]['orig']).dropna()
            ax   = axes[r, c]

            # histogram
            ax.hist(diff, bins=36, density=True,
                    alpha=0.75, color='tab:blue', edgecolor='white')

            # fitted normal curve
            if diff.std(ddof=1) > 0:          # avoid div/0 if all zeros
                x = np.linspace(diff.min(), diff.max(), 100)
                ax.plot(x,
                        stats.norm.pdf(x, diff.mean(), diff.std(ddof=1)),
                        color='tab:orange', lw=1.5)

            # cosmetics
            if r == 0:
                ax.set_title(f"{variant} – orig", fontsize=10)
            else:
                ax.set_title("")

            if c == 0:
                ax.set_ylabel(metric, rotation=0, labelpad=35, fontsize=9)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# ------------------ usage ------------------
hist_grid("query_metrics.csv")

# Checking for each metrics, how many were absolute zero differences.
def count_exact_ties(csv_path: str,
                     metrics=('dale_chall', 'fkgl', 'coleman_liau',
                              'toxicity', 'profanity', 'threat', 'insult'),
                     variants=('full', 'r1', 'r2', 'r3')):
    """
    Print, for each metric/variant pair, the number and percentage of
    exact ties (i.e., diff == 0 between variant and 'orig').

    Returns
    -------
    pd.DataFrame
        Columns: metric, variant, n_same, n_total, percent
    """
    df = pd.read_csv(csv_path)

    # wide-form dict as in the script
    wide = {m: df.pivot(index='index', columns='variant', values=m)
            for m in metrics}

    rows = []
    for metric, w in wide.items():
        for v in variants:
            paired = w[['orig', v]].dropna()
            diff   = paired[v] - paired['orig']
            n_same = (diff == 0).sum()
            n_tot  = len(diff)
            pct    = 100 * n_same / n_tot if n_tot else float('nan')
            rows.append([metric, v, n_same, n_tot, pct])

    ties = (pd.DataFrame(rows,
                         columns=['metric', 'variant',
                                  'n_same', 'n_total', 'percent'])
              .sort_values(['metric', 'variant']))

    # pretty print
    print("\n=== Exact-tie counts (variant vs. orig) ===")
    print(ties.to_string(index=False,
                         formatters={'percent': '{:.1f}%'.format}))
    return ties

count_exact_ties("query_metrics.csv")


# 6. Analysis of r1 and r2 ablations after removing the cases where the queries did not change. 
METRICS   = ['dale_chall', 'fkgl', 'coleman_liau',
             'toxicity', 'profanity', 'threat', 'insult']
VARIANTS  = ['r1', 'r2']

# ---------- helpers ----------
def normalise(q: str) -> str:
    """Lower-case + collapse whitespace + strip trailing ?!. for comparison."""
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip("?!.")

def wilcoxon_or_skip(arr):
    """Safely run Wilcoxon signed-rank; return (median, p)."""
    if (arr == 0).all():
        return np.median(arr), np.nan     # no variation
    stat, p = stats.wilcoxon(arr)
    return np.median(arr), p

# make sure snippets column is parsed into a list
df["snippets_parsed"] = df["snippets"].apply(ast.literal_eval)

for v in VARIANTS:
    diffs_by_metric   = {m: [] for m in METRICS}
    n_total           = 0     # filtered queries (query text changed)
    n_identical_snips = 0     # snippet sets still identical

    # iterate per original query id
    for qid, grp in df.groupby("index"):
        try:
            orig_row = grp.loc[grp["variant"] == "orig"].iloc[0]
            var_row  = grp.loc[grp["variant"] == v   ].iloc[0]
        except IndexError:
            continue  # variant missing

        # keep only queries whose text differs
        if normalise(orig_row["query"]) == normalise(var_row["query"]):
            continue

        n_total += 1

        # snippet-set comparison
        same_snips = set(orig_row["snippets_parsed"]) == set(var_row["snippets_parsed"])
        if same_snips:
            n_identical_snips += 1

        # collect metric deltas
        for m in METRICS:
            diffs_by_metric[m].append(var_row[m] - orig_row[m])

    pct_same = (n_identical_snips / n_total * 100) if n_total else float("nan")
    print(f"\n===== {v.upper()} =====")
    print(f"Queries with changed text: {n_total}")
    print(f"→ {pct_same:.1f}% still returned an identical snippet set\n")

    header = f"{'metric':15} {'n':>5} {'median_delta':>10} {'p_value':>12}"
    print(header)
    print("-" * len(header))

    for m, diffs in diffs_by_metric.items():
        diffs = np.asarray(diffs, dtype=float)
        if diffs.size < 2:
            print(f"{m:15} {diffs.size:5} {'NA':>10} {'NA':>12}")
            continue

        median_delta, p_val = wilcoxon_or_skip(diffs)
        print(f"{m:15} {diffs.size:5} {median_delta:10.4f} {p_val:12.6f}")
