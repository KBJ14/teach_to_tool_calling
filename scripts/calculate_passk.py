from math import comb
from typing import List, Optional, Dict, Iterable
import json
import os

def pass_at_k(successes: List[int], totals: List[int], k: int) -> Optional[float]:
    """
    TauBench/CodeX-style pass@k (at-least-one success).
    successes[i] = c_i
    totals[i]    = n_i
    k            = pick k trials (without replacement)

    per-task:
      p_i(k) = 1 - C(n_i - c_i, k) / C(n_i, k)

    final:
      pass@k = 평균_i p_i(k)   (단, n_i < k 인 task는 스킵)
    """
    assert len(successes) == len(totals)
    assert k >= 1

    values = []   # ← 여기 p_i(k)들을 모아두는 리스트
    
    for c, n in zip(successes, totals):
        if n < k:
            # n_i < k → k개 샘플링 자체 불가 → 이 task는 pass@k 계산 제외
            continue

        # C(n, k)
        denom = comb(n, k)
        if denom == 0:
            continue  # guard

        # C(n-c, k) = all-fail probability numerator
        all_fail = comb(n - c, k) / denom

        # p_i(k) = 1 - P(all fail)
        p = 1.0 - all_fail
        values.append(p)

    if not values:
        return None

    # ★★★ pass@k = task별 p_i(k)들의 평균 ★★★
    return sum(values) / len(values)


def pass_hat_k(successes, totals, k):
    values = []   # 여기에 task별 p_i(k)를 계속 append

    for c, n in zip(successes, totals):
        if n < k:
            continue    # 이 task는 pass^k 계산 불가능 → 스킵

        if c < k:
            values.append(0.0)   # C(c,k)=0
        else:
            values.append(comb(c, k) / comb(n, k))

    # ★★★ 여기서 평균을 계산한다 ★★★
    if not values:
        return None
    return sum(values) / len(values)   # ← pass^k 계산되는 줄


def load_trial_results(jsonl_path: str) -> Dict[str, bool]:
    """
    Load a trial accuracy jsonl and return a dict mapping instance_id -> correct bool.
    """
    results = {}
    if not os.path.exists(jsonl_path):
        return results

    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst_id = obj.get('instance_id')
            # Some rows might not have 'correct' (be cautious)
            correct = obj.get('correct', False)
            results[inst_id] = bool(correct)
    return results


def aggregate_trials_from_dir(exp_dir: str, file_pattern: str = 'selected_500_*_accuracy.jsonl') -> Dict[str, List[bool]]:
    """
    Read trial results from `exp_dir/trial0..trial9/` and aggregate per-instance correctness
    across all available trials found in the directory. Returns a mapping instance_id -> [booleans].
    """
    trials = []
    # look for directories named trialNN under exp_dir
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(exp_dir)

    for name in os.listdir(exp_dir):
        if not name.startswith('trial'):
            continue
        trial_dir = os.path.join(exp_dir, name)
        if not os.path.isdir(trial_dir):
            continue

        # prefer matching the exact filename pattern(s)
        for fname in os.listdir(trial_dir):
            if fname.endswith('_accuracy.jsonl') and fname.startswith('selected_500'):
                trials.append(os.path.join(trial_dir, fname))
                break

    trials = sorted(trials)
    # collect per-instance results
    aggregated: Dict[str, List[bool]] = {}
    for trial_file in trials:
        trial_results = load_trial_results(trial_file)
        for inst_id, correct in trial_results.items():
            aggregated.setdefault(inst_id, []).append(bool(correct))
    return aggregated


def compute_passk_for_experiment(exp_dir: str, k_min: int = 1, k_max: int = 5) -> Dict:
    """
    Compute pass^k and pass@k for k in [k_min..k_max] using the aggregated trials under exp_dir.
    Returns a dictionary with experiment name and pass metrics.
    """
    aggr = aggregate_trials_from_dir(exp_dir)
    instance_ids = sorted(aggr.keys())
    # successes: per-task c (sum correct across trials)
    successes = [sum(aggr[inst]) for inst in instance_ids]
    totals = [len(aggr[inst]) for inst in instance_ids]

    pass_hat = {}
    pass_at = {}
    for k in range(k_min, k_max + 1):
        pass_hat[k] = pass_hat_k(successes, totals, k)
        pass_at[k] = pass_at_k(successes, totals, k)

    name = os.path.basename(exp_dir.rstrip('/'))
    return {
        'name': name,
        'num_tasks': len(instance_ids),
        'n_per_task': None if not totals else max(totals),
        'pass_hat': pass_hat,
        'pass_at': pass_at,
    }


def write_jsonl(result: Dict, outpath: str):
    """Write a single-line JSONL containing the result mapping"""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', '--input_dir', dest='exp_dir', type=str, default='scripts/accuracy/gpt-4o_semantic_fc_edh',
                        help='Path to experiment directory containing trial subdirs (input_dir)')
    parser.add_argument('--name', type=str, default=None, help='Optional name to use for experiment output')
    parser.add_argument('--out', type=str, default=None, help='Path to output jsonl file (overrides name)')
    args = parser.parse_args()

    result = compute_passk_for_experiment(args.exp_dir, 1, 5)
    if args.name is not None:
        result['name'] = args.name

    if args.out is None:
        # default out in the scripts folder: all_passk.jsonl
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.join(script_dir, 'all_passk.jsonl')
    write_jsonl(result, args.out)
    print('Wrote passk to', args.out)

