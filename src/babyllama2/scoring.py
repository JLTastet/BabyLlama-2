from typing import Callable

def skim_task(res: dict):
    return {
        'acc': res['acc,none'],
        'std': res['acc_stderr,none'],
    }

def skim_results(raw_results: dict):
    # We keep a reasonably flat hierarchy to keep variable names short on W&B
    skimmed_results = dict()
    skimmed_results['groups'] = {group: skim_task(res) for (group, res) in raw_results['groups'].items()}
    for (group, subtasks) in raw_results['group_subtasks'].items():
        for subtask in subtasks:
            skimmed_results[subtask] = skim_task(raw_results['results'][subtask])
    return {'results': skimmed_results}

def string_agg_fn(x, y):
    return x + '/' + y

def flatten_dictionary(nested: dict, key_agg_fn: Callable=string_agg_fn):
    flat = dict()
    for (k_outer, v_outer) in nested.items():
        if isinstance(v_outer, dict):
            flattened_inner_dict = flatten_dictionary(v_outer, key_agg_fn)
            for (k_inner, v_inner) in flattened_inner_dict.items():
                flat[key_agg_fn(k_outer, k_inner)] = v_inner
        else:
            flat[k_outer] = v_outer
    return flat