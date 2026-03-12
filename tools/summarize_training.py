import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_event_scalars(event_file):
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        points = ea.Scalars(tag)
        data[tag] = {p.step: float(p.value) for p in points}
    return data


def load_root_scalars(exp_dir, subdir):
    event_files = sorted((exp_dir / subdir).glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    merged = {}
    for ef in event_files:
        scalars = load_event_scalars(ef)
        for tag, step_map in scalars.items():
            merged.setdefault(tag, {})
            merged[tag].update(step_map)
    return merged


def load_root_scalar(exp_dir, subdir, tag):
    scalars = load_root_scalars(exp_dir, subdir)
    return scalars.get(tag, {})


def scan_test_metrics(exp_dir):
    test_dir = exp_dir / "test"
    metrics = {}
    if not test_dir.exists():
        return metrics
    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        event_files = sorted(sub.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        m = re.match(r"^(minival|val_tot)_(BEV|3D)_conf_thr_([0-9.]+)_iou_([0-9.]+)_(.+)$", sub.name)
        if not m:
            continue
        split, metric_type, conf_thr, iou_thr, cls_key = m.groups()
        scalars = load_event_scalars(event_files[0])
        if not scalars:
            continue
        tag_name = f"{split}/{metric_type}_conf_thr_{conf_thr}"
        if tag_name not in scalars:
            tag_name = list(scalars.keys())[0]
        points = scalars.get(tag_name, {})
        key = {
            "split": split,
            "metric_type": metric_type,
            "conf_thr": conf_thr,
            "iou_thr": iou_thr,
            "cls_key": cls_key,
        }
        metrics[json.dumps(key, sort_keys=True)] = points
    return metrics


def parse_scores_from_pred_file(path):
    scores = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                scores.append(float(parts[-1]))
            except Exception:
                continue
    return scores


def scan_score_distribution(exp_dir, conf_thr):
    root = exp_dir / "test_kitti"
    out = {}
    if not root.exists():
        return out
    conf_name = str(conf_thr)
    for epoch_dir in sorted(root.glob("epoch_*_*")):
        m = re.match(r"^epoch_(\d+)_(subset|total)$", epoch_dir.name)
        if not m:
            continue
        epoch = int(m.group(1))
        split = m.group(2)
        pred_dir = epoch_dir / conf_name / "pred"
        if not pred_dir.exists():
            continue
        score_list = []
        for txt in pred_dir.glob("*.txt"):
            score_list.extend(parse_scores_from_pred_file(txt))
        if not score_list:
            continue
        arr = np.array(score_list, dtype=np.float64)
        out[(split, epoch)] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
        }
    return out


def to_sorted_items(step_to_value):
    return sorted(step_to_value.items(), key=lambda x: x[0])


def summarize_step_map(step_to_value, window=10):
    items = to_sorted_items(step_to_value)
    if not items:
        return None
    values = np.array([v for _, v in items], dtype=np.float64)
    n = min(window, len(values))
    first_window = float(values[:n].mean())
    last_window = float(values[-n:].mean())
    out = {
        "count": int(values.size),
        "first_step": int(items[0][0]),
        "last_step": int(items[-1][0]),
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "first_window_mean": first_window,
        "last_window_mean": last_window,
        "delta_last_first": float(values[-1] - values[0]),
        "delta_lastw_firstw": float(last_window - first_window),
        "ratio_lastw_firstw": float(last_window / max(first_window, 1e-12)),
    }
    return out


def first_last_mean(values, n=10):
    if not values:
        return (math.nan, math.nan)
    arr = np.array(values, dtype=np.float64)
    n = min(n, len(arr))
    return float(arr[:n].mean()), float(arr[-n:].mean())


def find_first_existing_series(tag_map, candidate_tags):
    for tag in candidate_tags:
        if tag in tag_map and tag_map[tag]:
            return tag, tag_map[tag]
    return None, {}


def align_step_maps(step_maps):
    if not step_maps:
        return []
    common = set(step_maps[0].keys())
    for sm in step_maps[1:]:
        common &= set(sm.keys())
    return sorted(common)


def analyze_loss_synergy(train_ce, train_bbox, train_con, window):
    out = {
        "available": False,
        "reason": "missing_required_series",
        "aligned_count": 0,
    }
    if not (train_ce and train_bbox and train_con):
        return out

    steps = align_step_maps([train_ce, train_bbox, train_con])
    if not steps:
        out["reason"] = "no_common_steps"
        return out

    ce = np.array([train_ce[s] for s in steps], dtype=np.float64)
    bbox = np.array([train_bbox[s] for s in steps], dtype=np.float64)
    con = np.array([train_con[s] for s in steps], dtype=np.float64)
    det = ce + bbox
    ratio = con / np.maximum(det, 1e-12)
    n = min(window, len(steps))
    corr = float(np.corrcoef(con, det)[0, 1]) if len(steps) > 1 else math.nan

    out.update(
        {
            "available": True,
            "reason": "",
            "aligned_count": int(len(steps)),
            "contrastive_over_det_ratio": {
                "first_window_mean": float(ratio[:n].mean()),
                "last_window_mean": float(ratio[-n:].mean()),
                "min": float(ratio.min()),
                "max": float(ratio.max()),
            },
            "corr_contrastive_vs_det": corr,
            "det_total": {
                "first_window_mean": float(det[:n].mean()),
                "last_window_mean": float(det[-n:].mean()),
            },
            "contrastive": {
                "first_window_mean": float(con[:n].mean()),
                "last_window_mean": float(con[-n:].mean()),
            },
        }
    )
    return out


def analyze_optional_train_metrics(train_iter_scalars, window):
    out = {
        "available_tags": sorted(train_iter_scalars.keys()),
        "condition_signals": {},
        "branch_signals": {},
        "missing_signals": [],
    }

    condition_candidates = {
        "condition_top1": [
            "train/condition_top1",
            "train/condition_acc",
            "train/condition_accuracy",
            "train/cond_top1",
            "train/acc_condition",
        ],
        "condition_margin": [
            "train/condition_margin",
            "train/embedding_margin",
            "train/cond_margin",
            "train/contrastive_margin",
        ],
    }
    for name, candidates in condition_candidates.items():
        tag, series = find_first_existing_series(train_iter_scalars, candidates)
        if tag is None:
            out["missing_signals"].append(name)
            continue
        out["condition_signals"][name] = {
            "tag": tag,
            "stats": summarize_step_map(series, window=window),
        }

    entropy_tag, entropy_series = find_first_existing_series(
        train_iter_scalars,
        ["train/branch_entropy", "train/gate_entropy", "train/selection_entropy"],
    )
    if entropy_tag is not None:
        out["branch_signals"]["branch_entropy"] = {
            "tag": entropy_tag,
            "stats": summarize_step_map(entropy_series, window=window),
        }
    else:
        out["missing_signals"].append("branch_entropy")

    branch_weight_tags = []
    for tag in train_iter_scalars.keys():
        lower = tag.lower()
        if not tag.startswith("train/"):
            continue
        if "loss" in lower:
            continue
        if ("branch" in lower or "gate" in lower) and ("weight" in lower or "prob" in lower):
            branch_weight_tags.append(tag)
    branch_weight_tags = sorted(branch_weight_tags)

    if len(branch_weight_tags) < 2:
        out["missing_signals"].append("branch_weight_distribution")
        out["branch_signals"]["branch_weight_distribution"] = {
            "available": False,
            "reason": "need_at_least_two_branch_weight_tags",
            "tags": branch_weight_tags,
        }
        return out

    step_maps = [train_iter_scalars[t] for t in branch_weight_tags]
    common_steps = align_step_maps(step_maps)
    if not common_steps:
        out["branch_signals"]["branch_weight_distribution"] = {
            "available": False,
            "reason": "no_common_steps_across_branch_weight_tags",
            "tags": branch_weight_tags,
        }
        return out

    probs = []
    for s in common_steps:
        vec = np.array([train_iter_scalars[t][s] for t in branch_weight_tags], dtype=np.float64)
        if np.all(vec >= 0.0) and vec.sum() > 0:
            p = vec / vec.sum()
        else:
            shifted = vec - vec.max()
            expv = np.exp(shifted)
            p = expv / np.maximum(expv.sum(), 1e-12)
        probs.append(p)
    probs = np.stack(probs, axis=0)

    max_w = probs.max(axis=1)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    entropy = entropy / math.log(probs.shape[1])

    n = min(window, len(common_steps))
    last_mean_by_branch = probs[-n:].mean(axis=0)
    dom_idx = int(last_mean_by_branch.argmax())

    out["branch_signals"]["branch_weight_distribution"] = {
        "available": True,
        "tags": branch_weight_tags,
        "aligned_count": int(len(common_steps)),
        "dominant_branch_tag_last_window": branch_weight_tags[dom_idx],
        "dominant_branch_mean_weight_last_window": float(last_mean_by_branch[dom_idx]),
        "mean_max_weight_last_window": float(max_w[-n:].mean()),
        "mean_entropy_last_window": float(entropy[-n:].mean()),
        "max_weight_stats": summarize_step_map({s: float(v) for s, v in zip(common_steps, max_w)}, window=window),
        "entropy_stats": summarize_step_map({s: float(v) for s, v in zip(common_steps, entropy)}, window=window),
    }
    return out


def analyze_metric_robustness(test_metrics, args):
    out = {
        "iou_sweep_same_setting": [],
        "metric_type_gap": [],
        "group_robustness": None,
    }

    # same split/type/conf/class, sweep IoU
    iou_series = []
    for raw_key, step_map in test_metrics.items():
        desc = json.loads(raw_key)
        if (
            desc["split"] == args.metric_split
            and desc["metric_type"] == args.metric_type
            and desc["conf_thr"] == str(args.metric_conf)
            and desc["cls_key"] == args.metric_cls
        ):
            items = to_sorted_items(step_map)
            if not items:
                continue
            iou_series.append(
                {
                    "iou_thr": float(desc["iou_thr"]),
                    "last_epoch": int(items[-1][0]),
                    "last_value": float(items[-1][1]),
                    "best_epoch": int(max(items, key=lambda x: x[1])[0]),
                    "best_value": float(max(items, key=lambda x: x[1])[1]),
                }
            )
    iou_series = sorted(iou_series, key=lambda x: x["iou_thr"])
    out["iou_sweep_same_setting"] = iou_series
    if iou_series:
        last_vals = np.array([x["last_value"] for x in iou_series], dtype=np.float64)
        out["iou_sweep_summary"] = {
            "num_iou_points": int(len(iou_series)),
            "last_value_max": float(last_vals.max()),
            "last_value_min": float(last_vals.min()),
            "last_value_gap_max_min": float(last_vals.max() - last_vals.min()),
            "last_value_std": float(last_vals.std()),
        }

    # same split/conf/iou/class, compare 3D vs BEV
    for iou in sorted({json.loads(k)["iou_thr"] for k in test_metrics.keys()}):
        vals = {}
        for mtype in ("3D", "BEV"):
            target = {
                "split": args.metric_split,
                "metric_type": mtype,
                "conf_thr": str(args.metric_conf),
                "iou_thr": str(iou),
                "cls_key": args.metric_cls,
            }
            key = json.dumps(target, sort_keys=True)
            if key not in test_metrics or not test_metrics[key]:
                continue
            vals[mtype] = to_sorted_items(test_metrics[key])[-1][1]
        if "3D" in vals and "BEV" in vals:
            out["metric_type_gap"].append(
                {
                    "iou_thr": float(iou),
                    "last_BEV": float(vals["BEV"]),
                    "last_3D": float(vals["3D"]),
                    "gap_BEV_minus_3D": float(vals["BEV"] - vals["3D"]),
                }
            )

    # same split/type/conf/iou, sweep cls_key (for grouped robustness if available)
    by_cls = {}
    for raw_key, step_map in test_metrics.items():
        desc = json.loads(raw_key)
        if (
            desc["split"] == args.metric_split
            and desc["metric_type"] == args.metric_type
            and desc["conf_thr"] == str(args.metric_conf)
            and desc["iou_thr"] == str(args.metric_iou)
        ):
            items = to_sorted_items(step_map)
            if items:
                by_cls[desc["cls_key"]] = float(items[-1][1])
    if len(by_cls) >= 2:
        vals = np.array(list(by_cls.values()), dtype=np.float64)
        out["group_robustness"] = {
            "available": True,
            "num_groups": int(len(by_cls)),
            "per_group_last": by_cls,
            "best_group": float(vals.max()),
            "worst_group": float(vals.min()),
            "gap_best_minus_worst": float(vals.max() - vals.min()),
            "mean_group": float(vals.mean()),
        }
    else:
        out["group_robustness"] = {
            "available": False,
            "reason": "need_at_least_two_cls_key_series_under_same_metric_setting",
            "num_groups": int(len(by_cls)),
            "per_group_last": by_cls,
        }
    return out


def select_metric(metrics, split, metric_type, conf_thr, iou_thr, cls_key):
    target = {
        "split": split,
        "metric_type": metric_type,
        "conf_thr": str(conf_thr),
        "iou_thr": str(iou_thr),
        "cls_key": cls_key,
    }
    key = json.dumps(target, sort_keys=True)
    if key in metrics:
        return target, metrics[key]
    fallback = None
    for raw_key, step_map in metrics.items():
        desc = json.loads(raw_key)
        if desc["split"] == split and desc["metric_type"] == metric_type and desc["cls_key"] == cls_key:
            fallback = (desc, step_map)
            if desc["conf_thr"] == str(conf_thr):
                return desc, step_map
    if fallback is not None:
        return fallback
    return None, {}


def summarize(exp_dir, args):
    exp_dir = Path(exp_dir).resolve()
    train_epoch_scalars = load_root_scalars(exp_dir, "train_epoch")
    train_iter_scalars = load_root_scalars(exp_dir, "train_iter")
    train_avg_loss = train_epoch_scalars.get("train/avg_loss", {})
    _, train_ce = find_first_existing_series(
        train_iter_scalars, ["train/loss_ce", "train/focal_loss_cls"]
    )
    _, train_bbox = find_first_existing_series(
        train_iter_scalars, ["train/loss_bbox", "train/loss_reg"]
    )
    train_con = train_iter_scalars.get("train/loss_contrastive", {})
    train_lr = train_iter_scalars.get("train/learning_rate", {})
    train_total = train_iter_scalars.get("train/total_loss", {})
    test_metrics = scan_test_metrics(exp_dir)
    score_dist = scan_score_distribution(exp_dir, args.score_conf)
    optional_train_metrics = analyze_optional_train_metrics(train_iter_scalars, window=args.window)
    contrastive_effectiveness = analyze_loss_synergy(train_ce, train_bbox, train_con, window=args.window)
    metric_robustness = analyze_metric_robustness(test_metrics, args)

    metric_desc, target_series = select_metric(
        test_metrics,
        args.metric_split,
        args.metric_type,
        args.metric_conf,
        args.metric_iou,
        args.metric_cls,
    )

    summary = {
        "experiment_dir": str(exp_dir),
        "target_metric": metric_desc,
        "target_series": to_sorted_items(target_series),
        "best_epoch_by_target": None,
        "topk_epochs_by_target": [],
        "last_epoch_by_target": None,
        "target_drop_from_best_to_last": None,
        "train_avg_loss": to_sorted_items(train_avg_loss),
        "train_iter_stats": {},
        "optional_train_metrics": optional_train_metrics,
        "contrastive_effectiveness": contrastive_effectiveness,
        "metric_robustness": metric_robustness,
        "score_distribution": [],
        "recommendation": {},
    }

    if target_series:
        sorted_pairs = to_sorted_items(target_series)
        best_epoch, best_val = max(sorted_pairs, key=lambda x: x[1])
        last_epoch, last_val = sorted_pairs[-1]
        summary["best_epoch_by_target"] = {"epoch": best_epoch, "value": best_val}
        summary["last_epoch_by_target"] = {"epoch": last_epoch, "value": last_val}
        summary["target_drop_from_best_to_last"] = best_val - last_val
        topk = sorted(sorted_pairs, key=lambda x: x[1], reverse=True)[: args.topk]
        summary["topk_epochs_by_target"] = [{"epoch": e, "value": v} for e, v in topk]

    iter_map = {
        "total_loss": train_total,
        "loss_ce": train_ce,
        "loss_bbox": train_bbox,
        "loss_contrastive": train_con,
        "learning_rate": train_lr,
    }
    for k, v in iter_map.items():
        arr = [x[1] for x in to_sorted_items(v)]
        if not arr:
            continue
        first_mean, last_mean = first_last_mean(arr, args.window)
        summary["train_iter_stats"][k] = {
            "count": len(arr),
            "first": float(arr[0]),
            "last": float(arr[-1]),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "first_window_mean": first_mean,
            "last_window_mean": last_mean,
        }

    for (split, epoch), stats in sorted(score_dist.items(), key=lambda x: (x[0][0], x[0][1])):
        summary["score_distribution"].append(
            {
                "split": split,
                "epoch": epoch,
                "conf_thr": str(args.score_conf),
                **stats,
            }
        )

    rec = {}
    if summary["best_epoch_by_target"] is not None:
        rec["recommended_epoch"] = summary["best_epoch_by_target"]["epoch"]
        rec["reason"] = "best_target_metric"
        rec["recommended_model_path"] = str(exp_dir / "models" / f"model_{rec['recommended_epoch']}.pt")
    summary["recommendation"] = rec
    return summary


def print_summary(summary):
    print(f"Experiment: {summary['experiment_dir']}")
    if summary["target_metric"] is not None:
        tm = summary["target_metric"]
        print(
            f"Target metric: {tm['split']} {tm['metric_type']} conf={tm['conf_thr']} iou={tm['iou_thr']} cls={tm['cls_key']}"
        )
    else:
        print("Target metric: not found")

    if summary["best_epoch_by_target"] is not None:
        b = summary["best_epoch_by_target"]
        l = summary["last_epoch_by_target"]
        print(f"Best epoch: {b['epoch']} value={b['value']:.6f}")
        print(f"Last epoch: {l['epoch']} value={l['value']:.6f}")
        print(f"Drop best->last: {summary['target_drop_from_best_to_last']:.6f}")
        print("Top epochs by target:")
        for item in summary["topk_epochs_by_target"]:
            print(f"  epoch {item['epoch']}: {item['value']:.6f}")

    if summary["train_avg_loss"]:
        first_epoch, first_loss = summary["train_avg_loss"][0]
        last_epoch, last_loss = summary["train_avg_loss"][-1]
        print(f"Epoch avg_loss: first={first_epoch}:{first_loss:.6f}, last={last_epoch}:{last_loss:.6f}")

    if summary["train_iter_stats"]:
        print("Train iter stats:")
        for k, v in summary["train_iter_stats"].items():
            print(
                f"  {k}: first={v['first']:.6f}, last={v['last']:.6f}, "
                f"first_w={v['first_window_mean']:.6f}, last_w={v['last_window_mean']:.6f}, "
                f"min={v['min']:.6f}, max={v['max']:.6f}, count={v['count']}"
            )

    if summary.get("contrastive_effectiveness"):
        ceff = summary["contrastive_effectiveness"]
        print("Contrastive effectiveness:")
        if not ceff.get("available", False):
            print(f"  unavailable: {ceff.get('reason', 'unknown')}")
        else:
            ratio = ceff["contrastive_over_det_ratio"]
            print(
                f"  con/det ratio: first_w={ratio['first_window_mean']:.6f}, "
                f"last_w={ratio['last_window_mean']:.6f}, min={ratio['min']:.6f}, max={ratio['max']:.6f}"
            )
            print(f"  corr(contrastive,det)={ceff['corr_contrastive_vs_det']:.6f}, aligned_count={ceff['aligned_count']}")

    if summary.get("optional_train_metrics"):
        otm = summary["optional_train_metrics"]
        print("Optional train signals:")
        if otm.get("condition_signals"):
            for name, info in otm["condition_signals"].items():
                st = info["stats"] or {}
                print(
                    f"  {name} ({info['tag']}): first_w={st.get('first_window_mean', math.nan):.6f}, "
                    f"last_w={st.get('last_window_mean', math.nan):.6f}"
                )
        if otm.get("branch_signals", {}).get("branch_entropy"):
            be = otm["branch_signals"]["branch_entropy"]
            st = be["stats"] or {}
            print(
                f"  branch_entropy ({be['tag']}): first_w={st.get('first_window_mean', math.nan):.6f}, "
                f"last_w={st.get('last_window_mean', math.nan):.6f}"
            )
        bwd = otm.get("branch_signals", {}).get("branch_weight_distribution", {})
        if bwd:
            if not bwd.get("available", False):
                print(f"  branch_weight_distribution unavailable: {bwd.get('reason', 'unknown')}")
            else:
                print(
                    f"  branch_weight_distribution: dominant={bwd['dominant_branch_tag_last_window']}, "
                    f"dominant_mean_w={bwd['dominant_branch_mean_weight_last_window']:.6f}, "
                    f"mean_max_w={bwd['mean_max_weight_last_window']:.6f}, "
                    f"mean_entropy={bwd['mean_entropy_last_window']:.6f}"
                )
        if otm.get("missing_signals"):
            print(f"  missing_signals: {', '.join(otm['missing_signals'])}")

    if summary.get("metric_robustness"):
        mr = summary["metric_robustness"]
        iou_sweep = mr.get("iou_sweep_same_setting", [])
        if iou_sweep:
            print("Target-setting IoU sweep (last values):")
            for item in iou_sweep:
                print(
                    f"  iou={item['iou_thr']:.2f}: last={item['last_value']:.6f} (epoch {item['last_epoch']}), "
                    f"best={item['best_value']:.6f} (epoch {item['best_epoch']})"
                )
            iou_sum = mr.get("iou_sweep_summary", {})
            if iou_sum:
                print(
                    f"  IoU sweep gap(max-min)={iou_sum['last_value_gap_max_min']:.6f}, "
                    f"std={iou_sum['last_value_std']:.6f}"
                )
        mtg = mr.get("metric_type_gap", [])
        if mtg:
            print("BEV-3D gap by IoU (last values):")
            for item in mtg:
                print(
                    f"  iou={item['iou_thr']:.2f}: BEV={item['last_BEV']:.6f}, 3D={item['last_3D']:.6f}, "
                    f"gap={item['gap_BEV_minus_3D']:.6f}"
                )
        gr = mr.get("group_robustness")
        if gr:
            if gr.get("available", False):
                print(
                    f"Group robustness: groups={gr['num_groups']}, "
                    f"worst={gr['worst_group']:.6f}, best={gr['best_group']:.6f}, "
                    f"gap={gr['gap_best_minus_worst']:.6f}"
                )
            else:
                print(f"Group robustness unavailable: {gr.get('reason', 'unknown')}")

    if summary["score_distribution"]:
        print("Score distribution per epoch:")
        for item in summary["score_distribution"]:
            print(
                f"  {item['split']} epoch {item['epoch']} conf={item['conf_thr']} "
                f"count={item['count']} mean={item['mean']:.6f} std={item['std']:.6f} "
                f"p10={item['p10']:.6f} p50={item['p50']:.6f} p90={item['p90']:.6f}"
            )

    if summary["recommendation"]:
        r = summary["recommendation"]
        print(
            f"Recommended checkpoint: epoch {r['recommended_epoch']} -> {r['recommended_model_path']}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--metric-split", type=str, default="val_tot", choices=["val_tot", "minival"])
    parser.add_argument("--metric-type", type=str, default="3D", choices=["3D", "BEV"])
    parser.add_argument("--metric-conf", type=str, default="0.3")
    parser.add_argument("--metric-iou", type=str, default="0.3")
    parser.add_argument("--metric-cls", type=str, default="sed")
    parser.add_argument("--score-conf", type=str, default="0.3")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize(args.exp, args)
    print_summary(summary)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary json: {args.json_out}")


if __name__ == "__main__":
    main()
