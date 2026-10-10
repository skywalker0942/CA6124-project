"""Assignment tuner using Hyperopt with DaisyRec's existing data/model pipeline.

This extends DaisyRec's tune workflow while leaving its Optuna-based tune.py
untouched. Expected command-line options are the same as tune.py.
"""

import json
from copy import deepcopy
from logging import getLogger

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe

from daisy.model.Models import RecommenderModel
from daisy.utils.config import init_config, init_logger, init_seed, param_type_config
from daisy.utils.dataset import AEDataset, BasicDataset, CandidatesDataset, get_dataloader
from daisy.utils.fpmc_data import build_fpmc_loader, set_fpmc_eval_context
from daisy.utils.loader import Preprocessor, RawDataReader
from daisy.utils.metrics import metrics_config
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from daisy.utils.splitter import TestSplitter, ValidationSplitter
from daisy.utils.utils import (
    build_candidates_set, ensure_dir, get_history_matrix, get_inter_matrix, get_ur,
)


def build_search_space(param_dict, tunable_names):
    space = {}
    for name in tunable_names:
        if name not in param_dict:
            continue
        spec = param_dict[name]
        if isinstance(spec, list):
            space[name] = hp.choice(name, spec)
        elif isinstance(spec, dict):
            low, high, step = spec["min"], spec["max"], spec.get("step")
            kind = param_type_config[name]
            if kind == "int":
                space[name] = hp.quniform(name, low, high, step or 1)
            elif kind == "float":
                space[name] = hp.uniform(name, low, high) if step is None else hp.quniform(name, low, high, step)
            else:
                raise ValueError(f"Unsupported Hyperopt parameter type for {name}: {kind}")
        else:
            raise ValueError(f"Invalid search-space entry for {name}: {spec!r}")
    if not space:
        raise ValueError("No tunable parameters found; provide a non-empty tune_pack")
    return space


def cast_params(params):
    out = dict(params)
    for name in ("factors", "batch_size", "num_ng", "num_layers", "epochs", "maxk"):
        if name in out:
            out[name] = int(round(out[name]))
    return out


def train_model(config, train):
    algo = config["algo_name"].lower()
    if algo == "fpmc":
        set_fpmc_eval_context(config, train)
    model = RecommenderModel(algo)(config)
    if algo == "fpmc":
        loader = build_fpmc_loader(train, config, shuffle=True)
    elif algo in ("itemknn", "puresvd", "slim", "mostpop", "ease"):
        model.fit(train)
        return model
    elif algo == "multi-vae":
        ids, values, _ = get_history_matrix(train, config, row="user")
        config["history_item_id"], config["history_item_value"] = ids, values
        dataset = AEDataset(train, yield_col=config["UID_NAME"])
        loader = get_dataloader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    elif algo in ("mf", "fm", "neumf", "nfm", "ngcf", "lightgcn"):
        if algo in ("ngcf", "lightgcn"):
            config["inter_matrix"] = get_inter_matrix(train, config)
        samples = BasicNegtiveSampler(train.copy(), config).sampling()
        loader = get_dataloader(BasicDataset(samples), batch_size=config["batch_size"], shuffle=True, num_workers=0)
    elif algo == "item2vec":
        samples = SkipGramNegativeSampler(train.copy(), config).sampling()
        loader = get_dataloader(BasicDataset(samples), batch_size=config["batch_size"], shuffle=True, num_workers=0)
    else:
        raise NotImplementedError(f"No training path is registered for {algo}")
    model.fit(loader)
    return model


def main():
    config = init_config()
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config["logger"] = logger

    param_dict = json.loads(config["tune_pack"])
    algo = config["algo_name"].lower()
    tunable_names = RecommenderModel(algo).tunable_param_names
    space = build_search_space(param_dict, tunable_names)

    reader, processor = RawDataReader(config), Preprocessor(config)
    df = processor.process(reader.get_data())
    base_config = deepcopy(config)
    base_config["user_num"], base_config["item_num"] = processor.user_num, processor.item_num
    train_indices, _ = TestSplitter(base_config).split(df)
    train_set = df.iloc[train_indices].copy()
    validation_splitter = ValidationSplitter(base_config)

    def objective(sampled):
        trial_config = deepcopy(base_config)
        trial_config.update(cast_params(sampled))
        scores = []
        for train_indices, validation_indices in validation_splitter.split(train_set):
            train = train_set.iloc[train_indices].copy()
            validation = train_set.iloc[validation_indices].copy()
            train_ur, val_ur = get_ur(train), get_ur(validation)
            trial_config["train_ur"] = train_ur
            model = train_model(trial_config, train)
            users, candidates = build_candidates_set(val_ur, train_ur, trial_config)
            loader = get_dataloader(CandidatesDataset(candidates), batch_size=128,
                                    shuffle=False, num_workers=0)
            predictions = model.rank(loader)
            scores.append(metrics_config[trial_config["optimization_metric"]]["evaluator"](
                val_ur, predictions, users))
        mean_score = float(np.mean(scores))
        logger.info("Hyperopt trial score=%0.6f params=%s", mean_score, cast_params(sampled))
        return {"loss": -mean_score, "status": STATUS_OK, "score": mean_score,
                "params": cast_params(sampled)}

    trials = Trials()
    best_encoded = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=int(config["hyperopt_trail"]),
        trials=trials,
        rstate=np.random.default_rng(int(config["seed"])),
        show_progressbar=False,
    )
    best = cast_params(space_eval(space, best_encoded))
    ensure_dir("./tune_res/")
    prefix = f"{config['loss_type']}_{algo}_{config['dataset']}_{config['prepro']}_{config['val_method']}"
    trial_rows = [
        {"trial": i + 1, "validation_score": trial["result"]["score"],
         "params": json.dumps(trial["result"]["params"], sort_keys=True)}
        for i, trial in enumerate(trials.trials)
    ]
    import pandas as pd
    pd.DataFrame(trial_rows).to_csv(f"./tune_res/trials_{prefix}.csv", index=False)
    with open(f"./tune_res/best_params_{prefix}.json", "w", encoding="utf-8") as stream:
        json.dump({"best_validation_score": -float(trials.best_trial["result"]["loss"]),
                   "best_params": best}, stream, indent=2)
    logger.info("Best validation score=%0.6f params=%s",
                -float(trials.best_trial["result"]["loss"]), best)


if __name__ == "__main__":
    main()
