import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.lib.fast_data_loader import FastDataLoader

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score
)
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def softmax_tem(z,T):
    z=z/T
    max_z = torch.max(z,dim=1,keepdim=True)[0]
    exp_z = torch.exp(z-max_z)
    sum_exp_z = torch.sum(exp_z,dim=1,keepdim=True)
    y= exp_z/sum_exp_z
    return y
def nsoftmax_tem(z):

    max_z = torch.max(z,dim=1,keepdim=True)[0]
    exp_z = torch.exp(z-max_z)
    sum_exp_z = torch.sum(exp_z,dim=1,keepdim=True)
    y= exp_z/sum_exp_z
    return y
def accuracy_from_loader(algorithm, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()
    # algorithm1.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = algorithm.predict(x)
                loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()
    # if hasattr(algorithm,'featurizer'):
    #     lora.mark_only_attn_as_trainable(algorithm.featurizer)
    # else :
    #     lora.mark_only_attn_as_trainable(algorithm.module.featurizer)
    acc = correct / total
    loss = losssum / total
    return acc, loss

def accuracy_from_loader_en(algorithms, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    # algorithm.eval()
    # algorithm1.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits=0
            for i in range(len(algorithms)):
                logits += algorithms[i].predict(x)
            logits=logits/len(algorithms)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    acc = correct / total
    loss = losssum / total
    return acc, loss



def accuracy(algorithm, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, **kwargs)


def accuracy_en(algorithms, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader_en(algorithms, loader, weights, **kwargs)


class Evaluator_:
    def __init__(
        self, test_envs,train_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy(algorithm,loader_kwargs, weights, debug=self.debug)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy(algorithm,loader_kwargs, weights, debug=self.debug)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
        
    def evaluate_en(self, algorithms, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy_en(algorithms,loader_kwargs, weights, debug=self.debug)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries


# ------------------------------------------------------------------------------

def get_predictions(algorithm, loader_kwargs, weights, device):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)

    all_probs, all_preds, all_labels, all_weights, all_losses = [], [], [], [], []
    weights_offset = 0

    algorithm.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                p = algorithm.predict(x)

            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)

            p_f32 = p.float()  # estabilidad numérica

            if p.size(1) == 1:
                probs = torch.sigmoid(p).view(-1)
                preds = (p.view(-1) > 0).long()
                losses = F.binary_cross_entropy_with_logits(
                    p_f32.view(-1), y.float(), reduction="none"
                )
            else:
                probs = torch.softmax(p, dim=1)
                preds = p.argmax(1)
                losses = F.cross_entropy(p_f32, y, reduction="none")

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_weights.append(batch_weights.cpu())
            all_losses.append(losses.cpu())

    algorithm.train()

    return {
        "probs": torch.cat(all_probs).numpy(),
        "preds": torch.cat(all_preds).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "weights": torch.cat(all_weights).numpy(),
        "losses": torch.cat(all_losses).numpy(),
    }
    
def balanced_accuracy(results):
    return balanced_accuracy_score(
        results["labels"], results["preds"], sample_weight=results["weights"]
    )


def macro_f1(results):
    return f1_score(
        results["labels"],
        results["preds"],
        average="macro",
        sample_weight=results["weights"],
    )

def binary_f1(results, positive_class, num_classes=None):
    """F1 binario con `positive_class` como clase positiva.

    Devuelve nan si el problema no es binario, en vez de fallar.
    """
    if num_classes is not None and num_classes != 2:
        return float("nan")

    labels = np.asarray(results["labels"])
    preds = np.asarray(results["preds"])
    weights = np.asarray(results["weights"])

    # Red de seguridad: si aparece cualquier clase fuera de {0, 1}
    # (en labels o en preds), average="binary" lanzaría ValueError.
    if not np.isin(np.union1d(labels, preds), [0, 1]).all():
        return float("nan")

    return f1_score(
        labels,
        preds,
        average="binary",
        pos_label=positive_class,
        sample_weight=weights,
        zero_division=0,
    )


def make_binary_f1(positive_class, num_classes=None):
    def metric_fn(results):
        return binary_f1(results, positive_class, num_classes)
    return metric_fn

def _sensitivity_specificity_per_class(results):
    """Calcula sensitivity y specificity one-vs-rest para cada clase."""
    labels = results["labels"]
    preds = results["preds"]
    weights = results["weights"]

    out = {}

    for c in np.unique(labels):
        y_true_bin = (labels == c).astype(int)
        y_pred_bin = (preds == c).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin,
            y_pred_bin,
            sample_weight=weights,
            labels=[0, 1],
        ).ravel()

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else float("nan")
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else float("nan")
        )

        out[int(c)] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    return out

def mean_loss(results):
    return float(np.average(results["losses"], weights=results["weights"]))


def auroc(results):
    from torchmetrics.functional.classification import multiclass_auroc
    labels, probs, w = results["labels"], results["probs"], results["weights"]

    if len(np.unique(labels)) < 2:
        # AUROC no está definido si el dominio solo tiene una clase presente
        return float("nan")

    if probs.ndim == 1:
        # Red con salida escalar (p.size(1) == 1 en get_predictions)
        return roc_auc_score(labels, probs, sample_weight=w)
    elif probs.shape[1] == 2:
        # Softmax de 2 columnas pero problema binario -> solo prob. de la clase positiva
        return roc_auc_score(labels, probs[:, 1], sample_weight=w)
    else:
        # Multiclase real (3+ clases): macro-AUROC one-vs-rest robusto
        labels = np.asarray(labels).astype(np.int64).ravel()
        probs = np.asarray(probs, dtype=np.float64)   # float64 para evitar problemas de precisión
        if w is not None:
            w = np.asarray(w, dtype=np.float64).ravel()

        # Descartar filas con NaN/inf (una sola fila rompería toda la métrica)
        finite = np.isfinite(probs).all(axis=1)
        if not finite.all():
            print(f"[auroc] descartando {(~finite).sum()} filas con NaN/inf")
            labels, probs = labels[finite], probs[finite]
            if w is not None:
                w = w[finite]

        # Clases evaluables: con positivos y negativos en este dominio
        C = probs.shape[1]
        counts = np.bincount(labels, minlength=C)
        valid = (counts > 0) & (counts < len(labels))
        if not valid.any():
            return float("nan")

        uniform_w = w is None or np.allclose(w, w.flat[0])
        if uniform_w:
            # Sin pesos relevantes: torchmetrics (no exige que las filas sumen 1)
            per_class = multiclass_auroc(
                torch.from_numpy(probs),
                torch.from_numpy(labels),
                num_classes=C,
                average=None,
            )
            return float(per_class[torch.from_numpy(valid)].mean().item())
        else:
            # Pesos no uniformes: OvR manual con sklearn, que sí soporta sample_weight
            aucs = [
                roc_auc_score((labels == c).astype(int), probs[:, c], sample_weight=w)
                for c in np.flatnonzero(valid)
            ]
            return float(np.mean(aucs))


def _accuracy(results):
    return accuracy_score(
        results["labels"], results["preds"], sample_weight=results["weights"]
    )

def sensitivity(results):
    """Sensitivity one-vs-rest para cada clase."""
    per_class = _sensitivity_specificity_per_class(results)

    return {
        c: values["sensitivity"]
        for c, values in per_class.items()
    }

def specificity(results):
    """Specificity one-vs-rest para cada clase."""
    per_class = _sensitivity_specificity_per_class(results)

    return {
        c: values["specificity"]
        for c, values in per_class.items()
    }

def macro_sensitivity(results):
    """Media macro de sensitivity entre clases."""
    values = sensitivity(results).values()
    return float(np.nanmean(list(values)))

def macro_specificity(results):
    """Media macro de specificity entre clases."""
    values = specificity(results).values()
    return float(np.nanmean(list(values)))

class LODOVal_Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False,
        target_env=None, device="cuda", extra_metrics=None,
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.device = device

        # métricas adicionales a calcular por split, además de acc/loss
        # ej: {"balanced_acc": balanced_accuracy, "macro_f1": macro_f1, "auroc": auroc}
        self.extra_metrics = extra_metrics or {}

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False, ret_extra=False, ret_preds=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        summaries = collections.defaultdict(float)
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}
        extra = {}  # "env{N}_{in|out}_{metric}" -> valor
        predictions = {}

        # Se puede modificar para también considerar los casos en los que:
        # - Se tiene un único dominio sin split de out

        for name, loader_kwargs, weights in self.eval_meta:
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs

            # una sola pasada por el loader
            results = get_predictions(algorithm, loader_kwargs, weights, self.device)

            if ret_preds:
                predictions[name] = results

            acc = _accuracy(results)
            loss = mean_loss(results)

            accuracies[f"{name}_acc"] = acc          # ej: env0_in_acc
            losses[f"{name}_loss"] = loss

            if self.extra_metrics:
                for metric_name, metric_fn in self.extra_metrics.items():
                    extra[f"{name}_{metric_name}"] = metric_fn(results)  # ej: env0_out_f1

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        outputs = (accuracies, summaries)
        if ret_losses:
            outputs = outputs + (losses,)
        if ret_extra:
            outputs = outputs + (extra,)
        if ret_preds:
            outputs = outputs + (predictions,)
        return outputs