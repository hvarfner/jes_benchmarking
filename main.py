import json
import os
import time

import fire
import gpytorch
import pandas as pd
import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.utils import get_optimal_samples
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.test_functions import *
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from log_transformed_interval import LogTransformedInterval


def get_covar_module(d: int):
    # print(f"getting covar module")
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_constraint=LogTransformedInterval(0.031 , 1e2, initial_value=1.0),
        ),
        outputscale_constraint=LogTransformedInterval(0.031 , 1e2, initial_value=1),
    )
    return covar_module


def get_likelihood():
    likelihood = GaussianLikelihood(
        # NOTE implies std in [1e-2, 1e-1], shoud check for noisy experiments
        # if we allow std down to 1e-3, model fitting errors pop up for constraints
        noise_constraint=LogTransformedInterval(1e-4, 1e0, initial_value=1e-3)
    )
    return likelihood


def fit_gp_model(
    train_X: Tensor, train_Y: Tensor, bounds: Tensor, dim: int, noise_std: float | None,
):
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=torch.ones_like(train_Y) * noise_std**2,
        covar_module=get_covar_module(d=dim),
        likelihood=get_likelihood() if noise_std is None else None,
        input_transform=Normalize(d=dim, bounds=bounds),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def save_results(
    data: dict, function_name: str, dim: int, acquisition_function: str, seed: int, directory: str,
):
    directory = f"{directory}/{function_name}_{dim}/{acquisition_function}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/seed_{seed}.json", "w") as f:
        json.dump(data, f, indent=4)


def run_bo(
    acq: str,
    dim: int,
    f="Ackley",
    seed: int = 42,
    iters: int = 200,
    noise: float = 0.0,
    directory: str = "results",
):
    torch.manual_seed(seed)
    fun = globals()[f](dim=dim, negate=True, noise_std=noise)
    if f == "Ackley":
        bounds = torch.Tensor([(-32.768 / 4, 32.768 / 2) for _ in range(fun.dim)]).T
    elif f == "Levy":
        bounds = torch.Tensor([(-5, 5) for _ in range(fun.dim)]).T
    else:
        bounds = fun.bounds

    n_initial_points = 2 * dim
    train_X = (
        draw_sobol_samples(bounds, n=n_initial_points, q=1).squeeze(-2).to(torch.double)
    )
    train_Y = fun(train_X).unsqueeze(-1)

    results = []
    for i in range(iters):
        t0 = time.time()
        gp = fit_gp_model(train_X, train_Y, bounds, fun.dim, noise)

        t1 = time.time()
        if acq in ["jes", "pes"]:
            optimal_inputs, optimal_outputs = get_optimal_samples(
                model=gp, bounds=bounds, num_restarts=4, raw_samples=2048, num_optima=64
            )

        if acq == "jes":
            acqf = qJointEntropySearch(
                model=gp, optimal_inputs=optimal_inputs, optimal_outputs=optimal_outputs
            )
        elif acq == "pes":
            acqf = qPredictiveEntropySearch(model=gp, optimal_inputs=optimal_inputs)
        elif acq == "logei":
            acqf = qLogNoisyExpectedImprovement(
                model=gp, X_baseline=train_X, prune_baseline=True
            )
        else:
            raise ValueError(f"No such acquisition function '{acq}'")
        t2 = time.time()
        print(t2-t1)
        
        if acq == "logei":
            candidate, acqval = optimize_acqf(
                acqf,
                bounds,
                q=1,
                num_restarts=16,
                raw_samples=1024,
                options={
                    "sample_around_best": False,
                    "batch_limit": 128,
                },
            )

            candidate, acqval_sab = optimize_acqf(
                acqf,
                bounds,
                q=1,
                num_restarts=16,
                raw_samples=512,
                options={
                    "sample_around_best": True,
                    "batch_limit": 128,
                },
            )
        else:
            candidate, acqval = optimize_acqf(
                acqf,
                bounds,
                q=1,
                num_restarts=16,
                raw_samples=1024,
                options={
                    "sample_around_best": False,
                    "batch_limit": 128,
                },
            )

            candidate, acqval_sab = optimize_acqf(
                acqf,
                bounds,
                q=1,
                num_restarts=16,
                raw_samples=512,
                ic_generator=gen_batch_initial_conditions,
                options={
                    "sample_around_best": True,
                    "batch_limit": 128,
                },
            )


        t3 = time.time()

        inference_loc, _ = optimize_acqf(
            PosteriorMean(model=gp),
            bounds,
            q=1,
            num_restarts=16,
            raw_samples=4096,
            options={
                "sample_around_best": True,
                "batch_limit": 512,
                     
            },
        )

        new_X = candidate.detach()
        new_Y = fun(new_X).unsqueeze(-1)
        new_f = fun(new_X, noise=False).unsqueeze(-1)
        out_of_sample_f = fun(inference_loc, noise=False).unsqueeze(-1)

        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        best_f = fun(train_X, noise=False).max()
        best_X = train_X[gp.posterior(train_X).mean.argmax()]
        in_sample_f = fun(best_X, noise=False).unsqueeze(-1)

        results.append(
            {
                "iteration": i,
                "new_X": new_X.tolist(),
                "new_Y": new_Y.item(),
                "acqval": acqval.item(),
                "acqval_sab": acqval_sab.item(),
                "new_f": new_f.item(),
                "best_f": best_f.item(),
                "in_sample_f": in_sample_f.item(),
                "out_of_sample_f": out_of_sample_f.item(),
                "fit_time": t1 - t0,
                "setup_time": t2 - t1,
                "opt_time": t3 - t2,
            }
        )

        save_results(
            data=results,
            function_name=f,
            dim=dim,
            acquisition_function=acq,
            seed=seed,
            directory=directory,
        )
        print(f"Iteration {i}")


if __name__ == "__main__":
    fire.Fire(run_bo)
