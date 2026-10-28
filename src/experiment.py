"""Episode-disjoint imitation learning and paired closed-loop evaluation.

Run from the repository root: python -m src.experiment
"""
import json
import platform
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import differential_evolution
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['cos_theta', 'sin_theta', 'theta_dot']
CONFIG = dict(seed=42, gravity=9.81, dt=0.05, steps=200,
              expert_episodes=36, horizon=16, de_iterations=10, de_popsize=5,
              evaluation_per_scenario=20, bootstrap_replicates=2000)


def angle(x):
    return np.arctan2(x[..., 1], x[..., 0])


def transition(theta, velocity, torque):
    """Gymnasium Pendulum dynamics, mass=length=1; vectorized over candidates."""
    velocity = np.clip(velocity + (3 * CONFIG['gravity'] / 2 * np.sin(theta)
                                  + 3 * np.clip(torque, -2, 2)) * CONFIG['dt'], -8, 8)
    return theta + velocity * CONFIG['dt'], velocity


def mpc(obs):
    """Optimize a bounded action sequence; apply its first torque only."""
    def objective(sequences):
        theta = np.full(sequences.shape[1], angle(obs))
        velocity = np.full(sequences.shape[1], obs[2])
        costs = np.zeros(sequences.shape[1])
        for torque in sequences:
            wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
            costs += wrapped ** 2 + 0.1 * velocity ** 2 + 0.001 * torque ** 2
            theta, velocity = transition(theta, velocity, torque)
        return costs
    result = differential_evolution(objective, [(-2, 2)] * CONFIG['horizon'],
        maxiter=CONFIG['de_iterations'], popsize=CONFIG['de_popsize'],
        seed=CONFIG['seed'], vectorized=True, updating='deferred', polish=False)
    return float(result.x[0])


def episode(policy, initial_angle, initial_velocity, seed):
    env = gym.make('Pendulum-v1', g=CONFIG['gravity'])
    env.reset(seed=int(seed))
    env.unwrapped.state = np.array([initial_angle, initial_velocity], dtype=float)
    obs = np.array([np.cos(initial_angle), np.sin(initial_angle), initial_velocity], dtype=np.float32)
    rows = []
    for step in range(CONFIG['steps']):
        start = time.perf_counter()
        torque = float(np.clip(policy(obs), -2, 2))
        latency_ms = (time.perf_counter() - start) * 1000
        next_obs, reward, terminated, truncated, _ = env.step(np.array([torque], dtype=np.float32))
        rows.append(dict(step=step, cos_theta=obs[0], sin_theta=obs[1], theta_dot=obs[2],
            torque=torque, reward=reward, next_angle_deg=np.rad2deg(angle(next_obs)),
            next_velocity=next_obs[2], latency_ms=latency_ms))
        obs = next_obs
        if terminated or truncated:
            break
    env.close()
    return pd.DataFrame(rows)


def check_dynamics():
    """Compare the fast planning dynamics against the actual environment."""
    env = gym.make('Pendulum-v1', g=CONFIG['gravity'])
    rng = np.random.default_rng(707)
    for _ in range(100):
        theta, velocity, torque = rng.uniform(-np.pi, np.pi), rng.uniform(-8, 8), rng.uniform(-2, 2)
        env.reset(seed=1)
        env.unwrapped.state = np.array([theta, velocity])
        observed, reward, _, _, _ = env.step(np.array([torque], dtype=np.float32))
        predicted_theta, predicted_velocity = transition(theta, velocity, np.float32(torque))
        np.testing.assert_allclose(observed, [np.cos(predicted_theta), np.sin(predicted_theta), predicted_velocity], atol=1e-6)
        expected_cost = ((theta + np.pi) % (2*np.pi) - np.pi)**2 + .1*velocity**2 + .001*float(np.float32(torque))**2
        np.testing.assert_allclose(-reward, expected_cost, atol=1e-6)
    env.close()


def generate_demonstrations():
    rng = np.random.default_rng(CONFIG['seed'])
    episodes = []
    for i in range(CONFIG['expert_episodes']):
        a, v = rng.uniform(-np.deg2rad(20), np.deg2rad(20)), rng.uniform(-.1, .1)
        frame = episode(mpc, a, v, 1000+i)
        frame['episode'] = i
        episodes.append(frame)
        if (i+1) % 6 == 0:
            print(f'Generated {i+1}/{CONFIG["expert_episodes"]} expert episodes', flush=True)
    data = pd.concat(episodes, ignore_index=True)
    # Split entire episodes before fitting transforms; keep unsuccessful trajectories.
    ids = rng.permutation(CONFIG['expert_episodes'])
    mapping = {int(i): ('train' if k < 22 else 'validation' if k < 29 else 'test') for k, i in enumerate(ids)}
    data['split'] = data.episode.map(mapping)
    groups = [set(data.loc[data.split == split, 'episode']) for split in ['train', 'validation', 'test']]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return data


def regression_metrics(y, prediction):
    prediction = np.clip(prediction, -2, 2)
    return dict(MAE=float(mean_absolute_error(y, prediction)),
                RMSE=float(np.sqrt(mean_squared_error(y, prediction))), R2=float(r2_score(y, prediction)))


def train_models(data):
    train, val, test = [data[data.split == s] for s in ['train', 'validation', 'test']]
    models = {'Mean action': DummyRegressor(), 'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0))}
    candidates = {}
    for width in [32, 64]:
        candidates[f'MLP {width}x{width}'] = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(width, width), activation='tanh', alpha=.01,
            max_iter=250, batch_size=128, learning_rate_init=.001,
            early_stopping=False, random_state=CONFIG['seed'], tol=1e-6, n_iter_no_change=25))
    validation = []
    for name, model in {**models, **candidates}.items():
        model.fit(train[FEATURES], train.torque)
        validation.append(dict(model=name, **regression_metrics(val.torque, model.predict(val[FEATURES]))))
    scores = pd.DataFrame(validation)
    selected = scores[scores.model.isin(candidates)].sort_values('RMSE').iloc[0].model
    models['Neural network'] = candidates[selected]
    results = []
    predictions = []
    for name, model in models.items():
        for split, frame in [('train', train), ('validation', val), ('test', test)]:
            pred = np.clip(model.predict(frame[FEATURES]), -2, 2)
            results.append(dict(model=name, split=split, **regression_metrics(frame.torque, pred)))
            if split == 'test':
                predictions.append(pd.DataFrame(dict(episode=frame.episode, step=frame.step,
                    actual=frame.torque, predicted=pred, model=name)))
    return models, scores, pd.DataFrame(results), pd.concat(predictions), str(selected)


def model_policy(model):
    # NumPy forward pass avoids constructing 8,000 tiny DataFrames per controller.
    if hasattr(model, 'steps'):
        scaler, estimator = model.steps[0][1], model.steps[1][1]
        def policy(obs):
            x = (obs - scaler.mean_) / scaler.scale_
            if isinstance(estimator, MLPRegressor):
                for w, b in zip(estimator.coefs_[:-1], estimator.intercepts_[:-1]):
                    x = np.tanh(x @ w + b)
                return float((x @ estimator.coefs_[-1] + estimator.intercepts_[-1]).item())
            return float(x @ estimator.coef_ + estimator.intercept_)
        return policy
    return lambda obs: float(model.constant_.item())


def evaluate(models):
    rng = np.random.default_rng(2026)
    records, trajectories = [], []
    for scenario, degrees, velocity in [('Near upright', 20, .1), ('Wider initial states', 90, 1.0)]:
        for i in range(CONFIG['evaluation_per_scenario']):
            a, v = rng.uniform(-np.deg2rad(degrees), np.deg2rad(degrees)), rng.uniform(-velocity, velocity)
            random_rng = np.random.default_rng(9000+i)
            policies = {'Random': lambda obs: random_rng.uniform(-2, 2), 'Zero torque': lambda obs: 0.,
                        **{n: model_policy(m) for n, m in models.items()}, 'MPC': mpc}
            for name, policy in policies.items():
                frame = episode(policy, a, v, 3000+i)
                records.append(dict(scenario=scenario, episode=i, controller=name,
                    initial_angle_deg=np.rad2deg(a), initial_velocity=v, score=frame.reward.sum(),
                    upright_fraction=((frame.next_angle_deg.abs() < 10) & (frame.next_velocity.abs() < .5)).mean(),
                    mean_abs_torque=frame.torque.abs().mean(), final_abs_angle_deg=abs(frame.iloc[-1].next_angle_deg),
                    median_latency_ms=frame.latency_ms.median()))
                if i == 0:
                    trajectories.append(frame.assign(scenario=scenario, episode=i, controller=name))
            if (i+1) % 5 == 0:
                print(f'Evaluated {scenario}: {i+1}', flush=True)
    return pd.DataFrame(records), pd.concat(trajectories, ignore_index=True)


def bootstrap(values, seed=42):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(CONFIG['bootstrap_replicates'], len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975])


def summarize(evaluation):
    summaries, differences = [], []
    for (scenario, controller), group in evaluation.groupby(['scenario', 'controller']):
        lo, hi = bootstrap(group.score)
        summaries.append(dict(scenario=scenario, controller=controller, n=len(group),
            mean_score=group.score.mean(), ci_low=lo, ci_high=hi,
            upright_fraction=group.upright_fraction.mean(), median_latency_ms=group.median_latency_ms.median()))
    for scenario, group in evaluation.groupby('scenario'):
        pivot = group.pivot(index='episode', columns='controller', values='score')
        for baseline in ['Random', 'Zero torque', 'Ridge', 'MPC']:
            delta = pivot['Neural network'] - pivot[baseline]
            lo, hi = bootstrap(delta)
            differences.append(dict(scenario=scenario, contrast=f'Neural network minus {baseline}',
                mean_difference=delta.mean(), ci_low=lo, ci_high=hi))
    return pd.DataFrame(summaries), pd.DataFrame(differences)


def plot_results(data, models, validation, metrics, predictions, evaluation, summary, trajectories):
    images = ROOT/'images'; images.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(name):
        plt.tight_layout(); plt.savefig(images/name, bbox_inches='tight'); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(np.rad2deg(angle(data[FEATURES].to_numpy())), data.torque, s=3, alpha=.2, color='#247ba0')
    axes[0].set(xlabel='State angle (degrees)', ylabel='Expert torque', title='Expert state–action coverage')
    data.groupby(['split']).size().reindex(['train','validation','test']).plot.bar(ax=axes[1], color='#247ba0', rot=0)
    axes[1].set(title='Entire episodes stay in one split', ylabel='State–action rows', xlabel='')
    save('dataset_coverage.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(models['Neural network'].steps[-1][1].loss_curve_, color='#e76f51')
    axes[0].set(title='Selected MLP training loss', xlabel='Iteration', ylabel='Training loss')
    scores = metrics[metrics.split=='test']
    axes[1].bar(scores.model, scores.RMSE, color=['#9aa5b1','#247ba0','#e76f51'])
    axes[1].set(title='Episode-disjoint test error', ylabel='Torque RMSE')
    save('model_evaluation.png')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax, name in zip(axes, ['Ridge', 'Neural network']):
        frame = predictions[predictions.model == name]
        ax.scatter(frame.actual, frame.predicted, s=5, alpha=.2)
        ax.plot([-2,2],[-2,2],color='#e76f51',linestyle='--')
        ax.set(title=name, xlabel='MPC torque', ylabel='Predicted torque', xlim=(-2.1,2.1), ylim=(-2.1,2.1))
    save('predicted_vs_actual.png')
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax,(scenario,group) in zip(axes,summary.groupby('scenario',sort=False)):
        ax.barh(group.controller, group.mean_score, color=['#9aa5b1' if n!='Neural network' else '#e76f51' for n in group.controller])
        ax.errorbar(group.mean_score, np.arange(len(group)), xerr=[group.mean_score-group.ci_low,group.ci_high-group.mean_score],fmt='none',ecolor='black',capsize=3)
        ax.set(title=scenario, xlabel='Mean 200-step return (higher is better)')
    fig.suptitle('Matched initial conditions · 95% episode bootstrap intervals')
    save('controller_comparison.png')
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax,(scenario,group) in zip(axes,trajectories.groupby('scenario',sort=False)):
        for name in ['Random','Ridge','Neural network','MPC']:
            f=group[group.controller==name];ax.plot(f.step, f.next_angle_deg,label=name,alpha=.85)
        ax.axhline(0,color='grey',linestyle=':');ax.set(title=scenario, xlabel='Step',ylabel='Post-action angle (degrees)')
    axes[1].legend(fontsize=8);save('example_trajectories.png')
    fig, ax=plt.subplots(figsize=(9,4))
    group=summary[summary.scenario=='Near upright']
    ax.bar(group.controller,group.median_latency_ms,color='#247ba0');ax.set_yscale('log')
    ax.set(title='Policy latency on this machine (includes action clipping)',ylabel='Median milliseconds per action · log scale')
    plt.xticks(rotation=20);save('policy_latency.png')


def main():
    output=ROOT/'results';output.mkdir(exist_ok=True)
    check_dynamics()
    data=generate_demonstrations()
    data.to_csv(output/'demonstrations.csv',index=False)
    models,validation,metrics,predictions,selected=train_models(data)
    # Ensure the lightweight deployment forward pass matches fitted sklearn predictions.
    for name,model in models.items():
        for row in data[FEATURES].iloc[:20].to_numpy():
            expected=float(model.predict(pd.DataFrame([row],columns=FEATURES))[0])
            np.testing.assert_allclose(model_policy(model)(row),expected,atol=1e-5)
    evaluation,trajectories=evaluate(models)
    summary,differences=summarize(evaluation)
    for name,frame in [('demonstrations',data),('validation',validation),('regression_metrics',metrics),
        ('test_predictions',predictions),('controller_episodes',evaluation),('controller_summary',summary),
        ('paired_differences',differences),('example_trajectories',trajectories)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    manifest=dict(config=CONFIG,selected_model=selected,mlp_iterations=int(models['Neural network'].steps[-1][1].n_iter_),mlp_iteration_limit=250,split_episodes=data.groupby('split').episode.nunique().to_dict(),
        rows=len(data),versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,scikit_learn=sklearn.__version__,gymnasium=gym.__version__),
        notes=['Single training seed; confidence intervals describe evaluation initial states only.',
               'Near-upright training is not general swing-up training.', 'Latency is machine-dependent; no real-time guarantee.'])
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    model=models['Neural network']; scaler,mlp=model.steps[0][1],model.steps[1][1]
    np.savez(output/'policy_weights.npz',mean=scaler.mean_,scale=scaler.scale_,
        **{f'weight_{i}':w for i,w in enumerate(mlp.coefs_)}, **{f'bias_{i}':b for i,b in enumerate(mlp.intercepts_)})
    plot_results(data,models,validation,metrics,predictions,evaluation,summary,trajectories)
    print(metrics.to_string(index=False),flush=True)
    print(summary.to_string(index=False),flush=True)
    print('PASS: dynamics, episode separation and deployment parity checks.',flush=True)


if __name__=='__main__':
    main()
