"""Run model selection on validation days, then evaluate once on held-out days."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .data import load, features, split_days, TARGET, FEATURES


def predict(model, frame):
    return np.maximum(model.predict(features(frame)), 0)

def scores(y, p):
    return {'MAE': float(mean_absolute_error(y, p)), 'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'R2': float(r2_score(y, p))}

def main():
    report, models = Path('reports'), Path('models')
    report.mkdir(exist_ok=True); models.mkdir(exist_ok=True)
    data, audit = load()
    train, valid, test = split_days(data)
    candidates = {'Mean baseline': DummyRegressor(), 'Ridge regression': make_pipeline(StandardScaler(), Ridge(alpha=1)),
                  'Random forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=2)}
    validation = {}
    for name, model in candidates.items():
        model.fit(features(train), train[TARGET])
        validation[name] = scores(valid[TARGET], predict(model, valid))
    winner = min(validation, key=lambda n: validation[n]['RMSE'])
    fit_data = pd.concat([train, valid])
    final, test_scores, daylight_scores = {}, {}, {}
    daylight = test['IRRADIATION'] > .01
    for name, candidate in candidates.items():
        final[name] = clone(candidate).fit(features(fit_data), fit_data[TARGET])
        p = predict(final[name], test)
        test_scores[name] = scores(test[TARGET], p)
        daylight_scores[name] = scores(test.loc[daylight, TARGET], p[daylight])
    selected = final[winner]
    ranges = {c: [float(features(fit_data)[c].min()), float(features(fit_data)[c].max())] for c in FEATURES[:2]}
    joblib.dump({'model': selected, 'name': winner, 'ranges': ranges}, models / 'solar.joblib')
    summary = {'task': 'Same-time measured-weather AC power estimation; not day-ahead forecasting', 'selected_model': winner,
               'power_unit': 'source AC_POWER units (commonly described as kW; not independently calibrated)',
               'audit': audit, 'splits': {n: {'rows':len(d), 'start':str(d.DATE_TIME.min()), 'end':str(d.DATE_TIME.max())} for n,d in [('train',train),('validation',valid),('test',test)]},
               'validation':validation, 'test':test_scores, 'test_daylight':daylight_scores, 'daylight_rows':int(daylight.sum())}
    (report / 'metrics.json').write_text(json.dumps(summary, indent=2))
    pred = predict(selected, test)
    pd.DataFrame(test_scores).T.to_csv(report / 'model_comparison.csv', index_label='model')
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
    fig, ax = plt.subplots(2, 1, figsize=(12,8), layout='constrained')
    ax[0].plot(test.DATE_TIME, test[TARGET], color='#153b50', label='Measured', linewidth=1.5)
    ax[0].plot(test.DATE_TIME, pred, color='#ed9d32', label='Estimated', alpha=.85, linewidth=1.2)
    ax[0].set(title=f'Solar Energy Output Predictor | {winner}', ylabel='Plant AC power (source units)')
    ax[0].legend(frameon=False)
    names = list(test_scores)
    ax[1].barh(names, [test_scores[n]['RMSE'] for n in names], color=['#b5c4cc','#367e9b','#ed9d32'])
    ax[1].set(xlabel='Held-out RMSE (source power units; lower is better)', title='Final 7 days · chronological holdout')
    fig.savefig(report / 'results.png', dpi=160)
    plt.close(fig)
    rows=''.join(f'<tr><td>{n}</td><td>{s["MAE"]:,.1f}</td><td>{s["RMSE"]:,.1f}</td><td>{s["R2"]:.4f}</td></tr>' for n,s in test_scores.items())
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Solar Energy Output Predictor</title><style>body{{font:17px system-ui;max-width:1100px;margin:60px auto;padding:0 24px;background:#f5f7f8;color:#153b50}}h1{{font-size:44px}}.tag{{color:#946016;font-weight:700}}img{{width:100%;border-radius:16px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:16px;border-bottom:1px solid #ddd;text-align:left}}p{{line-height:1.6}}</style><p class="tag">ENERGY × MACHINE LEARNING</p><h1>Solar Energy Output Predictor</h1><p>A measured-weather estimator for one solar plant. Built with chronological model selection and a separate final holdout.</p><img src="results.png" alt="Actual versus estimated power and model RMSE comparison"><h2>Held-out results</h2><table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>{rows}</table><p>Selected on validation data: <b>{winner}</b>. Power metrics use original dataset units. Daylight-only metrics and data exclusions are in metrics.json.</p><h2>What this demonstrates</h2><p>Data validation, complete-inverter aggregation, cyclic time features, baseline comparison, and reproducible evaluation.</p><h2>Limits</h2><p>34 days from one plant, with incomplete timestamps excluded. Same-time measured sunlight and temperature are required. This is not a future-weather forecast, and results do not establish performance at other sites or seasons.</p></html>'''
    (report / 'index.html').write_text(html)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
