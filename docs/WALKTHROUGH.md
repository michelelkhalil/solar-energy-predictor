# Michel's project walkthrough

## 1. What did we build?

A program that estimates how much electrical power one solar plant produces at a particular moment. It looks at the sunlight measurement, air temperature, and time of day.

For example: given measured sunlight of 0.7 in the dataset's units, 30°C, and noon, it produces an estimated total AC-power reading for this specific plant. It does not predict tomorrow's weather or the output of an arbitrary rooftop system.

## 2. What are inputs and targets?

Inputs, also called features, are information the model can use. The target is the answer we want it to learn: total AC power. We show the model many historical examples of inputs with their measured target.

We leave out DC power because it is another contemporaneous electrical-output measurement closely related to the answer. We also exclude yield counters, which would make the task less useful and risk giving the model information unavailable for a clean weather-based estimate.

## 3. Why clean the data?

There are 22 inverters. If only 21 report at noon, summing their readings makes the plant look weaker even if it is operating normally. We keep timestamps where all 22 report valid values. This creates a consistent target, but it also means results do not cover missing-sensor situations.

## 4. Why three models?

- Mean baseline: always guesses the average training output. This establishes a basic reference.
- Ridge regression: learns a weighted relationship between the inputs and output. It is a useful, strong, simple model.
- Random forest: averages many decision trees that divide the input space into regions. It can capture nonlinear relationships.

The forest is not automatically better. We choose using validation performance. In this run it won and also outperformed Ridge on the final test.

## 5. Why train, validation, and test?

Training teaches the model. Validation selects among candidate models. Testing checks the selected approach on later examples held aside until the choice is finished.

Shuffling nearby measurements could make the task unrealistically easy. Here, all training dates precede validation dates, which precede test dates. We then retrain the chosen specification using train plus validation data before final testing.

## 6. What do the numbers mean?

MAE is the average absolute error in the same units as the target. RMSE penalizes large errors more strongly. Lower is better for both.

R² compares squared errors with a constant mean reference. An R² of 0.9958 does not mean that 99.58% of individual predictions are correct.

The forest test RMSE was 508.4 versus Ridge's 669.3: (669.3 − 508.4) / 669.3 is approximately 24%. This comparison is much more informative than only comparing with the weak constant baseline.

## 7. Why is performance so high?

Same-time sunlight is strongly informative about same-time solar power. Nighttime is comparatively easy. The dataset also covers only one plant and a short period. High scores here do not establish day-ahead forecasting ability or performance in Tampa, Lebanon, winter, or another year.

## 8. What to do yourself next

1. Install the dependencies, download the data, and run training.
2. Open the chart and locate periods where the estimate differs from measurements.
3. Read data.py and explain why incomplete timestamps are excluded.
4. Read train.py and identify exactly which data selects the model.
5. Run the prediction command with two valid daylight irradiation values and compare outputs.
6. Explain the main limitation in your own words: future forecasts need future weather inputs.

Once you change models repeatedly after seeing test results, that test period becomes part of development. Use a new future holdout for an honest final score.

## Interview explanation

“I worked on a solar-power estimation project connecting my electrical-engineering background with machine learning. It aggregates inverter measurements into a consistent plant-level target, uses sunlight, temperature, and cyclic time features, and compares three models using chronological validation. The random forest achieved about 24% lower test RMSE than Ridge on a seven-day holdout. The main limitation is that it uses measured weather at the target time, so it is an estimator rather than a day-ahead forecasting system.”

Use this only when you can explain and reproduce those choices. Be open about AI assistance and distinguish the initial generated implementation from changes and analysis you personally complete.
