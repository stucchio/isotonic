# Isotonic

Frequently in data science, we have a relationship between `X` and `y` where (probabilistically) `y` increases as `X` does.

There is a classical algorithm for solving this problem nonparametrically, specifically [Isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression). This simple algorithm is also implemented in [sklearn.isotonic](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html). The classic algorithm is based on a piecewise constant approximation - with nodes at every data point - as well as minimizing (possibly weighted) `l^2` error.

I'm a heavy user of isotonic regression, but unfortunately the version in sklearn does not meet my needs. Specific failings:

- My data is frequently binary. This means that each `y[i]` is either 0 or 1, but the probability that `y[i]==1` increasers as `x[i]` increases.
- My data is often noisy with fatter than normal tails, which means that minimizing `l^2` error overweights outliers.

Also, I frequently want a smoother curve than the piecewise constant curve described above.

I've come up with many hacks to deal with this problem, but this library is my attempt to solve it once and for all.

# Usage

Right now I've implemented a single regression class, `BinomialIsotonicRegression`. This is used to handle the case of binary data. Here's an example of it's use:

from isotonic.binomial_isotonic_regression import BinomialIsotonicRegression
from isotonic.curves import PiecewiseLinearIsotonicCurve, PiecewiseConstantIsotonicCurve


    import pandas as pd
    from bokeh.plotting import figure, output_notebook, show
    from bokeh.models import Span, LinearAxis, Range1d, ColumnDataSource
    output_notebook()

    plot = figure(
        tools="pan,box_zoom,reset,save,",
        y_axis_label="y", title="isotonic",
        x_axis_label='x'
    )

    M = 10
    x_cuts = np.quantile(x, np.arange(0,1,1/M))
    df = pd.DataFrame({'x': x, 'y': y, 'x_c': x_cuts[np.digitize(x, x_cuts)-1]})
    grouped = df.groupby('x_c')['y'].mean().reset_index()

    plot.circle(grouped['x_c'], grouped['y'], color='green', legend_label='true_frac')

    curve = BinomialIsotonicRegression(10, increasing=True, curve_algo=PiecewiseLinearIsotonicCurve).fit(x, y)
    curve2 = BinomialIsotonicRegression(10, increasing=True, curve_algo=PiecewiseConstantIsotonicCurve).fit(x, y)

    xx = np.arange(x.min(), x.max(), 0.01)
    plot.line(xx, curve.predict_proba(xx), color='red', legend_label='piecewise linear')
    plot.line(xx, curve2.predict_proba(xx), color='blue', legend_label='piecewise constant')

    plot.circle(x, y, color='black', alpha=0.01, legend_label='raw data')
    show(plot)

![Simple binomial plot](binomial_isotonic.png)
