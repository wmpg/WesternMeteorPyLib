""" The source-region probabilities of an orbit along an integration, from a model of the near-Earth
object or meteoroid population: the series over the outputs of wmpl.Rebound.REBOUND, and the report,
JSON, CSV and figure made from it. Shared by wmpl.Utils.NEOMODSourceRegions (NEOMOD2, Nesvorny et al.
2024) and wmpl.Utils.METEOMODSourceRegions (METEOMOD, Broz et al. 2024).

A model is described by a dict, which each of those modules builds with its sourceModel function:
    'name':            [str] The name of the model, as printed.
    'label':           [str] A short name, for the figure.
    'reference':       [str] The papers to cite, in full.
    'short_reference': [str] The same, short, for the foot of the figure.
    'sources':         [list] The keys of the sources.
    'source_names':    [dict] {key: name} to print.
    'evaluate':        [function] (a, e, incl) -> (probabilities, status): the probability of each source
                       (last axis over 'sources', NaN where there is none) and IN_MODEL, OUTSIDE or
                       EMPTY_BIN for each orbit, a in AU and incl in radians, arrays of any shape.
    'reason':          [function] (a, e, incl, status) -> why one orbit has no probabilities, or None.
    'heading':         [str] What the model was evaluated for, printed after its name.
    'warning':         [str] A warning to print with the results, or None.
    'title_note':      [str] Appended to the title of the figure.
    'print_min':       [float] Sources below this probability everywhere are left out of the tables.
    'name_width':      [int] The width of the column of the source names.
    'plot_groups':     [list] (label, colour, [keys]) of the stacked areas of the figure, in order; the
                       other sources are drawn together as one. None to draw the sources with the
                       largest probabilities of the series, as many as there are colours.
    'json':            [dict] Fields of the model added to its JSON.
    'type_groups':     [list] (type, [keys]) of the types of the sources, to give the probabilities summed
                       by type as well, or None. Optional.
"""

from __future__ import print_function, division, absolute_import

import textwrap

import numpy as np


# Where an orbit falls: in a bin of the model, outside the model, or in a bin of the model without bodies
IN_MODEL = 0
OUTSIDE = 1
EMPTY_BIN = 2

# The interval the clones are summarised by, in percent
CLONE_INTERVAL = (2.5, 97.5)

# The categorical palette of the figures, in order, and the colour of the sources drawn together
PLOT_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
PLOT_OTHER_COLOR = '#a3a29d'

# Beyond this many sources left out of a table, they are counted rather than named
MAX_NAMED_LEFT_OUT = 8



def wrapped(text, indent="  ", width=100):
    """ A text wrapped into lines, each starting with the indent. """

    return [indent + line for line in textwrap.wrap(text, width)]



def finiteList(values):
    """ A list of floats from an array, None where a value is not finite. """

    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float).ravel()]



def citationLines(model):
    """ The line asking to cite the papers of a model, wrapped, after a blank line. """

    return [""] + wrapped("Please cite {:s}.".format(model['reference']))



def _seriesOf(model, time_days, nominal, clones, interval):
    """ The source probabilities of the nominal solution and the clones at some outputs (see
        sourceRegionSeries). """

    time_days = np.asarray(time_days, dtype=float)
    nominal = np.asarray(nominal, dtype=float).reshape(-1, 3)

    probs, status = model['evaluate'](nominal[:, 0], nominal[:, 1], nominal[:, 2])

    out = {
        'model': model,
        'time_days': time_days,
        'sources': list(model['sources']),
        'nominal': probs,
        'nominal_reason': [model['reason'](*row, st) for row, st in zip(nominal, status)],
        'n_clones': 0,
        'interval': tuple(interval),
        'clones_mean': None, 'clones_low': None, 'clones_high': None, 'clones_in_model': None,
        }

    if (clones is None) or (len(clones) == 0):
        return out

    clones = np.asarray(clones, dtype=float)

    probs, status = model['evaluate'](clones[..., 0], clones[..., 1], clones[..., 2])

    in_model = status == IN_MODEL
    n_in = np.sum(in_model, axis=0)

    # The mean and the interval at each output over the clones in the model there
    n_out, n_sources = probs.shape[1], probs.shape[2]
    mean = np.full((n_out, n_sources), np.nan)
    low = np.full((n_out, n_sources), np.nan)
    high = np.full((n_out, n_sources), np.nan)

    for k in np.nonzero(n_in)[0]:
        values = probs[in_model[:, k], k]
        mean[k] = np.mean(values, axis=0)
        low[k], high[k] = np.percentile(values, interval, axis=0)

    out.update({'n_clones': int(len(clones)), 'clones_mean': mean, 'clones_low': low,
        'clones_high': high, 'clones_in_model': n_in/len(clones)})

    return out



def sourceRegionSeries(model, time_days, nominal, clones=None, interval=CLONE_INTERVAL, horizon=None,
        representative=None):
    """ The source probabilities of an integrated orbit at each output time, of its nominal solution and
        over its clones, and those of the orbits to use beyond the predictability horizon: the nominal at
        the horizon and the clone that represents the clones at the end.

    Arguments:
        model: [dict] The model (see the module docstring).
        time_days: [ndarray] The output times (days).
        nominal: [ndarray] a (AU), e and incl (rad) of the nominal solution at each output, one row each.

    Keyword arguments:
        clones: [ndarray] The same for the clones, (clone, output, element). None by default.
        interval: [tuple] Percentiles of the interval of the clones. CLONE_INTERVAL by default.
        horizon: [tuple] (time_days, nominal, clones) at the predictability horizon, the last output at
            which the measurement still sets the nominal orbit (see
            wmpl.Rebound.REBOUND.predictabilityHorizon and orbitAtHorizon): its time, a, e and incl of
            the nominal, and those of the clones, one row each. None by default.
        representative: [tuple] (name, elements) of the clone that represents the clones at the end (see
            wmpl.Rebound.REBOUND.representativeOrbit): its name, and a, e and incl at each output, one
            row each. None by default.

    Return:
        [dict] 'model'; 'time_days'; 'sources'; 'nominal', the probabilities (output, source), NaN
            outside the model; 'nominal_reason', None or why the nominal has none at each output;
            'n_clones'; with clones, 'clones_mean', 'clones_low' and 'clones_high' (output, source) and
            'clones_in_model', the share of the clones in the model at each output, else None;
            'horizon', the same keys for the one output at the horizon, or None; and 'representative',
            {'clone', 'probabilities' (output, source), 'reason'} or None.
    """

    out = _seriesOf(model, time_days, nominal, clones, interval)

    out['horizon'] = None
    if horizon is not None:

        t_h, nominal_h, clones_h = horizon
        clones_h = None if (clones_h is None) or (len(clones_h) == 0) else [[c] for c in clones_h]

        out['horizon'] = _seriesOf(model, [t_h], [nominal_h], clones_h, interval)

    out['representative'] = None
    if representative is not None:

        name, elements = representative
        elements = np.asarray(elements, dtype=float).reshape(-1, 3)

        if len(elements) != len(out['time_days']):
            raise ValueError("The representative clone has {:d} outputs, the series {:d}.".format(
                len(elements), len(out['time_days'])))

        probs, status = model['evaluate'](elements[:, 0], elements[:, 1], elements[:, 2])

        out['representative'] = {'clone': str(name), 'probabilities': probs,
            'reason': [model['reason'](*row, st) for row, st in zip(elements, status)]}

    return out



def reboundSourceRegionSeries(model, sim_outputs, sim_outputs_mc, ci_names, horizon, mode, time_to_days):
    """ The source probabilities along a run of wmpl.Rebound.REBOUND (see sourceRegionSeries): at every
        output, or at the end only, with the predictability horizon and the representative clone when the
        run has them.

    Arguments:
        model: [dict] The model (see the module docstring).
        sim_outputs: [list] Outputs of the nominal solution (see reboundSimulate), each row holding the
            time and the heliocentric orbit (a in AU) as its first and third items.
        sim_outputs_mc: [dict] {name: outputs} of the clones.
        ci_names: [list] The clones of the confidence intervals, those used at the end.
        horizon: [dict] Return of predictabilityHorizon, with "representative" if the run gives one, or
            None.
        mode: [str] 'all' for every output, 'final' for the end only.
        time_to_days: [function] Converts the time of an output to days.

    Return:
        [dict] Return of sourceRegionSeries.
    """

    # Every output of the clones that ran the whole span, or the end of those of the confidence intervals
    if mode == 'all':
        k_out = slice(None)
        clones = [rows for rows in sim_outputs_mc.values() if len(rows) == len(sim_outputs)]
    else:
        k_out = slice(-1, None)
        clones = [sim_outputs_mc[name] for name in ci_names]

    def _elements(rows):
        return [[row[2].a, row[2].e, row[2].inc] for row in rows[k_out]]

    # The nominal and the clones at the output of the predictability horizon, the last one at which the
    #   measurement still sets the nominal orbit, found as orbitAtHorizon finds it
    at_horizon = None
    if (horizon is not None) and (horizon.get('horizon_days') is not None):
        times = np.array([time_to_days(row[0]) for row in sim_outputs])
        k_h = int(np.argmin(np.abs(times - horizon['horizon_days'])))
        at_horizon = (times[k_h], [sim_outputs[k_h][2].a, sim_outputs[k_h][2].e, sim_outputs[k_h][2].inc],
            [[rows[k_h][2].a, rows[k_h][2].e, rows[k_h][2].inc] for rows in sim_outputs_mc.values()
            if len(rows) > k_h])

    # The clone that represents the clones at the end, if the run has one (see representativeOrbit)
    representative = None
    rep = None if horizon is None else horizon.get('representative')
    if (rep is not None) and (len(sim_outputs_mc.get(rep['clone'], [])) == len(sim_outputs)):
        representative = (rep['clone'], _elements(sim_outputs_mc[rep['clone']]))

    return sourceRegionSeries(model, [time_to_days(row[0]) for row in sim_outputs[k_out]],
        _elements(sim_outputs), clones=[_elements(rows) for rows in clones] or None, horizon=at_horizon,
        representative=representative)



def sourceRegionSeriesToJSON(series, mode=None):
    """ The return of sourceRegionSeries as plain lists, for a JSON file.

    Arguments:
        series: [dict] Return of sourceRegionSeries.

    Keyword arguments:
        mode: [str] 'all' or 'final', which outputs were computed. None by default.

    Return:
        [dict] The series, the probabilities as {source: [value per output]}, those at the horizon as
            {source: value}.
    """

    model = series['model']

    def _bySource(values):
        if values is None:
            return None
        return {s: finiteList(values[:, k]) for k, s in enumerate(series['sources'])}

    def _atOutput(values):
        if values is None:
            return None
        return {s: finiteList(values[0, k])[0] for k, s in enumerate(series['sources'])}

    out = {'model': model['name'], 'reference': model['reference'], 'mode': mode}
    out.update(model['json'])
    out.update({
        'sources': series['sources'],
        'source_names': [model['source_names'].get(s, s) for s in series['sources']],
        'time_days': finiteList(series['time_days']),
        'nominal': _bySource(series['nominal']),
        'nominal_reason': series['nominal_reason'],
        'n_clones': series['n_clones'],
        'clones_interval': list(series['interval']),
        'clones_mean': _bySource(series['clones_mean']),
        'clones_low': _bySource(series['clones_low']),
        'clones_high': _bySource(series['clones_high']),
        'clones_in_model': None if series['clones_in_model'] is None
            else finiteList(series['clones_in_model']),
        'horizon': None,
        'representative': None,
        })

    # The nominal and the clones at the predictability horizon
    horizon = series.get('horizon')
    if horizon is not None:
        out['horizon'] = {
            'time_days': float(horizon['time_days'][0]),
            'nominal': _atOutput(horizon['nominal']),
            'nominal_reason': horizon['nominal_reason'][0],
            'n_clones': horizon['n_clones'],
            'clones_mean': _atOutput(horizon['clones_mean']),
            'clones_low': _atOutput(horizon['clones_low']),
            'clones_high': _atOutput(horizon['clones_high']),
            'clones_in_model': None if horizon['clones_in_model'] is None
                else float(horizon['clones_in_model'][0]),
            }

    # The clone that represents the clones at the end, at every output
    representative = series.get('representative')
    if representative is not None:
        out['representative'] = {
            'clone': representative['clone'],
            'probabilities': _bySource(representative['probabilities']),
            'reason': representative['reason'],
            }

    # The probabilities summed by type
    out['by_type'] = None
    groups = model.get('type_groups')
    if groups:

        labels = [label for label, _ in groups]

        def _byType(values, at_output=False):
            if values is None:
                return None
            sums = groupSums(values, series['sources'], groups)
            if at_output:
                return {label: finiteList(sums[0, g])[0] for g, label in enumerate(labels)}
            return {label: finiteList(sums[:, g]) for g, label in enumerate(labels)}

        out['by_type'] = {
            'types': labels,
            'members': {label: keys for label, keys in groups},
            'nominal': _byType(series['nominal']),
            'clones_mean': _byType(series['clones_mean']),
            'horizon': None if horizon is None else {'nominal': _byType(horizon['nominal'], at_output=True),
                'clones_mean': _byType(horizon['clones_mean'], at_output=True)},
            'representative': None if representative is None
                else _byType(representative['probabilities']),
            }

    return out



def groupSums(values, sources, groups):
    """ Probabilities summed over groups of sources.

    Arguments:
        values: [ndarray] Probabilities, with a last axis over the sources.
        sources: [list] The keys of the sources, in the order of the last axis.
        groups: [list] (label, [keys]) of the groups.

    Return:
        [ndarray] The sums, with a last axis over the groups, NaN where the sources have no probabilities.
    """

    values = np.asarray(values, dtype=float)

    sums = []
    for _, keys in groups:
        columns = [sources.index(s) for s in keys if s in sources]
        sums.append(np.sum(values[..., columns], axis=-1))

    return np.stack(sums, axis=-1)



def _horizonIndex(series):
    """ The output of a series at its predictability horizon, or None if it is not one of them. """

    horizon = series.get('horizon')
    if horizon is None:
        return None

    k = int(np.argmin(np.abs(np.asarray(series['time_days']) - horizon['time_days'][0])))

    return k if series['time_days'][k] == horizon['time_days'][0] else None



def writeSourceRegionCSV(series, file_path):
    """ Save the return of sourceRegionSeries as a CSV file, one row per output time and source, with
        the representative clone's probabilities and a flag on the rows at the predictability horizon,
        added before the last output if the series does not hold it.

    Arguments:
        series: [dict] Return of sourceRegionSeries.
        file_path: [str] The CSV file.
    """

    def _cell(value):
        return "{:.4f}".format(value) if np.isfinite(value) else ""

    def _rows(part, k, representative, at_horizon):
        """ The rows of one output of a series. """

        rows = []
        for j, s in enumerate(part['sources']):

            row = [_cell(part['nominal'][k, j])]

            if part['clones_mean'] is None:
                row += ["", "", "", ""]
            else:
                row += [_cell(part['clones_mean'][k, j]), _cell(part['clones_low'][k, j]),
                    _cell(part['clones_high'][k, j]), "{:.4f}".format(part['clones_in_model'][k])]

            row.append("" if representative is None else _cell(representative[k, j]))
            row.append("1" if at_horizon else "0")

            rows.append("{:.6f},{:s},{:s}\n".format(part['time_days'][k], s, ",".join(row)))

        return rows

    representative = None if series.get('representative') is None \
        else series['representative']['probabilities']

    k_horizon = _horizonIndex(series)
    n_out = len(series['time_days'])

    with open(file_path, 'w') as f:

        f.write("time_days,source,nominal,clones_mean,clones_p{:g},clones_p{:g},clones_in_model,"
            "representative,horizon\n".format(*series['interval']))

        for k in range(n_out):

            # The horizon, when it is not one of the outputs, before the last one
            if (k == n_out - 1) and (series.get('horizon') is not None) and (k_horizon is None):
                f.writelines(_rows(series['horizon'], 0, None, True))

            f.writelines(_rows(series, k, representative, k == k_horizon))



def timeText(time_days):
    """ A time in days as text, in years beyond two years. """

    if abs(time_days) > 2*365.25:
        return "{:+.1f} yr".format(time_days/365.25)

    return "{:+.2f} d".format(time_days)



def _seriesColumn(series, key, k):
    """ The probabilities of one output of a series as {source: value}, or None. """

    values = series[key]

    if (values is None) or (not np.any(np.isfinite(values[k]))):
        return None

    return {s: float(values[k, j]) for j, s in enumerate(series['sources'])}



def orderedSources(sources, columns, print_min):
    """ The sources sorted by their largest probability in any of the columns, those under print_min
        everywhere left out.

    Arguments:
        sources: [list] The keys of the sources.
        columns: [list] {source: probability} dicts, or None.
        print_min: [float] The smallest probability printed.

    Return:
        [list] The keys kept, in order.
    """

    largest = {}
    for s in sources:
        values = [col[s] for col in columns if (col is not None) and np.isfinite(col[s])]
        largest[s] = max(values) if values else np.nan

    kept = [s for s in sources if np.isfinite(largest[s]) and (largest[s] >= print_min)]

    return sorted(kept, key=lambda s: -largest[s])



def leftOutLine(model, sources, kept):
    """ The line naming the sources left out of a table, or counting them if they are many, or None. """

    left_out = [model['source_names'].get(s, s) for s in sources if s not in kept]

    if not left_out:
        return None

    if len(left_out) > MAX_NAMED_LEFT_OUT:
        return "  ({:d} other sources: below {:g})".format(len(left_out), model['print_min'])

    return "  ({:s}: below {:g})".format(", ".join(left_out), model['print_min'])



def _probabilityText(value):
    """ A probability as text, n/a where there is none. """

    return "{:.3f}".format(value) if np.isfinite(value) else "n/a"



def _intervalText(series, k, j):
    """ The clones' mean and interval of one source at one output of a series, as text. """

    mean = series['clones_mean'][k, j]

    if not np.isfinite(mean):
        return "n/a"

    return "{:.3f} [{:.3f}, {:.3f}]".format(mean, series['clones_low'][k, j], series['clones_high'][k, j])



def sourceRegionReportLines(series, first=0, first_note=None, time_label=None):
    """ Lines of text summarising the return of sourceRegionSeries: the source probabilities at a first
        output and at the last, of the nominal solution and over the clones, and beyond the
        predictability horizon those of the nominal at the horizon and of the representative clone at the
        end, with the papers to cite.

    Arguments:
        series: [dict] Return of sourceRegionSeries.

    Keyword arguments:
        first: [int] The first output to give, for an integration the first one after the object left
            the Earth, where its heliocentric orbit is defined. 0 by default.
        first_note: [str] What the first output is, written next to its time. None by default.
        time_label: [function] Formats a time in days for the column headers. None by default.

    Return:
        [list] The lines.
    """

    model = series['model']
    names = model['source_names']
    width_name = model['name_width']

    if time_label is None:
        time_label = timeText

    n_out = len(series['time_days'])
    first = min(max(int(first), 0), n_out - 1)
    outputs = [first, n_out - 1] if n_out - 1 > first else [n_out - 1]

    horizon = series.get('horizon')
    representative = series.get('representative')

    lines = ["Source regions in {:s}, {:s}:".format(model['name'], model['heading'])]

    if model['warning'] is not None:
        lines += wrapped(model['warning'])

    has_clones = series['clones_mean'] is not None

    # Every column of the tables, to order the sources by their largest probability in any of them
    columns = []
    for k in outputs:
        columns.append(_seriesColumn(series, 'nominal', k))
        if has_clones:
            columns.append(_seriesColumn(series, 'clones_mean', k))

    if horizon is not None:
        columns += [_seriesColumn(horizon, 'nominal', 0), _seriesColumn(horizon, 'clones_mean', 0)]

    if representative is not None:
        columns.append(_seriesColumn({'probabilities': representative['probabilities'],
            'sources': series['sources']}, 'probabilities', n_out - 1))

    citation = citationLines(model)

    if all(col is None for col in columns):
        for k in outputs:
            lines.append("  At t = {:s}: {:s}.".format(time_label(series['time_days'][k]),
                series['nominal_reason'][k] or "no clone is in the model"))
        return lines + citation

    order = orderedSources(series['sources'], columns, model['print_min'])


    ### The first output and the last ###

    header = "  {:<{:d}s}".format("", width_name)
    sub = "  {:<{:d}s}".format("Source", width_name)
    width = 38 if has_clones else 10

    for k in outputs:
        label = "t = {:s}{:s}".format(time_label(series['time_days'][k]), " ({:s})".format(first_note)
            if (k == first) and (first_note is not None) else "")
        header += "  {:<{:d}s}".format(label, width)
        sub += "  {:>8s}".format("nominal")
        if has_clones:
            sub += "  {:<28s}".format("clones mean [{:g}-{:g}%]".format(*series['interval']))
        else:
            sub += "  "

    lines += [header.rstrip(), sub.rstrip()]

    for s in order:

        j = series['sources'].index(s)
        line = "  {:<{:d}s}".format(names.get(s, s), width_name)

        for k in outputs:
            line += "  {:>8s}".format(_probabilityText(series['nominal'][k, j]))
            line += "  {:<28s}".format(_intervalText(series, k, j)) if has_clones else "  "

        lines.append(line.rstrip())

    left_out = leftOutLine(model, series['sources'], order)
    if left_out is not None:
        lines.append(left_out)

    for k in outputs:

        if series['nominal_reason'][k] is not None:
            lines.append("  Nominal at t = {:s}: {:s}.".format(time_label(series['time_days'][k]),
                series['nominal_reason'][k]))

        if has_clones and (series['clones_in_model'][k] < 1.0):
            lines.append("  At t = {:s}, {:.1f}% of the {:d} clones are in the model.".format(
                time_label(series['time_days'][k]), 100*series['clones_in_model'][k], series['n_clones']))

    if has_clones:
        lines.append("  The clones' mean is the probability given the uncertainty of the orbit. Over a "
            "long run,")
        lines.append("  where the clones have spread, it is the value to quote, the nominal being one of "
            "them.")

    # The probabilities summed by type
    groups = model.get('type_groups')
    if groups:

        columns = []
        for k in outputs:
            when = "t = {:s}".format(time_label(series['time_days'][k]))
            columns.append(('nominal', series, k, when))
            if has_clones:
                columns.append(('clones_mean', series, k, when))

        lines.append("")
        lines.append("  By type, the sum of the families of each type:")
        lines += _groupTable(series, groups, columns, width_name, model['print_min'])


    ### Beyond the predictability horizon ###

    if horizon is not None:

        lines.append("")

        if representative is not None:
            lines.append("  Beyond the predictability horizon, t = {:s}, the orbits to use are the nominal "
                "at the horizon".format(time_label(horizon['time_days'][0])))
            lines.append("  and the clone that represents the clones at the end (see the orbit to use at "
                "the end):")
        else:
            lines.append("  At the predictability horizon, t = {:s}, the last time the measurement sets the "
                "nominal orbit:".format(time_label(horizon['time_days'][0])))

        has_horizon_clones = horizon['clones_mean'] is not None

        header = "  {:<{:d}s}  {:<{:d}s}".format("", width_name, "at the horizon",
            38 if has_horizon_clones else 10)
        sub = "  {:<{:d}s}  {:>8s}".format("Source", width_name, "nominal")
        if has_horizon_clones:
            sub += "  {:<28s}".format("clones mean [{:g}-{:g}%]".format(*horizon['interval']))

        if representative is not None:
            header += "  at the end, t = {:s}".format(time_label(series['time_days'][-1]))
            sub += "  representative ({:s})".format(representative['clone'])

        lines += [header.rstrip(), sub.rstrip()]

        for s in order:

            j = series['sources'].index(s)
            line = "  {:<{:d}s}  {:>8s}".format(names.get(s, s), width_name,
                _probabilityText(horizon['nominal'][0, j]))

            if has_horizon_clones:
                line += "  {:<28s}".format(_intervalText(horizon, 0, j))

            if representative is not None:
                line += "  {:>8s}".format(_probabilityText(representative['probabilities'][-1, j]))

            lines.append(line.rstrip())

        if horizon['nominal_reason'][0] is not None:
            lines.append("  Nominal at the horizon: {:s}.".format(horizon['nominal_reason'][0]))

        if (representative is not None) and (representative['reason'][-1] is not None):
            lines.append("  Representative clone at the end: {:s}.".format(representative['reason'][-1]))

        # The same by type
        if groups:

            columns = [('nominal', horizon, 0, "at the horizon")]
            if has_horizon_clones:
                columns.append(('clones_mean', horizon, 0, "at the horizon"))
            if representative is not None:
                columns.append(('probabilities', dict(representative, sources=series['sources']), n_out - 1,
                    "at the end"))

            lines.append("  By type:")
            lines += _groupTable(series, groups, columns, width_name, model['print_min'])

    return lines + citation



def _groupTable(series, groups, columns, width_name, print_min):
    """ Lines of a table of the probabilities summed by type, a column for each (key, part of the series,
        output, time), the types under print_min everywhere left out. """

    titles = {'nominal': "nominal", 'clones_mean': "clones mean", 'probabilities': "representative"}

    sums = []
    for key, part, k, _ in columns:
        values = part[key]
        sums.append(groupSums(values[k], series['sources'], groups) if values is not None
            else np.full(len(groups), np.nan))

    sums = np.array(sums)

    with np.errstate(invalid='ignore'):
        largest = np.nanmax(np.where(np.isfinite(sums), sums, -1.0), axis=0)

    order = [g for g in np.argsort(-largest, kind='stable') if largest[g] >= print_min]

    lines = ["  {:<{:d}s}".format("", width_name) + "".join("  {:>16s}".format(when)
        for _, _, _, when in columns)]
    lines.append("  {:<{:d}s}".format("Type", width_name) + "".join("  {:>16s}".format(titles[key])
        for key, _, _, _ in columns))

    for g in order:
        lines.append("  {:<{:d}s}".format(groups[g][0], width_name) + "".join("  {:>16s}".format(
            _probabilityText(value)) for value in sums[:, g]))

    left_out = [groups[g][0] for g in range(len(groups)) if g not in order]
    if left_out:
        lines.append("  ({:s}: below {:g})".format(", ".join(left_out), print_min))

    return lines



def _largestGroups(series):
    """ The sources with the largest probabilities anywhere in a series, nominal or clones' mean, as
        many as there are colours, each a group of its own. """

    model = series['model']
    stacks = [series['nominal']] + ([series['clones_mean']] if series['clones_mean'] is not None else [])

    with np.errstate(invalid='ignore'):
        peaks = np.nanmax(np.vstack([np.where(np.isfinite(v), v, -1.0) for v in stacks]), axis=0)

    order = [series['sources'][j] for j in np.argsort(-peaks, kind='stable')][:len(PLOT_COLORS)]

    return [(model['source_names'].get(s, s), color, [s]) for s, color in zip(order, PLOT_COLORS)]



def _plotGroups(plot_groups, series, values):
    """ The probabilities of the groups of sources drawn in the figure, and of all the others together. """

    sources = series['sources']

    groups = []
    grouped = set()
    for _, _, keys in plot_groups:
        columns = [sources.index(s) for s in keys if s in sources]
        groups.append(np.nansum(values[:, columns], axis=1) if columns else np.zeros(len(values)))
        grouped.update(keys)

    others = [k for k, s in enumerate(sources) if s not in grouped]
    groups.append(np.nansum(values[:, others], axis=1) if others else np.zeros(len(values)))

    return groups



def plotSourceRegionSeries(series, file_path, title=None):
    """ Draw the source probabilities over an integration as stacked areas against time, the nominal
        solution above and the clones' mean below, if there are clones.

    Arguments:
        series: [dict] Return of sourceRegionSeries.
        file_path: [str] The figure file.

    Keyword arguments:
        title: [str] The title. None by default.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    model = series['model']

    time_days = np.asarray(series['time_days'])

    # Years for a long run, days for a short one
    span = np.max(np.abs(time_days)) if len(time_days) else 0.0
    t_scale, t_unit = (365.25, "yr") if span > 2*365.25 else (1.0, "d")
    t = time_days/t_scale

    panels = [('nominal', "Nominal solution")]
    if series['clones_mean'] is not None:
        panels.append(('clones_mean', "Mean over {:d} clones".format(series['n_clones'])))

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.8*len(panels) + 0.9), sharex=True,
        squeeze=False)
    axes = axes[:, 0]

    plot_groups = model['plot_groups'] if model['plot_groups'] is not None else _largestGroups(series)

    colors = [color for _, color, _ in plot_groups] + [PLOT_OTHER_COLOR]
    labels = [label for label, _, _ in plot_groups] + ["other"]

    for ax, (key, panel_title) in zip(axes, panels):

        values = series[key]
        groups = _plotGroups(plot_groups, series, values)

        # Stacked areas with a thin gap of the background between them
        bottom = np.zeros(len(t))
        for g, color in zip(groups, colors):
            ax.fill_between(t, bottom, bottom + g, step='mid', color=color, lw=0.6, edgecolor='white')
            bottom = bottom + g

        # Where the orbit is outside the model, a light grey band
        outside = ~np.any(np.isfinite(values), axis=1)
        if np.any(outside):
            ax.fill_between(t, 0, 1, where=outside, step='mid', color='0.92', lw=0)

        # The predictability horizon
        if series.get('horizon') is not None:
            ax.axvline(series['horizon']['time_days'][0]/t_scale, color='k', ls='--', lw=1.0)

        ax.set_ylim(0, 1)
        ax.set_xlim(np.min(t), np.max(t))
        ax.set_ylabel("Probability")
        ax.set_title(panel_title, fontsize=10, loc='left')
        ax.grid(axis='y', color='0.85', lw=0.5)
        ax.set_axisbelow(True)

        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel("Time ({:s})".format(t_unit))

    # The legend in the order of the stack, top first
    handles = [Patch(facecolor='0.92', label="outside the model")]
    handles += [Patch(facecolor=c, edgecolor='white', label=lab) for c, lab in zip(colors, labels)][::-1]
    if series.get('horizon') is not None:
        handles.append(Line2D([], [], color='k', ls='--', lw=1.0, label="predictability\nhorizon"))
    fig.legend(handles=handles, loc='center right', frameon=False, fontsize=9)

    if title is None:
        title = "Source regions in {:s}".format(model['name'])

    fig.suptitle("{:s}{:s}".format(title, model['title_note']), fontsize=11)

    fig.text(0.01, 0.005, "{:s}: {:s}".format(model['label'], model['short_reference']), fontsize=7,
        color='0.4', ha='left', va='bottom')

    fig.tight_layout(rect=(0, 0.02, 0.83, 1))
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
