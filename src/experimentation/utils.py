from scipy import stats as st


def confidence_interval(data, confidence_level):
    freedom_degrees = len(data) - 1
    std_err = st.sem(data)
    margin = std_err * st.t.ppf((1 + confidence_level) / 2, freedom_degrees)
    # interval = mean(data) +- margin
    return margin


def t_confidence_interval_thunk(confidence_level):
    return lambda x: confidence_interval(x, confidence_level)
