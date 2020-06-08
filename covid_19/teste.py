import sys
import os
# insert at 1, 0 is the script path (or '' in REPL)
base_dir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.insert(0, base_dir)

from covid19.estimation import ReproductionNumber
import pandas as pd
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.conversion import localconverter
from copy import deepcopy

def run_Rt_estimation(incidence, prior_shape, prior_scale, mean_si, sd_si, t_max, window_width):
    
    si_pars = {'mean': mean_si, 'sd': sd_si}
    
    Rt = ReproductionNumber(incidence=incidence,
                            si_pars=si_pars,
                            prior_shape=prior_shape, 
                            prior_scale=prior_scale,
                            window_width=window_width
                           )
    
    Rt.compute_posterior_parameters()

    Rt_posterior_sample = Rt.sample_from_posterior(sample_size=N)
    Rt.compute_posterior_summaries(posterior_sample=Rt_posterior_sample, t_max=t_max)

    results = Rt
    
    return results

N = 100_000
data = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
data['date'] = data['date'].astype('datetime64[ns]')

t_max = 5
window_width = 6


def get_incidence_data(data, region):
    incidence = data[data['state'] == region][['date', 'newCases', 'totalCases']]
    incidence = incidence.set_index('date')
    
    incidence = (incidence.asfreq('d')
                 .assign(newCases=lambda x: x.newCases.fillna(0),
                         totalCases=lambda x: x.totalCases.fillna(method='ffill'))
                 .query('totalCases >= 50'))
    incidence = incidence.reset_index()
    incidence.columns = ['dates', 'incidence', 'totalCases']
    incidence = incidence.set_index('dates')
    
    return incidence[['incidence']]


brazil_incidence = get_incidence_data(data=data, region="TOTAL")

brazil_results = run_Rt_estimation(incidence=brazil_incidence,
                                   prior_shape=5.12, prior_scale=0.64,
                                   mean_si=7.5, sd_si=3.4,
                                   t_max=t_max, window_width=window_width)

print(brazil_incidence)
print(brazil_results)


last_known_day = brazil_results.posterior_summary.query('end_dates == "2020-03-26"')

most_recent_Rt = last_known_day['Rt_mean'].round(2).values[0]
upper = last_known_day['Rt_q0.975'].round(2).values[0]
lower = last_known_day['Rt_q0.025'].round(2).values[0]
brazil_posterior_shape = last_known_day['Rt_shape'].values[0]
brazil_posterior_scale = last_known_day['Rt_scale'].values[0]
title = f"most recent $R_t$ = {most_recent_Rt} (95% CI = [{lower}, {upper}], $k = {round(brazil_posterior_shape, 2)}$, $\\theta = {round(brazil_posterior_scale, 5)}$)"
print(upper)
print(brazil_results.posterior_summary)
brazil_results.plot_reproduction_number(title=title)
