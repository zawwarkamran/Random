import numpy as np
import statsmodels.api as sm
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

np.random.seed(19)
x = np.random.normal(0,1,100)
error = np.random.normal(0,1,100)
y = 1.0 + -0.1*x + 0.05*(x**2) + 0.75*(x**3) + error
cols = {'y':y, 'x':x, 'x**2':x**2, 'x**3':x**3, 
'x**4':x**4, 'x**5':x**5, 'x**6':x**6, 'x**7':x**7,
'x**8':x**8}
dat = pd.DataFrame(data=cols)
df = pd.read_csv('HW5.csv').drop('Unnamed: 0', axis=1)
advert = pd.read_csv('Advertising.csv').drop('Unnamed: 0', axis=1)
print(advert.columns)


def bestsubset(data, which_metric):

	x = 0
	if which_metric == 'adjr2':
		x=1
	elif which_metric == 'bic':
		x=2

	combs = []
	for i in range(0, len(data.drop(data.columns[0], axis=1).columns)+1):
		for subset in combinations(data.drop(data.columns[0], axis=1).columns, i):
			combs.append(list(subset))

	model_score = []
	for predictor in combs:
		model = sm.OLS(data.iloc[:,0], sm.add_constant(data[predictor].to_numpy())).fit()
		model_score.append((model.aic, model.rsquared_adj, model.bic, predictor
		 ,model.params, model))

	def chooser(list):
		subsets = {}
		for i in range(1, len(data.drop(data.columns[0], axis=1).columns)+1):
			best_from_subset = []
			for item in model_score:
				if len(item[3]) == i:
					best_from_subset.append(item)
			best_from_subset.sort(key=lambda tup: tup[x])
			subsets[i] = best_from_subset[0]
		return subsets

	answer = chooser(model_score)

	return answer


best = bestsubset(df, 'aic')
print(best)
best2 = bestsubset(dat, 'aic')
print(best2[3])















