import pandas as pd 
import statistics
locat = "Binary Data/A2G2.csv"
df = pd.read_csv(locat)
# # # FILTER OUT EMPTY 
df = df.filter(['timestamps', 'delta_absolute_1','delta_absolute_2', 'delta_absolute_3', 'delta_absolute_4', 'alpha_absolute_1','alpha_absolute_2', 'alpha_absolute_3', 'alpha_absolute_4', 'beta_absolute_1','beta_absolute_2', 'beta_absolute_3', 'beta_absolute_4', 'theta_absolute_1','theta_absolute_2', 'theta_absolute_3', 'theta_absolute_4', 'gamma_absolute_1','gamma_absolute_2', 'gamma_absolute_3', 'gamma_absolute_4'])
for x in (['delta_absolute_1','delta_absolute_2', 'delta_absolute_3', 'delta_absolute_4']):
	df[x] = df[x].shift(-4)
for x in (['beta_absolute_1','beta_absolute_2', 'beta_absolute_3', 'beta_absolute_4']):
	df[x] = df[x].shift(-2)
for x in (['gamma_absolute_1','gamma_absolute_2', 'gamma_absolute_3', 'gamma_absolute_4']):
	df[x] = df[x].shift(-10)
for x in (['theta_absolute_1','theta_absolute_2', 'theta_absolute_3', 'theta_absolute_4']):
	df[x] = df[x].shift(-7)
df2 = df[df['blanks'] == 20]
#df = df[df['delta_absolute_2'] != null]

# # AVERAGING
# # df['DELTA'] = df[['delta_absolute_1','delta_absolute_2', 'delta_absolute_3', 'delta_absolute_4']].mean(axis=1)
# # df['ALPHA'] = df[['alpha_absolute_1','alpha_absolute_2', 'alpha_absolute_3', 'alpha_absolute_4']].mean(axis=1)
# # df['BETA'] = df[['beta_absolute_1','beta_absolute_2', 'beta_absolute_3', 'beta_absolute_4']].mean(axis=1)
# # df['GAMMA'] = df[['gamma_absolute_1','gamma_absolute_2', 'gamma_absolute_3', 'gamma_absolute_4']].mean(axis=1)
# # df['DELTArel'] = df[['delta_relative_1', 'delta_relative_2', 'delta_relative_3', 'delta_relative_4']].mean(axis=1)
# # df['ALPHArel'] = df[['alpha_relative_1', 'alpha_relative_2', 'alpha_relative_3', 'alpha_relative_4']].mean(axis=1)
# # df['BETArel'] = df[['beta_relative_1', 'beta_relative_2', 'beta_relative_3', 'beta_relative_4']].mean(axis=1)
# # df['GAMMArel'] = df[['gamma_relative_1', 'gamma_relative_2', 'gamma_relative_3', 'gamma_relative_4']].mean(axis=1)


df.to_csv(locat)