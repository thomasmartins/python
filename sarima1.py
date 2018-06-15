import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


arrecad = pd.read_csv("C:\\Users\\thomas.martins\\Downloads\\arrec_liq_rgps.csv",sep=';')
beneficios = pd.read_csv("C:\\Users\\thomas.martins\\Downloads\\desp_ben_prev.csv",sep=';')

beneficios['Data'] = pd.to_datetime(beneficios['Data'],dayfirst=True)
arrecad['Data'] = pd.to_datetime(arrecad['Data'],dayfirst=True)

beneficios.set_index('Data',inplace=True)
arrecad.set_index('Data',inplace=True)

lnbeneficios = beneficios.apply(lambda x: np.log(x))
lnarrecad = arrecad.apply(lambda y: np.log(y))

dlnarrecad = lnarrecad.diff()
dlnbeneficios = lnbeneficios.diff()

sdlnarrecad = lnarrecad.diff(12)
sdlnbeneficios = lnbeneficios.diff(12)

from statsmodels.tsa.vector_ar.var_model import VAR, FEVD

varseries1 = pd.concat([lnarrecad,lnbeneficios],axis=1)
VAR1 = VAR(varseries1)
varseries2 = pd.concat([lnbeneficios,lnarrecad],axis=1)
VAR2 = VAR(varseries2)
fit1 = VAR1.fit(11)
fit2 = VAR2.fit(11)

varseries1.reset_index()
varseries1['month'] = ""

for i in range(0,len(varseries1)):
  varseries1.loc[i,varseries1.columns[3]] = varseries1.loc[i,varseries1.columns[0]].month()
  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(dlnarrecad.dropna(), lags=36)
plt.title('FAC LNARRECAD')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\acf_arrecad.png')
plt.clf()
plot_acf(dlnbeneficios.dropna(), lags=36)
plt.title('FAC LNBENEFICIOS')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\acf_beneficios.png')
plt.clf()
plot_pacf(dlnarrecad.dropna(), lags=36)
plt.title('FACP LNARRECAD')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\pacf_arrecad.png')
plt.clf()
plot_pacf(dlnbeneficios.dropna(), lags=36)
plt.title('FACP LNBENEFICIOS')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\pacf_beneficios.png')
plt.clf()
plot_acf(sdlnarrecad.dropna(), lags=36)
plt.title('FAC LNARRECAD com diferença sazonal 12')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\acf_arrecad_s12.png')
plt.clf()
plot_acf(sdlnbeneficios.dropna(), lags=36)
plt.title('FAC LNBENEFICIOS com diferença sazonal 12')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\acf_beneficios_s12.png')
plt.clf()
plot_pacf(sdlnarrecad.dropna(), lags=36)
plt.title('FACP LNARRECAD com diferença sazonal 12')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\pacf_arrecad_s12.png')
plt.clf()
plot_pacf(sdlnbeneficios.dropna(), lags=36)
plt.title('FACP LNBENEFICIOS com diferença sazonal 12')
plt.savefig('C:\\Users\\thomas.martins\\Desktop\\pacf_beneficios_s12.png')
plt.clf()

from statsmodels.tsa.statespace.sarimax import SARIMAX

model1=SARIMAX(endog=lnarrecad,order=(0,1,1),seasonal_order=(1,1,1,12),trend='c',enforce_invertibility=False)
results1=model1.fit()
results1.summary()

model2=SARIMAX(endog=lnbeneficios,order=(0,1,1),seasonal_order=(1,1,1,12),trend='c',enforce_invertibility=False)
results2=model2.fit()
results2.summary()

model3=SARIMAX(endog=lnarrecad,order=(0,1,0),seasonal_order=(1,1,1,12),trend='c',enforce_invertibility=False)
results3=model3.fit()
results3.summary()

model4=SARIMAX(endog=lnbeneficios,order=(0,1,0),seasonal_order=(1,1,1,12),trend='c',enforce_invertibility=False)
results4=model4.fit()
results4.summary()

model5=SARIMAX(endog=lnarrecad,order=(0,1,1),seasonal_order=(0,1,1,12),trend='c',enforce_invertibility=False)
results5=model5.fit()
results5.summary()

model6=SARIMAX(endog=lnbeneficios,order=(0,1,1),seasonal_order=(0,1,1,12),trend='c',enforce_invertibility=False)
results6=model6.fit()
results6.summary()


