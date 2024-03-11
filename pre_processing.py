import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from os import chdir
sns.set()
"""import data"""

chdir("./data")
churn = pd.read_csv('churn_data.csv')

# print(churn.shape) # shows the no. of rows/cols

"""Pre-processing"""

del churn['customerID']

print((churn.isnull()).sum())
print(churn.eq(' ').any(axis=0))
print(churn['TotalCharges'].str.count(' ').sum())

churn_blankTC = churn[churn['TotalCharges'] == ' ']
# print(churn_blankTC)

churn['TotalCharges'] = churn['TotalCharges'].str.replace(' ','0')
churn['TotalCharges'] = churn['TotalCharges'].astype(float)

#
#%%  Primary visualization  of  Overall Churn percentage

sizes = churn['Churn'].value_counts(sort = True)
labels = 'Does not Churn', 'Churns'
colors = ['lightskyblue', 'lightcoral']
explode = (0.1, 0)  # explode Churns

fig = plt.figure()
plt.pie(sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        shadow=True,
        startangle=-180)

plt.axis('equal')
plt.title('% of Churn dataset')
plt.show()
fig.savefig('Churn_pi-chart.pdf')

"""Tenure"""

#%%   tenure -- density plot

fig = plt.figure()
ax = sns.kdeplot(churn.tenure[(churn["Churn"] == 'Yes') ],
                color="salmon", fill = True)
ax = sns.kdeplot(churn.tenure[(churn["Churn"] == 'No') ],
                ax=ax, color="royalblue", fill= True)
ax.set_xlabel("tenure")
ax.set_ylabel("Frequency")
plt.title('Variation of "tenure" for [Un-]Churned customers')
ax = ax.legend(["Churned","UnChurned"])
fig.savefig('Churn_tenure_density-plot.pdf')


#%%  MonthlyCharges -- density plot

fig= plt.figure()
ax = sns.kdeplot(churn.MonthlyCharges[(churn["Churn"] == 'Yes') ],
                color="orangered", fill = True)
ax = sns.kdeplot(churn.tenure[(churn["Churn"] == 'No') ],
                ax=ax, color="cornflowerblue", fill= True)
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Frequency")
plt.title('Variation of "Monthly Charges" for [Un-]Churned customers')
ax = ax.legend(["Churned","UnChurned"])
fig.savefig('Churn_MnthlyChrgs_density-plot.pdf')


#%%  TotalCharges -- density plot

fig= plt.figure()
ax = sns.kdeplot(churn.TotalCharges[(churn["Churn"] == 'Yes') ],
                color="lightcoral", fill= True)
ax = sns.kdeplot(churn.TotalCharges[(churn["Churn"] == 'No') ],
                ax=ax, color="steelblue", fill= True)
ax.set_xlabel("Total Charges")
ax.set_ylabel("Frequency")
plt.title('Variation of "Total Charges" for [Un-]Churned customers')
ax = ax.legend(["Churned","UnChurned"])
fig.savefig('Churn_TotalChrgs_density-plot.pdf')



#%%  Senrion Citizen -- Stacked bar chart

colors = ['slategrey','saddlebrown']
churn_sczn = churn.groupby(['SeniorCitizen','Churn']).size().unstack()
fig= plt.figure()

ax = (churn_sczn.T*100.0/
      churn_sczn.T.sum()).T.plot(kind='bar',
                                 width = 0.4,
                                 stacked = True,
                                 rot = 0,
                                 figsize = (6,6),
                                 color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='upper right',prop={'size':12},title = 'Churn')
ax.set_ylabel('% Customers')
ax.set_title('% Churn in Senior Citizens = 0, 1',size = 13)

# Add data labels to the bar diagram
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:.1f}%'.format(height),
                (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
                weight = 'normal',
                size =14)
fig.savefig('Churn_Sen-Cit_stacked-bar.pdf')
# plt.show()

#%%




