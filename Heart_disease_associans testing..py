# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:15:03 2022

@author: Sulav
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency
# load data
heart = pd.read_csv('heart_disease.csv')
print(heart.head())

# association between thalach value and presence of heart disease in a boxplot.
sns.boxplot(x=heart.thalach, y=heart.heart_disease)
plt.show()

# from the boxplot we can see mean thalach value for people who don't have heart disease is around 160 whereas it is around 140 for people who do.

thalach_hd = heart.thalach[heart.heart_disease =="presence"]
thalach_no_hd = heart.thalach[heart.heart_disease =="absence"]

# calculating mean difference and median between thalach levels between people who have heart disease and people who don't.

thalach_mean_difference = np.mean(thalach_no_hd) - np.mean(thalach_hd)
thalach_median_difference = np.median(thalach_no_hd) - np.median(thalach_hd)

## Hypothesis testing ## Two Sample T-test
# Null_Hpothesis = Average thalach for person with and without heart disease is equal
# Alt_Hypothesis = Average thalach for person with and without heart disease is not equal.

ttest, pval = ttest_ind(thalach_hd,thalach_no_hd)
print(pval)
#Since pval is a lot less than significance threshold of 0.05, there is a significant difference between average thalach value between people who have heart disease and people who don't.

### Association between cholestrol level and heart disease.
cholestrol_hd = heart.chol[heart.heart_disease =="presence"]
cholestrol_no_hd = heart.chol[heart.heart_disease =="absence"]
plt.clf()
sns.boxplot(x=heart.chol, y=heart.heart_disease)
plt.show()


# Two sample t-test for  cholestrol and heart disease.
# Null_Hpothesis = Average cholestrol level for person with and without heart disease is equal
# Alt_Hypothesis = Average cholestrol level for person with and without heart disease is not equal.

tstat, pval = ttest_ind(cholestrol_hd, cholestrol_no_hd)
print(pval)
#Since pval is greater than significance threshold of 0.05, there is a no significant difference between average cholestrol level between people who have heart disease and people who don't.

## Relationship between quantitative variable and non binary categorical variable using Tukey's range test.
# relationship between thalach and type of chest pain a person experiences.
plt.clf()
sns.boxplot(x=heart.thalach, y=heart.cp)
plt.show()
#box plot shows the relationship between mean thalach level and asymptomatic chest pain is significantly lower than mean thalach level and other types of chest pain.
thalach_typical = heart.thalach[heart.cp =="typical angina"] 
thalach_asymptom = heart.thalach[heart.cp =="asymptomatic"] 
thalach_nonangin = heart.thalach[heart.cp =="non-anginal pain"] 
thalach_atypical = heart.thalach[heart.cp =="atypical angina"]

#Tukey's Range test.
# Hypotheses:
#Null: People with typical angina, non-anginal pain, atypical angina, and asymptomatic people all have the same average thalach.
#Alternative: People with typical angina, non-anginal pain, atypical angina, and asymptomatic people do not all have the same average thalach.
tukey_results = pairwise_tukeyhsd(heart.thalach, heart.cp, 0.05)
print(tukey_results)

#Results:
# p-val less than 0.05 relationships:
#[asymptomatic and atypical angina, asymptomatic and non-anginal pain, asymptomatic and atypical angina] so reject null hypothesis for these relationships.
#p-val greater than 0.05 relationships:
#[atypical angina and non-anginal pain, atypical angina and typical angina, non-anginal pain and typical angina] so accept the null hypothesis for them.

##### Using chi-square method for association between two categorical variables.
# Null: There is NOT an association between chest pain type and whether or not someone is diagnosed with heart disease.
# Alternative: There is an association between chest pain type and whether or not someone is diagnosed with heart disease.
chest_pain = heart.cp
heart_disease = heart.heart_disease
Xtab = pd.crosstab(chest_pain, heart_disease)
print(Xtab)
chi2, pval, dof, expected = chi2_contingency(Xtab)
print(pval)
# since pval is lower than significance threshold of 0.05 we conclude that there is an association between type of chest pain and heart disease.
