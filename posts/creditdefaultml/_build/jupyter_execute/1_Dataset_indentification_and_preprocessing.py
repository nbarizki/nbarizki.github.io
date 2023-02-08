#!/usr/bin/env python
# coding: utf-8

# # LendingClub Dataset - A collection of Peer-to-Peer Individual Loan History

# ## 1. Introduction

# This notebook is an introduction to *loan dataset* that will be utilized as a case example in building classification machine learning model. This notebook is purposed to show each of the features characteristic hence no in-depth analysis or assumption will be performed in this notebook.

# #### **Basic Settings**

# In[1]:


import pandas as pd
import numpy as np

pd.set_option('display.float', '{:.2f}'.format)
pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 75)

loan_df = pd.read_csv(
    'dataset\lc_2010-2015.csv', 
    dtype={'desc': 'str', 'verification_status_joint': 'str'}
    )


# ## 2. A Brief Explanation about the Data

# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market.
# 
# LendingClub enables borrowers to create unsecured personal loans between $1,000 and $40,000. The standard loan period is three years. Investors can search and browse the loan listings on Lending Club website and select loans that they want to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors make money from interest. Lending Club makes money by charging borrowers an origination fee and investors a service fee.
# 
# However, peer-to-peer individual loan currently is not available.

# #### **Source**

# The dataset is obtained from: https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans

# ## 3. Identifying Dataset Characteristic

# #### **Duplicated Records**

# Before going further in identifying the dataset dimension, let's clear the duplicated observations first, if any.

# In[2]:


# duplicated loan record
loan_df[loan_df.duplicated()]


# #### **Features and Dimension**

# Let's discover each fields and records of the *Lending Club* datasets!

# In[3]:


loan_df.info()


# Takeaway:
# 
# - No. of rows/observations = 887,379
# - No. of columns/features = 72

# #### **More about the Dataset Features**

# To understand more about context of each fields, it is best generalized them into 3 main categories:
# 
# - Data fields related to **applicant's profile**, including their past credit history. This includes below fields:
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: left;">
#       <th></th>
#       <th>Loan Data Field Label</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>1</th>
#       <td>member_id</td>
#       <td>A unique LC assigned Id for the borrower member.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>delinq_2yrs</td>
#       <td>The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>inq_last_6mths</td>
#       <td>The number of inquiries in past 6 months (excluding auto and mortgage inquiries)</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>mths_since_last_delinq</td>
#       <td>The number of months since the borrower's last delinquency.</td>
#     </tr>
#     <tr>
#       <th>28</th>
#       <td>mths_since_last_record</td>
#       <td>The number of months since the last public record.</td>
#     </tr>
#     <tr>
#       <th>29</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>31</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>32</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.  </td>
#     </tr>
#     <tr>
#       <th>33</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>48</th>
#       <td>collections_12_mths_ex_med</td>
#       <td>Number of collections in 12 months excluding medical collections</td>
#     </tr>
#     <tr>
#       <th>49</th>
#       <td>mths_since_last_major_derog</td>
#       <td>Months since most recent 90-day or worse rating</td>
#     </tr>
#     <tr>
#       <th>50</th>
#       <td>policy_code</td>
#       <td>publicly available policy_code=1. new products not publicly available policy_code=2</td>
#     </tr>
#     <tr>
#       <th>52</th>
#       <td>annual_inc_joint</td>
#       <td>The total number of credit lines currently in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>53</th>
#       <td>dti_joint</td>
#       <td>A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income. </td>
#     </tr> 
#     <tr>
#       <th>54</th>
#       <td>verification_status_joint</td>
#       <td>Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified. </td>
#     </tr>
#     <tr>
#       <th>55</th>
#       <td>acc_now_delinq</td>
#       <td>The number of accounts on which the borrower is now delinquent.</td>
#     </tr>
#     <tr>
#       <th>56</th>
#       <td>tot_coll_amt</td>
#       <td>Total collection amounts ever owed</td>
#     </tr>
#     <tr>
#       <th>59</th>
#       <td>tot_cur_bal</td>
#       <td>Total current balance of all accounts</td>
#     </tr>
#     <tr>
#       <th>58</th>
#       <td>open_acc_6m</td>
#       <td>Number of open trades in last 6 months</td>
#     </tr>
#     <tr>
#       <th>59</th>
#       <td>open_il_12m</td>
#       <td>Number of installment accounts opened in past 12 months</td>
#     </tr>
#     <tr>
#       <th>60</th>
#       <td>open_il_24m</td>
#       <td>Number of installment accounts opened in past 24 months</td>
#     </tr>
#     <tr>
#       <th>61</th>
#       <td>mths_since_rcnt_il</td>
#       <td>Months since most recent installment accounts opened</td>
#     </tr>
#     <tr>
#       <th>62</th>
#       <td>total_bal_il</td>
#       <td>Total current balance of all installment accounts</td>
#     </tr>
#     <tr>
#       <th>63</th>
#       <td>il_util</td>
#       <td>Ratio of total current balance to high credit/credit limit on all install acct</td>
#     </tr>
#     <tr>
#       <th>64</th>
#       <td>open_rv_12m</td>
#       <td>Number of revolving trades opened in past 12 months</td>
#     </tr>
#     <tr>
#       <th>65</th>
#       <td>open_rv_24m</td>
#       <td>Number of revolving trades opened in past 24 months</td>
#     </tr>
#     <tr>
#       <th>66</th>
#       <td>max_bal_bc</td>
#       <td>Maximum current balance owed on all revolving accounts</td>
#     </tr>
#     <tr>
#       <th>67</th>
#       <td>all_util</td>
#       <td>Balance to credit limit on all trades</td>
#     </tr>
#     <tr>
#       <th>68</th>
#       <td>total_rev_hi_lim</td>
#       <td>Total revolving high credit/credit limit</td>
#     </tr>
#     <tr>
#       <th>69</th>
#       <td>inq_fi</td>
#       <td>Number of personal finance inquiries</td>
#     </tr>
#     <tr>
#       <th>70</th>
#       <td>total_cu_tl</td>
#       <td>Number of finance trades</td>
#     </tr>
#     <tr>
#       <th>71</th>
#       <td>inq_last_12m</td>
#       <td>Number of credit inquiries in past 12 months</td>
#     </tr>                                                                          
#   </tbody>
# </table>

# 
# - Data fields related to applicant's **loan profile** that registered. This includes below fields:
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: left;">
#       <th></th>
#       <th>Loan Data Field Label</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>id</td>
#       <td>A unique LC assigned ID for the loan listing.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>pymnt_plan</td>
#       <td>Indicates if a payment plan has been put in place for the loan</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>desc</td>
#       <td>Loan description provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request. </td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>50</th>
#       <td>policy_code</td>
#       <td>publicly available policy_code=1, new products not publicly available policy_code=2</td>
#     </tr>
#     <tr>
#       <th>51</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#         <tr>
#       <th>38</th>
#       <td>total_pymnt</td>
#       <td>Payments received to date for total amount funded</td>
#     </tr>
#     <tr>
#       <th>39</th>
#       <td>total_pymnt_inv</td>
#       <td>Payments received to date for portion of total amount funded by investors</td>
#     </tr>
#   </tbody>
# </table>

# 
# - Data fields related to **post-originated loan data**, which describes each of the loan's performance after originated to the borrower.
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: left;">
#       <th></th>
#       <th>Loan Data Field Label</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>3</th>
#       <td>funded_amnt</td>
#       <td>The total amount committed to that loan at that point in time.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>funded_amnt_inv</td>
#       <td>The total amount committed by investors for that loan at that point in time.</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>35</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W (Whole), F (Fractional)</td>
#     </tr>
#     <tr>
#       <th>36</th>
#       <td>out_prncp</td>
#       <td>Remaining outstanding principal for total amount funded</td>
#     </tr>
#     <tr>
#       <th>37</th>
#       <td>out_prncp_inv</td>
#       <td>Remaining outstanding principal for portion of total amount funded by investors</td>
#     </tr>
#     <tr>
#       <th>40</th>
#       <td>total_rec_prncp</td>
#       <td>Principal received to date</td>
#     </tr>
#     <tr>
#       <th>41</th>
#       <td>total_rec_int</td>
#       <td>Interest received to date</td>
#     </tr>
#     <tr>
#       <th>42</th>
#       <td>total_rec_late_fee</td>
#       <td>Late fees received to date</td>
#     </tr>
#     <tr>
#       <th>43</th>
#       <td>recoveries</td>
#       <td>post charge off gross recovery</td>
#     </tr>
#     <tr>
#       <th>44</th>
#       <td>collection_recovery_fee</td>
#       <td>post charge off collection fee</td>
#     </tr>
#     <tr>
#       <th>45</th>
#       <td>last_pymnt_d</td>
#       <td>Last month payment was received</td>
#     </tr>
#     <tr>
#       <th>46</th>
#       <td>last_pymnt_amnt</td>
#       <td>Last total payment amount received</td>
#     </tr>
#     <tr>
#       <th>47</th>
#       <td>next_pymnt_d</td>
#       <td>Next scheduled payment date</td>
#     </tr>
#     <tr>
#       <th>48</th>
#       <td>last_credit_pull_d</td>
#       <td>The most recent month LC pulled credit for this loan</td>
#     </tr> 
#   </tbody>
# </table>

# #### **Checking the Datatypes of Each Features**

# If we inspect features description from previous section thoroughly, we can identify clearly which features contains these characteristics:
# 
# 1. For numerical data, which are the discrete and continuous data.
# 2. For string object, which are the categorical data and which are the unique string (most often recorded by manual input e.g. from typing)
# 
# ##### *Numerical Data*

# We already identified that `pandas` has parsed the numerical features resulting into `int` and `float`, which indicate that `pandas` has tried to *idetify* those features accordingly. We only need to inspect *wrongly-identified* datatypes by inspecting each of `int` and `float` data.
# 
# For `int`:

# In[4]:


data_types = loan_df.aggregate(lambda x: x.dtype)
data_types[data_types == 'int64']


# For `float`:

# In[5]:


data_types[data_types == 'float64']


# Takeaway:
# 
# - Feature `delinq_2yrs`, `open_acc`, `pub_rec`,`total_acc`, `collections_12_mths_ex_med`, `policy_code`, `acc_now_delinq`,  should be `int`.

# In[6]:


cast_to_int = [
    'delinq_2yrs', 'open_acc', 'pub_rec',
    'total_acc', 'collections_12_mths_ex_med', 'policy_code',
    'acc_now_delinq'
    ]

for int_feature in cast_to_int:
    loan_df.loc[:, int_feature] = loan_df.loc[:, int_feature].astype('int64', errors='ignore')


# ##### *Object Data*

# For string object data, it is easily recognizable since categorical data should contain small unique observations, while manually-input data should have high-cardinality.

# In[7]:


loan_df.select_dtypes(include='object').aggregate(
    lambda x: x.drop_duplicates().count()
    ).sort_values(ascending=False)


# After inspecting each of the features:
# 
# - High-cardinality object. Feature `desc` has a very specific context, since it contains the purpose of the loan.

# In[8]:


high_cardinal_features = ['emp_title', 'title', 'desc', 'zip_code', 'addr_state']
for feature in high_cardinal_features:
    loan_df.loc[:, feature] = loan_df.loc[:, feature].str.strip()

loan_df[['emp_title', 'title', 'desc', 'zip_code', 'addr_state']].sort_values('desc', ascending=True).head()


# - Some of the features is time-related observations:

# In[9]:


loan_df[['earliest_cr_line', 'last_credit_pull_d', 'last_pymnt_d', 'issue_d', 'next_pymnt_d']].sample(5)


# In[10]:


datetime_features = [
    'earliest_cr_line', 'last_credit_pull_d', 'last_pymnt_d', 
    'issue_d', 'next_pymnt_d'
    ]
categorical_features = [
    'sub_grade', 'purpose', 'emp_length', 'loan_status', 
    'grade', 'home_ownership', 'verification_status_joint', 'verification_status', 
    'initial_list_status', 'pymnt_plan', 'application_type', 'term'
    ]
# casting dtype
for datetime_feature in datetime_features:
    loan_df.loc[:, datetime_feature] = \
        pd.to_datetime(loan_df.loc[:, datetime_feature], format='%b-%Y')
for categorical_feature in categorical_features:
    loan_df.loc[:, categorical_feature] = loan_df.loc[:, categorical_feature].astype('category', errors='ignore')


# Most of the time, categorical input is generally system-regulated and duplication is rarely to be found. Often dirty duplication is caused by capitalization so let's check them out! 

# In[11]:


category_columns = \
    loan_df.select_dtypes(include='category').columns

for column in category_columns:
    categories = list(loan_df[column].dtypes.categories)
    value_counts = loan_df[column].value_counts(sort=False)
    cat_dict = dict(zip(categories, value_counts))
    print(column, '. Categories: ', cat_dict, '\n')


# Feature `emp_length` is clearly an ordinal category but is not ordered correctly.

# In[12]:


emp_length_order = [
    '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
    ]

loan_df['emp_length'] = pd.Categorical(
    values=loan_df['emp_length'],
    categories=emp_length_order,
    ordered=True
    )


# #### **Target Feature: `loan_status`**

# Feature `loan_status` is the possible features as a target of supervised learning to predict whether particular loan is good or bad.

# In[13]:


loan_status_count = loan_df.loan_status.value_counts()
loan_status_percent = loan_df.loan_status.value_counts(normalize=True, )
loan_status = pd.DataFrame(
    [loan_status_count, loan_status_percent * 100]
    ).T
loan_status.columns = ['count', '%']
loan_status


# Based on above status, we can identify that:
# 
# - On-going loan has status: `Current`, `Late`, and `In Grace Period`
# - Finished loan has status either `Fully Paid`, `Charged Off`, and `Default`. 
# 
# Most of the loan listed in our dataset is on-going loan (`loan_status` = `Current`). However, we can identify some loan that faces some difficulties in paying-off debt on the due date: `Late` and `In Grace Period`. 
# 
# Let's see an example of `loan_status` recorded as `Current` and each of their principal loan status!

# In[14]:


loan_df\
    .loc[loan_df.loan_status == 'Current', ['issue_d', 'loan_amnt', 'total_rec_prncp', 'out_prncp', 'loan_status']]\
    .sample(5, random_state=99)\
    .assign(
        portion_paid=lambda x: x.total_rec_prncp / x.loan_amnt * 100
    )


# It is clear that `total_rec_prncp` is the total principal that has been paid, and `out_prncp` is the remaining principal to be paid. How about `portion_paid` for `loan_status` = `Charged Off` or `Default`?

# In[15]:


loan_df\
    .loc[loan_df.loan_status.isin(['Charged Off', 'Default']), ['issue_d', 'loan_amnt', 'total_rec_prncp', 'out_prncp', 'loan_status']]\
    .sample(5, random_state=5)\
    .assign(
        portion_paid=lambda x: x.total_rec_prncp / x.loan_amnt * 100
    )


# More in-depth analysis will be encouraged in the next chapter: **Exploration of Lending Club Dataset**.

# ## 4. Custom Transformer for Dataset Preprocessing

# In[16]:


from sklearn.base import BaseEstimator, TransformerMixin

class LoanDataPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.float_features_ = [
            'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
            'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',
            'inq_last_6mths', 'mths_since_last_delinq',
            'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
            'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
            'total_rec_int', 'total_rec_late_fee', 'recoveries',
            'collection_recovery_fee', 'last_pymnt_amnt',
            'collections_12_mths_ex_med', 'mths_since_last_major_derog',
            'policy_code', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq',
            'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_12m',
            'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
            'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
            'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m'
            ]
        self.int_features_ = [
            'id', 'member_id', 'delinq_2yrs', 'open_acc', 'pub_rec',
            'total_acc', 'collections_12_mths_ex_med', 'policy_code',
            'acc_now_delinq'
            ]
        self.datetime_features_ = [
            'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
            'last_credit_pull_d'
            ]
        self.string_features_ = ['emp_title', 'title', 'desc', 'zip_code', 'addr_state']
        self.categorical_features_ = [
            'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
            'verification_status', 'loan_status', 'pymnt_plan', 'purpose',
            'initial_list_status', 'application_type', 'verification_status_joint'
            ]
        self.emp_length_order_ = [
            '< 1 year', '1 year', '2 years', '3 years', '4 years', 
            '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
            ]
        return self
        
    def transform(self, X, y=None):
        # numerical datatype casting
        for int_feature in self.int_features_ :
            X.loc[:, int_feature] = X.loc[:, int_feature].astype('int64', errors='ignore')
        for float_feature in self.float_features_:
            X.loc[:, float_feature] = X.loc[:, float_feature].astype('float64', errors='ignore')
        # strip string
        for string_feature in self.string_features_:
            X.loc[:, string_feature] = X.loc[:, string_feature].str.strip()
        # datetime datatype casting
        for datetime_feature in self.datetime_features_:
            X.loc[:, datetime_feature] = \
                pd.to_datetime(X.loc[:, datetime_feature], format='%b-%Y')
        # categorical datatype casting
        for categorical_feature in self.categorical_features_:
            X.loc[:, categorical_feature] = \
                X.loc[:, categorical_feature].astype('category', errors='ignore')
        # ordinal category order
        X['emp_length'] = pd.Categorical(
            values=X['emp_length'],
            categories=self.emp_length_order_,
            ordered=True
            )
        return X


# Let's try the transformer for our hold-out dataset, the `Loan Data 2016-2017`, which we will utilize as the validation set for our machine learning model.

# In[17]:


loan_df_heldout = pd.read_csv(
    'dataset/lc_2016-2017.csv', 
    dtype={'desc': 'str', 'verification_status_joint': 'str'}
    )
loan_data_preprocess = LoanDataPreprocess()
loan_data_preprocess.fit(loan_df)
loan_df_heldout = loan_data_preprocess.transform(loan_df_heldout)

loan_df_heldout.dtypes


# ## 5. What's Next

# The interesting part will come on the next two chapters. We will explore the dataset in a hope to find some interesting insight and later we will develop a classification model to demonstrate the usefulness of machine learning model to deliver informed decision.
