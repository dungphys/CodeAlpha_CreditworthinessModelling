------
# <font color="orange"> Creditworthiness Predictive Modelling </font>

Machine Learning Classifiers for CreditWorthiness Indicator.     
***Author***: Anh Dung Le  
***Date***: November 2026  
@CodeAlpha internship


## <font color="orange"> 1. Objective </font>

This is a hands-on project within the one-month **CodeAlpha** 2026 internship program in machine learning. The project aims to explore an available credit risk dataset and build a predictive model to determine the creditworthiness (Good/Bad) of customers based on it.  

**What is Creditworthiness?** Creditworthiness is an appraisal of a potential borrower's ability and willingness to repay debts by a lender such as a bank (see [Jim Akin (2023) - Experian](https://www.experian.com/blogs/ask-experian/what-is-creditworthiness/)). This can be determined from credit report and demographic information. 

**Why does this matter?** This indicator helps lenders (e.g., banks) make decisions and minimize the financial risks on bab debts. A creditworthy customer will find it easier and cheaper to access credit (and vice versa).  

## <font color="orange"> 2. Data </font>

### **Metadata**

The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/alexdister/credit-risk-dataset). It is given in a CSV file. Some information is given below:

- Name: Credit Risk Dataset
- Owner: Alex Dister 
- License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- Last update: 2026
- Sources: This dataset was collected from a *simulated* Credit Risk Analytics project in the banking and financial sector, *inspired by real-world* credit assessment processes used by financial institutions such as Nova Bank.
 

### **Attributes**
- Number of entries: 32581
- Number of features (target exclusive): 28
- Target: `loan-status`
<center>

| No.| Feature | Description | Group |
|---|---|---|---|
| 1 | cliend_ID | client unique id number | Identifier
| 2 | person_age | age of borrower in years | Demographics
| 3 | gender | gender of borrower | Demographics
| 4 | marital_status | marital status of borrower | Demographics 
| 5 | education_level | highest education level attained | Demographics
| 6 | person_income | borrower's annual income (USD) | Demographics
| 7 | person_emp_length | employment length of the borrower | Demographics 
| 8 | employment_type | type of employment | Demographics
| 9 | person_home_ownership | house ownership status | Demographics
| 10 | country | country of residence | Demographics
| 11 | state | state/province of residence | Demographics
| 12 | city | city of residence | Demographics
| 13 | city_latitude | latitude coordinate of the city | Demographics
| 14 | city_longitude | longitude coordinate of the city | Demographics
| 15 | loan_amnt | requested loan amount in USD | Finance
| 16 | loan_intent | purpose of the loan | Finance
| 17 | loan_grade | redit grade assigned to the loan (A–G) | Finance
| 18 | loan_int_rate | loan interest rate applied (%) | Finance
| 19 | loan_term_months | term of the loan in months | Finance
| 20 | loan_percent_income | percentage of income allocated to loan payment | Finance
| 21 | loan_to_income_ratio | ratio of requested loan to total income | Finance
| 22 | other_debt | amount of other existing debts the borrower has | Finance
| 23 | debt_to_income_ratio | total debt to income ratio | Finance
| 24 | open_accounts | number of open credit accounts | Finance
| 25 | credit_utilization_ratio | credit utilization ratio | Finance
| 26 | cb_person_default_on_file | whether the borrower has defaulted previously (Y/N) | Finance
| 27 | cb_person_cred_hist_length | length of the borrower's credit history in years | Finance
| 28 | past_delinquencies | number of past loan delinquencies | Finance
| ***29*** | ***loan_status*** | ***loan repayment status (0 = non-default, 1 = default) $\to$ Creditworthiness Flag*** | ***Target***

</center>

## <font color="orange"> 2. Data analysis </font>

### **Basic observation**
- 29 Attributes:  float (9), int (8), str (12)
- There are abnormal data in the columns ``person_age`` and ``person_emp_length`` (> 100 years). A check reveals that these abnormal rows are small in size (few rows). So it is safe to remove theme.  
- No duplicates found
- Information on city's longitude and latitude can be (and should be) ignored. They just encode the city itself. (By the way, the city is usually not a useful information, as in this case.)  
- Name normalization:
    + `person_age` --> `person_age_years`
    + `person_income` --> `person_income_usd`
    + `person_emp_length` --> `person_emp_length_years`
    + `loan_amnt` --> `loan_amount_usd`
    + `other_debt` --> `other_debt_usd`
    + `cb_person_cred_hist_length` --> `cb_person_cred_hist_length_years`

### **Missing value analysis**
Two columns `loan_int_rate` and `person_emp_length_years` have missing values in their contents. The statistics of missing values is shown below:

<center>

| feature |  null_count  | null_pct (%) |
|--|--|--|  
|loan_int_rate | 3116| 9.563856|  
|  person_emp_length_years|         895|  2.747000|


<img src="./images/null_distrib.png"
     alt="nulls distribution" width="560"/>

</center>

So, missing values take a small fraction of total content, but not negligible.

Univariate tests (Mann-Whittney U test for numerics and chi-squared test for categories), supplemented by Benjamini-Hochberg false discovery rate (fdr) correction, show that, with alpha level of 0.05,
- in the case of `loan_int_rate` one cannot reject H0 that assumes the same distributions/rates among null row and non-null rows -> the missingness pattern in `loan_int_rate` is not likely to associate with other observed variables. The null pattern is likely MCAR. 
- in the `person_emp_length_years`, one can reject H0 for some variables, such as `person_income_usd` or `other_debt_usd`. There are certain correlation here. The null pattern is likely MAR.

The size of missingness together with random patterns suggest that one might use kNN for imputation.

### **Target distribution**

<center>
<img src="./images/target_distribution.png"
     alt="nulls distribution" width="400"/>
</center>

The default-to-non-default rate is about 1:3.58. So, this target is imbalanced. However, the class imbalance is mild.  

### **Target distribution by demographic features**

<center>
<img src="./images/target_on_cat_dem.png"
     alt="nulls distribution" width="600"/>
<img src="./images/target_on_num_dem.png"
     alt="nulls distribution" width="600"/>
</center>

The distribution of the loan status is the (nearly) same for different categories in following demographic features: marital status, gender, country, employment type, education level. These features therefore would not help in modelling, better to remove.

The differences in target distribution among categories in city and state are small. These two features have rather high cardinality, so with OHE encoding, it would increase the feature space, make data look very sparse, and make the model training and inference slow. So, it would be harmful rather than helpful. Good idea to remove them.

Home ownership carries useful information related to the target: borrowers with owned house or mortgage are less likely to default. This feature is a useful predictor.

Income has a clear correlation with the loan status: non-default borrowers are likely to have higher income. This os also a good predictor.

### **Target distribution by financial features**

<center>
<img src="./images/target_on_cat_fin.png"
     alt="nulls distribution" width="600"/>
<img src="./images/target_on_num_fin.png"
     alt="nulls distribution" width="600"/>
</center>

There are visible correlations with the target for most of financial features. Some typical relations:
- Better loan grade, lower default rate. There are sharp changes between C and D, and between F and G. 
- Borrowerd with previous default history are more likely to default
- Higher interest rate more likely leads to default.
- Borrowers with other debts are less likely to default.
- Higher loan to income or debt to income ratio, more likely to default.

### **Correlation heatmap**

<center>
<img src="./images/correlation_heatmap.png"
     alt="nulls distribution" width="600"/>
</center>

There is a noticable strong correlation (approx. 1.0) between `loan_percent_income` and `loan_to_income_ratio`. A follow-up analysis showed that one should keep only 1 of these two (say, `loan_to_income_ratio`) and the their ratio for linear model (logistic regression) to avoid multicollinearity. For tree-based models, it is fine to keep both, and also their ratio. 

Consequently, we have two datasets: 1 with both (df_trees), and 1 with only `loan_to_income_ratio` (df_logit).






