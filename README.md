------
# <font color="orange"> CreditWorthiness Predictive Modelling </font>

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




