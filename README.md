# Wage_Predictions
Use PUMS Data from Census.gov to Predict Wages for individuals
![](media/image1.png){width="2.803274278215223in"
height="1.9479166666666667in"}

**Predicting Annual Wages of Individuals Based on Public Use Microdata
Samples:**

**A Big Data Analytics Approach**

**MIS 749: Business Analytics**

**Project Report**

**San Diego State University**

**May 2, 2018**

**Team: Mean Squares**

Mayuri Kudale

Amit Gupte

Kanchan Pathak

Monish Thakore

Table of Contents

[Part 1 Executive Summary 3](#_Toc450490166)

[Part 2 Discovery 4](#_Toc450490167)

[Part 3 Data Preparation 6](#_Toc450490174)

[Part 4 Model Planning 12](#_Toc450490181)

[Part 5 Model Building 12](#_Toc450490182)

[Part 6 Results and Performance 16](#_Toc450490189)

[Part 7 Discussions and Recommendations 20](#_Toc450490194)

[References 22](#_Toc450490195)

[Appendix: R Code 22](#_Toc450490196)


**Executive Summary**

Salary estimator tool plays a significant role in today's world, it
helps the jobseeker in knowing the average wages paid according to their
job title. Currently job search engines like Glassdoor, Simply Hired,
Monster and many more are providing the salary estimator which takes
into consideration only two variables -- Job title and Job location.

By predicting wages based on the individual's details like education,
age, hours worked per week, and so on, tools can provide better and
accurate wage predictions specifically for that individual. It will also
help people understand why certain people make more than others and what
an individual can do to set themselves up for greater earning potential.

The dataset was provided from Census website which is collected by
American Community Survey (ACS). The American Community Survey (ACS)
helps local officials, community leaders, and businesses understand the
changes taking place in their communities. It is the premier source for
detailed population and housing information in the USA. The American
Community Survey (ACS) Public Use Microdata Sample (PUMS) files are a
set of untabulated records about individual people or housing units.
PUMS files are perfect for people like students as it gives great
accessibility to inexpensive data for research projects. Social
scientists often use the PUMS for regression analysis and modeling
applications.

**Project Goals**

-   To develop a predictive model that can best predict wages based on
    the other details of the individual like education, age, hours
    worked per week, and so on.

-   Identify factors driving salary of an individual.

-   Aid jobseekers in calculating their personalized salary estimate
    using their own person details.

**Key Points**

-   The dataset is summarized in below table based on the total number
    of observations, total number of training set observations, and
    number of test set observations. The dataset observations were split
    between training and test set, 70% and 30% respectively.

  **Dataset**                                   
  --------------------------------------------- ---------
  Number of observations in total dataset:      112,810
  Number of observations in training dataset:   78967
  Number of observations in test dataset:       33843

-   R was used to implementing 7 various model building techniques:

i)  Linear Regression.

ii) Lasso Regression

iii) Random Forest

iv) Decision Tree

v)  Generalized Additive Model (GAM)

vi) Boosting

vii) Bagging.

-   The Random Forest model provided the best modeling technique for
    predicting the wages with a training dataset RMSE of \#\#, and test
    dataset RMSE of \#\#.

**Discovery**

The dataset used in the report is data of California population only.
Our group located it on the census website. Data was provided in a .csv
(comma-separated values) format, which contained 112,810 observations
and 200 attributes.

Column WAGP which represented annual wages was taken as the response
variable. The other variables included details like schooling level,
age, gender, Industry etc. Distribution of variables can be explained
visually in the figure 1.0.

![A screenshot of a cell phone Description generated with very high
confidence](media/image2.png){width="5.952180664916885in"
height="3.3481014873140857in"}

*Figure 1.0: Distribution of all Variables*

After inspecting all the variables, as seen in the above plots, we found
below details:

-   AGEP (Age of a Person): The graph represents that the data is dense
    for the people from the age group of 15-78 years old.

-   CIT(Citizenship): This field shows that maximum people in California
    have a citizenship code equal to 1 representing people born in US.

-   COW (Class of Workers): This field shows that maximum people in
    California work for private and for-profit organizations firms
    represented by code=1.

-   ESR (Employment Status recode): This field shows that maximum people
    in California have a ESR=1 representing employed civilians.

-   INDP (Industry recode): This field shows that people in California
    work across different industries.

-   MSP (Marital Status): This field shows that maximum people in
    California are married.

-   RAC1P (Race Code): This field shows that maximum people in
    California are White followed by Asians and other Pacific Islanders.

-   SCHL (Schooling Level): This field shows that maximum people in
    California have a bachelor's Degree, followed by others who are high
    school graduates.

-   SEX: Maximum people working in California are Male.

-   WAGP (Annual Salary): This is our response variable. It represents
    salary of people in California.

-   WKHP (Number of Hours worked/week): This field shows that maximum
    people in California work 40 hours a week.

We then tried to visualize the data to check if there are any extreme
values/outliers. As shown in figure 1.1, this boxplot helped us identify
the outliers:

![A close up of a door Description generated with high
confidence](media/image3.png){width="5.550633202099737in"
height="3.122823709536308in"}

*Figure 1.1: Outliers*

We also discovered that the most significant issue with the dataset was
the missing data in multiple fields like schooling level, citizenship
etc. about 36%.

Below are some of the core Hypotheses we were focusing on:

**Null Hypothesis:** The annual salary earned by people in California
has no relationship to the personal data collected by the census.

**Alternative Hypothesis:** The annual salary earned by people in
California has relationship to the personal data collected by the census
and can be strongly predicted based on this data.

**Data Preparation**

The group began the data preparation process by downloading the dataset
from Census website. The original file type was in .csv format and
zipped. We started preparing our data for model building by pruning
variables for several reasons. The original dataset contained multiple
variables.

As a part of data preparation process, the activities consisted of:

1.  Saving the dataset into relational database system file (RDS)

2.  Identifying and graphing correlated predictors and removing them.

3.  Centering and scaling the data

4.  Running descriptive stats using describe, summary, and str

5.  Approximated missing values using Imputation.

6.  Extreme input variables were removed (outlier detection and removal)

7.  Data transformation for categorical data

8.  Removal of Near zero variance variables

9.  Splitting the dataset into training and test sets.

For the data preparation steps, tools which were implemented include R,
Tableau, SQL, and Microsoft Excel.

![](media/image4.JPG){width="4.395833333333333in" height="3.53125in"}

*Figure 2.0: Correlation Plot*

**Model Planning**

For model training, a training set was constructed out of 70% of the
sample, with the remaining 30% used as the test data. Once constructed,
R summary functions were used to construct correlation plots and
accuracy plots among the various modeling techniques.

We started our modelling phase with a simple Decision Tree
implementation to get a fair idea about the important predictors.

![A close up of a device Description generated with high
confidence](media/image5.png){width="4.133916229221347in"
height="2.8003958880139983in"}

*Figure 3.0: Decision Tree*

The Decision Tree output showed that the most important predictor for
predicting salary was schooling level. Other important predictors turned
out to be Age(AgeP) and Working hours/week(WKHP)

Resampling was done using Cross validation with 5 folds. The following
models were planned for training/testing:

-   Linear Regression

-   Lasso Regression

-   Random Forest

-   Decision Tree

-   GAM

-   Boosting

-   Bagging

-   Support Vector Machines

-   Neural Network

**Model Building**

For modeling validation, a training set was constructed from 70% of the
sample data, with remaining 30% used for the test set.

After Feature selection techniques implemented using CARET, all the
models mentioned below were constructed over the important set of
predictors. The team did not build alternate versions of these models
based on Principal Component Analysis because after pre-processing only
10 important predictors were left to run models.

![A close up of a map Description generated with very high
confidence](media/image6.png){width="5.802083333333333in"
height="2.911111111111111in"}

*Figure 4.0: Variable Importance*

All models were constructed using the CARET package and tuning
parameters were selected to minimize training RMSE.

Model that worked for our dataset was Boosting with a depth of 8. Other
different models used were:

-   Linear Regression

-   Lasso Regression

-   Random Forest

-   Decision Tree

-   GAM

-   Boosting

-   Bagging

**Results and Performance**

Based on model results, we calculated Root Mean Square Error (RMSE) and
R squared (R\^2) values to assess and measure performances and accuracy
of models. They are shown below in Figure 5, Figure 5.1. Boosting
provided the most accurate results.

  **Data Modeling Techniques**   **Tuning Parameter**   **R\^2**   **Error (Train)**   **Error (Test)**
  ------------------------------ ---------------------- ---------- ------------------- ------------------
  Linear Regression                                     26         RMSE =35000         RMSE =35670
  Lasso Regression                                      25.6       RMSE =38000         RMSE =39090
  Random Forest                                         43         RMSE =32000         RMSE =32091
  Decision Tree                                         18         RMSE =31300         RMSE =33102
  GAM                                                   32         RMSE =35000         RMSE =36070
  Bagging                                               29         RMSE =32700         RMSE =32905
  Boosting                                              42         RMSE=29000          RMSE=29311

However, the difference between Train and Test error was huge. As per
the Professors' suggestion we ran our modes on a range of data were the
residuals were zero for Random Forest. This gave us a split of dataset
where salary was between \$20,000 to \$200,000 per annum.

After re-running the models on the modified split of the dataset we were
able to achieve a better R-square and lower RMSE. And the modified Root
Mean Square Error (RMSE) and R squared (R\^2) are as below:

  **Data Modeling Techniques**   **Tuning Parameter**   **R\^2**   **Error (Train)**   **Error (Test)**
  ------------------------------ ---------------------- ---------- ------------------- ------------------
  Linear Regression                                     23         RMSE =33100         RMSE =32850
  Lasso Regression                                      23         RMSE =33000         RMSE =32885
  Random Forest                                         42         RMSE =29000         RMSE =29697
  Decision Tree                                         22         RMSE =33300         RMSE =33102
  GAM                                                   32         RMSE =31200         RMSE =31218
  Bagging                                               24         RMSE =32790         RMSE =32707
  Boosting                                              43         RMSE=28700          RMSE=28839

The best model that fit the data was Boosting with an R-square of 43 and
a test RMSE of \$28839. Figure 5 shows the results of boosting with
different depths.

![A close up of a map Description generated with high
confidence](media/image7.png){width="5.46805227471566in"
height="3.7151891951006126in"}

*Figure 5: Boosting Performance vs Iterations*

![A screenshot of a cell phone Description generated with very high
confidence](media/image8.png){width="5.204037620297463in"
height="3.525317147856518in"}

*Figure 5.1: Train RMSE for all models*

![](media/image9.png){width="5.166666666666667in" height="3.5in"}

*Figure 5.2: Train R squares for all models*

**\
**

**Discussion**

Based on the data analysis, our team concluded that methods like Random
Forest and Boosting can provide the best fit for outcome of a Salary
Prediction.

 From our initial data analysis, we realized that dataset was extremely
skewed and contained a lot of missing data. Imputation packages in R,
like Mice, work well to fill in the missing values and make the records
usable for analysis. We have also used CARET package for Data Partition
and Model Building. However, the extreme skewed nature of the target
variables, the samples provided required further processing to be able
to more accurately predict the salary.

**Recommendations**

Another recommendation is potentially selling the results to other job
search websites, which could help target specific job seekers. A pilot
program to implement these models in predicting the wages and making job
recommendations for jobseekers is suggested.

**Prospects**

We also recommend running models like Support Vector Machines and Neural
Network which can improve the accuracy and lead to increase in revenues
for the job search websites. Lastly, we recommend including the
observations from all the states of USA to get better wage prediction
which can help the job search websites in all regions of the USA.

**References**

1.  American Community Survey Data Published by Census Board (2016)
    Retrieved from:

    <https://www.census.gov/programs-surveys/acs/data/pums.html>

2.  Caret Official Documentation as References for Code:

    <https://cran.r-project.org/web/packages/caret/caret.pdf>

3.  Ggplot Documentation for Visualization:

    <https://cran.r-project.org/web/packages/ggplot2/ggplot2.pdf>
