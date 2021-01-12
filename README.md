# NYPD Complaints Exploration

## Goal
The goal of this project was to explore the recently released CCRB dataset from [Pro Publica](https://www.propublica.org/datastore/dataset/civilian-complaints-against-new-york-city-police-officers), to investigate trends in CCRB complaint characteristics and rulings, and use classification methodologies to see whether CCRB rulings can be predicted, and build understanding of what factors influence CCRB decisions.

## Context
The NYC Civilian Complaint Review Board (CCRB) is the oversight agency of NYPD. It investigates complaints of misconduct against America's largest police force. CCRB data was inaccessible to the public until recently. In the wake of national protests against police brutality, a state law protecting their secrecy was repealed in June 2020, enabling the public to access and review decades of complaints against the NYPD. My hope in exploring this data is to build understanding and awareness of how this mechanism for police accountability works (or falls short).

## Methodologies
1. Created a postgreSQL database to store the complaints data (accessed from Pro Publica) locally.
2. Pulled precinct-level demographic data from the US Census using the [CensusData](https://pypi.org/project/CensusData/) package. I also used [this key created by John Keefe](https://johnkeefe.net/nyc-police-precinct-and-census-data) to link block-level census data to police precincts.
3. Used pandas for data processing, feature engineering, and exploratory data analysis.
4. Tested the performance of six different classification models on the data; creating two different models to better represent the structure of CCRB rulings. Random Forest performed best on both models.
5. Optimized, evaluated, and selected the best model, which was Random Forest for both models.
6. Explored feature importance.
7. Created some interactive visualizations in D3 using observablehq to help folks better more intuitively understand some of my findings.
    - [Sankey Diagram of complaints pipeline](https://observablehq.com/d/1d11da02acbb7e56)
    - [Bar Chart Race of total complaints per officer](https://observablehq.com/d/38930dd9f2e4dd7c)

## Findings and Conclusions
Ultimately I was able to predict whether the CCRB would rule that a given complaint occurred (ie there was enough evidence to confirm the complaint occurred) with 68% accuracy, and was able to predict whether (of the complaints that occurred) an officer's conduct was ruled to have violated rules with 78% accuracy. The goal of this project was inferential (this predictive model would not have much real world value), and indeed I'm glad to see that CCRB rulings could not be very accurately predicted with the features available; had that been the case, it may have suggested some kind of biases in this independent investigative body.

However exploratory data analysis and discovery of feature importance in my models did point to some trends that may be worth further investigeation.  And some of the most important takeaways from this dataset are descriptive: 
- Some officers have accumulated over 60  complaints from civilians in less than 20 years on the force. These officers are all still active NYPD members as of Summer 2020.
- Black and and Hispanic complainants are overrepresented compared to the overall NYC population.
- Complaints filed by Black and Hispanic folks were less likely to be ruled as officers violating rules. However this disparity did not hold when controlling for all other variables (see logistic regression analysis in [this notebook](https://github.com/labb0t/nypd-complaints/blob/main/logreg_racial_disparity_rulings.ipynb)) 

## Tools and Approaches Used
**Classification Algorithms and Metrics:**
- KNN
- Logistic Regression
- Random Forest
- Gaussian Naive Bayes
- SVC
- ROC-AUC curve
- Confusion matrix

**Other:**
- PostgreSQL
- Python
- Scikit-learn
- Statsmodel
- Pandas
- Matplotlib
- Seaborn
- D3 (via Observable)
- CensusData
