# Implementing Machine Learning Model to Predict and Select Loan

Executing an informed-decision by gathering and evaluation some of relevant information must help us to reduce the risk of regret, but sometimes is highly time-consuming. In this occasion, I will demonstrate an acitivity to develop a machine learning model for solving classification problem, that may serves as *decision support system*. I will utilize LendingClub Loan data as an example in this analysis.

### **What's Inside**

This analysis is divided into 3 parts (see below). First part is focusing on inspecting and preprocessing the dataset without performing any analysis. The second part is the exploratory data analytics, to gather valuable insight that may helps to understand the dataset and building a good model. The last part is focusing on building an *XGboost* classification model, from preparing the features up to hyperparameter tuning.

### **The output**

We will implement the model to build a simple algorithm that decide which loan should we invested on, this is to demonstrate utilizing the model as some sort of informed-decision. The algorithm involves some optimizing technique, which will be performed using *Pyomo* as the API to instantiate the solver. Of course, we want to discover if the model may helps us at all, so we will demonstrate several kind of model as a measure and compare them!.

P.S. This analysis is rather a simplistic approach and solely to demonstrate the possible advantage of implementing machine learning model, hence doesn't encourage anyone to implement this kind of analysis outside of this. Investment should involve expert in their area.

Check out the content pages bundled with this book to see more.

```{tableofcontents}
```