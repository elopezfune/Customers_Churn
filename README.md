# Customers Churn

Customers churn refers to the amount of customers of a given company that stop using products or services during a certain time frame. One can calculate the churn rate by dividing the number of customers lost during that time period -- say a quarter -- by the number of existing customers at the beginning of that time period. For example, starting the quarter with 400 customers and ending with 380, the churn rate is 5% because 5% of your customers dropped off.

For Business Intelligence, this is one of the most important metrics to look at since loosing clients now-a-days is very easy, compared to retain the existing ones. Companies should aim for a churn rate that is as close to 0% as possible. In order to do this, the company has to be on top of its churn rate at all times and treat it as a top priority.

Three Ways to Reduce Customer Churn:

1. Focus the attention on the best customers. 
Rather than simply focusing on offering incentives to customers who are considering churning, it could be even more beneficial to pool the resources into the loyal, profitable customers.

2. Analyze churn as it occurs. 
Use the churned customers as a means of understanding why customers are leaving. Analyze how and when churn occurs in a customer's lifetime with the company, and use that data to put into place preemptive measures.

3. Show the customers that you care. 
Instead of waiting to connect with the customers until they reach out to you, try a more proactive approach. Communicate with them all the perks you offer and show them you care about their experience, and they'll be sure to stick around.

In this project, I will use several tools from Survival Analysis to focus on a customer retention program from the Telco company (https://www.telco.com/company-profile). Each row represents a customer, each column contains customer's attributes described on the column Metadata.

The data set includes information about:

    CustomerID: A unique ID that identifies each customer.
    Gender: The customer’s gender: Male, Female
    Age: The customer’s current age, in years, at the time the fiscal quarter ended.
    Senior Citizen: Indicates if the customer is 65 or older: Yes, No
    Married (Partner): Indicates if the customer is married: Yes, No
    Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
    Number of Dependents: Indicates the number of dependents that live with the customer.
    Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No
    Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
    Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
    Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
    Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
    Device Protection Plan: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
    Premium Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
    Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
    Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
    Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
    Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No
    Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
    Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.
    Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
    Tenure: Indicates the total amount of months that the customer has been with the company.
    Churn: Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.

After loading the training data and preprocessing it, I produced a predictive model using the Conditional Survival Forest model to spot the main risks of churning and make individual predictions. The Machine Learning Pipeline is stored in the directory "source", where there is a Jupyter notebook with the explanations of the model and data analytics.