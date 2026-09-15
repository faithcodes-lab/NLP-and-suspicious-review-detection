# Exploratory Data Analysis (issue #20 )

Exploratory Data Analysis (EDA) is an essential step in understanding and preparing the dataset for sentiment analysis and suspicious review detection in Natural Language Processing (NLP). Although the dataset consists of text data and categorical variables, EDA can still provide valuable insights. Following are the basic steps of EDA that will taken in this project.

1. Data Collection: Gather the dataset and examine its structure, including column names and data types.

2. Data Exploration: Explore the data in its raw form wich helps inform the cleaning process, feature engineering for suspicous review detection.

3. Data Cleaning: Perform data cleaning tasks specific to text data. This includes removing any irrelevant or redundant information, handling missing values, and dealing with inconsistencies in the text.

4. Descriptive Statistics: While the dataset may not have numerical variables, we can still calculate descriptive statistics for the text data. This includes determining the length of the text, identifying the most frequent words, or analyzing the distribution of the text length.

5. Data Visualization: Use visualizations to gain insights into the text data. For example, word clouds can be created to visualize the most commonly occurring words in both real and suspicious reviews. Bar plots or pie charts can also illustrate the distribution of different categories or labels in the dataset.

6. Data Distribution: Explore the distribution of categorical variables such as the "verified purchase" column. This will help understand the proportion of verified  and un verified purchase in the dataset and determine if there is a class imbalance.

7. Feature Engineering: Explore potential features that can be derived from the existing data to improve the performance of fake news detection models. This may involve creating new features such as sentiment scores, readability metrics, or linguistic features based on the text data.


8. Data Preparation: Prepare the data for further analysis and modeling tasks. This involves techniques such as text preprocessing, including tokenization, stop-word removal, and stemming/lemmatization. Additionally, encoding categorical variables and splitting the dataset into training and testing sets are crucial steps.

By conducting EDA, we can gain a deeper understanding of the dataset, uncover patterns and characteristics specific to fake news, and prepare the data for subsequent modeling and analysis. These insights will aid in developing robust and accurate fake news detection models.


### Step by step approach

After dataset collection, we encounterd a big data issue whereby we have a data that contained 233 million rows
- how do we handle a dataset this size?
- how do we update a dataset this along the project timeline if there is any need?
- how do we share this data set among the team memebers?
- how do we analyze this dataset without crashing the notebook we will be using or our various laptops?
- This is a university group project and we are tight on incurring cost
- what are cost effective ways we go around with this project iven the project deadline too?

We researched on different method

we concluded on making use of DuckDB and Python.

Exploring the raw dataset, we have uncovered the following:

## Duplicate Review Anomaly Investigation

During the exploratory data analysis (EDA), a detailed inspection of reviewer activity revealed the presence of duplicated review entries in the dataset. The following steps outline how this anomaly was identified and evaluated.

1. Inspecting Reviewer Activity
To better understand reviewer behavior, the dataset was grouped by asin (product ID), category and reviewerID. This made it possible to observe how many reviews each user posted for the same product.

This step revealed that some reviewers appeared multiple times for the same product, which is unusual because Amazon typically allows a user to post only one review per product.

2. Examining Individual Review Records
To investigate further, individual review records were manually inspected. By filtering the dataset for a specific reviewer and product combination, several rows were found to contain nearly identical values across all fields.

For example, the following attributes were identical across multiple rows:
- asin
- reviewerID
- category
- reviewText
- overall rating
- reviewTime
- unixReviewTime
- verified status

In some cases, the only slight variation observed was in the vote count or minor changes in the summary field.

3. Comparing Review Text
A closer comparison of the reviewText column confirmed that several entries contained exactly the same content. In other instances, the review text was identical while the summary contained minor wording differences such as punctuation changes (e.g., “Huge Chunks of Mold!!” vs. “Huge Mold!!! vs Toxic Mold in Bag”).

This indicated that the actual review content had been duplicated.

## Example Case

For product ASIN: B000W5QSYA, reviewer A2OY938ZNE1GTB (“Susan”) appeared 12 times, while only 6 unique review texts were present.

This indicates that the duplication is not necessarily the result of abnormal reviewer behaviour but rather artifacts introduced during data collection or processing.
 
Possible Causes of Duplicate Reviews

## a. Data Scraping Duplicates

Some rows were identical across all columns, including:
- reviewText
- summary
- unixReviewTime
- reviewerID
- vote

Example patterns observed:
- Rows 0 and 1
- Rows 4 and 5
- Rows 7 and 10

These rows represent the exact same review repeated multiple times.

Likely Cause

During large-scale data scraping, the same review may be collected multiple times if:
- the product appears in multiple category listings
- the crawler revisits the same review page
- deduplication was not applied during dataset creation

As a result, identical rows are stored multiple times in the dataset.


## b. Review Updates Captured as Separate Entries

Some rows share the same reviewer and timestamp but contain slightly different text.

Example:

|reviewerID    |unixReviewTime |summary |reviewText|
|---|---|---|---|
|A2OY938ZNE1GTB| 1434758400 |Five Stars|always buy this kink of dog food and my dog seems to prefer it over other kinds.|
|A2OY938ZNE1GTB| 1434758400|…taste of the wild…|always buy taste of the wild and my dog seems to like it over other brands so will continue to buy it . mix it with canned food and she enjoys it.|


4. Identifying Repetition Patterns:
The duplication pattern was further analyzed across multiple reviewers. Similar repetitions were observed for different users and products, where the same review appeared multiple times with identical timestamps and content.

The presence of these repetitions across multiple reviewers suggested that the issue was not isolated to a single user.

5. Evaluating Whether the Behavior Was Genuine
Several factors indicated that the duplication was unlikely to be the result of user behavior:
- The same reviewer posted identical reviews for the same product at the exact same timestamp.
- The review text was repeated verbatim across multiple rows.
- The same pattern appeared across multiple reviewers and products.
- Amazon normally prevents users from submitting identical reviews for the same product multiple times.

6. Conclusion
Based on these observations, the duplicated reviews are most likely a dataset issue introduced during the data collection or scraping process, rather than intentional reviewer behavior.

Possible causes include:
- multiple scraping passes capturing the same review
- updates to review vote counts being recorded as new entries
- review edits captured as separate records

To prevent these duplicates from biasing further analysis, duplicate reviews were identified using combinations of asin, category, reviewerID, reviewText, and unixReviewTime. This allowed the dataset to be cleaned before proceeding with feature engineering and suspicious review detection.

Recognizing and addressing such anomalies is an important step in ensuring the reliability of subsequent analyses and machine learning models.