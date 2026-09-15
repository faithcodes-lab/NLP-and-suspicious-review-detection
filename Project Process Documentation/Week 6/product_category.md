### Product Category Identification for Highly Reviewed Products (issue #31)

In order to better understand the types of products generating the highest review activity, it was necessary to identify the product categories associated with the most reviewed ASINs. The review dataset itself does not contain category information, therefore additional methods were explored to retrieve product category metadata.

---

### Initial Approach: Amazon Product Advertising API

The first approach considered was the **Amazon Product Advertising API (Amazon Associates API)**, which allows developers to retrieve structured product information such as product title, price, category, and metadata directly from Amazon's database.

To access this API, an **Amazon Associates account** had to be created. This involved several steps:

1. Registering for the Amazon Associates Program.
2. Providing a website or platform where Amazon affiliate links would be used.
3. Creating a unique **Associate ID**.
4. Gaining access to the Associates dashboard.
5. Attempting to generate API credentials.

However, several limitations were encountered with this approach:

- Access to the Product Advertising API requires **approved API credentials**, which are typically granted only after an account generates qualifying affiliate sales.
- The API imposes **strict request limits** and authentication requirements.
- The setup process requires additional configuration including **Access Keys, Secret Keys, and request signing**, which significantly increases complexity.
- The API is designed primarily for **affiliate marketing applications**, not large-scale research or data analysis.

Due to these restrictions and the requirement for qualifying sales before full API access is granted, the API approach was deemed **impractical for the purposes of this analysis**.

---

### Alternative Approach: SNAP Amazon Metadata Dataset

The next approach explored was the **Amazon SNAP metadata dataset**, which is a publicly available research dataset containing product metadata including categories, titles, and additional product attributes.

This dataset is commonly used in academic research related to recommender systems and review analysis.

However, several challenges were encountered:

- The SNAP metadata dataset only covers **a subset of Amazon products**, while the review dataset contains over **15 million unique ASINs**.
- Matching between the review dataset and the metadata dataset resulted in **very low coverage**, with only a small fraction of products successfully matched.
- Many ASINs present in the review dataset did not exist in the metadata dataset.

As a result, this approach did not provide sufficient coverage for identifying the categories of the most reviewed products.

---

### Final Approach: Web Scraping Using BeautifulSoup

Due to the limitations of the previous methods, a third approach was implemented using **web scraping with Python's BeautifulSoup library**.

This method involved:

1. Identifying the **top 30 most reviewed products** based on the number of reviews.
2. Constructing the product page URL using the ASIN format:
https://www.amazon.com/dp/{ASIN}

3. Sending HTTP requests to retrieve the product page.
4. Parsing the HTML page content using **BeautifulSoup**.
5. Extracting the product category information from the **page title**.

---

### Challenges Encountered During Web Scraping

Several challenges were encountered during this process:

**1. Amazon Bot Detection**

Amazon frequently blocks automated requests to prevent scraping. When this occurs, the server returns a generic page instead of the actual product page, resulting in missing category information.

This was observed when the returned page title appeared as: Amazon.com
rather than a product-specific title.

---

**2. Inconsistent Page Structures**

Amazon product pages do not always follow the same HTML structure. Initially, category information was extracted using the page breadcrumb navigation. However, this method failed for many pages due to structural inconsistencies.

To resolve this, category information was instead extracted from the **page title**, which consistently contains the category in the format: Product Name : Amazon.com: Category

This method proved to be significantly more reliable.

---

**3. Request Rate Limitations**

Frequent requests triggered Amazon's anti-bot protections. To solve this, a delay was introduced between requests to reduce the likelihood of blocking.

---

### Final Outcome

Using the BeautifulSoup-based approach, category information was successfully retrieved for the majority of the top reviewed products. The extracted categories revealed that most of the highly reviewed items belong to media-related categories such as:

- **Books**
- **Audible Books & Originals**
- **Movies & TV**

A smaller number of products were also identified in categories such as:

- **Electronics**
- **Amazon Devices & Accessories**
- **Pet Supplies**

Some products returned missing category values due to request blocking or page inconsistencies. However, the extracted results still provide valuable insight into the dominant product categories represented among the most reviewed items in the dataset.

Therefore the remaining missing categories where manually searched on amazon.com and where inserted in the data ensuring a complete top 30 products 

This process ultimately allowed the integration of product category information into the analysis despite the limitations of the available data sources.


### Sources and Tools Referenced

Several external resources and tools were consulted during the development of the product category extraction process.

#### Amazon Product Advertising API
Initial attempts to retrieve product metadata were based on the Amazon Product Advertising API documentation, which describes how product information such as titles, categories, and pricing can be retrieved programmatically.

Reference:
- Amazon Product Advertising API Documentation  
https://webservices.amazon.com/paapi5/documentation/

However, due to access restrictions and the requirement for qualifying affiliate sales before full API access is granted, this approach could not be used in this project.

---

#### BeautifulSoup Web Scraping Library
The final implementation relied on the Python **BeautifulSoup** library to parse HTML content retrieved from Amazon product pages and extract category information from the page title.

Reference:
- BeautifulSoup Documentation  
https://www.crummy.com/software/BeautifulSoup/bs4/doc/

BeautifulSoup was used to analyze the HTML structure of product pages and retrieve relevant metadata fields when available.

---

#### Use of AI Assistance for Debugging and Research
During the development process, AI tools such as **ChatGPT 5.0** were used to assist with debugging code, exploring alternative data collection approaches, and identifying potential solutions to issues encountered when interacting with the Amazon platform.

AI assistance was primarily used for:
- debugging Python and DuckDB code
- researching alternative methods for retrieving product metadata
- understanding limitations of the Amazon Product Advertising API
- improving web scraping logic and error handling

These tools were used as a **supporting research and debugging aid**, while all final implementation, analysis decisions, and interpretation of results were performed independently.

Reference:
- OpenAI ChatGPT  
https://chat.openai.com/