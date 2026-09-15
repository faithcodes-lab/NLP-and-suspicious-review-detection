## Week 3 – Managing Large Dataset (100M+ Rows)  
*Period:* 10 February – 16 February 2026  (issue #13)


### Problem
- The new dataset contains approximately *233 million rows*, making it too large to process on standard laptops.
- Attempting to load the dataset locally caused crashes, memory errors, and slow performance.
- The team required a shared solution that allows all members to access and query the dataset efficiently using Python.


### Solutions Researched

*DuckDB + Parquet*
- Installed DuckDB using Homebrew.
- Converted the dataset from CSV (87GB) to Parquet (46GB) to improve query speed and compression.
- Loaded the dataset into DuckDB for structured SQL queries.
- Stored the database on an *external/network drive* due to local disk limitations.
- DuckDB enables querying large datasets without loading the entire file into memory.

Apache Spark
- Investigated Spark as a distributed data processing framework.
- Supports large scale datasets and parallel computation using PySpark.
- Can process data without loading everything into memory.
- Could run locally or through cloud environments such as Google Colab or shared virtual machines.

*MySQL (UWE Hosting)*
- Attempted to configure a MySQL database through UWE hosting.
- Account and credentials were successfully created.
- Installed and connected to the **UWE VPN (Ivanti Secure Access Client).
- Connection attempts through phpMyAdmin and MySQL Workbench failed.
- Issue likely due to restricted remote access or server configuration problems.

Cloud-Based Solutions Considered
- Google BigQuery – provides free credits but may become costly once credits expire.
- Cloud storage solutions such as AWS S3 or Google Drive for shared dataset access.
- MotherDuck (cloud version of DuckDB)** considered as a hybrid cloud solution.

---

### Recommended Workflow

- Convert the dataset into Parquet format to reduce storage size and improve analytics performance.
- Store the Parquet file in shared cloud storage or network drive
- Use DuckDB with Python to query only the required portions of the dataset.
- Execute analysis through shared notebooks (Jupyter or Google Colab) to avoid overloading local machines.


---

### Outcome
- The team agreed to use DuckDB with Python as the primary method for managing and querying the dataset.
- This solution met the project requirements for large scale data processing, collaboration, and cost efficiency.