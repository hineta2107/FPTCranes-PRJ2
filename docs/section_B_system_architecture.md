# B - SYSTEM ARCHITECTURE & TECHNICAL REQUIREMENTS

## B.1. Hardware and Infrastructure Requirements
- **Processing Power (CPU):** A standard modern multi-core processor (e.g., Intel Core i3/i5 or AMD Ryzen equivalent) is fully sufficient. The training dataset is relatively small (approx. 1,500 rows), meaning that complex deep learning GPUs are not required.
- **Memory (RAM):** Minimum 4GB of RAM (8GB recommended) to comfortably load the dataset into memory using Pandas, execute feature engineering, and train the Random Forest ensemble models.
- **Storage:** Less than 500MB of free disk space is required to store the source code, the raw dataset, generated visualisations (PNGs), and the exported model artifacts (`.joblib` files).
- **Infrastructure:** The entire pipeline is designed for local execution (Local Machine). It does not require distributed computing clusters (like Apache Spark) or cloud-based GPU instances.

## B.2. Software Environment
- **Operating System:** Cross-platform compatibility. The project can be run on Windows, macOS, or Linux distributions.
- **Programming Language:** Python 3.10 or higher is strictly required due to the use of modern syntax features (e.g., type hinting, new dictionary methods) in the source code.
- **Environment Isolation:** It is highly recommended to run the project inside a virtual environment (using `venv`, `virtualenv`, or `conda`) to prevent dependency conflicts with other system-level Python packages.

## B.3. Technology Stack
The project relies on a robust, industry-standard Python data science stack:
- **Data Manipulation & Analysis:** `Pandas` and `NumPy` are used for loading the raw CSV files, handling missing values, filtering corrupted data, and performing fast numerical operations.
- **Machine Learning & Preprocessing:** `Scikit-learn` is the core modeling engine. It is used for feature encoding (OneHotEncoder, OrdinalEncoder), data splitting (TimeSeriesSplit), and constructing the `Pipeline` that trains various regression algorithms (Linear Regression, Ridge, Random Forest, Gradient Boosting).
- **Data Visualisation:** `Matplotlib` is utilized in the offline pipeline to generate static plots (e.g., correlation bar charts, actual vs. predicted scatter plots, feature importance charts) which are saved as PNG artifacts.
- **Web Application / Dashboard:** `Streamlit` powers the frontend, providing a reactive and interactive user interface for role-based authentication, viewing pipeline metrics, and submitting instant salary predictions.
- **Model Serialization:** `joblib` is used to efficiently serialize and save the trained pipeline and preprocessor to disk, allowing the Streamlit app to load the artifacts instantly without retraining.
- **Configuration & Utilities:** `PyYAML` is used for parsing project settings, and `pytest` handles automated testing to ensure pipeline constraints are met.
