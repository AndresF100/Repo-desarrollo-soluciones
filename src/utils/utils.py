import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Plotting and visualization
import seaborn as sns  # Advanced plotting and visualization
from scipy import stats  # Statistical functions

def load_and_examine_data(file_path: str) -> pd.DataFrame:
    """
    Loads an Excel file into a pandas DataFrame and performs
    initial examination, printing basic dataset information.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    # Read the Excel file, setting the first column as index
    df = pd.read_excel(file_path, index_col=0)

    # Print basic dataset information
    print("\n=== Basic Dataset Information ===")
    print("\nShape of dataset:", df.shape)
    print("\nColumns in dataset:", df.columns.tolist())
    print("\nData types of columns:")
    print(df.dtypes)

    return df

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes missing values in a DataFrame, calculating the count
    and percentage of missing values for each column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: DataFrame containing missing values count and
                      percentage for columns with missing data.
    """
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100

    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentages
    })

    print("\n=== Missing Values Analysis ===")
    missing_info_filtered = missing_info[
        missing_info['Missing Values'] > 0
    ]
    print(missing_info_filtered)

    return missing_info_filtered

def analyze_numerical_columns(df: pd.DataFrame) -> list[str]:
    """
    Performs statistical analysis on numerical columns of a DataFrame,
    printing descriptive statistics, skewness, and kurtosis.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        list[str]: A list of numerical column names.
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    print("\n=== Numerical Columns Analysis ===")
    print("\nDescriptive Statistics:")
    print(df[numerical_cols].describe())

    # Check for skewness and kurtosis
    print("\nSkewness:")
    print(df[numerical_cols].skew())
    print("\nKurtosis:")
    print(df[numerical_cols].kurtosis())

    return numerical_cols.tolist()

def analyze_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Analyzes categorical columns of a DataFrame, printing value counts
    and the number of unique values for each categorical column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        list[str]: A list of categorical column names.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns

    print("\n=== Categorical Columns Analysis ===")
    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts().head())
        print(f"Number of unique values: {df[col].nunique()}")

    return categorical_cols.tolist()

def plot_distributions(df: pd.DataFrame, numerical_cols: list[str]) -> None:
    """
    Creates and displays distribution plots (histograms and box plots)
    for each numerical column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        numerical_cols (list[str]): List of numerical column names
                                     to plot distributions for.
    """
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels

        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')

        plt.tight_layout()
        plt.show()

def correlation_analysis(df: pd.DataFrame, numerical_cols: list[str]) -> None:
    """
    Performs correlation analysis on specified numerical columns
    of a DataFrame and displays a heatmap of the correlation matrix.

    Args:
        df (pd.DataFrame): The DataFrame.
        numerical_cols (list[str]): List of numerical column names
                                     to include in the correlation analysis.
    """
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, annot=True, cmap='coolwarm', center=0
        )
        plt.title('Correlation Matrix')
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels
        plt.yticks(rotation=0) # Keep y-axis labels horizontal
        plt.tight_layout()
        plt.show()

def perform_eda(file_path: str) -> pd.DataFrame:
    """
    Main function to perform Exploratory Data Analysis (EDA) on
    an Excel file. It loads data, checks for missing values,
    analyzes numerical and categorical columns, and generates
    distribution plots and a correlation matrix.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame after performing EDA steps.
    """
    # Load and examine data
    df = load_and_examine_data(file_path)

    # Check missing values
    check_missing_values(df) # missing_info is not returned anymore

    # Analyze numerical and categorical columns
    numerical_cols = analyze_numerical_columns(df)
    categorical_cols = analyze_categorical_columns(df)

    # Create visualizations
    plot_distributions(df, numerical_cols)
    correlation_analysis(df, numerical_cols)

    return df

def plot_correlation_matrix(
    df: pd.DataFrame, min_correlation: float = 0.3
) -> plt.Figure:
    """
    Generates an improved and more readable correlation matrix heatmap,
    excluding the index column and filtering correlations by a minimum
    absolute value. Returns the figure object.

    Args:
        df (pd.DataFrame): DataFrame with the data.
        min_correlation (float, optional): Minimum correlation value
                                           to display (absolute value).
                                           Defaults to 0.3.

    Returns:
        plt.Figure: The matplotlib Figure object of the heatmap,
                    or None if no significant correlations are found.
    """
    # Select numerical columns, excluding 'Unnamed: 0' if present
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Unnamed: 0' in numeric_cols:
        numeric_cols.remove('Unnamed: 0')

    # Calculate the correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Filter for significant correlations
    corr_matrix_filtered = corr_matrix.copy()
    corr_matrix_filtered[abs(corr_matrix) < min_correlation] = 0

    # Remove rows and cols without significant correlations
    cols_to_keep = (abs(corr_matrix_filtered) >= min_correlation).any()
    corr_matrix_filtered = corr_matrix_filtered.loc[
        cols_to_keep, cols_to_keep
    ]

    # Return None if no significant correlations found
    if corr_matrix_filtered.empty:
        print(
            f"No correlations found with absolute value >= {min_correlation}"
        )
        return None

    # Create the visualization
    fig, ax = plt.subplots(
        figsize=(
            max(15, len(corr_matrix_filtered.columns) * 0.8),
            max(10, len(corr_matrix_filtered.index) * 0.5),
        )
    )

    # Generate heatmap with improved parameters
    sns.heatmap(
        corr_matrix_filtered,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 10},
        cbar_kws={'shrink': .8} # Shrink colorbar to fit better
    )

    # Adjust labels and title
    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(
        'Matriz de Correlación\n(Correlaciones ≥ {})'.format(min_correlation),
        pad=20,
        fontsize=14,
    )

    plt.tight_layout()
    return fig  # Return the figure

def analyze_correlations(df: pd.DataFrame) -> None:
    """
    Performs correlation analysis and displays a correlation matrix
    heatmap for correlations greater than or equal to 0.3
    (in absolute value), excluding the index column.
    Displays only the heatmap image.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    print("Note: 'Unnamed: 0' column excluded as it is just a counter.")

    # Correlation matrix with threshold of 0.3
    print("\n=== Correlation Matrix (Correlations ≥ 0.3) ===")
    correlation_fig = plot_correlation_matrix(df, min_correlation=0.3)
    if correlation_fig is not None:
        plt.show() # Show heatmap, but don't print matrix