import pandas as pd


def load_and_clean_data(df):
    """
    Load and clean the dataset.
    Parameters:
        df (pd.DataFrame): The Processed DataFrame for feature engineering.

    Returns:
        df_encoded (pd.DataFrame): The cleaned and encoded DataFrame ready
        for model training.
    """
    na_cols = [
        'Age', 'Session_Duration_Avg', 'Pages_Per_Session', 'Wishlist_Items',
        'Days_Since_Last_Purchase', 'Discount_Usage_Rate', 'Returns_Rate',
        'Email_Open_Rate', 'Customer_Service_Calls', 'Product_Reviews_Written',
        'Social_Media_Engagement_Score', 'Mobile_App_Usage',
        'Payment_Method_Diversity', 'Credit_Balance'
    ]

    for col in na_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 2. Encode Categoricals
    # Assuming Gender, Country, City, and Signup_Quarter are text/categorical
    cat_cols = ['Gender', 'Country', 'City', 'Signup_Quarter']
    existing_cat_cols = [col for col in cat_cols if col in df.columns]

    df_encoded = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

    # Ensure target variable is correctly named for downstream tasks
    if 'Churned' in df_encoded.columns:
        df_encoded.rename(columns={'Churned': 'Churn'}, inplace=True)

    return df_encoded
