import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

# --- 1. Data Loading and Model Setup (Cached for efficiency) ---

@st.cache_resource
def load_and_train_model():
    """Loads data, trains the preprocessor, and fits the NearestNeighbors model."""
    
    # Load the dataset
    df = pd.read_csv("cleaned_data_with_car_type.csv")
    
    # Define Features
    numerical_features = ['Price (in Lakhs)', 'Rating (out of 5)', 'Safety', 'Mileage', 'Seating Capacity']
    categorical_features = ['Car Type']
    
    X = df.drop(columns=['Car Name'])
    
    # Precompute numeric ranges for normalization used in Reason Generation
    num_ranges = {}
    for f in numerical_features:
        r = df[f].max() - df[f].min()
        num_ranges[f] = r if r != 0 else 1.0 # Avoid division by zero
    
    # Create Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create and Fit the KNN Pipeline
    knn_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('nn', NearestNeighbors(n_neighbors=6, algorithm='auto', metric='euclidean'))
    ])
    
    # Fit the pipeline
    knn_model.fit(X)
    
    return df, knn_model, numerical_features, categorical_features, num_ranges

# Load data and fit model once
df, knn_model, numerical_features, categorical_features, num_ranges = load_and_train_model()
Car_ID = df['Car Name'].values

# --- 2. Recommendation Function (Hybrid: Filter + KNN + Reasons) ---

def get_recommendations_with_reasons_and_range(df_full, min_price, max_price, user_input_df, model, 
                                              num_features, cat_features, ranges, top_k=5, top_reasons=3):
    """Finds the 5 most similar cars within a specified price range and provides reasons."""
    
    # A. Filter by Price Range
    price_col = 'Price (in Lakhs)'
    filtered_df = df_full[
        (df_full[price_col] >= min_price) & 
        (df_full[price_col] <= max_price)
    ].reset_index(drop=True)
    
    if filtered_df.empty:
        return []

    # B. Transform Filtered Data and User Input
    X_filtered = filtered_df.drop(columns=['Car Name'])
    Car_ID_filtered = filtered_df['Car Name'].values
    X_transformed_filtered = model.named_steps['preprocessor'].transform(X_filtered)
    user_input_transformed = model.named_steps['preprocessor'].transform(user_input_df)

    # C. Re-fit a temporary KNN model on the Filtered Data Subset
    num_neighbors = min(top_k + 1, len(X_transformed_filtered))
    nn_filtered = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto', metric='euclidean')
    nn_filtered.fit(X_transformed_filtered)
    
    distances, indices = nn_filtered.kneighbors(user_input_transformed)

    # D. Process Results and Generate Reasons
    all_indices = indices.flatten()
    all_distances = distances.flatten()
    
    # Skip the first neighbor if it's the query itself (distance near zero)
    start_index = 1 if all_distances[0] < 1e-6 and len(all_distances) > 1 else 0
    
    selected_indices = all_indices[start_index : start_index + top_k]
    selected_distances = all_distances[start_index : start_index + top_k]

    results = []
    user_row = user_input_df.iloc[0]

    for idx, dist in zip(selected_indices, selected_distances):
        # Index 'idx' is local to filtered_df
        neighbor_row = filtered_df.iloc[idx]
        
        feature_similarities = []

        # Numerical features: calculate similarity based on proximity
        for f in num_features:
            user_val = float(user_row[f])
            neighbor_val = float(neighbor_row[f])
            diff = abs(user_val - neighbor_val)
            normalized = diff / ranges[f]
            sim = max(0.0, 1.0 - normalized) # Similarity clipped at 0
            feature_similarities.append({
                'detail': f"**{f}**: User wanted {user_val}, Car has {neighbor_val}",
                'similarity': sim
            })

        # Categorical features: exact match
        for f in cat_features:
            user_val = user_row[f]
            neighbor_val = neighbor_row[f]
            sim = 1.0 if (str(user_val).strip().lower() == str(neighbor_val).strip().lower()) else 0.0
            feature_similarities.append({
                'detail': f"**{f}**: User wanted '{user_val}', Car is '{neighbor_val}'",
                'similarity': sim
            })

        # Sort features by descending similarity
        feature_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_reasons_list = [f['detail'] for f in feature_similarities[:top_reasons]]

        results.append({
            'car_name': Car_ID_filtered[idx],
            'distance': float(dist),
            'reasons': top_reasons_list
        })

    return results

# --- 3. Streamlit Application Layout ---

st.title("🚗 KNN Car Recommendation System with Reasons")
st.markdown("Enter your preferences to find the most similar cars within your budget.")

st.sidebar.header("User Preferences")

# Price Range Input
min_price = st.sidebar.number_input("Minimum Price (in Lakhs)", min_value=df['Price (in Lakhs)'].min(), max_value=df['Price (in Lakhs)'].max(), value=7.0, step=0.5)
max_price = st.sidebar.number_input("Maximum Price (in Lakhs)", min_value=df['Price (in Lakhs)'].min(), max_value=df['Price (in Lakhs)'].max(), value=15.0, step=0.5)

# Similarity Point and Feature Inputs
st.sidebar.markdown("---")
st.sidebar.subheader("Similarity Preferences")

ideal_price = st.sidebar.number_input("Ideal Price Point (in Lakhs)", 
                                      min_value=min_price, 
                                      max_value=max_price, 
                                      value=(min_price + max_price) / 2 if min_price <= max_price else min_price)

car_types = df['Car Type'].unique().tolist()
car_type_input = st.sidebar.selectbox("Car Type Preference", car_types)
seating_input = st.sidebar.slider("Seating Capacity", min_value=4, max_value=8, value=5)
safety_input = st.sidebar.slider("Safety Rating (1-5)", min_value=1, max_value=5, value=5)
rating_input = st.sidebar.slider("Customer Rating (out of 5)", min_value=3.0, max_value=5.0, value=4.5, step=0.1)
mileage_input = st.sidebar.slider("Mileage (in kmpl)", min_value=10.0, max_value=30.0, value=20.0, step=0.5)


if st.sidebar.button("Find Recommendations"):
    if min_price > max_price:
        st.error("Minimum Price must be less than or equal to Maximum Price.")
    else:
        # Create user input DataFrame for the similarity search
        user_preferences = pd.DataFrame([{
            'Price (in Lakhs)': ideal_price, 
            'Rating (out of 5)': rating_input,
            'Safety': safety_input,
            'Mileage': mileage_input,
            'Seating Capacity': seating_input,
            'Car Type': car_type_input
        }])

        with st.spinner("Finding similar cars and generating reasons..."):
            recommendation_objects = get_recommendations_with_reasons_and_range(
                df, min_price, max_price, user_preferences, knn_model, 
                numerical_features, categorical_features, num_ranges, top_k=5, top_reasons=3
            )

        st.subheader(f"Top Recommended Cars within {min_price:.2f} - {max_price:.2f} Lakhs")
        
        if not recommendation_objects:
             st.warning("No cars found within this price range.")
        else:
            st.success(f"Found {len(recommendation_objects)} Recommendations!")
            
            for i, rec in enumerate(recommendation_objects):
                st.markdown(f"### **{i+1}. {rec['car_name']}**")
                st.markdown(f"**Similarity Score (Distance):** {rec['distance']:.4f}")
                st.write("Top reasons for similarity:")
                for reason in rec['reasons']:
                    st.write(f"&nbsp; &nbsp; • {reason}")
                st.markdown("---")