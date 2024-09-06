import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Step 1: Load the Data
df = pd.read_parquet('metadata.parquet', engine="pyarrow")

# Display column names and sample data
print("Column names:", df.columns)
print("Sample data:\n", df.head())

# Step 2: Filter the Data
df['ModelYear'] = df['ModelYear'].astype(int)
models_to_include = ['F-150', 'Mustang', 'Bronco', 'Bronco Sport']
filtered_df = df[(df['Brand'] == 'Ford') & (df['Model'].isin(models_to_include)) & (df['ModelYear'] == 2023)]

# Display the filtered data
print("Filtered data for 2023:\n", filtered_df)

# Step 3: Create Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
filtered_df['embedding'] = filtered_df['caption'].apply(lambda x: model.encode(x))

# Display the DataFrame with embeddings
print("Data with embeddings:\n", filtered_df.head())

# Step 4: Perform the Search Query
def search_query(query, df, model, top_k=5):
    query_embedding = model.encode(query)
    df['similarity'] = df['embedding'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())
    top_results = df.sort_values(by='similarity', ascending=False).head(top_k)
    return top_results[['file_name', 'source_path', 'caption', 'similarity']]

# Define your search query
query = "Ford mustang black color close up shot"

# Perform the search query
results = search_query(query, filtered_df, model)

# Display the top results
print(f"Top results for the query '{query}':")
for index, row in results.iterrows():
    print(f"Image: {row['file_name']}, Similarity: {row['similarity']:.4f}, Path: {row['source_path']}")
    print(f"Description: {row['caption']}\n")

# Save the embeddings to a file (optional)
filtered_df.drop(columns=['embedding']).to_parquet('embedding_metadata.parquet')


# import pandas as pd

# # Load the Parquet file
# df = pd.read_parquet('metadata.parquet', engine='pyarrow')

# # Display the structure of the DataFrame
# print("DataFrame Structure:")
# print(df.info())

# # Display the first few rows of the DataFrame
# print("Sample Data:")
# print(df.head())

# # Filter the DataFrame for rows containing the keyword "bluecruise"
# keyword = "BlueCruise"
# filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)]

# # Display the filtered rows
# print(f"Rows containing the keyword '{keyword}':")
# print(filtered_df)



# import pandas as pd

# # Load the Parquet file
# df = pd.read_parquet('metadata.parquet', engine='pyarrow')


# # Extract unique values for 'Model' and 'ModelYear' columns
# unique_models = df['Model'].unique()
# unique_model_years = df['ModelYear'].unique()

# # Display the unique values
# print("Unique Models:")
# print(unique_models)

# print("Unique Model Years:")
# print(unique_model_years)


