import streamlit as st
import pandas as pd
from huggingface_hub import InferenceApi
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="AI-Powered Policy Impact Analyzer", page_icon="🤖", layout="wide")

# --- Hugging Face API Setup ---
api = InferenceApi(repo_id="distilbert-base-cased-distilled-squad")

# Function to Query Hugging Face API
def query_hf_api(question, context):
    payload = {"inputs": {"question": question, "context": context}}
    result = api(payload)
    return result.get("answer", "No answer found.")

# --- Load Data ---
@st.cache_data
def load_budget_data():
    file_path = "Combined_Financial_Data.csv"  # Budget Data file
    data = pd.read_csv(file_path)
    return data

@st.cache_data
def load_expenditure_data():
    file_path = "Expenditure.csv"  # Expenditure Data file
    data = pd.read_csv(file_path)
    return data

@st.cache_data
def load_income_tax_data():
    file_path = "Income Tax.csv"  # Income Tax Data file
    try:
        data = pd.read_csv(file_path, on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error loading Income Tax Data: {e}")
        data = pd.DataFrame()  # Return an empty DataFrame on error
    return data

# Load Data
budget_data = load_budget_data()
expenditure_data = load_expenditure_data()
income_tax_data = load_income_tax_data()

# --- App Layout ---
st.title("🤖 AI-Powered Policy Impact Analyzer")
st.write("""
Welcome to the **AI-Powered Policy Impact Analyzer**.  
Switch between datasets and explore insights dynamically.
""")
st.markdown("---")

# --- Dataset Type Dropdown ---
dataset_type = st.selectbox("Select Dataset Type", ["Budget Data", "Expenditure Data", "Income Tax Data"])

if dataset_type == "Budget Data":
    st.subheader("📋 Budget Data")
    # Source Dropdown for Budget Data
    budget_sources = budget_data["Source"].dropna().unique()  # Get unique sources
    selected_budget_source = st.selectbox("Select a Source to View Budget Data", options=budget_sources)
    
    # Filter and Display Budget Data
    filtered_budget_data = budget_data[budget_data["Source"] == selected_budget_source]
    st.dataframe(filtered_budget_data)
    
    # Visualization Section for Budget Data
    st.subheader(f"📊 Visualize the Budget Data for Source: {selected_budget_source}")
    chart_type = st.selectbox("Choose a chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
    x_axis = st.selectbox("Select X-axis", filtered_budget_data.columns)
    y_axis = st.selectbox("Select Y-axis", filtered_budget_data.columns)

    if chart_type == "Bar Chart":
        fig = px.bar(filtered_budget_data, x=x_axis, y=y_axis, title=f"Bar Chart for {selected_budget_source}")
        st.plotly_chart(fig)
    elif chart_type == "Line Chart":
        fig = px.line(filtered_budget_data, x=x_axis, y=y_axis, title=f"Line Chart for {selected_budget_source}")
        st.plotly_chart(fig)
    elif chart_type == "Pie Chart":
        fig = px.pie(filtered_budget_data, names=x_axis, values=y_axis, title=f"Pie Chart for {selected_budget_source}")
        st.plotly_chart(fig)

elif dataset_type == "Expenditure Data":
    st.subheader("📋 Expenditure Data")
    
    # Display column names
    st.write("**Columns in Expenditure Data:**")
    st.write(expenditure_data.columns)
    
    # Add an option to view the entire table
    item_groups = expenditure_data[expenditure_data.columns[0]].dropna().unique()
    dropdown_options = ["Show Full Table"] + list(item_groups)  # Add "Show Full Table" to the dropdown
    selected_item_group = st.selectbox("Select an Item Group (State/Region) to View Expenditure Data", options=dropdown_options)
    
    # Show filtered data or full table
    if selected_item_group == "Show Full Table":
        st.write("### Complete Expenditure Data")
        st.dataframe(expenditure_data)
    else:
        filtered_expenditure_data = expenditure_data[expenditure_data[expenditure_data.columns[0]] == selected_item_group]
        st.write(f"### Filtered Data for: {selected_item_group}")
        st.dataframe(filtered_expenditure_data)
    
    # Visualization Section
    st.subheader(f"📊 Visualize the Expenditure Data for: {selected_item_group if selected_item_group != 'Show Full Table' else 'Complete Data'}")
    chart_type = st.selectbox("Choose a chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
    x_axis = st.selectbox("Select X-axis", expenditure_data.columns)
    y_axis = st.selectbox("Select Y-axis", expenditure_data.columns)

    # Adjust visualization logic
    if selected_item_group == "Show Full Table":
        data_to_plot = expenditure_data
    else:
        data_to_plot = filtered_expenditure_data
    
    if chart_type == "Bar Chart":
        fig = px.bar(data_to_plot, x=x_axis, y=y_axis, title=f"Bar Chart for {selected_item_group}")
        st.plotly_chart(fig)
    elif chart_type == "Line Chart":
        fig = px.line(data_to_plot, x=x_axis, y=y_axis, title=f"Line Chart for {selected_item_group}")
        st.plotly_chart(fig)
    elif chart_type == "Pie Chart":
        fig = px.pie(data_to_plot, names=x_axis, values=y_axis, title=f"Pie Chart for {selected_item_group}")
        st.plotly_chart(fig)

elif dataset_type == "Income Tax Data":
    st.subheader("📉 Income Tax Data")
    
    # Display column names
    st.write("**Columns in Income Tax Data:**")
    st.write(income_tax_data.columns)
    
    # Add an option to view the entire table
    categories = income_tax_data[income_tax_data.columns[0]].dropna().unique()
    dropdown_options = ["Show Full Table"] + list(categories)  # Add "Show Full Table" to the dropdown
    selected_category = st.selectbox("Select a Category to View Income Tax Data", options=dropdown_options)
    
    # Show filtered data or full table
    if selected_category == "Show Full Table":
        st.write("### Complete Income Tax Data")
        st.dataframe(income_tax_data)
    else:
        filtered_income_tax_data = income_tax_data[income_tax_data[income_tax_data.columns[0]] == selected_category]
        st.write(f"### Filtered Data for: {selected_category}")
        st.dataframe(filtered_income_tax_data)
    
    # Visualization Section
    st.subheader(f"📊 Visualize the Income Tax Data for: {selected_category if selected_category != 'Show Full Table' else 'Complete Data'}")
    chart_type = st.selectbox("Choose a chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
    x_axis = st.selectbox("Select X-axis", income_tax_data.columns)
    y_axis = st.selectbox("Select Y-axis", income_tax_data.columns)

    # Adjust visualization logic
    if selected_category == "Show Full Table":
        data_to_plot = income_tax_data
    else:
        data_to_plot = filtered_income_tax_data
    
    if chart_type == "Bar Chart":
        fig = px.bar(data_to_plot, x=x_axis, y=y_axis, title=f"Bar Chart for {selected_category}")
        st.plotly_chart(fig)
    elif chart_type == "Line Chart":
        fig = px.line(data_to_plot, x=x_axis, y=y_axis, title=f"Line Chart for {selected_category}")
        st.plotly_chart(fig)
    elif chart_type == "Pie Chart":
        fig = px.pie(data_to_plot, names=x_axis, values=y_axis, title=f"Pie Chart for {selected_category}")
        st.plotly_chart(fig)

# --- Question & Prediction Section ---
st.markdown("---")
st.subheader("🤔 Ask Questions about the Data")
question = st.text_input("Enter your question about the data:")

if question:
    # Combine all data for context
    all_data = pd.concat([budget_data, expenditure_data, income_tax_data], ignore_index=True)
    context = all_data.to_string(index=False)  # Convert data to string context
    answer = query_hf_api(question, context)
    st.write(f"**Answer:** {answer}")

# Footer
st.markdown("---")
st.write("**Developed with ❤️ using Streamlit, Pandas, Plotly, and Hugging Face API**")
