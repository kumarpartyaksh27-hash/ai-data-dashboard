import streamlit as st
import pandas as pd
from google import genai

# Page Setup
st.set_page_config(page_title="AI Data Analytics Dashboard", page_icon="📊", layout="wide")

st.title("📊 AI Data Analytics & Insights Dashboard")
st.write("Apni CSV data file upload karo, data ka preview dekho, aur AI se automatic insights paao!")

# Sidebar for API key
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
st.sidebar.markdown("[Get Free Gemini API Key](https://aistudio.google.com/app/apikey)")

# File Uploader
uploaded_file = st.file_uploader("📤 Upload your CSV Dataset", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Dataset Preview (First 5 Rows)")
    st.dataframe(df.head())
    
    st.subheader("📈 Dataset Summary Statistics")
    st.write(df.describe())
    
    # AI Insights Button
    if st.button("🤖 Generate AI Data Insights", use_container_width=True):
        if not api_key:
            st.warning("⚠️ Kripya sidebar me Gemini API Key dalein!")
        else:
            try:
                client = genai.Client(api_key=api_key)
                with st.spinner("Data analyze ho raha hai..."):
                    # Convert dataframe summary to text for Gemini prompt
                    buffer = df.describe().to_string()
                    columns_info = ", ".join(df.columns)
                    
                    prompt = f"""
                    You are an expert Data Analyst. Analyze the following dataset summary and give key business insights, trends, and recommendations.
                    
                    Dataset Columns: {columns_info}
                    Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}
                    
                    Summary Statistics:
                    {buffer}
                    
                    Provide a clean Markdown report with:
                    1. **Dataset Overview**: What this data represents based on columns.
                    2. **Key Trends & Patterns**: Important observations from the stats.
                    3. **Actionable Recommendations**: What decisions can be made using this data.
                    """
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    
                    st.subheader("💡 AI Data Insights & Report")
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")