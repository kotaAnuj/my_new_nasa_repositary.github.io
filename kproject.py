import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import timeit
from google.generativeai import GenerativeModel, configure

# Replace with your API key
GOOGLE_API_KEY = "AIzaSyA5HtRnzGruiia-aKtMMLnBjJ0ovTh11nE"

class Tools:
    def __init__(self):
        self.available_tools = {
            "plotly_scatter_plot": {
                "function": self.plotly_scatter_plot,
                "description": "Generates an interactive scatter plot using Plotly."
            },
            "plotly_line_chart": {
                "function": self.plotly_line_chart,
                "description": "Creates an interactive line chart using Plotly."
            },
            "plotly_bar_chart": {
                "function": self.plotly_bar_chart,
                "description": "Generates an interactive bar chart using Plotly."
            },
            "plotly_3d_surface": {
                "function": self.plotly_3d_surface,
                "description": "Generates a 3D surface plot using Plotly."
            },
            "plotly_pie_chart": {
                "function": self.plotly_pie_chart,
                "description": "Creates an interactive pie chart using Plotly."
            },
            "plotly_histogram": {
                "function": self.plotly_histogram,
                "description": "Generates a histogram to display frequency distributions using Plotly."
            },
            "plotly_3d_scatter": {
                "function": self.plotly_3d_scatter,
                "description": "Creates a 3D scatter plot with Plotly."
            },
            "plotly_animated_chart": {
                "function": self.plotly_animated_chart,
                "description": "Generates an animated plot in Plotly to visualize changes over time."
            },
            "matplotlib_scatter_plot": {
                "function": self.matplotlib_scatter_plot,
                "description": "Creates a scatter plot using Matplotlib."
            },
            "matplotlib_line_plot": {
                "function": self.matplotlib_line_plot,
                "description": "Generates a line plot using Matplotlib."
            },
            "matplotlib_bar_chart": {
                "function": self.matplotlib_bar_chart,
                "description": "Creates a bar chart using Matplotlib."
            },
            "matplotlib_histogram": {
                "function": self.matplotlib_histogram,
                "description": "Generates a histogram using Matplotlib."
            },
            "matplotlib_heatmap": {
                "function": self.matplotlib_heatmap,
                "description": "Creates a heatmap using Matplotlib."
            },
            # New validation and performance tools
            "performance_metrics": {
                "function": self.calculate_performance_metrics,
                "description": "Calculates various performance metrics for data analysis"
            },
            "confusion_matrix_plot": {
                "function": self.plot_confusion_matrix,
                "description": "Generates confusion matrix visualization"
            },
            "algorithm_timing": {
                "function": self.measure_algorithm_performance,
                "description": "Measures and compares algorithm execution times"
            },
            "statistical_validation": {
                "function": self.statistical_validation,
                "description": "Performs statistical validation tests"
            },
            "correlation_heatmap": {
                "function": self.plot_correlation_heatmap,
                "description": "Generates correlation heatmap for features"
            },
            "box_whisker_plot": {
                "function": self.plot_box_whisker,
                "description": "Creates box and whisker plots for distribution analysis"
            },
            "violin_plot": {
                "function": self.plot_violin,
                "description": "Generates violin plots for distribution comparison"
            },
            "dimensionality_reduction": {
                "function": self.plot_dimensionality_reduction,
                "description": "Performs and visualizes dimensionality reduction"
            }
        }

    # Image Generation Tool (Existing tool)
    def image_generation_tool(self, prompt):
        API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.content  # Return image bytes
                else:
                    return f"Error: {response.status_code}, {response.text}"
            except Exception as e:
                return f"Exception occurred: {str(e)}"

        image_bytes = query({"inputs": prompt})
        if isinstance(image_bytes, bytes):
            with open(f"generated_image_{prompt[:10]}.png", "wb") as f:
                f.write(image_bytes)
            return f"Image generated and saved as: generated_image_{prompt[:10]}.png"
        else:
            return image_bytes

    # ----------------- Plotly Tools ---------------------
    
    # Plotly Scatter Plot
    def plotly_scatter_plot(self, x, y):
        try:
            fig = px.scatter(x=x, y=y, title="Plotly Scatter Plot", labels={'x': 'X-axis', 'y': 'Y-axis'})
            st.plotly_chart(fig)
            return "Plotly scatter plot displayed successfully."
        except Exception as e:
            return f"Error in creating scatter plot: {str(e)}"

    # Plotly Line Chart
    def plotly_line_chart(self, x, y):
        try:
            fig = px.line(x=x, y=y, title="Plotly Line Chart", labels={'x': 'X-axis', 'y': 'Y-axis'})
            st.plotly_chart(fig)
            return "Plotly line chart displayed successfully."
        except Exception as e:
            return f"Error in creating line chart: {str(e)}"
    
    # Plotly Bar Chart
    def plotly_bar_chart(self, x, y):
        try:
            fig = px.bar(x=x, y=y, title="Plotly Bar Chart", labels={'x': 'Categories', 'y': 'Values'})
            st.plotly_chart(fig)
            return "Plotly bar chart displayed successfully."
        except Exception as e:
            return f"Error in creating bar chart: {str(e)}"
    
    # Plotly Pie Chart
    def plotly_pie_chart(self, values, labels):
        try:
            fig = px.pie(values=values, names=labels, title="Plotly Pie Chart")
            st.plotly_chart(fig)
            return "Plotly pie chart displayed successfully."
        except Exception as e:
            return f"Error in creating pie chart: {str(e)}"
    
    # Plotly Histogram
    def plotly_histogram(self, x, bins=10):
        try:
            fig = px.histogram(x=x, nbins=bins, title="Plotly Histogram", labels={'x': 'Values', 'y': 'Frequency'})
            st.plotly_chart(fig)
            return "Plotly histogram displayed successfully."
        except Exception as e:
            return f"Error in creating histogram: {str(e)}"
    
    # Plotly 3D Surface Plot
    def plotly_3d_surface(self, x, y, z):
        try:
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
            fig.update_layout(title="3D Surface Plot", scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis"))
            st.plotly_chart(fig)
            return "3D surface plot displayed successfully."
        except Exception as e:
            return f"Error in creating 3D surface plot: {str(e)}"
    
    # Plotly 3D Scatter Plot
    def plotly_3d_scatter(self, x, y, z):
        try:
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
            fig.update_layout(title="3D Scatter Plot", scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis"))
            st.plotly_chart(fig)
            return "3D scatter plot displayed successfully."
        except Exception as e:
            return f"Error in creating 3D scatter plot: {str(e)}"
    
    # Plotly Animated Chart
    def plotly_animated_chart(self, x, y, frame_data, animation_type="scatter"):
        try:
            if animation_type == "scatter":
                fig = px.scatter(x=x, y=y, animation_frame=frame_data, title="Animated Scatter Plot")
            elif animation_type == "line":
                fig = px.line(x=x, y=y, animation_frame=frame_data, title="Animated Line Plot")
            else:
                return f"Unsupported animation type: {animation_type}"

            st.plotly_chart(fig)
            return f"Plotly {animation_type} animated plot displayed successfully."
        except Exception as e:
            return f"Error in creating animated plot: {str(e)}"

    # ----------------- Matplotlib Tools ---------------------

    # Matplotlib Scatter Plot
    def matplotlib_scatter_plot(self, x, y):
        try:
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            ax.set_title("Matplotlib Scatter Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            st.pyplot(fig)
            return "Matplotlib scatter plot displayed successfully."
        except Exception as e:
            return f"Error in creating scatter plot: {str(e)}"

    # Matplotlib Line Plot
    def matplotlib_line_plot(self, x, y):
        try:
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title("Matplotlib Line Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            st.pyplot(fig)
            return "Matplotlib line plot displayed successfully."
        except Exception as e:
            return f"Error in creating line plot: {str(e)}"

    # Matplotlib Bar Chart
    def matplotlib_bar_chart(self, categories, values):
        try:
            fig, ax = plt.subplots()
            ax.bar(categories, values)
            ax.set_title("Matplotlib Bar Chart")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Values")
            st.pyplot(fig)
            return "Matplotlib bar chart displayed successfully."
        except Exception as e:
            return f"Error in creating bar chart: {str(e)}"

    # Matplotlib Histogram
    def matplotlib_histogram(self, data, bins=10):
        try:
            fig, ax = plt.subplots()
            ax.hist(data, bins=bins)
            ax.set_title("Matplotlib Histogram")
            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            return "Matplotlib histogram displayed successfully."
        except Exception as e:
            return f"Error in creating histogram: {str(e)}"

    # Matplotlib Heatmap
    def matplotlib_heatmap(self, data):
        try:
            fig, ax = plt.subplots()
            cax = ax.imshow(data, cmap='hot', interpolation='nearest')
            fig.colorbar(cax)
            ax.set_title("Matplotlib Heatmap")
            st.pyplot(fig)
            return "Matplotlib heatmap displayed successfully."
        except Exception as e:
            return f"Error in creating heatmap: {str(e)}"

    # New validation and performance methods
    def calculate_performance_metrics(self, y_true, y_pred):
        """Calculate various performance metrics."""
        try:
            metrics = {
                'Precision': precision_score(y_true, y_pred, average='weighted'),
                'Recall': recall_score(y_true, y_pred, average='weighted'),
                'F1 Score': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Create metrics visualization
            fig = go.Figure(data=[
                go.Bar(x=list(metrics.keys()), y=list(metrics.values()))
            ])
            fig.update_layout(title="Performance Metrics")
            st.plotly_chart(fig)
            
            return metrics
        except Exception as e:
            return f"Error in calculating performance metrics: {str(e)}"

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix using seaborn."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)
            return "Confusion matrix plotted successfully"
        except Exception as e:
            return f"Error in plotting confusion matrix: {str(e)}"

    def measure_algorithm_performance(self, algorithm, input_sizes):
        """Measure algorithm performance across different input sizes."""
        try:
            times = []
            for size in input_sizes:
                time_taken = timeit.timeit(lambda: algorithm(size), number=5) / 5
                times.append(time_taken)
            
            fig = px.line(x=input_sizes, y=times, 
                         title='Algorithm Performance',
                         labels={'x': 'Input Size', 'y': 'Execution Time (s)'})
            st.plotly_chart(fig)
            return times
        except Exception as e:
            return f"Error in measuring algorithm performance: {str(e)}"

    def statistical_validation(self, data1, data2):
        """Perform statistical validation tests."""
        try:
            from scipy import stats
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(data1, data2)
            
            # Create results visualization
            results = {
                't-statistic': t_stat,
                'p-value': p_value
            }
            
            fig = go.Figure(data=[
                go.Bar(x=list(results.keys()), y=list(results.values()))
            ])
            fig.update_layout(title="Statistical Test Results")
            st.plotly_chart(fig)
            
            return results
        except Exception as e:
            return f"Error in statistical validation: {str(e)}"

    def plot_correlation_heatmap(self, df):
        """Plot correlation heatmap."""
        try:
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Feature Correlation Heatmap')
            st.pyplot(fig)
            return "Correlation heatmap plotted successfully"
        except Exception as e:
            return f"Error in plotting correlation heatmap: {str(e)}"

    def plot_box_whisker(self, df, columns=None):
        """Create box and whisker plots."""
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
            
            fig = go.Figure()
            for col in columns:
                fig.add_trace(go.Box(y=df[col], name=col))
            
            fig.update_layout(title="Box and Whisker Plots")
            st.plotly_chart(fig)
            return "Box and whisker plots created successfully"
        except Exception as e:
            return f"Error in creating box and whisker plots: {str(e)}"

    def plot_violin(self, df, columns=None):
        """Create violin plots."""
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
            
            fig = go.Figure()
            for col in columns:
                fig.add_trace(go.Violin(y=df[col], name=col, box_visible=True))
            
            fig.update_layout(title="Violin Plots")
            st.plotly_chart(fig)
            return "Violin plots created successfully"
        except Exception as e:
            return f"Error in creating violin plots: {str(e)}"

    def plot_dimensionality_reduction(self, df):
        """Perform and visualize dimensionality reduction."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = StandardScaler().fit_transform(df[numeric_cols])
            
            # Perform PCA
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            
            # Create visualization
            fig = px.scatter(x=components[:, 0], y=components[:, 1],
                           title="PCA Visualization",
                           labels={'x': 'First Principal Component',
                                 'y': 'Second Principal Component'})
            st.plotly_chart(fig)
            
            # Show explained variance
            explained_variance = pca.explained_variance_ratio_
            st.write(f"Explained variance ratio: {explained_variance}")
            
            return "Dimensionality reduction completed successfully"
        except Exception as e:
            return f"Error in dimensionality reduction: {str(e)}"
    
def analyze_and_validate_data(df, experiment_type, tools):
    """Enhanced analysis and validation of the dataset."""
    try:
        # Generate insights using Gemini API
        prompt = f"Analyze this dataset from a {experiment_type} biological experiment and provide insights:\n{df.head(20).to_string()}"
        model = GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        st.write(f"Gemini AI Summary for {experiment_type} experiment:")
        st.write(response.text)

        # Data Validation Section
        st.write("Data Validation and Quality Metrics:")
        
        # Basic data quality checks
        st.write("1. Data Quality Summary:")
        st.write(f"- Missing values: {df.isnull().sum().sum()}")
        st.write(f"- Duplicate rows: {df.duplicated().sum()}")
        st.write(f"- Number of features: {df.shape[1]}")
        st.write(f"- Number of samples: {df.shape[0]}")

        # Correlation analysis
        st.write("\n2. Feature Correlation Analysis:")
        tools.plot_correlation_heatmap(df)

        # Distribution analysis
        st.write("\n3. Distribution Analysis:")
        tools.plot_box_whisker(df)
        tools.plot_violin(df)

        # Dimensionality reduction
        st.write("\n4. Dimensionality Reduction Analysis:")
        tools.plot_dimensionality_reduction(df)

        # Statistical comparison (if earth data is available)
        if experiment_type == "space" and "earth_experiment_data" in st.session_state:
            st.write("\n5. Statistical Comparison with Earth Data:")
            earth_df = st.session_state["earth_experiment_data"]
            
            # Perform statistical tests
            for col in df.select_dtypes(include=[np.number]).columns:
                st.write(f"\nAnalysis for {col}:")
                tools.statistical_validation(df[col], earth_df[col])

        # Original visualizations (from previous code)
        st.write("\nStandard Visualizations:")
        tools.use_tool("plotly_scatter_plot", [df.iloc[:, 0], df.iloc[:, 1]])
        tools.use_tool("plotly_line_chart", [df.iloc[:, 0], df.iloc[:, 1]])
        tools.use_tool("plotly_bar_chart", [df.columns, df.iloc[:, 0]])
        tools.use_tool("plotly_pie_chart", [df.iloc[:, 0].value_counts().values, 
                                          df.iloc[:, 0].value_counts().index])
        tools.use_tool("plotly_histogram", [df.iloc[:, 0]])
        
        # Advanced visualizations for space vs earth comparison
        if experiment_type == "space" and "earth_experiment_data" in st.session_state:
            st.write("\nAdvanced Space vs Earth Comparisons:")
            earth_df = st.session_state["earth_experiment_data"]
            
            # Create combined dataset for comparison
            combined_df = pd.concat([
                df.assign(experiment_type='Space'),
                earth_df.assign(experiment_type='Earth')
            ])
            
            # Enhanced comparison visualizations
            for col in df.select_dtypes(include=[np.number]).columns:
                fig = go.Figure()
                
                # Add violin plots
                fig.add_trace(go.Violin(x=['Space']*len(df), y=df[col],
                                      name='Space', side='negative',
                                      line_color='blue'))
                fig.add_trace(go.Violin(x=['Earth']*len(earth_df), y=earth_df[col],
                                      name='Earth', side='positive',
                                      line_color='green'))
                
                fig.update_layout(title=f'Distribution Comparison: {col}',
                                violingap=0, violinmode='overlay')
                st.plotly_chart(fig)

    except Exception as e:
        st.write(f"An error occurred during analysis: {e}")

# Streamlit app
def main():
    st.title("Enhanced Space vs. Earth Biological Data Visualization")
    st.write("""
    This application provides comprehensive analysis and visualization of biological 
    experiments conducted in space and on Earth, including statistical validation 
    and performance metrics.
    """)

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())

            # Check if API key is defined
            if not GOOGLE_API_KEY:
                st.write("Please set your GOOGLE_API_KEY environment variable.")
            else:
                # Configure the Gemini API
                configure(api_key=GOOGLE_API_KEY)

                # Determine the experiment type
                experiment_type = st.selectbox("Select the experiment type", 
                                             ["space", "earth"])

                # Save earth experiment data for comparison
                if experiment_type == "earth":
                    st.session_state["earth_experiment_data"] = df

                # Initialize the Tools class
                tools = Tools()

                # Analyze and validate the data
                analyze_and_validate_data(df, experiment_type, tools)

        except Exception as e:
            st.write(f"An error occurred while loading the file: {e}")

if __name__ == "__main__":
    main()