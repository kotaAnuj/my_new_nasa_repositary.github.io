import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from google.generativeai import GenerativeModel, configure
import requests

# Replace with your API key
GOOGLE_API_KEY = "AIzaSyDpaOZq0jE6d4SdTpf1GyNk_lLkB75Kn_8"

class Tools:
    def __init__(self):
        self.available_tools = {
            "image_generation": {
                "function": self.image_generation_tool,
                "description": "Generates an image based on a given text prompt using Stable Diffusion API."
            },
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
            }
        }

    # Image Generation Tool
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

    # Plotly Tools
    def plotly_scatter_plot(self, x, y):
        try:
            fig = px.scatter(x=x, y=y, title="Plotly Scatter Plot", labels={'x': 'X-axis', 'y': 'Y-axis'})
            st.plotly_chart(fig)
            return "Plotly scatter plot displayed successfully."
        except Exception as e:
            return f"Error in creating scatter plot: {str(e)}"

    def plotly_line_chart(self, x, y):
        try:
            fig = px.line(x=x, y=y, title="Plotly Line Chart", labels={'x': 'X-axis', 'y': 'Y-axis'})
            st.plotly_chart(fig)
            return "Plotly line chart displayed successfully."
        except Exception as e:
            return f"Error in creating line chart: {str(e)}"
    
    def plotly_bar_chart(self, x, y):
        try:
            fig = px.bar(x=x, y=y, title="Plotly Bar Chart", labels={'x': 'Categories', 'y': 'Values'})
            st.plotly_chart(fig)
            return "Plotly bar chart displayed successfully."
        except Exception as e:
            return f"Error in creating bar chart: {str(e)}"
    
    def plotly_pie_chart(self, values, labels):
        try:
            fig = px.pie(values=values, names=labels, title="Plotly Pie Chart")
            st.plotly_chart(fig)
            return "Plotly pie chart displayed successfully."
        except Exception as e:
            return f"Error in creating pie chart: {str(e)}"
    
    def plotly_histogram(self, x, bins=10):
        try:
            fig = px.histogram(x=x, nbins=bins, title="Plotly Histogram", labels={'x': 'Values', 'y': 'Frequency'})
            st.plotly_chart(fig)
            return "Plotly histogram displayed successfully."
        except Exception as e:
            return f"Error in creating histogram: {str(e)}"
    
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

    # Matplotlib Tools
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

    def use_tool(self, tool_name, input_data):
        tool = self.available_tools.get(tool_name)
        if tool:
            return tool["function"](*input_data)
        else:
            return f"Tool {tool_name} is not available."

    def get_tool_description(self, tool_name):
        tool = self.available_tools.get(tool_name)
        if tool:
            return tool["description"]
        else:
            return f"No description available for tool {tool_name}."

def analyze_and_visualize_data(dfs, experiment_types, tools):
    """Analyzes and visualizes multiple datasets using Gemini API and various tools.
    Args:
      dfs: List of Pandas DataFrames.
      experiment_types: List of strings indicating the type of each experiment.
      tools: Instance of the Tools class.
    """
    try:
        # Generate insights using Gemini API for all datasets
        all_data_summary = "\n\n".join([f"Dataset {i+1} ({exp_type}):\n{df.head(20).to_string()}" for i, (df, exp_type) in enumerate(zip(dfs, experiment_types))])
        prompt = f"Analyze these datasets from biological experiments and provide insights, comparisons, and notable differences:\n{all_data_summary}"
        model = GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        st.write("Gemini AI Summary and Comparison:")
        st.write(response.text)

        # Visualizations for each dataset
        for i, (df, experiment_type) in enumerate(zip(dfs, experiment_types)):
            st.write(f"Data Visualizations for experiment {i+1} ({experiment_type}):")
            
            # Scatter Plot (Plotly)
            tools.use_tool("plotly_scatter_plot", [df.iloc[:, 0], df.iloc[:, 1]])

            # Line Chart (Plotly)
            tools.use_tool("plotly_line_chart", [df.iloc[:, 0], df.iloc[:, 1]])

            # Bar Chart (Plotly)
            tools.use_tool("plotly_bar_chart", [df.columns, df.iloc[:, 0]])

            # Pie Chart (Plotly)
            tools.use_tool("plotly_pie_chart", [df.iloc[:, 0].value_counts().values, df.iloc[:, 0].value_counts().index])

            # Histogram (Plotly)
            tools.use_tool("plotly_histogram", [df.iloc[:, 0]])

            # 3D Surface Plot (Plotly)
            tools.use_tool("plotly_3d_surface", [np.linspace(0, 10, 20), np.linspace(0, 10, 20), np.random.rand(20, 20)])

            # 3D Scatter Plot (Plotly)
            tools.use_tool("plotly_3d_scatter", [np.random.rand(50), np.random.rand(50), np.random.rand(50)])

            # Animated Scatter Plot (Plotly)
            tools.use_tool("plotly_animated_chart", [np.arange(10), np.random.rand(10), np.arange(10)])

            # Scatter Plot (Matplotlib)
            tools.use_tool("matplotlib_scatter_plot", [df.iloc[:, 0], df.iloc[:, 1]])

            # Line Plot (Matplotlib)
            tools.use_tool("matplotlib_line_plot", [df.iloc[:, 0], df.iloc[:, 1]])

            # Bar Chart (Matplotlib)
            tools.use_tool("matplotlib_bar_chart", [df.columns, df.iloc[:, 0]])

            # Histogram (Matplotlib)
            tools.use_tool("matplotlib_histogram", [df.iloc[:, 0]])

            # Heatmap (Matplotlib)
            tools.use_tool("matplotlib_heatmap", [np.random.rand(10, 10)])

        # Comparison Plots
        if len(dfs) > 1:
            st.write("Comparison Plots:")
            
            # Scatter Plot Comparison
            fig = go.Figure()
            for i, (df, exp_type) in enumerate(zip(dfs, experiment_types)):
                fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], mode='markers', name=f'{exp_type} {i+1}'))
            fig.update_layout(title="Scatter Plot Comparison", xaxis_title=dfs[0].columns[0], yaxis_title=dfs[0].columns[1])
            st.plotly_chart(fig)

            # Box Plot Comparison
            combined_df = pd.concat([df.assign(experiment=f"{exp_type} {i+1}") for i, (df, exp_type) in enumerate(zip(dfs, experiment_types))], ignore_index=True)
            fig = px.box(combined_df, x="experiment", y=combined_df.columns[1], title="Box Plot Comparison")
            st.plotly_chart(fig)

            # Correlation Heatmap
            correlation_matrices = [df.corr() for df in dfs]
            fig, axes = plt.subplots(1, len(dfs), figsize=(5*len(dfs), 4))
            for i, (corr_matrix, ax) in enumerate(zip(correlation_matrices, axes.flatten() if len(dfs) > 1 else [axes])):
                im = ax.imshow(corr_matrix, cmap='coolwarm')
                ax.set_title(f"{experiment_types[i]} {i+1}")
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr_matrix.columns)
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
            plt.tight_layout()
            st.pyplot(fig)

        # AI-driven comparison insights
        if len(dfs) > 1:
            comparison_prompt = f"Compare the datasets and provide insights on similarities, differences, and potential implications:\n{all_data_summary}"
            comparison_response = model.generate_content(comparison_prompt)
            st.write("AI-driven Comparison Insights:")
            st.write(comparison_response.text)

    except Exception as e:
        st.write(f"An error occurred: {e}")

# Streamlit app
def main():
    st.title("Multi-Experiment Biological Data Visualization and Comparison")

    # Upload multiple CSV files
    uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        try:
            dfs = [pd.read_csv(file) for file in uploaded_files]
            
            for i, df in enumerate(dfs):
                st.write(f"Data Preview for Dataset {i+1}:")
                st.dataframe(df.head())

            # Check if API key is defined
            if not GOOGLE_API_KEY:
                st.write("Please set your GOOGLE_API_KEY environment variable.")
            else:
                # Configure the Gemini API
                configure(api_key=GOOGLE_API_KEY)

                # Determine the experiment types
                experiment_types = [st.selectbox(f"Select the experiment type for Dataset {i+1}", ["space", "earth"]) for i in range(len(dfs))]

                # Initialize the Tools class
                tools = Tools()

                # Analyze and visualize the data
                if st.button("Analyze and Visualize Data"):
                    analyze_and_visualize_data(dfs, experiment_types, tools)

        except Exception as e:
            st.write(f"An error occurred while loading the files: {e}")

if __name__ == "__main__":
    main()
