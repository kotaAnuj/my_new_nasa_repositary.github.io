import sqlite3
import google.generativeai as genai
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os
import playsound

class SpaceBiologyAgent:
    def __init__(self, name, api_key):
        self.name = name
        self.role = "Space Biology Research Assistant"
        self.memory = sqlite3.connect('agent_memory.db')
        self.cursor = self.memory.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS interactions
                              (id INTEGER PRIMARY KEY, task TEXT, response TEXT)''')
        self.memory.commit()

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        self.recognizer = sr.Recognizer()

    def remember(self, task, response):
        self.cursor.execute("INSERT INTO interactions (task, response) VALUES (?, ?)",
                            (task, response))
        self.memory.commit()

    def recall(self, task):
        self.cursor.execute("SELECT response FROM interactions WHERE task = ?", (task,))
        return self.cursor.fetchone()

    def process_task(self, task):
        recalled = self.recall(task)
        if recalled:
            return recalled[0]
        
        prompt = f"As a {self.role} named {self.name}, respond to: {task}"
        response = self.model.generate_content(prompt).text
        self.remember(task, response)
        return response

    def visualize_data(self, data, chart_type):
        if chart_type == "scatter":
            fig = px.scatter(data, x=data.columns[0], y=data.columns[1],
                             title="Scatter Plot of Space Biology Data")
        elif chart_type == "line":
            fig = px.line(data, x=data.columns[0], y=data.columns[1],
                          title="Line Chart of Space Biology Data")
        elif chart_type == "bar":
            fig = px.bar(data, x=data.columns[0], y=data.columns[1],
                         title="Bar Chart of Space Biology Data")
        elif chart_type == "heatmap":
            fig = px.imshow(data.corr(), title="Correlation Heatmap of Space Biology Data")
        elif chart_type == "3d_scatter":
            fig = px.scatter_3d(data, x=data.columns[0], y=data.columns[1], z=data.columns[2],
                                title="3D Scatter Plot of Space Biology Data")
        elif chart_type == "box":
            fig = px.box(data, title="Box Plot of Space Biology Data")
        else:
            return "Unsupported chart type"
        
        return fig

    def compare_datasets(self, datasets, chart_type):
        fig = go.Figure()
        for i, data in enumerate(datasets):
            if chart_type == "scatter":
                fig.add_trace(go.Scatter(x=data[data.columns[0]], y=data[data.columns[1]],
                                         mode='markers', name=f'Dataset {i+1}'))
            elif chart_type == "line":
                fig.add_trace(go.Scatter(x=data[data.columns[0]], y=data[data.columns[1]],
                                         mode='lines', name=f'Dataset {i+1}'))
            elif chart_type == "bar":
                fig.add_trace(go.Bar(x=data[data.columns[0]], y=data[data.columns[1]],
                                     name=f'Dataset {i+1}'))
        fig.update_layout(title="Comparison of Space Biology Datasets")
        return fig

    def analyze_experiment(self, data):
        description = data.describe().to_string()
        prompt = f"As a {self.role}, analyze this space biology experiment data:\n{description}"
        analysis = self.model.generate_content(prompt).text
        return analysis

    def compare_experiments(self, data1, data2):
        desc1 = data1.describe().to_string()
        desc2 = data2.describe().to_string()
        prompt = f"As a {self.role}, compare these two space biology experiments:\nExperiment 1:\n{desc1}\n\nExperiment 2:\n{desc2}"
        comparison = self.model.generate_content(prompt).text
        return comparison

    def suggest_next_steps(self, current_findings):
        prompt = f"As a {self.role}, based on these current findings from a space biology experiment, suggest the next steps for research:\n{current_findings}"
        suggestions = self.model.generate_content(prompt).text
        return suggestions

    def explain_concept(self, concept):
        prompt = f"As a {self.role}, explain this space biology concept in simple terms: {concept}"
        explanation = self.model.generate_content(prompt).text
        return explanation

    def listen(self):
        with sr.Microphone() as source:
            st.write("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
        
        try:
            text = self.recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            st.write(f"Error with speech recognition: {e}")
            return None

    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)

def main():
    st.title("Space Biology AI Research Assistant")
    
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if not api_key:
        st.warning("Please enter your Google API Key to continue.")
        return

    agent = SpaceBiologyAgent("Astra", api_key)

    st.sidebar.title("Input Options")
    input_method = st.sidebar.radio("Choose input method:", ("Text", "Voice"))

    uploaded_files = st.file_uploader("Upload your space biology experiment data (CSV)", type="csv", accept_multiple_files=True)
    datasets = []
    if uploaded_files:
        for file in uploaded_files:
            data = pd.read_csv(file)
            datasets.append(data)
            st.write(f"Data Preview for {file.name}:")
            st.write(data.head())

        for i, data in enumerate(datasets):
            analysis = agent.analyze_experiment(data)
            st.write(f"AI Analysis for Dataset {i+1}:")
            st.write(analysis)

        if len(datasets) > 1:
            comparison = agent.compare_experiments(datasets[0], datasets[1])
            st.write("Comparison of Experiments:")
            st.write(comparison)

        chart_type = st.selectbox("Select visualization type:", 
                                  ["scatter", "line", "bar", "heatmap", "3d_scatter", "box"])
        
        if len(datasets) == 1:
            fig = agent.visualize_data(datasets[0], chart_type)
        else:
            fig = agent.compare_datasets(datasets, chart_type)
        
        st.plotly_chart(fig)

    st.write("Ask the AI assistant about space biology:")
    if input_method == "Text":
        user_question = st.text_input("Your question:")
        if user_question:
            response = agent.process_task(user_question)
            st.write("AI Response:")
            st.write(response)
            agent.speak(response)
    else:
        if st.button("Start Listening"):
            user_question = agent.listen()
            if user_question:
                response = agent.process_task(user_question)
                st.write("AI Response:")
                st.write(response)
                agent.speak(response)

    if st.button("Suggest Next Research Steps"):
        if datasets:
            suggestions = agent.suggest_next_steps(analysis)
            st.write("Suggested Next Steps:")
            st.write(suggestions)
            agent.speak(suggestions)
        else:
            st.write("Please upload and analyze data first.")

if __name__ == "__main__":
    main()