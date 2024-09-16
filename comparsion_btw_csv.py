import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score

class DataComparator:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
        self.df1 = None
        self.df2 = None
        self.common_columns = None
        self.accuracy_scores = {}
        self.absent_data = {}
        self.present_data = {}

    def load_data(self):
        """Loads the CSV files into pandas DataFrames"""
        self.df1 = pd.read_csv(self.file1, encoding='ISO-8859-1')
        self.df2 = pd.read_csv(self.file2, encoding='ISO-8859-1')

    def preprocess_columns(self):
        """Preprocess column names by standardizing them (removing spaces, making lowercase)"""
        self.df1.columns = self.df1.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("'", '')
        self.df2.columns = self.df2.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("'", '')

    def find_common_columns(self):
        """Finds the columns that are common between the two CSV files"""
        self.common_columns = [col for col in self.df1.columns if col in self.df2.columns]

    def calculate_absent_and_present(self):
        """Calculates the absent (missing) and present values for file1"""
        for col in self.common_columns:
            total_values = len(self.df1[col])
            absent_values = self.df1[col].isna().sum()
            present_values =total_values - absent_values
            
            self.absent_data[col] = absent_values
            self.present_data[col] = present_values

    def calculate_accuracy(self):
        """Calculates the accuracy of matching values between the two files for each common column"""
        for col in self.common_columns:
            col_data1 = self.df1[col].fillna('missing').astype(str).values
            col_data2 = self.df2[col].fillna('missing').astype(str).values

            # Ensure the lengths are the same for comparison
            min_length = min(len(col_data1), len(col_data2))
            col_data1, col_data2 = col_data1[:min_length], col_data2[:min_length]

            accuracy = accuracy_score(col_data1, col_data2)
            self.accuracy_scores[col] = accuracy * 100  # Convert accuracy to percentage

    def plot_histogram(self):
        """Generates a histogram of absent, present values, and accuracy"""
        names = list(self.common_columns)
        absent_values = [self.absent_data[col] for col in self.common_columns]
        present_values = [self.present_data[col] for col in self.common_columns]
        accuracy_values = [self.accuracy_scores[col] for col in self.common_columns]

        fig = go.Figure()

        # Histogram for Absent Values
        fig.add_trace(go.Bar(
            x=names,
            y=absent_values,
            name='Absent Values',
            marker_color='red',
            text=absent_values,  
            textposition='outside',
            textfont=dict(size=12, color='black', family='Arial', weight='bold')
        ))

        # Histogram for Present Values
        fig.add_trace(go.Bar(
            x=names,
            y=present_values,
            name='Present Values',
            marker_color='blue',
            text=present_values,
            textposition='outside',
            textfont=dict(size=12, color='black', family='Arial', weight='bold')
        ))

        # Histogram for Accuracy
        fig.add_trace(go.Bar(
            x=names,
            y=accuracy_values,
            name='Accuracy (%)',
            marker_color='yellow',
            text=accuracy_values,
            textposition='outside',
            textfont=dict(size=12, color='black', family='Arial', weight='bold')
        ))

        fig.update_layout(
            title='Comparison of Absent, Present Values, and Accuracy for Each Column',
            xaxis=dict(title='Column Names', tickvals=np.arange(len(names)), ticktext=names, tickangle=-90),
            yaxis=dict(title='Values (Counts / %)'),
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("CSV File Comparison: Absent, Present Values & Accuracy")

    file1 = st.file_uploader("Upload CSV file 1  Python Generated", type="csv")
    file2 = st.file_uploader("Upload CSV file 2 Manually", type="csv")

    if file1 and file2:
        comparator = DataComparator(file1, file2)
        comparator.load_data()
        comparator.preprocess_columns()
        comparator.find_common_columns()
        comparator.calculate_absent_and_present()
        comparator.calculate_accuracy()
        comparator.plot_histogram()

if __name__ == "__main__":
    main()
