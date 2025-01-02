import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

# Set Page Config for an Agriculture-Themed Dashboard
st.set_page_config(
    page_title="Agriculture Dashboard",
    page_icon="🌾",
    layout="wide",
)

# Paths to files
model_path = os.path.join(os.getcwd(), 'linear_model_log.pkl')
data_path = os.path.join(os.getcwd(), 'agriculture_data_replaced.csv')

# Load the trained linear regression model
try:
    linear_model_log = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Ensure 'linear_model_log.pkl' exists in the same folder as this script.")
    st.stop()

# Load the dataset
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("Dataset file not found. Ensure 'agriculture_data_replaced.csv' exists in the same folder.")
    st.stop()

# Define mappings for Area and Item
area_mapping = {
    'Argentina': 0, 'Australia': 1, 'Austria': 2, 'Belgium': 3, 'Brazil': 4,
    'Canada': 5, 'China': 6, 'Denmark': 7, 'Finland': 8, 'France': 9,
    'Germany': 10, 'Hungary': 11, 'India': 12, 'Ireland': 13, 'Italy': 14,
    'Netherlands (Kingdom of the)': 15, 'New Zealand': 16, 'Poland': 17,
    'Romania': 18, 'Spain': 19, 'Sweden': 20, 'United States of America': 21
}
item_mapping = {
    'Barley': 0, 'Cereals, primary': 1, 'Hen eggs in shell, fresh': 2,
    'Meat, Total': 3, 'Milk, Total': 4, 'Raw milk of cattle': 5, 'Wheat': 6
}

# Create aggregated data for visualizations (global definition)
map_summary = data.groupby(['Year', 'Area'], as_index=False).agg({
    'Production Value (t)': 'sum',
    'Export Value (1000 USD)': 'sum',
    'Import Value (1000 USD)': 'sum'
})

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Page", ["📈Prediction and Forecasting", "📊Interactive Visualizations"])

### PREDICTION AND FORECASTING ###
if page == "📈Prediction and Forecasting":
    st.title("Export Value Prediction and Forecast Tool")

    # Sidebar for user inputs
    st.sidebar.header("Inputs")
    selected_area = st.sidebar.selectbox("Select Area", list(area_mapping.keys()))
    selected_item = st.sidebar.selectbox("Select Item", list(item_mapping.keys()))

    # Manual input for Export Quantity
    export_quantity = st.sidebar.number_input(
        "Export Quantity (t)",
        min_value=0,  
        max_value=102646398,  
        value=10000,  
        step=1000  
    )
    st.sidebar.text(f"Selected: {export_quantity:,} t")  

    # Manual input for Producer Price
    producer_price = st.sidebar.number_input(
        "Producer Price (USD/tonne)",
        min_value=0,  
        max_value=4042,  
        value=500,  
        step=10 
    )
    st.sidebar.text(f"Selected: ${producer_price:,}/tonne") 

    # Manual input for Production Value
    production_value = st.sidebar.number_input(
        "Production Value (t)",
        min_value=0,  
        max_value=635092927,  
        value=50000,  
        step=1000
    )
    st.sidebar.text(f"Selected: {production_value:,} t") 

    # Filter data for the selected area and item
    filtered_data = data[(data["Area"] == selected_area) & (data["Item"] == selected_item)]
    st.dataframe(filtered_data[["Year", "Export Quantity (t)", "Producer Price (USD/tonne)", "Production Value (t)", "Export Value (1000 USD)"]])


    # Add a download button for the table
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df_to_csv(filtered_data)
    st.download_button(
        label="Download Historical Data as CSV",
        data=csv_data,
        file_name=f"{selected_area}_{selected_item}_historical_data.csv",
        mime='text/csv'
    )




    # Map selections to numeric codes
    area_code = area_mapping[selected_area]
    item_code = item_mapping[selected_item]

    # Placeholder for additional features
    year = 2025
    additional_feature_placeholder = 0

    # Log-transform inputs
    log_inputs = np.log1p([production_value, export_quantity, producer_price])

    # Prepare input for prediction
    final_inputs = [year, area_code, item_code] + list(log_inputs) + [additional_feature_placeholder]

    # Predict export value
    try:
        log_predicted_export_value = linear_model_log.predict([final_inputs])[0]
        predicted_export_value = np.expm1(log_predicted_export_value)

        # Display prediction
        st.subheader("Predicted Export Value")
        st.write(f"${predicted_export_value:,.2f} (in 1000 USD)")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.stop()

    # Allow user to use the predicted value for forecasting
    st.sidebar.subheader("Forecasting Options")
    use_predicted = st.sidebar.checkbox("Use Predicted Export Value for Forecast", value=True)
    forecast_input_value = st.sidebar.number_input(
        "Export Value for Forecast (1000 USD):",
        min_value=0,
        value=int(predicted_export_value) if use_predicted else 10000
    )

    # Button to trigger forecast generation
    generate_forecast = st.sidebar.button("Generate Forecast")

    # Forecasting Logic
    if generate_forecast:
        st.subheader("Forecast Results")

        # Filter data
        filtered_data = data[(data['Area'] == selected_area) & (data['Item'] == selected_item)]
        filtered_data = filtered_data.sort_values("Year")
        filtered_data.set_index("Year", inplace=True)

        # Append user-provided export value
        export_values = filtered_data['Export Value (1000 USD)'].copy()
        next_year = export_values.index.max() + 1
        export_values.loc[next_year] = forecast_input_value

        # Forecast with SARIMA
        best_order = (0, 1, 0)
        best_seasonal_order = (1, 1, 0, 12)
        model = SARIMAX(export_values, order=best_order, seasonal_order=best_seasonal_order)
        model_fit = model.fit(disp=False)
        forecast_steps = 20
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_years = range(next_year + 1, next_year + 1 + forecast_steps)

        # Combine historical and forecast data for continuity
        combined_years = list(export_values.index) + [forecast_years[0]]
        combined_values = list(export_values.values) + [forecast.iloc[0]]

        # Create plotly figure
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=export_values.index,
            y=export_values.values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))

        # Connecting point (last historical to first forecast)
        fig.add_trace(go.Scatter(
            x=combined_years[-2:],  # Last historical year and first forecast year
            y=combined_values[-2:],  # Corresponding values
            mode='lines',
            name='Connection',
            line=dict(color='blue', dash='dot')  # Dotted line for connection
        ))

        # Forecast data
        fig.add_trace(go.Scatter(
            x=list(forecast_years),
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=f"SARIMA Forecast for {selected_item} in {selected_area}",
            xaxis_title="Year",
            yaxis_title="Export Value (1000 USD)",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )

        # Display the figure
        st.plotly_chart(fig)

### INTERACTIVE VISUALIZATIONS ###
elif page == "📊Interactive Visualizations":
    st.title("Interactive Data Visualizations")

    # Sidebar selection
    visualization_type = st.sidebar.selectbox(
        "Choose Visualization",
        [
            "Production Trends Map",
            "Export Trends Map",
            "Import Trends Map",
            "Trade Balance Comparison",
            "Production Items Grouping",
            "Export Value Line Chart",
            "Export Value Treemap"
        ]
    )

    # Conditional Rendering of Visualizations
    if visualization_type == "Production Trends Map":
        fig_production = px.choropleth(
            map_summary,
            locations='Area',
            locationmode='country names',
            color='Production Value (t)',
            hover_name='Area',
            animation_frame='Year',
            title='Production Trends by Country (2000-2022)',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_production)

    elif visualization_type == "Export Trends Map":
        fig_export = px.choropleth(
            map_summary,
            locations='Area',
            locationmode='country names',
            color='Export Value (1000 USD)',
            hover_name='Area',
            animation_frame='Year',
            title='Export Trends by Country (2000-2022)',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_export)

    elif visualization_type == "Import Trends Map":
        fig_import = px.choropleth(
            map_summary,
            locations='Area',
            locationmode='country names',
            color='Import Value (1000 USD)',
            hover_name='Area',
            animation_frame='Year',
            title='Import Trends by Country (2000-2022)',
            color_continuous_scale=px.colors.sequential.Cividis
        )
        st.plotly_chart(fig_import)

    elif visualization_type == "Trade Balance Comparison":
        # Calculate Trade Balance
        data['Trade Balance (1000 USD)'] = (
            data['Export Value (1000 USD)'] - data['Import Value (1000 USD)']
        )
        
        # Group data by Year and Area
        grouped_data = data.groupby(['Year', 'Area'], as_index=False)['Trade Balance (1000 USD)'].sum()
        
        # Dropdown for Country Selection
        countries = grouped_data['Area'].unique()
        selected_country = st.sidebar.selectbox("Select a Country", ["All"] + list(countries))

        # Create Plotly figure
        fig = go.Figure()
        if selected_country == "All":
            # Add all countries
            for country in countries:
                country_data = grouped_data[grouped_data['Area'] == country]
                fig.add_trace(go.Scatter(
                    x=country_data['Year'],
                    y=country_data['Trade Balance (1000 USD)'],
                    mode='lines+markers',
                    name=country
                ))
        else:
            # Add only the selected country
            country_data = grouped_data[grouped_data['Area'] == selected_country]
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['Trade Balance (1000 USD)'],
                mode='lines+markers',
                name=selected_country
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Trade Balance Comparison ({selected_country})" if selected_country != "All" else "Trade Balance Comparison for All Countries",
            xaxis_title="Year",
            yaxis_title="Trade Balance (1000 USD)",
            legend_title="Countries",
            width=1000,
            height=600
        )

        # Display chart
        st.plotly_chart(fig)

    elif visualization_type == "Production Items Grouping":
        # Define the list of EU countries
        eu_countries = [
            "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
            "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
            "Netherlands (Kingdom of the)", "Poland", "Portugal", "Romania",
            "Slovakia", "Slovenia", "Spain", "Sweden"
        ]

        # List of all countries in the dataset
        all_countries = data['Area'].unique()

        # Separate countries into Worldwide (non-EU countries) and EU countries
        worldwide_countries = [country for country in all_countries if country not in eu_countries]

        # Group data by Area, Item, and Year
        comparison_data_dynamic = data.groupby(['Area', 'Item', 'Year'])[['Production Value (t)']].sum().reset_index()

        # Sidebar dropdown for group selection
        group_selection = st.sidebar.selectbox("Select Country Group", ["All Countries", "EU Countries", "Worldwide Countries"])

        # Filter countries based on selection
        if group_selection == "EU Countries":
            filtered_countries = eu_countries
        elif group_selection == "Worldwide Countries":
            filtered_countries = worldwide_countries
        else:  # All Countries
            filtered_countries = all_countries

        # Filter data for selected group
        filtered_data = comparison_data_dynamic[comparison_data_dynamic['Area'].isin(filtered_countries)]

        # Sidebar filter for year
        selected_year = st.sidebar.slider("Select Year", int(filtered_data['Year'].min()), int(filtered_data['Year'].max()), step=1)

        # Filter data by selected year
        filtered_data = filtered_data[filtered_data['Year'] == selected_year]

        # Get the top items by production value
        top_items = filtered_data.groupby('Item')['Production Value (t)'].sum().nlargest(10).index
        filtered_data = filtered_data[filtered_data['Item'].isin(top_items)]

        # Create Plotly bar chart
        fig_dynamic_grouped = go.Figure()

        for country in filtered_countries:
            country_data = filtered_data[filtered_data['Area'] == country]
            fig_dynamic_grouped.add_trace(
                go.Bar(
                    x=country_data['Item'],
                    y=country_data['Production Value (t)'],
                    name=country
                )
            )

        # Update layout for the bar chart
        fig_dynamic_grouped.update_layout(
            title=f"Top Production Items for {group_selection} in {selected_year}",
            xaxis_title="Production Items",
            yaxis_title="Production Value (tons)",
            barmode="group",
            width=1000,
            height=600,
            yaxis=dict(
                tickformat=",",  # Force commas for thousands separator
                title="Production Value (tons)"
            )
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig_dynamic_grouped)

    elif visualization_type == "Export Value Line Chart":
        export_values = data[['Year', 'Area', 'Export Value (1000 USD)']].groupby(['Year', 'Area']).sum().reset_index()
        fig_line_chart = px.line(
            export_values,
            x='Year',
            y='Export Value (1000 USD)',
            color='Area',
            title='Export Value Trends Over Time'
        )
        fig_line_chart.update_layout(width=1000, height=600)
        st.plotly_chart(fig_line_chart)

    elif visualization_type == "Export Value Treemap":
        export_values = data[['Year', 'Area', 'Export Value (1000 USD)']].groupby(['Year', 'Area']).sum().reset_index()
        fig_treemap = px.treemap(
            export_values,
            path=['Area', 'Year'],
            values='Export Value (1000 USD)',
            title='Treemap of Export Value by Country and Year'
        )
        st.plotly_chart(fig_treemap)

# Footer or Branding Section
st.sidebar.markdown("---")
st.sidebar.write("Powered by 🌾 Agriculture Insights")
st.sidebar.markdown("Created by **Federico Ariton**")