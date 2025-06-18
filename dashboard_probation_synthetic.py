import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

st.set_page_config(layout="wide")

# Generate synthetic data
@st.cache_data
def generate_data():
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
        'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
        'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
        'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
        'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
        'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    years = np.arange(1994, 2020)
    np.random.seed(42)
    data = []
    for state in states:
        base_prob = np.random.uniform(200, 500)
        base_spending = np.random.uniform(800, 1500)
        for year in years:
            year_effect = (year - 1994) * np.random.uniform(-5, 5)
            spending = base_spending + year_effect + np.random.normal(0, 50)
            probation_pop = base_prob + (spending * 0.2) + np.random.normal(0, 20)
            cpi = 100 + (year - 1994) * 2.5
            spending_real = spending * 100 / cpi
            data.append([state, year, spending, spending_real, probation_pop])
    df = pd.DataFrame(data, columns=['State', 'Year', 'Spending', 'Spending_Real', 'Probation_Pop'])
    return df

# Load data
df = generate_data()

# Sidebar controls
st.sidebar.header("Filter")
state_choice = st.sidebar.selectbox("Select a State", sorted(df['State'].unique()))
year_range = st.sidebar.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), (2000, 2015))

# Filtered data
df_filtered = df[(df['State'] == state_choice) & (df['Year'].between(*year_range))]

# Main dashboard
st.title("Justice System Dashboard: Synthetic Analysis")
st.markdown(f"### State: {state_choice} | Years: {year_range[0]}–{year_range[1]}")

# Line plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Corrections Spending Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(data=df_filtered, x='Year', y='Spending', ax=ax)
    ax.set_ylabel("Nominal Spending (Millions USD)")
    st.pyplot(fig)

with col2:
    st.subheader("Inflation-Adjusted Spending Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(data=df_filtered, x='Year', y='Spending_Real', ax=ax, color='green')
    ax.set_ylabel("Real Spending (1994 USD)")
    st.pyplot(fig)

# Regression analysis
st.markdown("### Inflation-Adjusted Spending and Probation Population Regression")
df_lag = df.copy()
df_lag['Lagged_Real_Spending'] = df_lag.groupby('State')['Spending_Real'].shift(1)
df_lag = df_lag.dropna()

model = smf.ols('Probation_Pop ~ Spending_Real + Lagged_Real_Spending + C(State) + C(Year)', data=df_lag).fit()
st.write(model.summary().tables[1])

# Final chart: Predicted vs actual
st.subheader("Actual vs. Predicted Probation Population")
df_lag['Predicted'] = model.predict(df_lag)
df_display = df_lag[df_lag['State'] == state_choice]
fig, ax = plt.subplots()
ax.plot(df_display['Year'], df_display['Probation_Pop'], label='Actual')
ax.plot(df_display['Year'], df_display['Predicted'], label='Predicted', linestyle='--')
ax.set_ylabel("Probation Population")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("Data and model are fully synthetic. Replace with BJS data for real analysis.")

# Section: 50-State Facet Plot
st.markdown("### 50-State Visualization: Spending vs Probation Pop")

sns.set(style="whitegrid")
g = sns.FacetGrid(df, col="State", col_wrap=5, height=2.2, sharey=False)
g.map_dataframe(sns.lineplot, x="Year", y="Spending", label="Spending (Mil)")
g.map_dataframe(sns.lineplot, x="Year", y="Probation_Pop", color='orange', label="Probation Pop")
g.set_titles(col_template="{col_name}", size=8)
g.add_legend()
plt.subplots_adjust(top=0.92)
g.fig.suptitle("Corrections Spending and Probation Population by State (1994–2019)", fontsize=14)

st.pyplot(g.fig)
