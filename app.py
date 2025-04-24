import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# === Load Data ===
df = pd.read_csv(r"https://drive.google.com/uc?export=download&id=1XbbQg3cyd3XHk0AWepjC8QK4jc1IHP-b")

# Use only relevant columns
columns_to_use = [
    'sex', 'diet_group', 'age_group',
    'mean_ghgs', 'mean_land', 'mean_watuse',
    'mean_acid', 'mean_eut', 'mean_bio'
]
df = df[columns_to_use].dropna()

# Separate numeric and categorical
numeric_cols = ['mean_ghgs', 'mean_land', 'mean_watuse', 'mean_acid', 'mean_eut', 'mean_bio']
cat_cols = ['sex', 'diet_group', 'age_group']

# Scale numeric
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === App Setup ===
app = Dash(__name__)
app.title = "Sustainability Dashboard"

app.layout = html.Div([  
    html.H1("🌱 Diet & Environmental Impact Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Gender"),
        dcc.Dropdown(
            id='gender',
            options=[{'label': g, 'value': g} for g in ['All'] + list(df_scaled['sex'].unique())],
            value='All'
        ),
        html.Label("Age Group"),
        dcc.Dropdown(
            id='age',
            options=[{'label': a, 'value': a} for a in ['All'] + list(df_scaled['age_group'].unique())],
            value='All'
        )
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.Label("Choose Chart Type"),
        dcc.Tabs(id='chart-tabs', value='all', children=[
            dcc.Tab(label='All Charts', value='all'),
            dcc.Tab(label='Radar Chart', value='radar'),
            dcc.Tab(label='Violin Plot', value='violin'),
            dcc.Tab(label='Parallel Coordinates', value='parallel'),
            dcc.Tab(label='Sunburst', value='sunburst'),
            dcc.Tab(label='Heatmap', value='heatmap'),
            dcc.Tab(label='3D Clustering', value='cluster'),
            dcc.Tab(label='Density Heatmap', value='density')
        ])
    ], style={'width': '68%', 'display': 'inline-block', 'paddingLeft': '20px'}),

    html.Div(id='main-graph', style={'marginTop': '30px'})
])

# === Callback ===
@app.callback(
    Output('main-graph', 'children'),
    Input('gender', 'value'),
    Input('age', 'value'),
    Input('chart-tabs', 'value')
)
def update_chart(gender, age, chart_type):
    # Filter data based on the "All" option
    if gender != 'All':
        filtered = df_scaled[df_scaled['sex'] == gender]
    else:
        filtered = df_scaled

    if age != 'All':
        filtered = filtered[filtered['age_group'] == age]

    # Radar
    grouped = filtered.groupby('diet_group')[numeric_cols].mean().reset_index()
    radar_fig = go.Figure()
    for _, row in grouped.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=row[numeric_cols],
            theta=numeric_cols,
            fill='toself',
            name=row['diet_group']
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
        showlegend=True,
        title="Radar Chart of Environmental Impact by Diet",
        autosize=True
    )

    # Violin
    df_v = filtered.melt(id_vars=['diet_group'], value_vars=numeric_cols,
                         var_name='Impact Type', value_name='Value')
    violin_fig = px.violin(df_v, x='Impact Type', y='Value', color='diet_group', box=True,
                           points='all', title="Violin Plot of Environmental Metrics by Diet")

    # Parallel
    parallel_fig = px.parallel_coordinates(filtered,
                                           dimensions=numeric_cols,
                                           color=filtered['diet_group'].astype('category').cat.codes,
                                           labels={col: col for col in numeric_cols},
                                           color_continuous_scale=px.colors.diverging.Tealrose,
                                           title="Parallel Coordinates Plot")

    # Sunburst
    sunburst_fig = px.sunburst(
        df,
        path=['sex', 'age_group', 'diet_group'],
        values='mean_ghgs',
        color='mean_ghgs',
        color_continuous_scale='RdBu',
        title='🌞 GHG Emissions by Gender → Age → Diet'
    )
    sunburst_fig.update_layout(title_x=0.5)

    # Correlation Heatmap
    corr = df[numeric_cols].corr()
    heatmap_fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='🔥 Correlation Heatmap of Environmental Metrics'
    )
    heatmap_fig.update_layout(title_x=0.5)

    # Clustering
    cluster_df = df_scaled[numeric_cols].copy()
    kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
    df_scaled['Cluster'] = kmeans.fit_predict(cluster_df)

    cluster_fig = px.scatter_3d(
        df_scaled, x='mean_ghgs', y='mean_land', z='mean_watuse',
        color='Cluster',
        symbol='diet_group',
        title='🧬 3D Clustering of Environmental Impact',
        color_continuous_scale='Viridis'
    )
    cluster_fig.update_layout(title_x=0.5)

    # Density
    density_fig = px.density_heatmap(
        df,
        x='age_group',
        y='diet_group',
        z='mean_land',
        histfunc='avg',
        color_continuous_scale='Viridis',
        title='🌍 Avg Land Use by Age and Diet Group'
    )
    density_fig.update_layout(title_x=0.5)
    
    # Set config for high-quality rendering
    config = {
        'displayModeBar': True,
        'scrollZoom': True,
        'responsive': True,
        'toImageButtonOptions': {'format': 'png', 'width': 1500, 'height': 1000}
    }

    # Return selected or all charts
    if chart_type == 'radar':
        return dcc.Graph(figure=radar_fig, config=config)
    elif chart_type == 'violin':
        return dcc.Graph(figure=violin_fig, config=config)
    elif chart_type == 'parallel':
        return dcc.Graph(figure=parallel_fig, config=config)
    elif chart_type == 'sunburst':
        return dcc.Graph(figure=sunburst_fig, config=config)
    elif chart_type == 'heatmap':
        return dcc.Graph(figure=heatmap_fig, config=config)
    elif chart_type == 'cluster':
        return dcc.Graph(figure=cluster_fig, config=config)
    elif chart_type == 'density':
        return dcc.Graph(figure=density_fig, config=config)
    else:  # 'all'
        return [
            dcc.Graph(figure=radar_fig, config=config),
            dcc.Graph(figure=violin_fig, config=config),
            dcc.Graph(figure=parallel_fig, config=config),
            dcc.Graph(figure=sunburst_fig, config=config),
            dcc.Graph(figure=heatmap_fig, config=config),
            dcc.Graph(figure=cluster_fig, config=config),
            dcc.Graph(figure=density_fig, config=config)
        ]

server = app.server


#gunicorn app:server