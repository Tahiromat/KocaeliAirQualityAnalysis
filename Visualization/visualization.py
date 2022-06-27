class VisualizationTypesClass:

    def histogram_visualization(df, st, px, y_axis):
        df = df.resample('M').mean()
        fig = px.histogram(df, x=y_axis)
        fig.layout.update(title_text="Histogram Visualization For ------ " + y_axis, xaxis_rangeslider_visible=False, width=750, height=600)
        st.plotly_chart(fig)

    def line_visualization(df, st, go, y_axis):
        df = df.resample('M').mean()
        fig = go.Figure()
        fig.add_trace(go.Line(x=df.index, y=df[y_axis]))
        fig.layout.update(title_text="Line Visualization For ------ " + y_axis, xaxis_rangeslider_visible=False, width=750, height=600)
        st.plotly_chart(fig)

    def pie_visualization(df, st, pd, px, parameters):
        mean_values = []
        for param in parameters:
            data_m = df[param].mean()
            mean_values.append(data_m)
        data_mean = pd.DataFrame()
        data_mean['Params'] = parameters
        data_mean['Values'] = mean_values
        fig = px.pie(data_mean, values='Values', names='Params')
        fig.layout.update(width=1500, height=600)
        st.plotly_chart(fig)

    def map_visualization(st, pdk, lat, lon):
        st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=12.5,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                # data=df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))