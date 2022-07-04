def add_dashed_line_for_limit(fig, selected_param):
        if selected_param == "PM10 ( µg/m3 )":
            fig.add_hline(y=45, line_width=2, line_dash='dash')
        elif selected_param == "SO2 ( µg/m3 )":
            fig.add_hline(y=40, line_width=2, line_dash='dash')
        elif selected_param == "NO2 ( µg/m3 )":
            fig.add_hline(y=25, line_width=2, line_dash='dash')
        elif selected_param == "O3 ( µg/m3 )":
            fig.add_hline(y=60, line_width=2, line_dash='dash')
        elif selected_param == "PM 2.5 ( µg/m3 )":
            fig.add_hline(y=15, line_width=2, line_dash='dash')
        elif selected_param == "CO ( µg/m3 )":
            fig.add_hline(y=24, line_width=2, line_dash='dash')
        else:
            pass

class AnalysisTypesClass:
    
    def daily_analysis(df, st, go, selected_param):
        df['day_of_month'] = df.index.day
        df = df.groupby('day_of_month').mean()

        fig = go.Figure()
        fig.add_trace(go.Line(x=df.index, y=df[selected_param]))
        add_dashed_line_for_limit(fig, selected_param)
        fig.layout.update(title_text="Daily Analysis For ------ " + selected_param, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

    def monthly_analysis(df, st, pd, go, selected_param):      
        df['month'] = df.index.month
        df = df.groupby('month').mean()

        fig = go.Figure()
        fig.add_trace(go.Line(x=df.index, y=df[selected_param]))
        add_dashed_line_for_limit(fig, selected_param)
        fig.layout.update(title_text="Monthly Analysis For ------ " + selected_param, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

    def annual_analysis(df, st, go, selected_param):
        df['year'] = df.index.year
        df = df.groupby('year').mean()

        fig = go.Figure()
        fig.add_trace(go.Line(x=df.index, y=df[selected_param]))
        add_dashed_line_for_limit(fig, selected_param)
        fig.layout.update(title_text="Annual Analysis For ------ " + selected_param, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

        '''
            PM10 ( µg/m3 )
            SO2 ( µg/m3 )
            NO2 ( µg/m3 )
            NOX ( µg/m3 )
            O3 ( µg/m3 )
            PM 2.5 ( µg/m3 )
            CO ( µg/m3 )

        '''