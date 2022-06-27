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
        df['daily'] = df.Date.dt.day
        daily_data_mean = df[[selected_param, 'daily']].groupby('daily').mean()
        fig = go.Figure()
        fig.add_trace(go.Line(x=daily_data_mean.index, y=df[selected_param]))
        add_dashed_line_for_limit(fig, selected_param)
        fig.layout.update(title_text="Daily Analysis For ------ " + selected_param, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)


    def monthly_analysis(df, st, pd, go, selected_param):
        df['monthsofyear'] = pd.Categorical(df.Date.dt.month, categories=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], ordered=True)
        monthly_data_mean = df[[selected_param, 'monthsofyear']].groupby('monthsofyear').mean()
        fig = go.Figure()
        fig.add_trace(go.Line(x=monthly_data_mean.index, y=df[selected_param]))
        add_dashed_line_for_limit(fig, selected_param)
        fig.layout.update(title_text="Monthly Analysis For ------ " + selected_param, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

    def annual_analysis(df, st, go, selected_param):
        df['annual'] = df.Date.dt.year
        annual_data_mean = df[[selected_param, 'annual']].groupby('annual').mean()
        fig = go.Figure()
        fig.add_trace(go.Line(x=annual_data_mean.index, y=df[selected_param]))
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