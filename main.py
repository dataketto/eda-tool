import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pivottablejs import pivot_ui
import glob
import os
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode,DataReturnMode, JsCode # pip install streamlit-aggrid==0.2.3
# from st_aggrid import AgGrid
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(page_title="Cool App",page_icon=":shark:",layout="wide")
st.markdown("""
<style>
[data-testid="column"] column:first-child {
    box-shadow: rgb(0 0 0 / 20%) 0px 2px 1px -1px, rgb(0 0 0 / 14%) 0px 1px 1px 0px, rgb(0 0 0 / 12%) 0px 1px 3px 0px;
    border-radius: 15px;
    padding: 1% 1% 1% 1%;
} 
[data-testid="column"] column:third-child {
    box-shadow: rgb(0 0 0 / 20%) 0px 2px 1px -1px, rgb(0 0 0 / 14%) 0px 1px 1px 0px, rgb(0 0 0 / 12%) 0px 1px 3px 0px;
    border-radius: 15px;
    padding: 1% 1% 1% 1%;
}
</style>""", unsafe_allow_html=True)

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def upload():
    with st.sidebar:
        # uploaded_file = st.sidebar.file_uploader("Choose a Foldr",type=['csv'])
        file_path = st.text_input("Paste a Folder Path (e.g. D:\Download\ ) ")
        
        if file_path:
            filename = file_selector(folder_path=file_path)
            st.write('You selected `%s`' % filename)
            # csv files in the path
            files = glob.glob(file_path + "/*.csv")
            files_xlsx = glob.glob(file_path + "/*.xlsx")
            files_xml = glob.glob(file_path + "/*.xml")
            for file in files_xlsx:
                files.append(file)
            for file in files_xml:
                files.append(file)
            latest_file = sorted(files, key=os.path.getctime)
            # file name without extension
            # file name with extension
            file_names = []
            for file in latest_file:
                file_name = os.path.basename(file)
                file_names.append(os.path.splitext(file_name)[0]+os.path.splitext(file_name)[1])
            st.write(file_names)
            # file_selected_1 = st.selectbox("Select file",file_names)
            # st.write(file_selected_1)
            # file_selected = file_path+file_selected_1
            file_selected = st.text_input("Paste a Folder Path  D:\Download\ ) ")
            if file_selected is not None:
                if ".xlsx" in file_selected:
                    data_xls = pd.read_excel(f"{file_selected}", index_col=None)
                elif ".xml" in file_selected:
                    data_xls = pd.read_xml(f"{file_selected}", index_col=None)
                else:
                    data_xls = pd.read_csv(f"{file_selected}", index_col=None)
                st.session_state['file_upload'] = True
                return data_xls
        else :
            st.session_state['file_upload'] = False
    

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def operation_onchange():
    st.session_state['operation_'] = st.session_state.op

def tab1(df):
    # drop where all nan row 
    df = df.dropna(how = 'all')
    # drop where all nan col 
    df = df.dropna(axis=1, how='all')
    # clean up data
    df.replace('[^\x00-\x7F]','',regex=True, inplace = True)
    # \W for non-word character replace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace('\W', '', regex=True)
    # select purpose
    select_section = st.selectbox("select section",("Pivot table","EDA" ,"Final column selection"))
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # remove duplicates
    duplicates_check_box = st.checkbox("Remove Duplicates")

    #Example controlers
    st.sidebar.subheader("St-AgGrid example options")
    sample_size = 30
    grid_height = 800
    return_mode = st.sidebar.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
    return_mode_value = DataReturnMode.__members__[return_mode]
    update_mode = st.sidebar.selectbox("Update Mode", list(GridUpdateMode.__members__), index=len(GridUpdateMode.__members__)-1)
    update_mode_value = GridUpdateMode.__members__[update_mode]
    #enterprise modules
    enable_enterprise_modules = True
    enable_sidebar =True
    #features
    fit_columns_on_grid_load = False
    enable_selection=False
    if enable_selection:
        st.sidebar.subheader("Selection options")
        selection_mode = st.sidebar.radio("Selection Mode", ['single','multiple'], index=1)

        use_checkbox = st.sidebar.checkbox("Use check box for selection", value=True)
        if use_checkbox:
            groupSelectsChildren = st.sidebar.checkbox("Group checkbox select children", value=True)
            groupSelectsFiltered = st.sidebar.checkbox("Group checkbox includes filtered", value=True)

        if ((selection_mode == 'multiple') & (not use_checkbox)):
            rowMultiSelectWithClick = st.sidebar.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
            if not rowMultiSelectWithClick:
                suppressRowDeselection = st.sidebar.checkbox("Suppress deselection (while holding CTRL)", value=False)
            else:
                suppressRowDeselection=False
    enable_pagination =True
    if enable_pagination:
        paginationAutoSize =True
        if not paginationAutoSize:
            paginationPageSize = st.sidebar.number_input("Page size", value=5, min_value=0, max_value=sample_size)


    

    if duplicates_check_box:
        """###### Duplicates"""\
        # dropping duplicate values
        before_len = len(df)
        df.drop_duplicates(keep=False, inplace=True)
        after_len = len(df)
        if before_len-after_len==0:
            st.write("No Duplicates")
        else:
            st.success(f"{before_len-after_len} duplicates rows removed")
        # st.write(before_len)
        # st.write(after_len)

    


    nan_check_box = st.checkbox("Check Nulls")
    if nan_check_box:
        """###### Check Nulls"""
        before_len = len(df)
        # Applying the method
        check_nan = df.isnull().values.any()
        st.dataframe(df.isna().sum().sort_values(ascending=False))
        nan_remove = st.checkbox("Remove Nulls ")
        if nan_remove:
            user_cat_i = st.multiselect(
                        f"Values for ",
                        df.columns[df.isna().any()].sort_values()
                        # default=list(df.columns[df.isna().any()].sort_values()),
                    )
            for col in user_cat_i:
                df = df.dropna(subset=col)
            after_len = len(df)
            if before_len-after_len==0:
                st.write("No Rows Removed")
            else:
                st.success(f"{before_len-after_len} rows removed")
    filter_col_check_box = st.checkbox("Filter Columns")
    if filter_col_check_box:
        user_cat_i = st.multiselect(
                        f"Values for ",
                        df.columns,
                        default=list(df.columns),
                    )
        df = df[user_cat_i]

    outlier_check_box = st.checkbox("Remove Outliers")
    if outlier_check_box:
        # Calculate the Interquartile Range(IQR)
        before_len = len(df)
        col1,col2 = st.columns(2)
        select_outlier_col = col1.multiselect(
                            f"Select Outlier column ",
                            df.select_dtypes(include=numerics).columns,
                            default=list(df.select_dtypes(include=numerics).columns),
                        )
        outlier_slider = col2.slider('Select a Interquartile Range',0.00, 1.00,step=0.05,value= (0.25, 0.75))
        Q1 = df.quantile(outlier_slider[0]) 
        Q3 = df.quantile(outlier_slider[1]) 
        IQR = Q3 - Q1
        st.dataframe(IQR)
        #Identify outliers #Identify outliers (values outside of Q1-1.5IQR to Q3+1.5IQR range) 
        df = df[~((df[select_outlier_col] < (Q1 - 1.5 * IQR)) |(df[select_outlier_col] > (Q3 + 1.5 * IQR))).any(axis=1)]
        after_len = len(df)
        if before_len-after_len==0:
                st.write("No Outliers Removed")
        else:
            st.success(f"{before_len-after_len} Outliers rows removed")

    # purpose
    if select_section == "Pivot table":
        """### Pivot table"""
        if len(df)<=200000:
            t = pivot_ui(df)
            with open(t.src) as t:
                components.html(t.read(), width=1800, height=800, scrolling=True)
        else :
            #Infer basic colDefs from dataframe types
            gb = GridOptionsBuilder.from_dataframe(df)
            #customize gridOptions
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
            #configures last row to use custom styles based on cell's value, injecting JsCode on components front end
            cellsytle_jscode = JsCode("""
            function(params) {
                if (params.value == 'A') {
                    return {
                        'color': 'white',
                        'backgroundColor': 'darkred'
                    }
                } else {
                    return {
                        'color': 'black',
                        'backgroundColor': 'white'
                    }
                }
            };
            """)
            gb.configure_column("group", cellStyle=cellsytle_jscode)

            if enable_sidebar:
                gb.configure_side_bar()
            if enable_selection:
                gb.configure_selection(selection_mode)
                if use_checkbox:
                    gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)
            if enable_pagination:
                if paginationAutoSize:
                    gb.configure_pagination(paginationAutoPageSize=True)
                else:
                    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

            gb.configure_grid_options(domLayout='normal')
            gridOptions = gb.build()
            grid_response = AgGrid(
                df, 
                gridOptions=gridOptions,
                height=grid_height, 
                width='100%',
                data_return_mode=return_mode_value, 
                update_mode=update_mode_value,
                fit_columns_on_grid_load=fit_columns_on_grid_load,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                enable_enterprise_modules=enable_enterprise_modules
                )
            df = grid_response['data']
            selected = grid_response['selected_rows']
            selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')
            
    elif select_section== "EDA" :
        """### EDA"""
        if len(df)<=80000:
            t = pivot_ui(df)
            with open(t.src) as t:
                components.html(t.read(), width=1800, height=800, scrolling=True)
        else :
            #Infer basic colDefs from dataframe types
            gb = GridOptionsBuilder.from_dataframe(df)
            #customize gridOptions
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
            #configures last row to use custom styles based on cell's value, injecting JsCode on components front end
            cellsytle_jscode = JsCode("""
            function(params) {
                if (params.value == 'A') {
                    return {
                        'color': 'white',
                        'backgroundColor': 'darkred'
                    }
                } else {
                    return {
                        'color': 'black',
                        'backgroundColor': 'white'
                    }
                }
            };
            """)
            gb.configure_column("group", cellStyle=cellsytle_jscode)

            if enable_sidebar:
                gb.configure_side_bar()
            if enable_selection:
                gb.configure_selection(selection_mode)
                if use_checkbox:
                    gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)
            if enable_pagination:
                if paginationAutoSize:
                    gb.configure_pagination(paginationAutoPageSize=True)
                else:
                    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

            gb.configure_grid_options(domLayout='normal')
            gridOptions = gb.build()
            grid_response = AgGrid(
                df, 
                gridOptions=gridOptions,
                height=grid_height, 
                width='100%',
                data_return_mode=return_mode_value, 
                update_mode=update_mode_value,
                fit_columns_on_grid_load=fit_columns_on_grid_load,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                enable_enterprise_modules=enable_enterprise_modules
                )
            df = grid_response['data']
            selected = grid_response['selected_rows']
            selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')

        with st.spinner("Displaying results..."):
            #displays the chart
            col1,col2,col3 = st.columns(3)
            column_X = col1.selectbox("X",df.columns)
            column_Y = col2.selectbox("Y",df.columns)
            try:
                chart_data = df[[column_X,column_Y]]
                chart_1 = alt.Chart(chart_data).mark_circle(size=60).encode(
                    x=alt.X(column_X),
                    y=alt.Y(column_Y),
                    # color='Origin'
                    tooltip= list(df[[column_X,column_Y]].columns)
                ).interactive().properties(
                width=150,
                height=400
                )
                st.altair_chart(chart_1, use_container_width=True)
            except Exception as e: st.write(e)

            # Boxplot with Min/Max Whiskers
            boxplot  = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                x=alt.X(column_X),
                y=alt.Y(column_Y)
            ).properties(
                width=150,
                height=400
            ).interactive()
            st.altair_chart(boxplot, use_container_width=True)

            # bar
            col1,col2,col3,col4 = st.columns(4)
            column_X_ = col1.selectbox("X _ ",df.columns)
            if df.columns[0] ==column_X_:
                column_Y_ = df.columns[1]
            else : 
                column_Y_ = df.columns[0]
            # column_Y_ = col2.selectbox("Y ",df.columns)
            # value = col3.selectbox( "Operation_value_", ("count","sum","mean",  "median", "min", "max" ))
            # value_ = col4.selectbox( "data type", ("a continuous real-valued quantity","a discrete ordered quantity","a discrete unordered category",  "a time or date value" ))
            # if value_ =="a time or date value":
            #     datatype = 'T'
            # elif value_ == "a continuous real-valued quantity":
            #     datatype = 'Q'
            # elif value_ =="a discrete ordered quantity":
            #     datatype = 'O'
            # elif value_ == "a discrete unordered category":
            #     datatype = 'N'
            chart_data = df[[column_X_,column_Y_]]
            chart_data = pd.melt(chart_data,id_vars=column_X_, var_name=column_Y_, value_name="quantity")
            # st.dataframe(chart_data)
            chart = alt.Chart(data=chart_data).mark_bar().encode(
                x=alt.X(column_X_),
                y=alt.Y(f"column_Y_:Q",aggregate="count")
            ).properties(
                width=150,
                height=400
            ).interactive()
            alt.vconcat(
                chart.encode(color='column_Y_:Q').properties(title='quantitative'),
                chart.encode(color='column_Y_:O').properties(title='ordinal'),
                chart.encode(color='column_Y_:N').properties(title='nominal'),
            )
            st.altair_chart(chart, use_container_width=True)

            

            # base = alt.Chart(df).encode(
            #     theta=alt.Theta("column_X:Q", stack=True),
            #     radius=alt.Radius("column_X:Q", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
            #     color="column_X:N",
            # )
            # st.altair_chart(base, use_container_width=True)
    submit = st.button('Final Select')
        
    # t = pivot_ui(df)
    # with open(t.src) as t:
    #     components.html(t.read(), width=900, height=1000, scrolling=True)
    # df_filtered = st.dataframe(filter_dataframe(df))
    # st.dataframe(df_filtered)
    # section_container  = st.container()
    # select_section = section_container.selectbox("select section",("Pivot table","Column Selection" ,"Final column selection"))

    # with section_container:
    #     for i in range(st.session_state['no_of_section']):
    #         if select_section == "Pivot table":
    #             """### Pivot table"""
    #             col1,col2 = st.columns([1,5])
    #             column = col1.multiselect("Column_",options=df.columns)
    #             # if column :
    #             #     st.session_state['column_main_'] = column
    #             # elif not st.session_state['column_main_']:
    #             #     st.session_state['column_main_'] = df.select_dtypes(include=object).columns[0] 
    #             row = col1.multiselect("Row_",options=df.columns)
    #             value_col = col1.multiselect("Value_",options=df.select_dtypes(include=numerics).columns)
    #             if value_col:
    #                 st.session_state['value_main_'] = value_col
    #             elif not st.session_state['value_main_']:
    #                 st.session_state['value_main_'] = df.select_dtypes(include=numerics).columns[0]
    #             value = col1.selectbox( "Operation_value_", ("Count", "Distinct","Sum","Mean"),key='op',on_change=operation_onchange)
    #             if value :
    #                 st.session_state['operation_'] = value
    #             # st.write(value)
    #             # st.write(st.session_state['operation_'])
    #             if st.session_state['operation_'] == 'Sum':
    #                 arg_ = np.sum
    #             elif st.session_state['operation_'] == 'Count':
    #                 arg_ = 'count'
    #             elif st.session_state['operation_'] == 'Distinct':
    #                 arg_ =  pd.Series.nunique
    #             elif st.session_state['operation_'] == 'Mean':
    #                 arg_ =  np.mean
    #             df_filtered = st.dataframe(filter_dataframe(df))
    #             df_query = pd.pivot_table(df_filtered, values=st.session_state['value_main_'], index=st.session_state['column_main_'],
    #                                 columns=row, aggfunc=arg_)
    #             # container
    #             container_1 = col2.container()
    #             with container_1:
    #                 if  st.session_state['value_main_']:
    #                     st.dataframe(df_query)
    #                 else:
    #                     e = RuntimeError('Must select - Column, value in sidebar')
    #                     st.exception(e)
    #     add_section_btn = st.button("add section")
    #     if add_section_btn:
    #         st.session_state['no_of_section']+=1
    #     remove_section_btn = st.button("remove section")
    #     if remove_section_btn:
    #         st.session_state['no_of_section']-=1

        # """ ### Categorical """
        # """###### Describe"""
        # st.dataframe(df[df.select_dtypes(include=object).columns].describe())
        # col1,col2,col3 = st.columns(3)
        # column = col1.multiselect("Column ",options=df.select_dtypes(include=object).columns)
        # row = col2.multiselect("Row ",options=df.columns)

        # var = 'OverallQual'
        
        # if column:
        #     # st.dataframe(df[column].describe())
        #     # Create distplot with custom bin_size
        #     st.line_chart(df[column[0]])
        #     st.bar_chart(df[column[0]])
        #     chart = alt.Chart(df).mark_bar().encode(
        #         alt.X(column[0]),
        #         y='count()',
        #     )
        #     st.altair_chart(chart)
        #     chart = alt.Chart(df).mark_bar().encode(
        #         alt.X(column[0]),
        #         y=row[0],
        #     )
        #     st.altair_chart(chart)

        #     chart_1 = alt.Chart(df).mark_circle().encode(
        #     alt.X(alt.repeat("column"), type='quantitative'),
        #     alt.Y(alt.repeat("row"), type='quantitative'),
        #     color='Origin:N'
        #     ).properties(
        #         width=150,
        #         height=150
        #     ).repeat(
        #         row=row,
        #         column=column
        #     ).interactive()
        #     st.altair_chart(chart_1)
        # else :
        #     st.dataframe(df[df.columns[0]].describe())
        # """ ### Numerical """
        # """###### Describe"""
        # st.dataframe(df[df.select_dtypes(include=numerics).columns].describe())
        # """###### Duplicates"""
        # duplicates_row = df.groupby(df.columns.tolist(),as_index=False).size()
        # if duplicates_row['size'].sum()==0:
        #     st.write("No Duplicates")
        # else:
        #     st.dataframe(duplicates_row)

        # """###### Null"""
        # # Applying the method
        # check_nan = df.isnull().values.any()
        # st.dataframe(df.isna().sum())

        
        # value_ = st.multiselect("Select Value ",options=df.select_dtypes(include=numerics).columns)
        # if value_:
            #histogram
            # Histogram
            # st.bar_chart(data=df[value_[0]])

            #histogram
            # hist_data = [df[value_[0]],df[value_[1]]]
            # group_labels = [value_]
            # fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 25])
            # st.plotly_chart(fig, use_container_width=True)

            # x = [1, 2, 3, 4, 5]
            # y = [6, 7, 2, 4, 5]

            # p = figure(
            #     title='simple line example',
            #     x_axis_label='x',
            #     y_axis_label='y')

            # p.line(x, y, legend_label='Trend', line_width=2)

            # st.bokeh_chart(p, use_container_width=True)
            # scale = alt.Scale(
            #     domain=["sun", "fog", "drizzle", "rain", "snow"],
            #     range=["#e7ba52", "#a7a7a7", "#aec7e8", "#1f77b4", "#9467bd"],
            # )
            # color = alt.Color("weather:N", scale=scale)

            # # We create two selections:
            # # - a brush that is active on the top panel
            # # - a multi-click that is active on the bottom panel
            # brush = alt.selection_interval(encodings=["x"])
            # click = alt.selection_multi(encodings=["color"])
            # # Bottom panel is a bar chart of weather type
            # color=alt.value("#f4a582")
            # bars = (
            #     alt.Chart(df).mark_bar().encode(
            #         alt.X(alt.repeat('column'), type='quantitative'),
            #         alt.Y(alt.repeat('row'), type='quantitative')
            #         # x=df[column[0]],
            #         # y=df[value_[0]]
            #     )
            # )
            # st.altair_chart(bars, use_container_width=True)

            # fig, ax = plt.subplots( figsize=(10, 2))
            # sns.histplot( df[value_[0]],bins='auto',fill=True)
            # st.pyplot(fig)
            # fig, ax = plt.subplots( figsize=(10, 2))
            # sns.distplot( df[value_[0]],bins='auto',ax=ax)
            # st.pyplot(fig)
            # st.plotly_chart(sns.distplot(df[value_[0]]))
            # data = pd.concat([df[value_[0]], df[value_[1]]], axis=1)
            # f, ax = plt.subplots(figsize=(8, 6))
            # fig = sns.boxplot(x=df[value_[1]], y=df[value_[0]], data=data)
            # fig.axis(ymin=0, ymax=800000)
            # st.write(fig)

            #correlation matrix
            # corrmat = df.corr()
            # f, ax = plt.subplots(figsize=(15, 2))
            # sns.heatmap(corrmat, vmax=.8)
            # st.pyplot(f)
            
            # k = 10 #number of variables for heatmap
            # cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
            # cm = np.corrcoef(df[cols].values.T)
            # sns.set(font_scale=1.25)
            # sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
            # st.pyplot(fig)
            # fig, (ax1, ax2) = plt.subplots(2)
            # fig.suptitle('Vertically stacked subplots')
            # ax1.plot(x, y)
            # ax2.plot(x, -y)
        

def main():
    # Storing the chat session state
    if 'file_upload' not in st.session_state:
        st.session_state['file_upload'] = False
    if 'tab_1' not in st.session_state:
        st.session_state['tab_1'] = 2
    # if 'column_main' not in st.session_state:
    #     st.session_state['column_main'] = []
    # if 'value_main' not in st.session_state:
    #     st.session_state['value_main'] = []
    # if 'operation' not in st.session_state:
    #     st.session_state['operation'] = []
    df = upload()
    if st.session_state['file_upload']:
        # store current selected 
        st.success("File Selected successfully!!!")
        tabs = st.tabs(["🗃 Data Preview", "📈 Analysis"])
        with tabs[0]:
            st.dataframe(df)
        with tabs[1]:
            if 'column_main_' not in st.session_state:
                st.session_state['column_main_'] = []
            if 'value_main_' not in st.session_state:
                st.session_state['value_main_'] = []
            if 'operation_' not in st.session_state:
                st.session_state['operation_'] = []
            if 'no_of_section' not in st.session_state:
                st.session_state['no_of_section'] = 1
            tab1(df)
    else :
        st.error("👈Select file from sidebar....ERROR: File not Selected!")


if __name__ == "__main__":
    main()

# import altair as alt
# import streamlit as st
# from vega_datasets import data

# source = data.seattle_weather()

# scale = alt.Scale(
#     domain=["sun", "fog", "drizzle", "rain", "snow"],
#     range=["#e7ba52", "#a7a7a7", "#aec7e8", "#1f77b4", "#9467bd"],
# )
# color = alt.Color("weather:N", scale=scale)

# # We create two selections:
# # - a brush that is active on the top panel
# # - a multi-click that is active on the bottom panel
# brush = alt.selection_interval(encodings=["x"])
# click = alt.selection_multi(encodings=["color"])

# # Top panel is scatter plot of temperature vs time
# points = (
#     alt.Chart()
#     .mark_point()
#     .encode(
#         alt.X("monthdate(date):T", title="Date"),
#         alt.Y(
#             "temp_max:Q",
#             title="Maximum Daily Temperature (C)",
#             scale=alt.Scale(domain=[-5, 40]),
#         ),
#         color=alt.condition(brush, color, alt.value("lightgray")),
#         size=alt.Size("precipitation:Q", scale=alt.Scale(range=[5, 200])),
#     )
#     .properties(width=550, height=300)
#     .add_selection(brush)
#     .transform_filter(click)
# )

# # Bottom panel is a bar chart of weather type
# bars = (
#     alt.Chart()
#     .mark_bar()
#     .encode(
#         x="count()",
#         y="weather:N",
#         color=alt.condition(click, color, alt.value("lightgray")),
#     )
#     .transform_filter(brush)
#     .properties(
#         width=550,
#     )
#     .add_selection(click)
# )

# chart = alt.vconcat(points, bars, data=source, title="Seattle Weather: 2012-2015")

# tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

# with tab1:
#     st.altair_chart(chart, theme="streamlit", use_container_width=True)
# with tab2:
#     st.altair_chart(chart, theme=None, use_container_width=True)


# if 'dummy_data' not in st.session_state.keys():
#         dummy_data = ['IND','USA','BRA','MEX','ARG']
#         st.session_state['dummy_data'] = dummy_data
#     else:
#         dummy_data = st.session_state['dummy_data']
# 
    # def get_selected_checkboxes():
    #     return [i.replace('dynamic_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_checkbox_') and st.session_state[i]]

#     def checkbox_container(data):
#         st.header('Select A country')
#         new_data = st.text_input('Enter country Code to add')
#         cols = st.columns(10)
#         if cols[0].button('Add Coutry'):
#             dummy_data.append(new_data)
#         if cols[1].button('Select All'):
#             for i in data:
#                 st.session_state['dynamic_checkbox_' + i] = True
#             st.experimental_rerun()
#         if cols[2].button('UnSelect All'):
#             for i in data:
#                 st.session_state['dynamic_checkbox_' + i] = False
#             st.experimental_rerun()
#         for i in data:
#             st.checkbox(i, key='dynamic_checkbox_' + i)
#     checkbox_container(dummy_data)

# col_count= 0
#         col1,col2,col3,col4,col5,col6 = st.columns(6)
#         for col in df.columns:
#             if col_count==0:
#                 col1.checkbox(f'{col}',value=True)
#                 col_count+=1
#             elif col_count==1:
#                 col2.checkbox(f'{col}',value=True)
#                 col_count+=1
#             elif col_count==2:
#                 col3.checkbox(f'{col}',value=True)
#                 col_count+=1
#             elif col_count==3:
#                 col4.checkbox(f'{col}',value=True)
#                 col_count+=1
#             elif col_count==4:
#                 col5.checkbox(f'{col}',value=True)
#                 col_count+=1
#             elif col_count==5:
#                 col6.checkbox(f'{col}',value=True)
#                 col_count= 0

# col1,col2,col3 = st.columns([15,1,1])
#         add_tabs =  col2.button("➕") 
#         if add_tabs and st.session_state['tab_1']<5:
#             st.session_state['tab_1']+=1  
#         sub_tabs =  col3.button("➖") 
#         if sub_tabs and st.session_state['tab_1']>2:
#             st.session_state['tab_1']-=1  
