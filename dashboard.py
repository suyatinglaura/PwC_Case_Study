# import libraries
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

app = Dash(__name__)

# Part 1: Sales Performance across Category
# read orders table
orders = pd.read_csv('006_lomo_orders_dataset.csv')
# filter out unavailable and canceled orders
orders_filtered = orders[(orders['order_status']!='unavailable') & (orders['order_status']!='canceled')]
orders_filtered = orders_filtered[orders_filtered.columns[:4]]

# read order items table
order_items = pd.read_csv('007_lomo_order_items_dataset.csv')
order_items_filtered = order_items[order_items.columns.drop(['shipping_limit_date', 'freight_value'])]

# read products table
products = pd.read_csv('004_lomo_products_dataset.csv')
products_filtered = products[products.columns[:2]]

# read category name table
category_name = pd.read_csv('005_lomo_product_category_name_translation.csv')

# merge the tables
order_categories = pd.merge(pd.merge(pd.merge(orders_filtered, order_items_filtered), products_filtered), category_name, left_on='product_category_name', right_on='product_category_name_portugese')
order_categories = order_categories[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_item_id', 'product_id', 'seller_id', 'price', 'product_category_name_english']]
order_categories['order_purchase_timestamp'] = list(order_categories['order_purchase_timestamp'].map(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')))

# identify the product categories with highest total sales
sale_amount = order_categories.groupby(['product_category_name_english']).sum()[['price']]
sale_amount = sale_amount.reset_index()
top_10_sale_amount = sale_amount.sort_values(by='price', ascending=False)[:10]
top_10_sale_amount.rename(columns = {'product_category_name_english':'Product Categories', 'price':'Total Sales'}, inplace = True)
# plot the total sales graph
fig = px.bar(top_10_sale_amount, x="Product Categories", y="Total Sales", title='Top 10 Product Categories with the Highest Total Sales')
fig.update_traces(marker_color='lightsalmon')
fig.update_layout(clickmode='event+select')

# identify the product categories with the highest sales growth rate
# create date separator
start_period = min(order_categories.order_purchase_timestamp)
separator1 = start_period + relativedelta(months=+6)
separator2 = separator1 + relativedelta(months=+6)
separator3 = separator2 + relativedelta(months=+6)
# create a period column
period=[]
for date in order_categories['order_purchase_timestamp']:
    if date > separator3:
        period.append(3)
    elif date > separator2:
        period.append(2)
    elif date > separator1:
        period.append(1)
    else:
        period.append(0)
order_categories['period'] = period
# compute the sales growth rate
sale_amount_dev = order_categories.groupby(['product_category_name_english', 'period']).sum()[['price']]
sale_amount_dev = sale_amount_dev.reset_index()
growth_rate = pd.DataFrame({'Product Categories':[], 'Period 1':[], 'Period 2':[], 'Period 3':[], 'Period 4':[], 'Average Growth Rate':[]})
categories = sale_amount_dev.product_category_name_english.unique()
for cat in categories:
    grow_check=True
    periods = sale_amount_dev[sale_amount_dev['product_category_name_english']==cat].period.values
    sales = sale_amount_dev[sale_amount_dev['product_category_name_english']==cat].price.values
    sales_growth = []
    prev = sales[0]
    # compute sales growth rate for each category
    for i in range(1, len(sales)):
        cur = sales[i]
        if cur < prev:
            grow_check=False
            break
        sales_growth.append((cur-prev)/prev)
        prev = cur
    # fill missing values with 0
    for i in range(4):
        if i not in periods:
            periods=np.insert(periods, i, i)
            sales=np.insert(sales, i, 0)
    if grow_check:
        growth_rate.loc[len(growth_rate.index)] = [cat]+list(sales)+[np.mean(sales_growth)] 
top_10_growth_rate = growth_rate.sort_values(by='Average Growth Rate', ascending=False)[:10]
# plot the sales growth rate graph
fig2 = px.bar(top_10_growth_rate, x="Product Categories", y="Average Growth Rate", title='Top 10 Product Categories with the Highest Average Growth Rate')
fig2.update_traces(marker_color='lightsalmon')
fig2.update_layout(clickmode='event+select')

# plot the sales trend for the category with the highest potential
top_growth_rate = top_10_growth_rate['Product Categories'].values[0]
fig3_data = growth_rate[growth_rate['Product Categories']==top_growth_rate].iloc[0][1:5]
labels = ['Period 1','Period 2','Period 3','Period 4']
fig3 = px.line(x=labels, y=fig3_data, title=f'Sales Trend of {top_growth_rate}')
fig3.update_layout(xaxis_title='Period', yaxis_title='Period Total Sales')
fig3.update_traces(textposition="bottom right")

# Part 2: Sales Performance across State
# read the customers table
customers = pd.read_csv('001_lomo_customers_dataset.csv')

# get the number of converted customers in each state
order_customers = pd.merge(order_categories, customers)
state_customers = order_customers.groupby('customer_state').nunique()['customer_unique_id'].reset_index()

# combine with the latitude and longitude of each state 
th = pd.read_csv('thailand.csv') # download from https://simplemaps.com/data/th-cities
state_customers = pd.merge(state_customers, th, left_on='customer_state', right_on='city')
state_customers.rename(columns = {'customer_unique_id':'Number of Customers', 'customer_state': 'Customer State'}, inplace = True)

# plot the state customer number graph
fig4 = px.scatter_geo(state_customers, lat=state_customers['lat'],lon=state_customers['lng'], color=state_customers['Number of Customers'], size=state_customers['Number of Customers'], 
                      hover_data={'lat': False, 'lng': False, 'Customer State': True})
fig4.update_geos(lonaxis_range= [min(th['lng']), max(th['lng'])], lataxis_range= [min(th['lat']), max(th['lat'])])
fig4.update_layout(clickmode='event+select')

# plot the state average spending graph
state_customers['Average Spending'] = list(map(lambda x: round(x, 2), order_customers.groupby('customer_state').sum()['price'].values/state_customers['Number of Customers'].values))
fig5 = px.scatter_geo(state_customers, lat=state_customers['lat'],lon=state_customers['lng'], color=state_customers['Average Spending'], size=state_customers['Average Spending'], size_max=10,
                     hover_data={'lat': False, 'lng': False, 'Average Spending': True, 'Customer State': True})
fig5.update_geos(lonaxis_range= [min(th['lng']), max(th['lng'])], lataxis_range= [min(th['lat']), max(th['lat'])])

# plot the state conversion rate graph
state_customers['Conversion Rate'] = list(map(lambda x: round(x, 4), state_customers['Number of Customers'].values/customers.groupby('customer_state').nunique()['customer_unique_id'].values))
fig6 = px.scatter_geo(state_customers, lat=state_customers['lat'],lon=state_customers['lng'], color=state_customers['Conversion Rate'], size=state_customers['Conversion Rate'], size_max=10,
                    hover_data={'lat': False, 'lng': False, 'Conversion Rate': True, 'Customer State': True})
fig6.update_geos(lonaxis_range= [min(th['lng']), max(th['lng'])], lataxis_range= [min(th['lat']), max(th['lat'])])

app.layout = html.Div(children=[
    html.H1(children='Sales Performance across Category'),

    dcc.Graph(
        id='best-performance',
        figure=fig
    ),
    
    html.Div(className='row', children=[
    html.Div(dcc.Graph(
        id='highest-potential',
        figure=fig2
    ),style={'width': '65%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(
        id='sales-change',
        figure={}
    ),style={'width': '35%', 'display': 'inline-block'})]),
    

    html.H1(children='Sales Performance across State'),

    html.Div(className='row', children=[
    html.Div(dcc.Graph(
        id='state-customers',
        figure=fig4
    ),style={'width': '30%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(
        id='state-spending',
        figure=fig5
    ),style={'width': '30%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(
        id='state-conversion',
        figure=fig6
    ),style={'width': '30%', 'display': 'inline-block'})])
])

# update the sales trend graph for selected category
@app.callback(
    Output('sales-change', 'figure'),
    Input('highest-potential', 'clickData'))
def update_sales_change(clickData):
    if clickData:
        # get the selected category
        clicked_cat = clickData['points'][0]['x']
        data = growth_rate[growth_rate['Product Categories']==clicked_cat].iloc[0][1:5]
        # plot the sales trend graph for the selected category
        labels = ['Period 1','Period 2','Period 3','Period 4']
        fig = px.line(x=labels, y=data, title=f'Sales Trend of {clicked_cat}')
        fig.update_layout(xaxis_title='Period', yaxis_title='Period Total Sales')
        fig.update_traces(textposition="bottom right")
        return fig
    else:
        # otherwise, display the sales trend graph of the category with the highest growth rate
        return fig3

# update the state spending and conversion plots for selected state
@app.callback(
    Output('state-spending', 'figure'),
    Output('state-conversion', 'figure'),
    Input('state-customers', 'clickData'))
def update_state_graph(clickData):
    if clickData:
        # get the selected state
        clicked_state = clickData['points'][0]['customdata']
        # annotate the average spending graph for the selected state
        fig1 = px.scatter_geo(state_customers, lat=state_customers['lat'],lon=state_customers['lng'], color=state_customers['Average Spending'], size=state_customers['Average Spending'], size_max=10,
                        hover_data={'lat': False, 'lng': False, 'Average Spending': True, 'Customer State': True})
        fig1.update_geos(lonaxis_range= [min(th['lng']), max(th['lng'])], lataxis_range= [min(th['lat']), max(th['lat'])])
        fig1.add_trace(go.Scattergeo(lat=[clicked_state[0]], lon=[clicked_state[1]], mode="text", text=[clicked_state[2]], textfont={"family":["Arial Black"]}, showlegend=False))
        # annotate the sales conversion rate for the selected state
        fig2 = px.scatter_geo(state_customers, lat=state_customers['lat'],lon=state_customers['lng'], color=state_customers['Conversion Rate'], size=state_customers['Conversion Rate'], size_max=10,
                        hover_data={'lat': False, 'lng': False, 'Conversion Rate': True, 'Customer State': True})
        fig2.update_geos(lonaxis_range= [min(th['lng']), max(th['lng'])], lataxis_range= [min(th['lat']), max(th['lat'])])
        fig2.add_trace(go.Scattergeo(lat=[clicked_state[0]], lon=[clicked_state[1]], mode="text", text=[clicked_state[2]], textfont={"family":["Arial Black"]}, showlegend=False))
        return fig1, fig2
    else:
        # otherwise, display the graphs without annotation
        return fig5, fig6

if __name__ == '__main__':
    app.run_server(debug=True)