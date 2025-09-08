#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import os
from sqlalchemy import create_engine
import logging
import time

logging.basicConfig(
    filename="logs/ingestion_db.log",
    level=logging.DEBUG,
    format="%(asctime)s -%(levelname)s -%(message)s",
    filemode="a"
)

engine = create_engine('sqlite:///inventory.db')

def ingest_db(df, table_name, engine):
    df.to_sql(table_name, con=engine, if_exists = 'replace', index=False)
    
def load_raw_data():
    start = time.time()
    for file in os.listdir('data'):
        if '.csv' in file:
            df = pd.read_csv('data/'+file)
            logging.info(f'Ingesting {file} in database')
            ingest_db(df, file[:-4], engine)
    end = time.time()
    total_time = (end-start)/60
    logging.info('--------Ingestion Complete--------')
    
    logging.info(f'\nTotal Time Taken: {total_time} minutes')
    
if __name__ == '__main__':
    load_raw_data()


# Exploratory Data Analysis:
# 
# Understanding the dataset to explore how the data is present in the databse and if there is a need of creating 
# aggregated tables that can help with:
# 1. Vendor Selection 
# 2. Product Pricing Optimization

# In[6]:


import pandas as pd
import sqlite3


# In[7]:


#Create database connection
conn = sqlite3.connect('inventory.db')


# In[8]:


#Check tables present in database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type = 'table'",conn)
tables


# In[15]:


for table in tables['name']:
    print('-'*50, f'{table}', '-'*50)
    print('Count of records:', pd.read_sql(f"select count(*) as count from {table}", conn)['count'].values[0])
    display(pd.read_sql(f"select * from {table} limit 5", conn))


# In[17]:


purchases = pd.read_sql_query("SELECT * FROM purchases WHERE VendorNumber =4466",conn)
purchases


# In[18]:


purchase_prices = pd.read_sql_query("SELECT * FROM purchase_prices WHERE VendorNumber =4466",conn)
purchase_prices


# In[19]:


vendor_invoice = pd.read_sql_query("SELECT * FROM vendor_invoice WHERE VendorNumber =4466",conn)
vendor_invoice


# In[22]:


sales = pd.read_sql_query("SELECT * FROM sales WHERE VendorNo =4466",conn)
sales


# In[23]:


purchases.groupby(['Brand','PurchasePrice'])[['Quantity','Dollars']].sum()


# In[24]:


purchase_prices


# In[26]:


vendor_invoice['PONumber'].nunique()


# In[30]:


vendor_invoice.columns


# In[28]:


purchases


# In[33]:


sales.groupby('Brand')[['SalesDollars','SalesPrice','SalesQuantity']].sum()


# In[34]:


purchases.groupby(['Brand','PurchasePrice'])[['Quantity','Dollars']].sum()


# The purchases table contains actual purchase data, including the date of purchase, products (brands) purchased by vendors, the amount paid (in dollars), and the quantity purchased.
# 
# 1. The purchase price column is derived from the purchase_prices table, which provides product-wise actual and purchase prices. The combination of vendor and brand is unique in this table.
# 
# 2. The vendor_invoice table aggregates data from the purchases table, summarizing quantity and dollar amounts, along with an additional column for freight. This table maintains uniqueness based on vendor and PO number.
# 
# 3. The sales table captures actual sales transactions, detailing the brands purchased by vendors, the quantity sold, the selling price, and the revenue earned.

# As the data that we need for analysis is distributed in different tables, we need to create a summary table containing:
# 
# 1. purchase transactions made by vendors
# 
# 2. sales transaction data
# 
# 3. freight costs for each vendor
# 
# 4. actual product prices from vendors

# In[35]:


vendor_invoice.columns


# In[38]:


freight_summary = pd.read_sql_query("""SELECT VendorNumber, SUM(Freight) as FreightCost
FROM vendor_invoice
GROUP BY VendorNumber""",conn)
freight_summary


# In[42]:


pd.read_sql_query("""SELECT 
    p.VendorNumber,
    p.VendorName,
    p.Brand,
    p.PurchasePrice,
    pp.Volume,
    pp.Price AS ActualPrice,
    SUM(p.Quantity) as TotalPurchseQuantity,
    SUM(p.Dollars) as TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp
    ON p.Brand = pp.Brand
    WHERE p.PurchasePrice>0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand
    ORDER BY TotalPurchaseDollars""", conn)


# In[43]:


sales.columns


# In[44]:


pd.read_sql_query("""SELECT
    VendorNo,
    Brand,
    SUM(SalesDollars) AS TotalSalesDollars,
    SUM(SalesPrice) AS TotalSalesPrice,
    SUM(SalesQuantity) AS TotalSalesQuantity,
    SUM(ExciseTax) AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo,Brand
    ORDER BY TotalSalesDollars""",conn)


# In[47]:


vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS (
    SELECT
        VendorNumber,
        SUM(Freight) AS FreightCost
    FROM vendor_invoice
    GROUP BY VendorNumber
),

PurchaseSummary AS (
    SELECT
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.Description,
        p.PurchasePrice,
        pp.Price AS ActualPrice,
        pp.Volume,
        SUM(p.Quantity) AS TotalPurchaseQuantity,
        SUM(p.Dollars) AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp
        ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, 
             p.PurchasePrice, pp.Price, pp.Volume
),

SalesSummary AS (
    SELECT
        VendorNo,
        Brand,
        SUM(SalesQuantity) AS TotalSalesQuantity,
        SUM(SalesDollars) AS TotalSalesDollars,
        SUM(SalesPrice) AS TotalSalesPrice,
        SUM(ExciseTax) AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo, Brand
)

SELECT
    ps.VendorNumber,
    ps.VendorName,
    ps.Brand,
    ps.Description,
    ps.PurchasePrice,
    ps.ActualPrice,
    ps.Volume,
    ps.TotalPurchaseQuantity,
    ps.TotalPurchaseDollars,
    ss.TotalSalesQuantity,
    ss.TotalSalesDollars,
    ss.TotalSalesPrice,
    ss.TotalExciseTax,
    fs.FreightCost
FROM PurchaseSummary ps
LEFT JOIN SalesSummary ss
    ON ps.VendorNumber = ss.VendorNo
    AND ps.Brand = ss.Brand
LEFT JOIN FreightSummary fs
    ON ps.VendorNumber = fs.VendorNumber
ORDER BY ps.TotalPurchaseDollars DESC""",conn)


# In[48]:


vendor_sales_summary


# This query generates a vendor-wise sales and purchase summary which is valuable for:
# 
# **Performance Optimization**:
# 
# 1. The query involves heavy joins and aggregations on large datasets like sales and purchases.
# 
# 2. Storing the pre-aggregated results avoids repeated expensive computations.
# 
# 3. Helps in analyzing sales, purchases, and pricing for different vendors and brands.
# 
# 4. Future Benefits of Storing this data for faster Dashboarding & Reporting.
# 
# 5. Instead of running expensive queries each time, dashboards can fetch data quickly from vendor_sales_summary.

# In[59]:


vendor_sales_summary.dtypes


# In[58]:


vendor_sales_summary.isnull().sum()


# In[57]:


vendor_sales_summary['VendorName'].unique()


# In[53]:


vendor_sales_summary['Description'].unique()


# In[54]:


vendor_sales_summary['Volume'] = vendor_sales_summary['Volume'].astype('float64')


# In[55]:


vendor_sales_summary.fillna(0, inplace=True)


# In[56]:


vendor_sales_summary['VendorName'] = vendor_sales_summary['VendorName'].str.strip()


# In[62]:


vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']


# In[63]:


vendor_sales_summary['GrossProfit'].min()


# In[76]:


vendor_sales_summary['ProfitMargin'] = (vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars'])*100


# In[65]:


vendor_sales_summary['StockTurnover'] = vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']


# In[66]:


vendor_sales_summary['SalesToPurchaseRation'] = vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']


# In[69]:


vendor_sales_summary.columns


# In[67]:


cursor = conn.cursor()


# In[70]:


cursor.execute("""CREATE TABLE vendor_sales_summary (
   VendorNumber INT,
   VendorName VARCHAR(100),
   Brand INT,
   Description VARCHAR(100),
   PurchasePrice DECIMAL(10,2),
   ActualPrice DECIMAL(10,2),
   Volume,
   TotalPurchaseQuantity INT,
   TotalPurchaseDollars DECIMAL(15,2),
   TotalSalesQuantity INT,
   TotalSalesDollars DECIMAL(15,2),
   TotalSalesPrice DECIMAL(15,2),
   TotalExciseTax DECIMAL(15,2),
   FreightCost DECIMAL(15,2),
   GrossProfit DECIMAL(15,2),
   ProfitMargin DECIMAL(15,2),
   StockTurnover DECIMAL(15,2),
   SalesToPurchaseRatio DECIMAL(15,2),
   PRIMARY KEY (VendorNumber, Brand)
);
""")


# In[73]:


pd.read_sql("SELECT * FROM vendor_sales_summary",conn)


# In[72]:


vendor_sales_summary.to_sql('vendor_sales_summary',conn, if_exists = 'replace', index = False)


# In[78]:


import sqlite3
import pandas as pd
import logging
from ingestion_db import ingest_db
import time 

logging.basicConfig(
    filename="logs/get_vendor_summary.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

def create_vendor_summary(conn):
    '''this function will merge the different tables to get the overall vendor summary and adding new columns in the resultant data'''
    vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS (
            SELECT
                VendorNumber,
                SUM(Freight) AS FreightCost
            FROM vendor_invoice
            GROUP BY VendorNumber
        ), 
        
        PurchaseSummary AS (
            SELECT
            p.VendorNumber,
            p.VendorName,
            p.Brand,
            p.Description,
            p.PurchasePrice,
            pp.Price AS ActualPrice,
            pp.Volume,
            SUM(p.Quantity) AS TotalPurchaseQuantity,
            SUM(p.Dollars) AS TotalPurchaseDollars
        FROM purchases p
        JOIN purchase_prices pp
            ON p.Brand = pp.Brand
        WHERE p.PurchasePrice > 0
        GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
    ),
    
    SalesSummary AS (
        SELECT
            VendorNo,
            Brand,
            SUM(SalesQuantity) AS TotalSalesQuantity,
            SUM(SalesDollars) AS TotalSalesDollars,
            SUM(SalesPrice) AS TotalSalesPrice,
            SUM(ExciseTax) AS TotalExciseTax
        FROM sales
        GROUP BY VendorNo, Brand
    )
    
    SELECT
        ps.VendorNumber,
        ps.VendorName,
        ps.Brand,
        ps.Description,
        ps.PurchasePrice,
        ps.ActualPrice,
        ps.Volume,
        ps.TotalPurchaseQuantity,
        ps.TotalPurchaseDollars,
        ss.TotalSalesQuantity,
        ss.TotalSalesDollars,
        ss.TotalSalesPrice,
        ss.TotalExciseTax,
        fs.FreightCost
    FROM PurchaseSummary ps
    LEFT JOIN SalesSummary ss
        ON ps.VendorNumber = ss.VendorNo
       AND ps.Brand = ss.Brand
    LEFT JOIN FreightSummary fs
        ON ps.VendorNumber = fs.VendorNumber
    ORDER BY ps.TotalPurchaseDollars DESC
    """, conn)
    
    return vendor_sales_summary

def clean_data(df):
    '''this function will clean the data'''

    # changing datatype to float
    df['Volume'] = df['Volume'].astype('float')

    # filling missing value with 0
    df.fillna(0, inplace=True)

    # removing spaces from categorical columns
    df['VendorName'] = df['VendorName'].str.strip()
    df['Description'] = df['Description'].str.strip()

    # creating new columns for better analysis
    vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']
    vendor_sales_summary['ProfitMargin'] = (vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars'])*100
    vendor_sales_summary['StockTurnover'] = vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']
    vendor_sales_summary['SalesToPurchaseRation'] = vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']

    return df

if __name__ == '__main__':
    # creating database connection
    conn = sqlite3.connect('inventory.db')

    logging.info('Creating Vendor Summary Table.....')
    summary_df = create_vendor_summary(conn)
    logging.info(summary_df.head())

    logging.info('Cleaning Data.....')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())

    logging.info('Ingesting data.....')
    ingest_db(clean_df, 'vendor_sales_summary', conn)
    logging.info('Completed')


# In[80]:


import sqlite3
import pandas as pd
import logging
from ingestion_db import ingest_db
import time 

logging.basicConfig(
    filename="logs/get_vendor_summary.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

def create_vendor_summary(conn):
    '''this function will merge the different tables to get the overall vendor summary and adding new columns in the resultant data'''
    vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS (
            SELECT
                VendorNumber,
                SUM(Freight) AS FreightCost
            FROM vendor_invoice
            GROUP BY VendorNumber
        ), 
        
        PurchaseSummary AS (
            SELECT
            p.VendorNumber,
            p.VendorName,
            p.Brand,
            p.Description,
            p.PurchasePrice,
            pp.Price AS ActualPrice,
            pp.Volume,
            SUM(p.Quantity) AS TotalPurchaseQuantity,
            SUM(p.Dollars) AS TotalPurchaseDollars
        FROM purchases p
        JOIN purchase_prices pp
            ON p.Brand = pp.Brand
        WHERE p.PurchasePrice > 0
        GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
    ),
    
    SalesSummary AS (
        SELECT
            VendorNo,
            Brand,
            SUM(SalesQuantity) AS TotalSalesQuantity,
            SUM(SalesDollars) AS TotalSalesDollars,
            SUM(SalesPrice) AS TotalSalesPrice,
            SUM(ExciseTax) AS TotalExciseTax
        FROM sales
        GROUP BY VendorNo, Brand
    )
    
    SELECT
        ps.VendorNumber,
        ps.VendorName,
        ps.Brand,
        ps.Description,
        ps.PurchasePrice,
        ps.ActualPrice,
        ps.Volume,
        ps.TotalPurchaseQuantity,
        ps.TotalPurchaseDollars,
        ss.TotalSalesQuantity,
        ss.TotalSalesDollars,
        ss.TotalSalesPrice,
        ss.TotalExciseTax,
        fs.FreightCost
    FROM PurchaseSummary ps
    LEFT JOIN SalesSummary ss
        ON ps.VendorNumber = ss.VendorNo
       AND ps.Brand = ss.Brand
    LEFT JOIN FreightSummary fs
        ON ps.VendorNumber = fs.VendorNumber
    ORDER BY ps.TotalPurchaseDollars DESC
    """, conn)
    
    return vendor_sales_summary

def clean_data(df):
    '''this function will clean the data'''
    # changing datatype to float
    df['Volume'] = df['Volume'].astype('float')
    # filling missing value with 0
    df.fillna(0, inplace=True)
    # removing spaces from categorical columns
    df['VendorName'] = df['VendorName'].str.strip()
    df['Description'] = df['Description'].str.strip()
    # creating new columns for better analysis
    df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']  # Changed here
    df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars'])*100     # Changed here
    df['StockTurnover'] = df['TotalSalesQuantity'] / df['TotalPurchaseQuantity'] # Changed here
    df['SalesToPurchaseRatio'] = df['TotalSalesDollars'] / df['TotalPurchaseDollars'] # Changed here (also fixed typo: Riation → Ratio)
    return df

if __name__ == '__main__':
    # creating database connection
    conn = sqlite3.connect('inventory.db')

    logging.info('Creating Vendor Summary Table.....')
    summary_df = create_vendor_summary(conn)
    logging.info(summary_df.head())

    logging.info('Cleaning Data.....')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())

    logging.info('Ingesting data.....')
    ingest_db(clean_df, 'vendor_sales_summary', conn)
    logging.info('Completed')


# In[ ]:




