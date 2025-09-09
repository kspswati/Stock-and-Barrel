# Stock & Barrel 📊
*Inventory Management & Vendor Performance Analysis System*

*Link for the Dashboard:* https://lookerstudio.google.com/s/tROmcEHsVd8

## About the Project

Stock & Barrel is a comprehensive data analysis project designed to optimize inventory management and vendor performance through advanced analytics. The system processes multiple data sources including purchases, sales, vendor invoices, and pricing data to provide actionable business insights for strategic decision-making.

## Problem Statement

Effective inventory and sales management are critical for profitability. Companies risk incurring losses due to:

- **Inefficient pricing** - Products may be priced too high or too low, affecting competitiveness and margins
- **Poor inventory turnover** - Capital locked in slow-moving inventory reduces cash flow
- **Vendor dependency** - Over-reliance on specific vendors creates supply chain risks
- **Suboptimal procurement** - Lack of bulk purchasing strategies increases unit costs

## Solution

Our solution provides a comprehensive analytics framework that:

1. **Data Integration**: Consolidates data from multiple sources (purchases, sales, vendor invoices, pricing)
2. **Automated ETL Pipeline**: Processes and cleans data with automated ingestion and transformation
3. **Performance Analytics**: Creates aggregated summary tables for fast querying and analysis
4. **Statistical Analysis**: Performs hypothesis testing and confidence interval analysis
5. **Visualization Dashboard**: Generates insightful charts and graphs for decision-making

### Key Components

- **Data Ingestion Module** (`ingestion_db.py`): Automated CSV-to-SQLite database loading
- **Exploratory Data Analysis** (`Exploratory Data Analysis.py`): Data exploration and summary table creation
- **Vendor Performance Analysis** (`Vendor Performance Analysis.py`): Advanced analytics and statistical testing

## Key Findings

### 🎯 Business Performance Metrics
- **Top 10 vendors contribute 85.5%** of total procurement spending
- **Average profit margin difference**: Low-performing vendors (41.55%) vs Top-performing vendors (31.18%)
- **Bulk purchasing impact**: Up to **72% reduction** in unit costs for large orders
- **Capital efficiency**: Identified vendors with slow inventory turnover requiring attention

### 📈 Vendor Performance Insights
- **High-margin, low-sales products** identified for promotional campaigns
- **Inventory turnover analysis** revealed products with stock turnover < 1
- **Purchase volume correlation** confirmed bulk buying advantages
- **Statistical significance** proven between vendor performance categories (p < 0.05)

## Business Insights

### 🔍 Strategic Recommendations

1. **Vendor Diversification**
   - Reduce dependency risk by expanding vendor base beyond top 10
   - Current top vendors control 85.5% of procurement spend

2. **Pricing Optimization**
   - Target high-margin, low-sales brands for promotional strategies
   - Implement dynamic pricing based on inventory turnover rates

3. **Inventory Management**
   - Focus on vendors with stock turnover < 1 for inventory reduction
   - Implement just-in-time procurement for slow-moving items

4. **Procurement Strategy**
   - Encourage bulk purchasing to achieve up to 72% unit cost savings
   - Negotiate volume discounts with key suppliers

## Visual Insights

The analysis includes comprehensive visualizations:

- **📊 Vendor Contribution Analysis**: Pareto charts showing vendor purchase contributions
- **📈 Performance Scatter Plots**: Brand performance matrix (sales vs. profit margin)
- **📉 Inventory Turnover Analysis**: Identification of slow-moving inventory
- **🎯 Confidence Interval Comparisons**: Statistical validation of vendor performance differences
- **💰 Cost-Volume Relationships**: Bulk purchasing impact visualization

## Key Results

### Performance Metrics
| Metric | Value | Insight |
|--------|--------|---------|
| **Vendor Concentration** | 85.5% (Top 10) | High dependency risk |
| **Average Profit Margin Gap** | 10.37% | Significant performance variance |
| **Bulk Purchase Savings** | Up to 72% | Strong volume discount opportunity |
| **Statistical Significance** | p < 0.05 | Proven vendor performance differences |

### Data Quality Improvements
- ✅ Automated data cleaning and validation
- ✅ Elimination of negative profit transactions
- ✅ Standardized vendor name formatting
- ✅ Comprehensive null value handling


### Key Features
- **SQLite Database**: Efficient local data storage and querying
- **Pandas Integration**: Advanced data manipulation and analysis
- **Statistical Testing**: Scipy-based hypothesis testing
- **Visualization**: Matplotlib and Seaborn for comprehensive charts
- **Automated Logging**: Comprehensive process tracking and error handling

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy sqlalchemy
```

### Running the Analysis
```bash
# 1. Data Ingestion
python ingestion_db.py

# 2. Exploratory Analysis
python "Exploratory Data Analysis.py"

# 3. Vendor Performance Analysis  
python "Vendor Performance Analysis.py"
```

### Data Requirements
- **purchases.csv**: Transaction-level purchase data
- **sales.csv**: Sales transaction records
- **vendor_invoice.csv**: Invoice and freight information
- **purchase_prices.csv**: Product pricing data

## Project Structure
```
stock-barrel/
│
├── data/                          # CSV data files
├── logs/                          # Process logs
├── ingestion_db.py               # Data loading module
├── Exploratory Data Analysis.py  # EDA and data preparation
├── Vendor Performance Analysis.py # Advanced analytics
├── inventory.db                  # SQLite database
└── vendor_sales_summary_final.csv # Final analysis output
```

## Future Enhancements

- 🔮 **Predictive Analytics**: Demand forecasting models
- 📱 **Real-time Dashboard**: Interactive web-based reporting
- 🤖 **Automated Alerts**: Performance threshold notifications  
- 📊 **Advanced ML Models**: Customer segmentation and recommendation engines
- 🔄 **API Integration**: Real-time data pipeline connections

## Contributing

Please feel free to submit issues, feature requests, or pull requests to help improve Stock & Barrel.

---

*Built with Python, Pandas, SQLite, and advanced statistical analysis techniques for data-driven inventory management decisions.*
