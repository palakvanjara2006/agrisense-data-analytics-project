# Week 2: Data Collection Strategy and Cleaning Plans

A strategy and cleaning plan for a simulated integrated farm-to-market agribusiness, **AgriSense Farms Pvt Ltd**, operating in Gujarat and Madhya Pradesh (wheat, rice and soybean). Builds on the Week 1 data landscape report.

## What the report covers

- Introduction to the data collection approach
- Review of public data sources (Agmarknet/OGD, CEDA, NASA POWER, IMD, Soil Health Card, DES, FAOSTAT)
- Data collection methodology and storage layout
- Data profiling plan
- Detailed cleaning plan: missing values, outliers, units and format standardisation
- Tools and step-by-step procedures
- Seasonal, geographical and market considerations
- Reliability and representativeness, challenges and solutions
- Risk management and quality assurance
- Appendices: data dictionary, sample code, cleaning log template

## Files

- `Week2_Data_Collection_and_Cleaning_Plan.docx`: the full report
- `agri_cleaning_pipeline.py`: runnable pipeline (NASA POWER collection, nine-stage mandi price cleaning, profiling, cleaning log)

## Run the demo

    pip install pandas numpy requests
    python agri_cleaning_pipeline.py --demo

## Author

Junior Data Analyst task submission, November 2026.
