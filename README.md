# Globlal-weather-metrics
Astro-Meteorological Data Pipeline & Predictive ETL Framework
Overview
A high-performance Python-based ETL pipeline designed to synthesize heterogeneous data sources—including Earth observation telemetry (ERA5), celestial mechanics (Skyfield), and spatial regional masks—into a unified, analysis-ready dataset. This framework is engineered for scale, utilizing multiprocessing to handle large-scale GRIB file decoding and complex feature engineering for predictive modeling.

Core Engineering Features
Multiprocessing ETL Engine: Implements Python’s multiprocessing library to parallelize GRIB file extraction and data transformation, significantly reducing processing latency for high-resolution climate data.
Celestial Feature Engineering: Leverages skyfield and ecliptic_frame to calculate precise solar and planetary positions, identifying astronomical seasons and keystone dates that drive climatic variance.
Heterogeneous Data Fusion: Programmatically merges multi-format data streams, including .grib, .csv, and SQL-relational data using SQLAlchemy and xarray.
Advanced Spatial Masking: Utilizes region_mask and xarray for vectorized spatial indexing, mapping global climate coordinates to specific political and geographical boundaries.
Fault-Tolerant Processing: Built with robust error-handling and automated resource cleanup (OS-level file management) to ensure pipeline stability during 80+ year historical data loops.

Technical Stack
Core Logic: Python (OOP, Vectorized operations).
Data Processing: Pandas, NumPy, xarray, cfgrib.
Physics & Geometry: Skyfield, Scipy.stats (Skewness, Kurtosis, Z-score).
Infrastructure: SQLAlchemy for persistence and multiprocessing for compute optimization.

