import pandas as pd
import numpy as np

def transform_to_standard_format(input_file, output_file='Formatted_Dataset.csv'):
    """
    Transforms house price datasets into the standard format.
    Standard Headers: location, size, total_sqft, bath, balcony, price
    """
    df = pd.read_csv(input_file)
    
    # Map Price and Scale to Lakhs (if raw rupees are detected)
    price_cols = [c for c in df.columns if 'price' in c.lower() or 'cost' in c.lower()]
    if price_cols:
        price_val = pd.to_numeric(df[price_cols[0]], errors='coerce')
        df['price'] = price_val.apply(lambda x: x / 100000 if x > 10000 else x)
    
    # Map Area
    area_cols = [c for c in df.columns if 'area' in c.lower() or 'sqft' in c.lower()]
    df['total_sqft'] = pd.to_numeric(df[area_cols[0]], errors='coerce') if area_cols else np.nan
            
    # Map Bedrooms to 'BHK' format (e.g., 3 -> '3 BHK')
    size_cols = [c for c in df.columns if 'bedroom' in c.lower() or 'bhk' in c.lower()]
    if size_cols:
        df['size'] = df[size_cols[0]].apply(lambda x: f"{int(float(x))} BHK" if pd.notnull(x) else x)
            
    # Map Location
    loc_cols = [c for c in df.columns if 'loc' in c.lower() or 'address' in c.lower()]
    if loc_cols:
        df['location'] = df[loc_cols[0]].apply(lambda x: str(x).split(',')[0].strip())
            
    target_cols = ['location', 'size', 'total_sqft', 'bath', 'balcony', 'price']
    df[target_cols].to_csv(output_file, index=False)
    print(f"File saved as {output_file}")