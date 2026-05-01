import pandas as pd
import numpy as np

def flexible_transform(df):
    transformed = pd.DataFrame()
    
    price_cols = [c for c in df.columns if 'price' in c.lower() or 'cost' in c.lower() or 'amount' in c.lower()]
    if not price_cols: 
        return None, "Missing 'Price' column."
    
    price_val = pd.to_numeric(df[price_cols[0]], errors='coerce')
    transformed['price'] = price_val.apply(lambda x: x / 100000 if x > 10000 else x)
    
    area_cols = [c for c in df.columns if 'area' in c.lower() or 'sqft' in c.lower()]
    if not area_cols: 
        return None, "Missing 'Area/Sqft' column."
    transformed['total_sqft'] = pd.to_numeric(df[area_cols[0]], errors='coerce')
    
    size_cols = [c for c in df.columns if 'bedroom' in c.lower() or 'bhk' in c.lower() or 'size' in c.lower()]
    if size_cols:
        def format_bhk(x):
            if pd.isnull(x):
                return "2 BHK"
            if isinstance(x, str) and 'BHK' in x.upper():
                return x
            try:
                val = int(float(str(x).split(' ')[0]))
                return f"{val} BHK"
            except (ValueError, TypeError):
                return str(x)
        
        transformed['size'] = df[size_cols[0]].apply(format_bhk)
    else:
        transformed['size'] = "2 BHK"
        
    loc_cols = [c for c in df.columns if 'loc' in c.lower() or 'address' in c.lower()]
    transformed['location'] = df[loc_cols[0]].apply(lambda x: str(x).split(',')[0].strip()) if loc_cols else "Unknown"
    
    transformed['bath'] = df.get('bath', df.get('Bathrooms', 2))
    transformed['balcony'] = df.get('balcony', df.get('Balcony', 0))
    
    transformed['area_type'] = df.get('area_type', 'Super built-up Area')
    transformed['availability'] = df.get('availability', 'Ready To Move')
    transformed['society'] = df.get('society', np.nan)
    
    return transformed.dropna(subset=['total_sqft', 'price', 'size']), None