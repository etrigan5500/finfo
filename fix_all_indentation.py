import json
import re

# Read the notebook
with open('training/notebooks/train_models.ipynb', 'r') as f:
    nb = json.load(f)

# Get cell 11 content
cell_content = ''.join(nb['cells'][11]['source'])

# Fix all the indentation issues systematically
lines = cell_content.split('\n')
fixed_lines = []

# Track context for proper indentation
in_get_real_news = False
in_cluster_articles = False
in_collect_training = False

for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Fix search_queries list indentation
    if stripped.startswith('search_queries = ['):
        fixed_lines.append('            search_queries = [')
    elif stripped == 'f"{company_name} financial news",':
        fixed_lines.append('                f"{company_name} financial news",')
    elif stripped == 'f"{company_name} stock prediction news"':
        fixed_lines.append('                f"{company_name} stock prediction news"')
    elif stripped == ']' and i > 0 and 'search_queries' in lines[i-3]:
        fixed_lines.append('            ]')
    
    # Fix if len(all_articles) >= 6 block
    elif stripped == 'if len(all_articles) >= 6:  # Need at least 6 articles for 3 clusters':
        fixed_lines.append('            if len(all_articles) >= 6:  # Need at least 6 articles for 3 clusters')
    elif stripped.startswith('return self._cluster_articles'):
        fixed_lines.append('                return self._cluster_articles(all_articles, num_clusters=3, articles_per_cluster=num_articles//3)')
    elif stripped == 'else:' and i > 0 and 'if len(all_articles) >= 6' in lines[i-1]:
        fixed_lines.append('            else:')
    elif stripped == '# Return what we have if not enough for clustering':
        fixed_lines.append('                # Return what we have if not enough for clustering')
    elif stripped == 'return all_articles[:num_articles]' and 'Return what we have' in lines[i-1]:
        fixed_lines.append('                return all_articles[:num_articles]')
    
    # Fix _cluster_articles method
    elif stripped == 'for cluster_id in range(num_clusters):':
        fixed_lines.append('            for cluster_id in range(num_clusters):')
    elif stripped == 'cluster_indices = np.where(cluster_labels == cluster_id)[0]':
        fixed_lines.append('                cluster_indices = np.where(cluster_labels == cluster_id)[0]')
    elif stripped == 'if len(cluster_indices) > 0:':
        fixed_lines.append('                if len(cluster_indices) > 0:')
    elif stripped.startswith('for idx in sorted_indices[:articles_per_cluster]:'):
        fixed_lines.append('                        for idx in sorted_indices[:articles_per_cluster]:')
    elif stripped == 'clustered_articles.append(articles[idx])':
        fixed_lines.append('                            clustered_articles.append(articles[idx])')
    
    # Fix collect_training_data_efficient indentation issues
    elif stripped == 'if len(hist) < 10:  # Need at least 10 data points':
        fixed_lines.append('            if len(hist) < 10:  # Need at least 10 data points')
    elif stripped == 'print(f"Insufficient data for {ticker}, skipping...")' and 'if len(hist) < 10' in cell_content[max(0, cell_content.rfind('\n', 0, cell_content.find(line))-100):cell_content.find(line)]:
        fixed_lines.append('                print(f"Insufficient data for {ticker}, skipping...")')
    elif stripped == 'continue' and i > 0 and 'Insufficient data' in lines[i-1]:
        fixed_lines.append('                continue')
    
    elif stripped == 'if interval != resample_period and resample_period != \'no_resample\':':
        fixed_lines.append('            if interval != resample_period and resample_period != \'no_resample\':')
    elif stripped == 'hist_resampled = hist.resample(resample_period).agg({':
        fixed_lines.append('                hist_resampled = hist.resample(resample_period).agg({')
    elif stripped == "'Open': 'first',":
        fixed_lines.append('                    \'Open\': \'first\',')
    elif stripped == "'High': 'max',":
        fixed_lines.append('                    \'High\': \'max\',')
    elif stripped == "'Low': 'min',":
        fixed_lines.append('                    \'Low\': \'min\',')
    elif stripped == "'Close': 'last',":
        fixed_lines.append('                    \'Close\': \'last\',')
    elif stripped == "'Volume': 'sum'":
        fixed_lines.append('                    \'Volume\': \'sum\'')
    elif stripped == '}).dropna()':
        fixed_lines.append('                }).dropna()')
    elif stripped == 'else:' and i > 0 and 'resample_period' in lines[i-1]:
        fixed_lines.append('            else:')
    elif stripped == 'hist_resampled = hist.copy()':
        fixed_lines.append('                hist_resampled = hist.copy()')
    
    elif stripped == 'if len(hist_resampled) < sequence_length + 1:':
        fixed_lines.append('            if len(hist_resampled) < sequence_length + 1:')
    elif stripped == 'print(f"Insufficient resampled data for {ticker} ({len(hist_resampled)} points), skipping...")':
        fixed_lines.append('                print(f"Insufficient resampled data for {ticker} ({len(hist_resampled)} points), skipping...")')
    
    elif stripped == 'try:' and i > 0 and 'Get financial data' in lines[i-1]:
        fixed_lines.append('            try:')
    elif stripped == 'info = stock.info':
        fixed_lines.append('                info = stock.info')
    elif stripped == 'except:' and i > 0 and 'info = stock.info' in lines[i-1]:
        fixed_lines.append('            except:')
    elif stripped == 'info = {}':
        fixed_lines.append('                info = {}')
    
    elif stripped == 'if available_samples > num_samples_per_ticker:':
        fixed_lines.append('            if available_samples > num_samples_per_ticker:')
    elif stripped == 'sample_indices = np.random.choice(available_samples, num_samples_per_ticker, replace=False)':
        fixed_lines.append('                sample_indices = np.random.choice(available_samples, num_samples_per_ticker, replace=False)')
    elif stripped == 'sample_indices = sorted(sample_indices + sequence_length)  # Adjust for sequence length':
        fixed_lines.append('                sample_indices = sorted(sample_indices + sequence_length)  # Adjust for sequence length')
    elif stripped == 'else:' and i > 0 and 'available_samples > num_samples_per_ticker' in lines[i-1]:
        fixed_lines.append('            else:')
    elif stripped == 'sample_indices = range(sequence_length, len(hist_resampled) - 1)':
        fixed_lines.append('                sample_indices = range(sequence_length, len(hist_resampled) - 1)')
    
    elif stripped == 'for i in sample_indices:':
        fixed_lines.append('            for i in sample_indices:')
    elif stripped == 'if not np.isnan(hist_resampled[\'Future_Return\'].iloc[i]) and sample_count < num_samples_per_ticker:':
        fixed_lines.append('                if not np.isnan(hist_resampled[\'Future_Return\'].iloc[i]) and sample_count < num_samples_per_ticker:')
    
    # Default case - keep the line as is
    else:
        fixed_lines.append(line)

# Join the fixed content
fixed_content = '\n'.join(fixed_lines)

# Update the cell
nb['cells'][11]['source'] = fixed_content.split('\n')
if nb['cells'][11]['source'][-1] == '':
    nb['cells'][11]['source'] = nb['cells'][11]['source'][:-1]

# Add newlines back
nb['cells'][11]['source'] = [line + '\n' for line in nb['cells'][11]['source']]
if nb['cells'][11]['source']:
    nb['cells'][11]['source'][-1] = nb['cells'][11]['source'][-1].rstrip('\n')

# Write back
with open('training/notebooks/train_models.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed all indentation issues!') 