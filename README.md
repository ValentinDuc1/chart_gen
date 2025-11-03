[README.md](https://github.com/user-attachments/files/23308120/README.md)
# Chart Generator for Automated Report Generation

A Python desktop application for generating charts with matching JSON metadata files. Perfect for automated report generation workflows.

## Features

- **Classic Chart Types**: Line, Bar, Horizontal Bar, Pie, and Scatter plots
- **Paired Output**: Each chart generates both a PNG image and JSON metadata file with matching filenames
- **Extensible Design**: Easy to add new chart types
- **Batch Processing**: Generate multiple charts at once
- **Customizable**: Full control over titles, labels, colors, and styling
- **High Quality**: 300 DPI output by default for professional reports

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from chart_generator import ChartGenerator

# Initialize
generator = ChartGenerator(output_dir="./my_charts")

# Generate a simple bar chart
data = {
    'x': ['Q1', 'Q2', 'Q3', 'Q4'],
    'y': [100, 120, 115, 140]
}

png_path, json_path = generator.generate_chart(
    chart_type='bar',
    data=data,
    filename_root='quarterly_sales',
    title='Quarterly Sales 2024',
    xlabel='Quarter',
    ylabel='Sales ($K)'
)

print(f"Chart saved to: {png_path}")
print(f"Metadata saved to: {json_path}")
```

## Supported Chart Types

### 1. Line Chart

```python
# Single line
data = {
    'x': ['Jan', 'Feb', 'Mar', 'Apr'],
    'y': [10, 15, 13, 17]
}

# Multiple lines
data = {
    'x': ['Jan', 'Feb', 'Mar', 'Apr'],
    'y': [[10, 15, 13, 17], [8, 12, 11, 14]],
    'labels': ['Series 1', 'Series 2']
}

generator.generate_chart(
    chart_type='line',
    data=data,
    filename_root='my_line_chart',
    title='My Line Chart',
    xlabel='Time',
    ylabel='Value'
)
```

### 2. Bar Chart (Vertical)

```python
data = {
    'x': ['A', 'B', 'C', 'D'],
    'y': [25, 40, 30, 50]
}

generator.generate_chart(
    chart_type='bar',
    data=data,
    filename_root='my_bar_chart',
    title='My Bar Chart',
    xlabel='Category',
    ylabel='Value',
    colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
)
```

### 3. Horizontal Bar Chart

```python
data = {
    'categories': ['Category A', 'Category B', 'Category C'],
    'values': [45, 38, 52]
}

generator.generate_chart(
    chart_type='horizontal_bar',
    data=data,
    filename_root='my_hbar_chart',
    title='My Horizontal Bar Chart',
    xlabel='Value',
    ylabel='Category'
)
```

### 4. Pie Chart

```python
data = {
    'labels': ['Slice A', 'Slice B', 'Slice C'],
    'values': [30, 45, 25],
    'explode': [0.1, 0, 0]  # Optional: explode first slice
}

generator.generate_chart(
    chart_type='pie',
    data=data,
    filename_root='my_pie_chart',
    title='My Pie Chart',
    colors=['#FF9999', '#66B2FF', '#99FF99']
)
```

### 5. Scatter Plot

```python
data = {
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 5, 7, 9],
    'sizes': [50, 100, 75, 150, 125],  # Optional: point sizes
    'colors': 'blue'  # Optional: point colors
}

generator.generate_chart(
    chart_type='scatter',
    data=data,
    filename_root='my_scatter',
    title='My Scatter Plot',
    xlabel='X Axis',
    ylabel='Y Axis'
)
```

## Advanced Features

### Adding Metadata

```python
generator.generate_chart(
    chart_type='bar',
    data=data,
    filename_root='sales_report',
    title='Sales Report',
    xlabel='Month',
    ylabel='Revenue',
    metadata={
        'department': 'Sales',
        'region': 'North America',
        'year': 2024,
        'generated_by': 'AutoReport System'
    }
)
```

### Batch Generation

```python
batch_specs = [
    {
        'chart_type': 'bar',
        'data': {'x': ['A', 'B', 'C'], 'y': [10, 20, 15]},
        'filename_root': 'chart1',
        'title': 'Chart 1'
    },
    {
        'chart_type': 'line',
        'data': {'x': [1, 2, 3], 'y': [5, 10, 8]},
        'filename_root': 'chart2',
        'title': 'Chart 2'
    }
]

results = generator.batch_generate(batch_specs)
```

### Custom Styling

```python
generator.generate_chart(
    chart_type='line',
    data=data,
    filename_root='styled_chart',
    title='Styled Chart',
    xlabel='X',
    ylabel='Y',
    figsize=(12, 8),      # Custom figure size
    dpi=600,              # Higher resolution
    linewidth=3           # Thicker lines
)
```

## Output Format

### PNG Files
- High-quality images (300 DPI by default)
- Professional formatting with titles and labels
- Optimized for report inclusion

### JSON Files
Each JSON file contains:
```json
{
  "filename_root": "quarterly_sales",
  "chart_type": "bar",
  "title": "Quarterly Sales 2024",
  "xlabel": "Quarter",
  "ylabel": "Sales ($K)",
  "data": {
    "x": ["Q1", "Q2", "Q3", "Q4"],
    "y": [100, 120, 115, 140]
  },
  "generated_at": "2024-11-03T10:30:45.123456",
  "png_file": "quarterly_sales.png",
  "metadata": {}
}
```

## File Naming

Both files share the same root filename:
- `somechart123.png` - The chart image
- `somechart123.json` - The chart metadata

This makes it easy to:
- Match charts with their data
- Regenerate charts from JSON
- Archive and version control your reports

## Running Examples

Run the example script to see all chart types in action:

```bash
python example_usage.py
```

This will generate sample charts in the `./output_charts` directory.

## Adding New Chart Types

To add a new chart type:

1. Add the chart type name to `SUPPORTED_CHART_TYPES`
2. Implement a `_create_<chart_type>_chart` method in the `ChartGenerator` class

Example:
```python
def _create_heatmap_chart(self, ax, data, title, xlabel, ylabel, **kwargs):
    """Create a heatmap."""
    import numpy as np
    matrix = np.array(data.get('matrix', []))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.colorbar(im, ax=ax)
```

## Use Cases

- **Automated Reporting**: Generate charts for daily/weekly/monthly reports
- **Data Pipelines**: Integrate chart generation into ETL workflows
- **Dashboard Exports**: Create static chart exports from dynamic data
- **Documentation**: Generate charts for technical documentation
- **Archival**: Store chart data and images together for future reference

## License

MIT License - feel free to use in your projects!
