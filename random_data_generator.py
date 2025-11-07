"""
Random Data Generator for Charts
Generates realistic random data for each chart type
"""

import random
import string
from typing import Dict, Any, List, Optional


class RandomDataGenerator:
    """Generate random data suitable for different chart types."""
    
    # Sample categories and labels for realistic data
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']
    YEARS = ['2020', '2021', '2022', '2023', '2024', '2025']
    REGIONS = ['North', 'South', 'East', 'West', 'Central', 'Northeast', 'Southeast', 'Northwest', 'Southwest']
    PRODUCTS = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    DEPARTMENTS = ['Sales', 'Marketing', 'Engineering', 'HR', 'Finance', 'Operations', 'Customer Service']
    CATEGORIES = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    DEVICE_TYPES = ['Desktop', 'Mobile', 'Tablet', 'Smart TV', 'Gaming Console', 'Wearable']
    SATISFACTION_METRICS = ['Product Quality', 'Customer Service', 'Delivery Speed', 'Pricing', 'User Experience']
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random data generator.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
    
    def generate_line_data(
        self,
        num_points: int = None,
        num_series: int = 1,
        x_type: str = 'months',
        y_range: tuple = (10, 100),
        trend: str = 'random'
    ) -> Dict[str, Any]:
        """
        Generate data for line charts.
        
        Args:
            num_points: Number of data points (auto-determined if None)
            num_series: Number of lines to generate
            x_type: Type of x-axis ('months', 'quarters', 'years', 'numeric', 'categories')
            y_range: Range for y values (min, max)
            trend: 'random', 'increasing', 'decreasing', 'fluctuating'
            
        Returns:
            Dictionary with 'x', 'y', and optionally 'labels' keys
        """
        # Determine x-axis values
        if x_type == 'months':
            x = self.MONTHS[:num_points] if num_points else self.MONTHS
        elif x_type == 'quarters':
            x = self.QUARTERS[:num_points] if num_points else self.QUARTERS
        elif x_type == 'years':
            x = self.YEARS[:num_points] if num_points else self.YEARS[-6:]
        elif x_type == 'numeric':
            count = num_points or 10
            x = list(range(1, count + 1))
        else:  # categories
            count = num_points or 8
            x = [f'Point {i+1}' for i in range(count)]
        
        actual_num_points = len(x)
        
        # Generate y values based on trend
        if num_series == 1:
            y = self._generate_series(actual_num_points, y_range, trend)
            return {'x': x, 'y': y}
        else:
            y_series = []
            labels = []
            for i in range(num_series):
                series_data = self._generate_series(actual_num_points, y_range, trend)
                y_series.append(series_data)
                labels.append(random.choice(self.PRODUCTS))
            return {'x': x, 'y': y_series, 'labels': labels}
    
    def generate_bar_data(
        self,
        num_bars: int = None,
        x_type: str = 'categories',
        y_range: tuple = (10, 100)
    ) -> Dict[str, Any]:
        """
        Generate data for bar charts.
        
        Args:
            num_bars: Number of bars (auto-determined if None)
            x_type: Type of x-axis ('categories', 'regions', 'products', 'departments', 'months')
            y_range: Range for y values (min, max)
            
        Returns:
            Dictionary with 'x' and 'y' keys
        """
        # Determine categories
        if x_type == 'regions':
            categories = self.REGIONS
        elif x_type == 'products':
            categories = self.PRODUCTS
        elif x_type == 'departments':
            categories = self.DEPARTMENTS
        elif x_type == 'months':
            categories = self.MONTHS
        else:  # categories
            categories = self.CATEGORIES
        
        if num_bars:
            categories = categories[:num_bars]
        
        values = [random.randint(y_range[0], y_range[1]) for _ in categories]
        
        return {'x': categories, 'y': values}
    
    def generate_horizontal_bar_data(
        self,
        num_bars: int = None,
        category_type: str = 'satisfaction',
        value_range: tuple = (1, 5)
    ) -> Dict[str, Any]:
        """
        Generate data for horizontal bar charts.
        
        Args:
            num_bars: Number of bars (auto-determined if None)
            category_type: Type of categories ('satisfaction', 'departments', 'products', 'custom')
            value_range: Range for values (min, max)
            
        Returns:
            Dictionary with 'categories' and 'values' keys
        """
        if category_type == 'satisfaction':
            categories = self.SATISFACTION_METRICS
            # For satisfaction, use float values
            values = [round(random.uniform(value_range[0], value_range[1]), 1) 
                     for _ in categories]
        elif category_type == 'departments':
            categories = self.DEPARTMENTS
            values = [random.randint(value_range[0], value_range[1]) for _ in categories]
        elif category_type == 'products':
            categories = self.PRODUCTS
            values = [random.randint(value_range[0], value_range[1]) for _ in categories]
        else:  # custom
            count = num_bars or 5
            categories = [f'Item {i+1}' for i in range(count)]
            values = [random.randint(value_range[0], value_range[1]) for _ in categories]
        
        if num_bars:
            categories = categories[:num_bars]
            values = values[:num_bars]
        
        return {'categories': categories, 'values': values}
    
    def generate_pie_data(
        self,
        num_slices: int = None,
        label_type: str = 'devices',
        total: int = 100,
        explode_largest: bool = True
    ) -> Dict[str, Any]:
        """
        Generate data for pie charts.
        
        Args:
            num_slices: Number of slices (auto-determined if None)
            label_type: Type of labels ('devices', 'products', 'regions', 'custom')
            total: Total to distribute (will be normalized to percentages)
            explode_largest: Whether to explode the largest slice
            
        Returns:
            Dictionary with 'labels', 'values', and optionally 'explode' keys
        """
        if label_type == 'devices':
            labels = self.DEVICE_TYPES
        elif label_type == 'products':
            labels = self.PRODUCTS
        elif label_type == 'regions':
            labels = self.REGIONS
        else:  # custom
            count = num_slices or 5
            labels = [f'Slice {i+1}' for i in range(count)]
        
        if num_slices:
            labels = labels[:num_slices]
        
        # Generate random values that sum to total
        values = [random.randint(5, 50) for _ in labels]
        
        # Normalize to sum to total
        current_sum = sum(values)
        values = [round((v / current_sum) * total, 1) for v in values]
        
        result = {'labels': labels, 'values': values}
        
        if explode_largest:
            max_idx = values.index(max(values))
            explode = [0.1 if i == max_idx else 0 for i in range(len(values))]
            result['explode'] = explode
        
        return result
    
    def generate_scatter_data(
        self,
        num_points: int = None,
        x_range: tuple = (0, 100),
        y_range: tuple = (0, 100),
        correlation: str = 'random',
        size_variation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate data for scatter plots.
        
        Args:
            num_points: Number of points (default: 20)
            x_range: Range for x values (min, max)
            y_range: Range for y values (min, max)
            correlation: 'random', 'positive', 'negative', 'none'
            size_variation: Whether to vary point sizes
            
        Returns:
            Dictionary with 'x', 'y', and optionally 'sizes' and 'colors' keys
        """
        count = num_points or 20
        
        # Generate x values
        x = [random.randint(x_range[0], x_range[1]) for _ in range(count)]
        
        # Generate y values based on correlation
        if correlation == 'positive':
            # Positive correlation with some noise
            y = [x_val + random.randint(-15, 15) for x_val in x]
        elif correlation == 'negative':
            # Negative correlation with some noise
            max_val = x_range[1] + y_range[1]
            y = [max_val - x_val + random.randint(-15, 15) for x_val in x]
        elif correlation == 'none':
            # No correlation
            y = [random.randint(y_range[0], y_range[1]) for _ in range(count)]
        else:  # random
            # Some correlation with lots of noise
            base = [(x_val * 0.5) for x_val in x]
            y = [int(b + random.randint(-30, 30)) for b in base]
        
        # Clamp y values to range
        y = [max(y_range[0], min(y_range[1], val)) for val in y]
        
        result = {'x': x, 'y': y}
        
        if size_variation:
            sizes = [random.randint(50, 300) for _ in range(count)]
            result['sizes'] = sizes
        
        return result
    
    def generate_grouped_bar_data(
        self,
        num_categories: int = None,
        num_groups: int = 3,
        category_type: str = None,
        group_type: str = None,
        value_range: tuple = (20, 100)
    ) -> Dict[str, Any]:
        """
        Generate data for grouped bar charts (multi-dimensional data).
        Perfect for: Products over Time, Products by Region, etc.
        
        Args:
            num_categories: Number of categories on x-axis (auto-determined if None)
            num_groups: Number of groups/series to compare (default: 3)
            category_type: Type of x-axis ('quarters', 'months', 'regions', 'years') - random if None
            group_type: Type of groups ('products', 'regions', 'departments') - random if None
            value_range: Range for values (min, max)
            
        Returns:
            Dictionary with 'categories', 'groups', and 'values' keys
        """
        # Randomize category_type if not specified
        if category_type is None:
            category_type = random.choice(['quarters', 'months', 'regions', 'years', 'products', 'departments'])
        
        # Randomize group_type if not specified (ensure it's different from category_type)
        if group_type is None:
            # All possible group types
            all_types = ['quarters', 'months', 'regions', 'years', 'products', 'departments']
            # Remove the category_type to avoid duplication
            available_types = [t for t in all_types if t != category_type]
            group_type = random.choice(available_types)
        
        # Determine categories (x-axis)
        if category_type == 'quarters':
            categories = self.QUARTERS[:num_categories] if num_categories else self.QUARTERS
        elif category_type == 'months':
            categories = self.MONTHS[:num_categories] if num_categories else self.MONTHS[:6]
        elif category_type == 'regions':
            categories = self.REGIONS[:num_categories] if num_categories else self.REGIONS[:5]
        elif category_type == 'years':
            categories = self.YEARS[:num_categories] if num_categories else self.YEARS[-4:]
        elif category_type == 'products':
            categories = self.PRODUCTS[:num_categories] if num_categories else self.PRODUCTS[:4]
        elif category_type == 'departments':
            categories = self.DEPARTMENTS[:num_categories] if num_categories else self.DEPARTMENTS[:5]
        else:
            count = num_categories or 4
            categories = [f'Period {i+1}' for i in range(count)]
        
        # Determine groups (series to compare)
        if group_type == 'products':
            groups = self.PRODUCTS[:num_groups]
        elif group_type == 'regions':
            groups = self.REGIONS[:num_groups]
        elif group_type == 'departments':
            groups = self.DEPARTMENTS[:num_groups]
        elif group_type == 'quarters':
            groups = self.QUARTERS[:num_groups]
        elif group_type == 'months':
            groups = self.MONTHS[:num_groups]
        elif group_type == 'years':
            groups = self.YEARS[:num_groups] if num_groups <= len(self.YEARS) else self.YEARS
        else:
            groups = [f'Group {i+1}' for i in range(num_groups)]
        
        # Generate values for each group across all categories
        values = []
        for _ in range(num_groups):
            group_values = [random.randint(value_range[0], value_range[1]) 
                          for _ in categories]
            values.append(group_values)
        
        return {
            'categories': categories,
            'groups': groups,
            'values': values
        }
    
    def _generate_series(
        self,
        num_points: int,
        value_range: tuple,
        trend: str
    ) -> List[float]:
        """Generate a series of values with specified trend."""
        min_val, max_val = value_range
        
        if trend == 'increasing':
            base = list(range(num_points))
            increment = (max_val - min_val) / num_points
            values = [min_val + (i * increment) + random.randint(-5, 5) 
                     for i in base]
        elif trend == 'decreasing':
            base = list(range(num_points))
            decrement = (max_val - min_val) / num_points
            values = [max_val - (i * decrement) + random.randint(-5, 5) 
                     for i in base]
        elif trend == 'fluctuating':
            values = []
            current = random.randint(min_val, max_val)
            for _ in range(num_points):
                change = random.randint(-15, 15)
                current = max(min_val, min(max_val, current + change))
                values.append(current)
        else:  # random
            values = [random.randint(min_val, max_val) for _ in range(num_points)]
        
        return [max(min_val, min(max_val, int(v))) for v in values]
    
    def generate_random_chart_spec(
        self,
        chart_type: Optional[str] = None,
        filename_root: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete random chart specification.
        
        Args:
            chart_type: Type of chart (randomly chosen if None)
            filename_root: Base filename (auto-generated if None)
            
        Returns:
            Complete chart specification dictionary ready for ChartGenerator
        """
        chart_types = ['line', 'bar', 'horizontal_bar', 'pie', 'scatter', 'grouped_bar', 'stacked_bar']
        
        if chart_type is None:
            chart_type = random.choice(chart_types)
        
        if filename_root is None:
            random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            filename_root = f"{chart_type}_chart_{random_id}"
        
        if chart_type == 'line':
            num_series = random.choice([1, 1, 1, 2, 3])  # Bias toward single series
            data = self.generate_line_data(
                num_series=num_series,
                x_type=random.choice(['months', 'quarters', 'numeric']),
                trend=random.choice(['random', 'increasing', 'fluctuating'])
            )
            return {
                'chart_type': 'line',
                'data': data,
                'filename_root': filename_root,
                'title': f'{random.choice(["Sales", "Revenue", "Performance", "Growth"])} Over Time',
                'xlabel': random.choice(['Month', 'Quarter', 'Time Period']),
                'ylabel': random.choice(['Value', 'Amount ($K)', 'Units', 'Score']),
                'metadata': {'generated': 'random', 'series_count': num_series}
            }
        
        elif chart_type == 'bar':
            x_type = random.choice(['regions', 'products', 'departments', 'categories'])
            data = self.generate_bar_data(x_type=x_type)
            
            # Map x_type to appropriate labels - automatically matches data to labels
            label_map = {
                'regions': ('Regional Performance', 'Region'),
                'products': ('Product Performance', 'Product'),
                'departments': ('Department Performance', 'Department'),
                'categories': ('Category Performance', 'Category'),
                'months': ('Monthly Performance', 'Month')
            }
            title, xlabel = label_map.get(x_type, ('Category Performance', 'Category'))
            
            return {
                'chart_type': 'bar',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': xlabel,
                'ylabel': random.choice(['Sales ($K)', 'Revenue', 'Units Sold', 'Value']),
                'metadata': {'generated': 'random', 'bar_count': len(data['x']), 'x_type': x_type}
            }
        
        elif chart_type == 'horizontal_bar':
            data = self.generate_horizontal_bar_data(
                category_type=random.choice(['satisfaction', 'departments', 'products'])
            )
            return {
                'chart_type': 'horizontal_bar',
                'data': data,
                'filename_root': filename_root,
                'title': random.choice(['Customer Satisfaction Ratings', 'Performance Metrics', 'Rating Summary']),
                'xlabel': random.choice(['Rating', 'Score', 'Value']),
                'ylabel': random.choice(['Category', 'Metric', 'Department']),
                'metadata': {'generated': 'random', 'category_count': len(data['categories'])}
            }
        
        elif chart_type == 'pie':
            data = self.generate_pie_data(
                label_type=random.choice(['devices', 'products', 'regions'])
            )
            return {
                'chart_type': 'pie',
                'data': data,
                'filename_root': filename_root,
                'title': f'{random.choice(["Market", "Usage", "Distribution", "Share"])} Breakdown',
                'metadata': {'generated': 'random', 'slice_count': len(data['labels'])}
            }
        
        elif chart_type == 'scatter':
            data = self.generate_scatter_data(
                num_points=random.randint(15, 30),
                correlation=random.choice(['random', 'positive', 'none'])
            )
            return {
                'chart_type': 'scatter',
                'data': data,
                'filename_root': filename_root,
                'title': f'{random.choice(["Correlation", "Relationship", "Distribution"])} Analysis',
                'xlabel': random.choice(['Variable X', 'Age', 'Time', 'Input']),
                'ylabel': random.choice(['Variable Y', 'Score', 'Value', 'Output']),
                'metadata': {'generated': 'random', 'point_count': len(data['x'])}
            }
        
        elif chart_type == 'grouped_bar':
            # Generate with random category and group types
            data = self.generate_grouped_bar_data(
                num_groups=random.choice([2, 3, 3, 4]),  # Bias toward 3 groups
                value_range=(30, 120)
            )
            
            # Create adaptive title and xlabel based on what was randomly chosen
            # Extract the types from the generated data
            first_category = data['categories'][0]
            first_group = data['groups'][0]
            
            # Determine category type from first category
            if first_category in self.QUARTERS:
                category_label = 'Quarter'
                title_prefix = 'Quarterly'
            elif first_category in self.MONTHS:
                category_label = 'Month'
                title_prefix = 'Monthly'
            elif first_category in self.REGIONS:
                category_label = 'Region'
                title_prefix = 'Regional'
            elif first_category in self.YEARS:
                category_label = 'Year'
                title_prefix = 'Annual'
            elif first_category in self.PRODUCTS:
                category_label = 'Product'
                title_prefix = 'Product'
            elif first_category in self.DEPARTMENTS:
                category_label = 'Department'
                title_prefix = 'Department'
            else:
                category_label = 'Category'
                title_prefix = 'Category'
            
            # Determine group type from first group
            if first_group in self.PRODUCTS:
                group_label = 'Product'
            elif first_group in self.REGIONS:
                group_label = 'Regional'
            elif first_group in self.DEPARTMENTS:
                group_label = 'Department'
            elif first_group in self.QUARTERS:
                group_label = 'Quarterly'
            elif first_group in self.MONTHS:
                group_label = 'Monthly'
            elif first_group in self.YEARS:
                group_label = 'Annual'
            else:
                group_label = 'Group'
            
            return {
                'chart_type': 'grouped_bar',
                'data': data,
                'filename_root': filename_root,
                'title': f'{group_label} Performance by {category_label}',
                'xlabel': category_label,
                'ylabel': random.choice(['Sales ($K)', 'Revenue', 'Units', 'Performance Score']),
                'metadata': {
                    'generated': 'random',
                    'num_groups': len(data['groups']),
                    'num_categories': len(data['categories'])
                }
            }
        
        elif chart_type == 'stacked_bar':
            # Generate with random category and group types (same as grouped_bar)
            data = self.generate_grouped_bar_data(
                num_groups=random.choice([2, 3, 3, 4]),  # Bias toward 3 groups
                value_range=(30, 120)
            )
            
            # Create adaptive title and xlabel (same logic as grouped_bar)
            first_category = data['categories'][0]
            first_group = data['groups'][0]
            
            # Determine category type from first category
            if first_category in self.QUARTERS:
                category_label = 'Quarter'
                title_prefix = 'Quarterly'
            elif first_category in self.MONTHS:
                category_label = 'Month'
                title_prefix = 'Monthly'
            elif first_category in self.REGIONS:
                category_label = 'Region'
                title_prefix = 'Regional'
            elif first_category in self.YEARS:
                category_label = 'Year'
                title_prefix = 'Annual'
            elif first_category in self.PRODUCTS:
                category_label = 'Product'
                title_prefix = 'Product'
            elif first_category in self.DEPARTMENTS:
                category_label = 'Department'
                title_prefix = 'Department'
            else:
                category_label = 'Category'
                title_prefix = 'Category'
            
            # Determine group type from first group
            if first_group in self.PRODUCTS:
                group_label = 'Product'
            elif first_group in self.REGIONS:
                group_label = 'Regional'
            elif first_group in self.DEPARTMENTS:
                group_label = 'Department'
            elif first_group in self.QUARTERS:
                group_label = 'Quarterly'
            elif first_group in self.MONTHS:
                group_label = 'Monthly'
            elif first_group in self.YEARS:
                group_label = 'Annual'
            else:
                group_label = 'Group'
            
            return {
                'chart_type': 'stacked_bar',
                'data': data,
                'filename_root': filename_root,
                'title': f'{group_label} Distribution by {category_label}',
                'xlabel': category_label,
                'ylabel': random.choice(['Total Sales ($K)', 'Total Revenue', 'Total Units', 'Combined Performance']),
                'metadata': {
                    'generated': 'random',
                    'num_groups': len(data['groups']),
                    'num_categories': len(data['categories'])
                }
            }
        
        return {}