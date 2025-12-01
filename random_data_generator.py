"""
Random Data Generator for Charts
Generates realistic random data for each chart type
"""

import random
import string
import math
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
    RATING_LABELS = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
    SCORE_LABELS = ['A', 'B', 'C', 'D', 'F']
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random data generator.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            random.seed(seed)
    
    def _random_date_range(self, start='2000-01', end='2024-12'):
        """
        Generate a random date range within bounds.
        
        Args:
            start: Start bound as 'YYYY-MM'
            end: End bound as 'YYYY-MM'
            
        Returns:
            Tuple of (start_date, end_date) as 'YYYY-MM' strings
        """
        # Parse bounds
        sy, sm = map(int, start.split('-'))
        ey, em = map(int, end.split('-'))
        
        # Random start
        y1 = random.randint(sy, ey)
        m1 = random.randint(1, 12)
        
        # Random end after start
        y2 = random.randint(y1, ey)
        m2 = random.randint(1, 12)
        
        # Ensure end is after start
        if y2 == y1 and m2 < m1:
            m2 = m1
        
        return (f'{y1}-{m1:02d}', f'{y2}-{m2:02d}')
    
    def generate_line_data(
        self,
        num_points: int = None,
        num_series: int = None,
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
    
    def generate_box_data(
        self,
        num_boxes: int = None,
        label_type: str = None,
        value_range: tuple = (10, 100),
        num_samples: int = None
    ) -> Dict[str, Any]:
        """
        Generate data for box plots (distribution analysis).
        
        Args:
            num_boxes: Number of boxes to create (auto-determined if None)
            label_type: Type of labels ('products', 'regions', 'departments', 'quarters', 'months')
            value_range: Range for values (min, max)
            num_samples: Number of data points per box (default: random 15-30)
            
        Returns:
            Dictionary with 'labels' and 'data' keys
        """
        # Randomize label_type if not specified
        if label_type is None:
            label_type = random.choice(['products', 'regions', 'departments', 'quarters', 'months'])
        
        # Determine labels
        if label_type == 'products':
            labels = self.PRODUCTS[:num_boxes] if num_boxes else self.PRODUCTS[:4]
        elif label_type == 'regions':
            labels = self.REGIONS[:num_boxes] if num_boxes else self.REGIONS[:5]
        elif label_type == 'departments':
            labels = self.DEPARTMENTS[:num_boxes] if num_boxes else self.DEPARTMENTS[:5]
        elif label_type == 'quarters':
            labels = self.QUARTERS[:num_boxes] if num_boxes else self.QUARTERS
        elif label_type == 'months':
            labels = self.MONTHS[:num_boxes] if num_boxes else self.MONTHS[:6]
        else:
            count = num_boxes or 4
            labels = [f'Group {i+1}' for i in range(count)]
        
        # Generate data points for each box
        data = []
        for _ in labels:
            # Random number of samples per box (or use specified)
            n_samples = num_samples if num_samples else random.randint(15, 30)
            
            # Generate data with some distribution characteristics
            # Use normal distribution for more realistic box plots
            range_size = value_range[1] - value_range[0]
            
            # Calculate mean safely for any range size
            if range_size <= 1:
                # Very small range, use midpoint
                mean = (value_range[0] + value_range[1]) / 2
            elif range_size <= 10:
                # Small range, use uniform random
                mean = random.uniform(value_range[0], value_range[1])
            else:
                # Larger range, use middle portion with margin
                margin = max(int(range_size * 0.2), 1)
                # Ensure start < stop for randint
                start = value_range[0] + margin
                stop = value_range[1] - margin
                if start >= stop:
                    mean = (value_range[0] + value_range[1]) / 2
                else:
                    mean = random.randint(start, stop)
            
            std_dev = max(range_size / 6, 0.5)  # At least 0.5 std dev
            
            samples = []
            for _ in range(n_samples):
                # Generate value with normal distribution
                value = random.gauss(mean, std_dev)
                # Clamp to range
                value = max(value_range[0], min(value_range[1], value))
                samples.append(round(value, 1))
            
            # Add a few potential outliers (30% chance)
            if random.random() < 0.3:
                n_outliers = random.randint(1, 2)
                for _ in range(n_outliers):
                    try:
                        if random.random() < 0.5:
                            # High outlier
                            high_val = int(mean + 2*std_dev)
                            if high_val < value_range[1]:
                                outlier = random.randint(high_val, value_range[1])
                                samples.append(outlier)
                        else:
                            # Low outlier
                            low_val = int(mean - 2*std_dev)
                            if low_val > value_range[0]:
                                outlier = random.randint(value_range[0], low_val)
                                samples.append(outlier)
                    except (ValueError, OverflowError):
                        # Skip outlier if range is too small
                        pass
            
            data.append(samples)
        
        return {
            'labels': labels,
            'data': data
        }
    
    def generate_area_data(
        self,
        num_points: int = None,
        area_type: str = 'range',
        x_type: str = 'months',
        y_range: tuple = (10, 100),
        num_series: int = 2,
        step: bool = False
    ) -> Dict[str, Any]:
        """
        Generate data for area charts (fill_between).
        
        Args:
            num_points: Number of data points (auto-determined if None)
            area_type: 'single', 'range', 'stacked', 'stacked_100', 'overlapping', 'step'
            x_type: Type of x-axis ('months', 'quarters', 'numeric', 'years')
            y_range: Range for y values (min, max)
            num_series: Number of series for multi-series areas
            step: Create step transitions (for step area)
            
        Returns:
            Dictionary with appropriate keys for area chart type
        """
        # Determine x-axis values
        if x_type == 'months':
            x = self.MONTHS[:num_points] if num_points else self.MONTHS[:6]
        elif x_type == 'quarters':
            x = self.QUARTERS[:num_points] if num_points else self.QUARTERS
        elif x_type == 'years':
            x = self.YEARS[:num_points] if num_points else self.YEARS[-5:]
        elif x_type == 'numeric':
            count = num_points or 10
            x = list(range(1, count + 1))
        else:
            count = num_points or 8
            x = [f'Period {i+1}' for i in range(count)]
        
        actual_num_points = len(x)
        
        if area_type == 'single':
            # Single area from 0 to y
            y = self._generate_series(actual_num_points, y_range, 'fluctuating')
            result = {'x': x, 'y': y}
            if step:
                result['step'] = True
            return result
            
        elif area_type == 'range':
            # Range/band chart with y1 and y2
            mid_range = (y_range[0] + y_range[1]) / 2
            spread = (y_range[1] - y_range[0]) / 4
            
            # Generate middle line (convert to int for _generate_series)
            middle = self._generate_series(actual_num_points, 
                                          (int(mid_range - spread), int(mid_range + spread)), 
                                          'fluctuating')
            
            # Create range around it
            y1 = [max(y_range[0], m - random.randint(5, 15)) for m in middle]
            y2 = [min(y_range[1], m + random.randint(5, 15)) for m in middle]
            
            result = {'x': x, 'y1': y1, 'y2': y2}
            if step:
                result['step'] = True
            return result
            
        elif area_type in ['stacked', 'stacked_100', 'overlapping']:
            # Multiple areas
            areas = []
            labels = random.sample(self.PRODUCTS, min(num_series, len(self.PRODUCTS)))
            
            for i in range(num_series):
                # Generate smaller values for stacking
                adjusted_range = (y_range[0] // num_series, y_range[1] // num_series)
                y = self._generate_series(actual_num_points, adjusted_range, 'fluctuating')
                areas.append({
                    'y': y,
                    'label': labels[i] if i < len(labels) else f'Series {i+1}'
                })
            
            result = {'x': x, 'areas': areas, 'area_type': area_type}
            if step:
                result['step'] = True
            return result
            
        else:  # Fallback to stacked
            # Multiple stacked areas
            areas = []
            labels = random.sample(self.PRODUCTS, min(num_series, len(self.PRODUCTS)))
            
            for i in range(num_series):
                adjusted_range = (y_range[0] // num_series, y_range[1] // num_series)
                y = self._generate_series(actual_num_points, adjusted_range, 'fluctuating')
                areas.append({
                    'y': y,
                    'label': labels[i] if i < len(labels) else f'Series {i+1}'
                })
            
            result = {'x': x, 'areas': areas, 'area_type': 'stacked'}
            if step:
                result['step'] = True
            return result
    
    def generate_discrete_distribution_data(
        self,
        distribution_type: str = 'range',
        num_values: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate data for discrete probability distributions.
        
        Args:
            distribution_type: Type of distribution 
                - 'binomial': Binomial distribution (n trials, p probability)
                - 'poisson': Poisson distribution (lambda rate)
                - 'uniform': Discrete uniform distribution
                - 'custom': Custom distribution with specified probabilities
                - 'dice': Dice roll (standard 6-sided die)
                - 'rating': Rating distribution (1-5 stars)
                - 'score': Letter grade distribution (A, B, C, D, F)
            num_values: Number of discrete values (auto-determined for most types)
            **kwargs: Distribution-specific parameters
            
        Returns:
            Dictionary with 'values'/'categories' and 'probabilities' keys
        """
        import math
        
        if distribution_type == 'binomial':
            n = kwargs.get('n', 10)
            p = kwargs.get('p', 0.5)
            values = list(range(n + 1))
            probabilities = []
            for k in values:
                prob = (math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)))
                probabilities.append(prob)
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            return {
                'values': values,
                'probabilities': probabilities,
                'labels': [f'X={k}' for k in values],
                'distribution_params': {'type': 'binomial', 'n': n, 'p': p}
            }
        
        elif distribution_type == 'poisson':
            lambda_rate = kwargs.get('lambda_rate', 3.0)
            max_value = kwargs.get('max_value', 15)
            values = list(range(max_value + 1))
            probabilities = []
            for k in values:
                prob = ((lambda_rate ** k) * math.exp(-lambda_rate)) / math.factorial(k)
                probabilities.append(prob)
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            return {
                'values': values,
                'probabilities': probabilities,
                'labels': [f'X={k}' for k in values],
                'distribution_params': {'type': 'poisson', 'lambda': lambda_rate}
            }
        
        elif distribution_type == 'uniform':
            min_val = kwargs.get('min_val', 1)
            max_val = kwargs.get('max_val', 6)
            values = list(range(min_val, max_val + 1))
            n = len(values)
            probabilities = [1.0 / n] * n
            return {
                'values': values,
                'probabilities': probabilities,
                'labels': [f'X={v}' for v in values],
                'distribution_params': {'type': 'uniform', 'min': min_val, 'max': max_val}
            }
        
        elif distribution_type == 'dice':
            values = list(range(1, 7))
            probabilities = [1/6] * 6
            return {
                'values': values,
                'probabilities': probabilities,
                'labels': [f'Roll {i}' for i in values],
                'distribution_params': {'type': 'dice'}
            }
        
        elif distribution_type == 'rating':
            categories = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
            skew_type = kwargs.get('skew', 'positive')
            if skew_type == 'positive':
                base_probs = [0.05, 0.08, 0.15, 0.35, 0.37]
            elif skew_type == 'negative':
                base_probs = [0.37, 0.35, 0.15, 0.08, 0.05]
            else:
                base_probs = [0.15, 0.20, 0.30, 0.20, 0.15]
            probabilities = []
            for p in base_probs:
                variation = random.uniform(-0.05, 0.05)
                probabilities.append(max(0.01, p + variation))
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            return {
                'categories': categories,
                'probabilities': probabilities,
                'values': list(range(1, 6)),
                'distribution_params': {'type': 'rating', 'skew': skew_type}
            }
        
        elif distribution_type == 'score':
            categories = ['A', 'B', 'C', 'D', 'F']
            grade_type = kwargs.get('grade_type', 'normal')
            if grade_type == 'easy':
                base_probs = [0.35, 0.35, 0.20, 0.07, 0.03]
            elif grade_type == 'hard':
                base_probs = [0.10, 0.20, 0.35, 0.25, 0.10]
            else:
                base_probs = [0.20, 0.30, 0.30, 0.15, 0.05]
            probabilities = []
            for p in base_probs:
                variation = random.uniform(-0.03, 0.03)
                probabilities.append(max(0.01, p + variation))
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            return {
                'categories': categories,
                'probabilities': probabilities,
                'distribution_params': {'type': 'score', 'grade_type': grade_type}
            }
        
        elif distribution_type == 'custom':
            categories = kwargs.get('categories', self.CATEGORIES[:5])
            probabilities = kwargs.get('probabilities', None)
            if probabilities is None:
                n = len(categories)
                probabilities = [random.uniform(0.1, 1.0) for _ in range(n)]
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            return {
                'categories': categories,
                'probabilities': probabilities,
                'distribution_params': {'type': 'custom'}
            }
        
        else:
            return self.generate_discrete_distribution_data('binomial', **kwargs)
    
    def generate_cumulative_distribution_data(
        self,
        distribution_type: str = 'range',
        num_values: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate data for cumulative distribution function (CDF) charts.
        
        Args:
            distribution_type: Type of distribution (same as discrete_distribution_data)
                - 'binomial', 'poisson', 'uniform', 'dice', 'rating', 'score', 'custom'
                - 'normal_approx': Continuous normal approximation
            num_values: Number of discrete values (auto-determined for most types)
            **kwargs: Distribution-specific parameters
            
        Returns:
            Dictionary with 'values', 'probabilities', 'cdf', and 'labels' keys
        """
        import numpy as np
        
        # Special case: continuous normal approximation
        if distribution_type == 'normal_approx':
            mean = kwargs.get('mean', 50)
            std = kwargs.get('std', 15)
            num_points = kwargs.get('num_points', 100)
            
            # Generate x values
            x_min = mean - 3 * std
            x_max = mean + 3 * std
            x_values = np.linspace(x_min, x_max, num_points)
            
            # Calculate CDF using error function
            from scipy import special
            cdf_values = 0.5 * (1 + special.erf((x_values - mean) / (std * np.sqrt(2))))
            
            return {
                'x': x_values.tolist(),
                'cdf': cdf_values.tolist(),
                'distribution_params': {'type': 'normal', 'mean': mean, 'std': std}
            }
        
        # For discrete distributions, get PMF first
        pmf_data = self.generate_discrete_distribution_data(
            distribution_type=distribution_type,
            num_values=num_values,
            **kwargs
        )
        
        # Calculate cumulative distribution
        probabilities = pmf_data['probabilities']
        cdf = np.cumsum(probabilities).tolist()
        
        # Build result with both PMF and CDF
        result = pmf_data.copy()
        result['cdf'] = cdf
        
        return result
    
    def generate_time_series_histogram_data(
        self,
        num_time_points: int = None,
        samples_per_time: int = None,
        value_range: tuple = (0, 100),
        trend: str = None,
        volatility: str = 'medium',
        bins: int = 20
    ) -> Dict[str, Any]:
        """
        Generate data for time series histogram charts.
        Shows how a distribution evolves over time.
        
        Args:
            num_time_points: Number of time steps (default: 12)
            samples_per_time: Samples at each time point (default: 200)
            value_range: Range of values (min, max)
            trend: Distribution trend over time
                - 'random': Random walk
                - 'increasing': Mean increases over time
                - 'decreasing': Mean decreases over time
                - 'cyclical': Oscillating pattern
                - 'volatility_increase': Spread increases over time
                - 'volatility_decrease': Spread decreases over time
            volatility: Amount of randomness ('low', 'medium', 'high')
            bins: Number of histogram bins
            
        Returns:
            Dictionary with 'time_points', 'data_series', 'bins', 'labels'
        """
        import numpy as np
        
        n_times = num_time_points or 12
        n_samples = samples_per_time or 200
        
        time_points = list(range(n_times))
        data_series = []
        
        # Set volatility level
        vol_levels = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
        vol_factor = vol_levels.get(volatility, 1.0)
        
        # Initial distribution parameters
        initial_mean = (value_range[0] + value_range[1]) / 2
        initial_std = (value_range[1] - value_range[0]) / 6
        
        # Generate data for each time point
        for t in range(n_times):
            t_normalized = t / max(1, n_times - 1)  # 0 to 1
            
            if trend == 'increasing':
                # Mean increases linearly
                mean = initial_mean + (value_range[1] - initial_mean) * t_normalized * 0.8
                std = initial_std * vol_factor
                
            elif trend == 'decreasing':
                # Mean decreases linearly
                mean = initial_mean - (initial_mean - value_range[0]) * t_normalized * 0.8
                std = initial_std * vol_factor
                
            elif trend == 'cyclical':
                # Oscillating mean
                mean = initial_mean + np.sin(2 * np.pi * t_normalized * 2) * (value_range[1] - value_range[0]) * 0.25
                std = initial_std * vol_factor
                
            elif trend == 'volatility_increase':
                # Variance increases over time
                mean = initial_mean
                std = initial_std * (0.5 + 1.5 * t_normalized) * vol_factor
                
            elif trend == 'volatility_decrease':
                # Variance decreases over time
                mean = initial_mean
                std = initial_std * (1.5 - 1.0 * t_normalized) * vol_factor
                
            else:  # random walk
                # Mean does random walk
                if t == 0:
                    mean = initial_mean
                else:
                    mean = means[-1] + np.random.normal(0, 5)
                    mean = np.clip(mean, value_range[0], value_range[1])
                std = initial_std * vol_factor
            
            # Generate samples
            samples = np.random.normal(mean, std, n_samples)
            
            # Clip to value range
            samples = np.clip(samples, value_range[0], value_range[1])
            
            data_series.append(samples.tolist())
            
            # Store means for random walk
            if trend == 'random':
                if t == 0:
                    means = [mean]
                else:
                    means.append(mean)
        
        # Generate time labels (could be months, quarters, etc.)
        label_options = [
            self.MONTHS,
            self.QUARTERS * 3,  # Repeat quarters
            [f'T{i+1}' for i in range(n_times)],
            [f'Week {i+1}' for i in range(n_times)]
        ]
        labels = random.choice(label_options)[:n_times]
        
        return {
            'time_points': time_points,
            'data_series': data_series,
            'bins': bins,
            'labels': labels,
            'trend': trend,
            'volatility': volatility
        }
    
    def generate_treemap_data(
        self,
        num_categories: int = None,
        category_type: str = None,
        value_range: tuple = (10, 1000),
        distribution: str = None,
        hierarchical: bool = False
    ) -> Dict[str, Any]:
        """
        Generate data for treemap charts (hierarchical rectangles).
        
        Args:
            num_categories: Number of categories (default: random 8-15)
            category_type: Type of categories ('products', 'regions', 'departments', 'technologies', 'custom')
            value_range: Range for values (min, max)
            distribution: Size distribution ('uniform', 'power_law', 'exponential')
            hierarchical: Whether to create hierarchical groups (default: True)
            
        Returns:
            Dictionary with 'labels', 'sizes', and optionally 'groups' keys
        """
        # Determine number of categories
        num_categories = num_categories or random.randint(8, 15)
        
        # Create hierarchical structure if requested
        if hierarchical:
            # Create 3-5 main groups
            num_groups = random.randint(3, 5)
            group_labels = [chr(65 + i) for i in range(num_groups)]  # A, B, C, D, E
            
            # Distribute categories among groups
            labels = []
            sizes = []
            groups = []
            
            items_per_group = [num_categories // num_groups] * num_groups
            # Distribute remainder
            for i in range(num_categories % num_groups):
                items_per_group[i] += 1
            
            for group_idx, (group_label, group_size) in enumerate(zip(group_labels, items_per_group)):
                # Generate sizes for this group
                if distribution == 'power_law':
                    # Power law within group
                    group_values = []
                    for i in range(group_size):
                        u = random.random()
                        x_min = value_range[0]
                        x_max = value_range[1]
                        x = x_min * ((x_max / x_min) ** u)
                        group_values.append(int(x))
                    group_values.sort(reverse=True)
                
                elif distribution == 'exponential':
                    group_values = []
                    for i in range(group_size):
                        decay_rate = 0.3
                        value = value_range[1] * math.exp(-decay_rate * i) + value_range[0]
                        group_values.append(int(value))
                    random.shuffle(group_values)
                
                else:  # uniform
                    group_values = [random.randint(value_range[0], value_range[1]) 
                                  for _ in range(group_size)]
                
                # Create labels for items in this group
                for i, val in enumerate(group_values):
                    labels.append(f'{group_label}-{i+1}')
                    sizes.append(val)
                    groups.append(group_label)
            
            return {
                'labels': labels,
                'sizes': sizes,
                'groups': groups
            }
        
        else:
            # Non-hierarchical (flat) structure
            # Determine category labels
            if category_type == 'products':
                base_labels = self.PRODUCTS + [f'Product {chr(65+i)}' for i in range(10)]
            elif category_type == 'regions':
                base_labels = self.REGIONS + ['Europe', 'Asia', 'Africa', 'Oceania', 'South America', 'North America']
            elif category_type == 'departments':
                base_labels = self.DEPARTMENTS
            elif category_type == 'technologies':
                base_labels = ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust', 'TypeScript', 
                              'Ruby', 'PHP', 'Swift', 'Kotlin', 'C#', 'R', 'MATLAB']
            elif category_type == 'market_segments':
                base_labels = ['Enterprise', 'SMB', 'Consumer', 'Education', 'Government', 
                              'Healthcare', 'Retail', 'Finance', 'Manufacturing', 'Technology']
            else:  # custom
                base_labels = [f'Category {chr(65+i)}' for i in range(26)]
            
            # Ensure unique labels
            if len(base_labels) < num_categories:
                base_labels.extend([f'Item {i}' for i in range(num_categories - len(base_labels))])
            
            labels = random.sample(base_labels, num_categories)
            
            # Generate sizes based on distribution
            if distribution == 'uniform':
                sizes = [random.randint(value_range[0], value_range[1]) for _ in range(num_categories)]
            
            elif distribution == 'power_law':
                sizes = []
                for i in range(num_categories):
                    u = random.random()
                    x_min = value_range[0]
                    x_max = value_range[1]
                    x = x_min * ((x_max / x_min) ** u)
                    sizes.append(int(x))
                sizes.sort(reverse=True)
            
            elif distribution == 'exponential':
                sizes = []
                for i in range(num_categories):
                    decay_rate = 0.3
                    value = value_range[1] * math.exp(-decay_rate * i) + value_range[0]
                    sizes.append(int(value))
                random.shuffle(sizes)
            
            else:
                sizes = [random.randint(value_range[0], value_range[1]) for _ in range(num_categories)]
                sizes.sort(reverse=True)
            
            return {
                'labels': base_labels,
                'sizes': sizes
            }
    
    def generate_surface3d_data(
        self,
        x_range: tuple = (-5, 5),
        y_range: tuple = (-5, 5),
        resolution: int = 50,
        function_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate data for 3D surface plots.
        
        Args:
            x_range: Range for x values (min, max)
            y_range: Range for y values (min, max)
            resolution: Number of points in each dimension
            function_type: Type of function ('gaussian', 'saddle', 'wave', 'peaks', 'random')
            
        Returns:
            Dictionary with 'x', 'y', 'z' keys
        """
        import numpy as np
        
        # Determine function type
        if function_type is None:
            function_type = random.choice(['gaussian', 'saddle', 'wave', 'peaks', 'ripple'])
        
        # Create mesh
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Generate Z based on function type
        if function_type == 'gaussian':
            # Gaussian peak
            Z = np.exp(-(X**2 + Y**2) / 10)
        
        elif function_type == 'saddle':
            # Saddle point (hyperbolic paraboloid)
            Z = X**2 - Y**2
        
        elif function_type == 'wave':
            # Wave pattern
            Z = np.sin(np.sqrt(X**2 + Y**2))
        
        elif function_type == 'peaks':
            # MATLAB peaks function
            Z = 3 * (1 - X)**2 * np.exp(-(X**2) - (Y + 1)**2) \
                - 10 * (X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) \
                - 1/3 * np.exp(-(X + 1)**2 - Y**2)
        
        elif function_type == 'ripple':
            # Ripple effect
            R = np.sqrt(X**2 + Y**2) + 0.01
            Z = np.sin(R) / R
        
        else:  # random
            # Random surface with smoothing
            Z = np.random.randn(resolution, resolution)
            try:
                from scipy.ndimage import gaussian_filter
                Z = gaussian_filter(Z, sigma=3)
            except ImportError:
                # Fallback: simple averaging smoothing
                kernel_size = 5
                from numpy import convolve
                for _ in range(2):
                    for i in range(resolution):
                        Z[i, :] = np.convolve(Z[i, :], np.ones(kernel_size)/kernel_size, mode='same')
                    for j in range(resolution):
                        Z[:, j] = np.convolve(Z[:, j], np.ones(kernel_size)/kernel_size, mode='same')
        
        return {
            'x': x.tolist(),
            'y': y.tolist(),
            'z': Z.tolist(),
            'function': function_type
        }
    
    def generate_scatter3d_data(
        self,
        num_points: int = None,
        x_range: tuple = (0, 100),
        y_range: tuple = (0, 100),
        z_range: tuple = (0, 100),
        distribution: str = None,
        num_clusters: int = 3
    ) -> Dict[str, Any]:
        """
        Generate data for 3D scatter plots.
        
        Args:
            num_points: Number of points (default: random 50-200)
            x_range: Range for x values (min, max)
            y_range: Range for y values (min, max)
            z_range: Range for z values (min, max)
            distribution: Type ('uniform', 'clustered', 'spherical', 'spiral')
            num_clusters: Number of clusters for 'clustered' type
            
        Returns:
            Dictionary with 'x', 'y', 'z', 'colors', 'sizes' keys
        """
        import numpy as np
        
        num_points = num_points or random.randint(50, 200)
        
        if distribution is None:
            distribution = random.choice(['uniform', 'clustered', 'spherical', 'spiral'])
        
        if distribution == 'uniform':
            # Uniform random distribution
            x = np.random.uniform(x_range[0], x_range[1], num_points)
            y = np.random.uniform(y_range[0], y_range[1], num_points)
            z = np.random.uniform(z_range[0], z_range[1], num_points)
            colors = z  # Color by z value
        
        elif distribution == 'clustered':
            # Multiple clusters
            x, y, z = [], [], []
            colors = []
            
            points_per_cluster = num_points // num_clusters
            
            for i in range(num_clusters):
                # Random cluster center
                cx = random.uniform(x_range[0], x_range[1])
                cy = random.uniform(y_range[0], y_range[1])
                cz = random.uniform(z_range[0], z_range[1])
                
                # Generate points around center
                cluster_x = np.random.normal(cx, (x_range[1] - x_range[0]) / 10, points_per_cluster)
                cluster_y = np.random.normal(cy, (y_range[1] - y_range[0]) / 10, points_per_cluster)
                cluster_z = np.random.normal(cz, (z_range[1] - z_range[0]) / 10, points_per_cluster)
                
                x.extend(cluster_x)
                y.extend(cluster_y)
                z.extend(cluster_z)
                colors.extend([i] * points_per_cluster)
            
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            colors = np.array(colors)
        
        elif distribution == 'spherical':
            # Points on or near a sphere
            phi = np.random.uniform(0, 2*np.pi, num_points)
            theta = np.random.uniform(0, np.pi, num_points)
            
            radius = (x_range[1] - x_range[0]) / 3
            center_x = (x_range[0] + x_range[1]) / 2
            center_y = (y_range[0] + y_range[1]) / 2
            center_z = (z_range[0] + z_range[1]) / 2
            
            x = center_x + radius * np.sin(theta) * np.cos(phi)
            y = center_y + radius * np.sin(theta) * np.sin(phi)
            z = center_z + radius * np.cos(theta)
            
            # Add some noise
            x += np.random.normal(0, radius/10, num_points)
            y += np.random.normal(0, radius/10, num_points)
            z += np.random.normal(0, radius/10, num_points)
            
            colors = z  # Color by height
        
        else:  # spiral
            # Spiral pattern
            t = np.linspace(0, 4*np.pi, num_points)
            
            scale_x = (x_range[1] - x_range[0]) / 2
            scale_y = (y_range[1] - y_range[0]) / 2
            scale_z = (z_range[1] - z_range[0]) / (4*np.pi)
            
            x = x_range[0] + scale_x + scale_x * t/(4*np.pi) * np.cos(t)
            y = y_range[0] + scale_y + scale_y * t/(4*np.pi) * np.sin(t)
            z = z_range[0] + scale_z * t
            
            colors = t  # Color by position along spiral
        
        # Generate sizes with variation
        sizes = np.random.uniform(30, 100, num_points)
        
        return {
            'x': x.tolist(),
            'y': y.tolist(),
            'z': z.tolist(),
            'colors': colors.tolist() if isinstance(colors, np.ndarray) else colors,
            'sizes': sizes.tolist()
        }
    
    def generate_bar3d_data(
        self,
        num_x: int = None,
        num_y: int = None,
        x_labels: list = None,
        y_labels: list = None,
        value_range: tuple = (10, 100)
    ) -> Dict[str, Any]:
        """
        Generate data for 3D bar charts.
        
        Args:
            num_x: Number of bars in x direction (default: random 3-6)
            num_y: Number of bars in y direction (default: random 3-6)
            x_labels: Labels for x axis (auto-generated if None)
            y_labels: Labels for y axis (auto-generated if None)
            value_range: Range for bar heights (min, max)
            
        Returns:
            Dictionary with 'x', 'y', 'z', 'x_labels', 'y_labels' keys
        """
        import numpy as np
        
        num_x = num_x or random.randint(3, 6)
        num_y = num_y or random.randint(3, 6)
        
        # Generate labels if not provided
        if x_labels is None:
            x_labels = random.choice([
                self.QUARTERS[:num_x],
                self.PRODUCTS[:num_x],
                self.REGIONS[:num_x],
                [f'X{i+1}' for i in range(num_x)]
            ])
        
        if y_labels is None:
            y_labels = random.choice([
                self.PRODUCTS[:num_y],
                self.REGIONS[:num_y],
                ['2021', '2022', '2023', '2024', '2025'][:num_y],
                [f'Y{i+1}' for i in range(num_y)]
            ])
        
        # Create grid of bar positions
        x_pos = []
        y_pos = []
        z_heights = []
        
        for i in range(num_x):
            for j in range(num_y):
                x_pos.append(i)
                y_pos.append(j)
                z_heights.append(random.randint(value_range[0], value_range[1]))
        
        return {
            'x': x_pos,
            'y': y_pos,
            'z': z_heights,
            'dx': 0.8,
            'dy': 0.8,
            'x_labels': x_labels,
            'y_labels': y_labels
        }
    def generate_line3d_data(
        self,
        num_points: int = None,
        num_lines: int = 1,
        trajectory_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate data for 3D line plots (trajectories).
        
        Args:
            num_points: Number of points per line (default: random 50-200)
            num_lines: Number of lines
            trajectory_type: 'spiral', 'helix', 'random_walk', 'lissajous', 'parametric'
            
        Returns:
            Dictionary with 'x', 'y', 'z' or 'lines' keys
        """
        import numpy as np
        
        num_points = num_points or random.randint(50, 200)
        
        if trajectory_type is None:
            trajectory_type = random.choice(['spiral', 'helix', 'random_walk', 'lissajous'])
        
        if num_lines == 1:
            t = np.linspace(0, 4*np.pi, num_points)
            
            if trajectory_type == 'helix':
                x = np.cos(t)
                y = np.sin(t)
                z = t / (2*np.pi)
            
            elif trajectory_type == 'spiral':
                r = t / (4*np.pi)
                x = r * np.cos(t * 2)
                y = r * np.sin(t * 2)
                z = t / (4*np.pi)
            
            elif trajectory_type == 'lissajous':
                a, b = random.randint(1, 4), random.randint(1, 4)
                x = np.sin(a * t)
                y = np.cos(b * t)
                z = np.sin((a + b) * t / 2)
            
            else:  # random_walk
                x = np.cumsum(np.random.randn(num_points)) * 0.1
                y = np.cumsum(np.random.randn(num_points)) * 0.1
                z = np.cumsum(np.random.randn(num_points)) * 0.1
            
            return {
                'x': x.tolist(),
                'y': y.tolist(),
                'z': z.tolist()
            }
        else:
            # Multiple lines
            lines = []
            for _ in range(num_lines):
                t = np.linspace(0, 4*np.pi, num_points)
                offset = random.uniform(-2, 2)
                
                x = np.cos(t) + offset
                y = np.sin(t) + offset
                z = t / (2*np.pi) + offset
                
                lines.append([x.tolist(), y.tolist(), z.tolist()])
            
            return {'lines': lines}
    
    def generate_wireframe3d_data(
        self,
        x_range: tuple = (-5, 5),
        y_range: tuple = (-5, 5),
        resolution: int = 30,
        function_type: str = None
    ) -> Dict[str, Any]:
        """Generate data for wireframe (same as surface but rendered differently)."""
        return self.generate_surface3d_data(x_range, y_range, resolution, function_type)
    
    def generate_quiver3d_data(
        self,
        grid_size: tuple = None,
        field_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate data for 3D vector field (quiver) plots.
        
        Args:
            grid_size: (nx, ny, nz) grid dimensions
            field_type: 'radial', 'circular', 'vortex', 'uniform', 'random'
            
        Returns:
            Dictionary with 'x', 'y', 'z', 'u', 'v', 'w' keys
        """
        import numpy as np
        
        grid_size = grid_size or (5, 5, 5)
        nx, ny, nz = grid_size
        
        if field_type is None:
            field_type = random.choice(['radial', 'circular', 'vortex', 'uniform'])
        
        # Create grid
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        z = np.linspace(-2, 2, nz)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Flatten for quiver
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        if field_type == 'radial':
            # Pointing outward from center
            u = x_flat
            v = y_flat
            w = z_flat
        
        elif field_type == 'circular':
            # Circular around z-axis
            u = -y_flat
            v = x_flat
            w = np.zeros_like(z_flat)
        
        elif field_type == 'vortex':
            # Vortex field
            r = np.sqrt(x_flat**2 + y_flat**2)
            u = -y_flat / (r + 0.1)
            v = x_flat / (r + 0.1)
            w = z_flat * 0.5
        
        elif field_type == 'uniform':
            # Uniform field in one direction
            direction = random.choice(['x', 'y', 'z'])
            if direction == 'x':
                u = np.ones_like(x_flat)
                v = np.zeros_like(y_flat)
                w = np.zeros_like(z_flat)
            elif direction == 'y':
                u = np.zeros_like(x_flat)
                v = np.ones_like(y_flat)
                w = np.zeros_like(z_flat)
            else:
                u = np.zeros_like(x_flat)
                v = np.zeros_like(y_flat)
                w = np.ones_like(z_flat)
        
        else:  # random
            u = np.random.randn(len(x_flat))
            v = np.random.randn(len(y_flat))
            w = np.random.randn(len(z_flat))
        
        return {
            'x': x_flat.tolist(),
            'y': y_flat.tolist(),
            'z': z_flat.tolist(),
            'u': u.tolist(),
            'v': v.tolist(),
            'w': w.tolist()
        }
    
    def generate_stem3d_data(
        self,
        num_points: int = None,
        x_range: tuple = (0, 10),
        y_range: tuple = (0, 10),
        z_range: tuple = (0, 100),
        pattern: str = None
    ) -> Dict[str, Any]:
        """
        Generate data for 3D stem plots.
        
        Args:
            num_points: Number of stems (default: random 15-30)
            x_range, y_range, z_range: Ranges for coordinates
            pattern: 'random', 'grid', 'decreasing'
            
        Returns:
            Dictionary with 'x', 'y', 'z' keys
        """
        import numpy as np
        
        num_points = num_points or random.randint(15, 30)
        
        if pattern is None:
            pattern = random.choice(['random', 'grid', 'decreasing'])
        
        if pattern == 'grid':
            # Regular grid
            n_side = int(np.sqrt(num_points))
            x = np.repeat(np.linspace(x_range[0], x_range[1], n_side), n_side)
            y = np.tile(np.linspace(y_range[0], y_range[1], n_side), n_side)
            z = np.random.uniform(z_range[0], z_range[1], n_side * n_side)
        
        elif pattern == 'decreasing':
            # Heights decrease with distance from center
            x = np.random.uniform(x_range[0], x_range[1], num_points)
            y = np.random.uniform(y_range[0], y_range[1], num_points)
            
            cx = (x_range[0] + x_range[1]) / 2
            cy = (y_range[0] + y_range[1]) / 2
            
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_dist = np.max(dist)
            z = z_range[1] * (1 - dist / max_dist)
        
        else:  # random
            x = np.random.uniform(x_range[0], x_range[1], num_points)
            y = np.random.uniform(y_range[0], y_range[1], num_points)
            z = np.random.uniform(z_range[0], z_range[1], num_points)
        
        return {
            'x': x.tolist(),
            'y': y.tolist(),
            'z': z.tolist()
        }
    
    def generate_parametric3d_data(
        self,
        surface_type: str = None,
        resolution: int = 50
    ) -> Dict[str, Any]:
        """
        Generate parametric surface data (torus, sphere, etc.).
        
        Args:
            surface_type: 'torus', 'sphere', 'mobius', 'klein_bottle', 'figure8'
            resolution: Grid resolution
            
        Returns:
            Dictionary with 'x', 'y', 'z', 'surface_type' keys
        """
        import numpy as np
        
        if surface_type is None:
            surface_type = random.choice(['torus', 'sphere', 'figure8', 'mobius'])
        
        if surface_type == 'torus':
            # Torus parameters
            R = 3  # Major radius
            r = 1  # Minor radius
            
            u = np.linspace(0, 2*np.pi, resolution)
            v = np.linspace(0, 2*np.pi, resolution)
            U, V = np.meshgrid(u, v)
            
            X = (R + r * np.cos(V)) * np.cos(U)
            Y = (R + r * np.cos(V)) * np.sin(U)
            Z = r * np.sin(V)
        
        elif surface_type == 'sphere':
            u = np.linspace(0, 2*np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            U, V = np.meshgrid(u, v)
            
            radius = 3
            X = radius * np.sin(V) * np.cos(U)
            Y = radius * np.sin(V) * np.sin(U)
            Z = radius * np.cos(V)
        
        elif surface_type == 'mobius':
            u = np.linspace(0, 2*np.pi, resolution)
            v = np.linspace(-1, 1, resolution//2)
            U, V = np.meshgrid(u, v)
            
            X = (2 + V * np.cos(U/2)) * np.cos(U)
            Y = (2 + V * np.cos(U/2)) * np.sin(U)
            Z = V * np.sin(U/2)
        
        elif surface_type == 'figure8':
            u = np.linspace(0, 2*np.pi, resolution)
            v = np.linspace(0, 2*np.pi, resolution)
            U, V = np.meshgrid(u, v)
            
            X = (2 + np.cos(V)) * np.cos(U)
            Y = (2 + np.cos(V)) * np.sin(U)
            Z = np.sin(V) * np.sin(U)
        
        else:  # default to torus
            R, r = 3, 1
            u = np.linspace(0, 2*np.pi, resolution)
            v = np.linspace(0, 2*np.pi, resolution)
            U, V = np.meshgrid(u, v)
            
            X = (R + r * np.cos(V)) * np.cos(U)
            Y = (R + r * np.cos(V)) * np.sin(U)
            Z = r * np.sin(V)
        
        return {
            'x': X.tolist(),
            'y': Y.tolist(),
            'z': Z.tolist(),
            'surface_type': surface_type
        }
    
    def generate_contour3d_data(
        self,
        x_range: tuple = (-5, 5),
        y_range: tuple = (-5, 5),
        resolution: int = 50,
        function_type: str = None
    ) -> Dict[str, Any]:
        """Generate data for 3D contour (same as surface)."""
        return self.generate_surface3d_data(x_range, y_range, resolution, function_type)
    
    def generate_volume3d_data(
        self,
        grid_size: tuple = None,
        volume_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate volumetric data (3D voxel data).
        
        Args:
            grid_size: (nx, ny, nz) voxel grid dimensions
            volume_type: 'sphere', 'gaussian', 'random', 'gradient'
            
        Returns:
            Dictionary with 'volume', 'threshold' keys
        """
        import numpy as np
        
        grid_size = grid_size or (10, 10, 10)
        nx, ny, nz = grid_size
        
        if volume_type is None:
            volume_type = random.choice(['sphere', 'gaussian', 'gradient'])
        
        # Create coordinate grids
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        z = np.linspace(-1, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        if volume_type == 'sphere':
            # Solid sphere
            R = np.sqrt(X**2 + Y**2 + Z**2)
            volume = (R < 0.8).astype(float)
        
        elif volume_type == 'gaussian':
            # Gaussian blob
            volume = np.exp(-(X**2 + Y**2 + Z**2) / 0.5)
        
        elif volume_type == 'gradient':
            # Linear gradient
            volume = (X + Y + Z + 3) / 6
        
        else:  # random
            volume = np.random.rand(nx, ny, nz)
        
        return {
            'volume': volume.tolist(),
            'threshold': 0.5
        }
    
    def generate_network3d_data(
        self,
        num_nodes: int = None,
        connection_probability: float = 0.15
    ) -> Dict[str, Any]:
        """
        Generate 3D network graph data.
        
        Args:
            num_nodes: Number of nodes (default: random 10-25)
            connection_probability: Probability of edge between nodes
            
        Returns:
            Dictionary with 'nodes', 'edges' keys
        """
        import numpy as np
        
        num_nodes = num_nodes or random.randint(10, 25)
        
        # Generate random node positions in 3D
        nodes = np.random.uniform(-5, 5, (num_nodes, 3))
        
        # Generate edges based on distance or random probability
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < connection_probability:
                    edges.append([i, j])
        
        # Generate node sizes (could be degree-based)
        node_degrees = [0] * num_nodes
        for edge in edges:
            node_degrees[edge[0]] += 1
            node_degrees[edge[1]] += 1
        
        node_sizes = [50 + deg * 20 for deg in node_degrees]
        
        return {
            'nodes': nodes.tolist(),
            'edges': edges,
            'node_sizes': node_sizes
        }
        
    def generate_hist2d_data(
        self,
        num_points: int = None,
        x_range: tuple = (0, 100),
        y_range: tuple = (0, 100),
        distribution: str = None,
        bins: list = None
    ) -> Dict[str, Any]:
        """
        Generate data for 2D histogram charts.
        
        Args:
            num_points: Number of data points (default: 200)
            x_range: Range for x values (min, max)
            y_range: Range for y values (min, max)
            distribution: Type of distribution 
                - 'random': Uniform random distribution
                - 'clustered': Multiple clusters/concentrations
                - 'diagonal': Concentrated along diagonal
                - 'circular': Circular/radial pattern
                - 'normal': Bivariate normal distribution
            bins: Number of bins [x_bins, y_bins] (default: [20, 20])
            
        Returns:
            Dictionary with 'x', 'y', and 'bins' keys
        """
        import math
        
        count = num_points or 200
        bins = bins or [20, 20]
        
        x = []
        y = []
        
        if distribution == 'random':
            # Uniform random distribution
            x = [random.uniform(x_range[0], x_range[1]) for _ in range(count)]
            y = [random.uniform(y_range[0], y_range[1]) for _ in range(count)]
            
        elif distribution == 'clustered':
            # Multiple clusters
            num_clusters = random.randint(2, 4)
            points_per_cluster = count // num_clusters
            
            for _ in range(num_clusters):
                # Random cluster center
                cx = random.uniform(x_range[0] + 10, x_range[1] - 10)
                cy = random.uniform(y_range[0] + 10, y_range[1] - 10)
                
                # Generate points around cluster center
                for _ in range(points_per_cluster):
                    # Use Gaussian distribution around center
                    px = random.gauss(cx, (x_range[1] - x_range[0]) / 10)
                    py = random.gauss(cy, (y_range[1] - y_range[0]) / 10)
                    
                    # Clamp to range
                    px = max(x_range[0], min(x_range[1], px))
                    py = max(y_range[0], min(y_range[1], py))
                    
                    x.append(px)
                    y.append(py)
            
            # Fill remaining points
            remaining = count - len(x)
            for _ in range(remaining):
                x.append(random.uniform(x_range[0], x_range[1]))
                y.append(random.uniform(y_range[0], y_range[1]))
        
        elif distribution == 'diagonal':
            # Concentrated along diagonal with noise
            for _ in range(count):
                t = random.uniform(0, 1)
                px = x_range[0] + t * (x_range[1] - x_range[0])
                py = y_range[0] + t * (y_range[1] - y_range[0])
                
                # Add noise
                noise_x = random.gauss(0, (x_range[1] - x_range[0]) / 15)
                noise_y = random.gauss(0, (y_range[1] - y_range[0]) / 15)
                
                px = max(x_range[0], min(x_range[1], px + noise_x))
                py = max(y_range[0], min(y_range[1], py + noise_y))
                
                x.append(px)
                y.append(py)
        
        elif distribution == 'circular':
            # Circular/radial pattern
            cx = (x_range[0] + x_range[1]) / 2
            cy = (y_range[0] + y_range[1]) / 2
            max_radius = min(x_range[1] - cx, y_range[1] - cy) * 0.8
            
            for _ in range(count):
                # Random angle and radius
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, max_radius)
                
                # Add some clustering at certain radii
                if random.random() < 0.3:
                    radius = random.uniform(max_radius * 0.6, max_radius * 0.9)
                
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                
                x.append(px)
                y.append(py)
        
        elif distribution == 'normal':
            # Bivariate normal distribution
            mean_x = (x_range[0] + x_range[1]) / 2
            mean_y = (y_range[0] + y_range[1]) / 2
            std_x = (x_range[1] - x_range[0]) / 6
            std_y = (y_range[1] - y_range[0]) / 6
            
            for _ in range(count):
                px = random.gauss(mean_x, std_x)
                py = random.gauss(mean_y, std_y)
                
                # Clamp to range
                px = max(x_range[0], min(x_range[1], px))
                py = max(y_range[0], min(y_range[1], py))
                
                x.append(px)
                y.append(py)
        
        else:
            # Default to random
            x = [random.uniform(x_range[0], x_range[1]) for _ in range(count)]
            y = [random.uniform(y_range[0], y_range[1]) for _ in range(count)]
        
        return {
            'x': x,
            'y': y,
            'bins': bins
        }
    
    def generate_cohere_data(
        self,
        signal_type: str = None,
        duration: float = 1.0,
        Fs: int = 1000,
        NFFT: int = 256
    ) -> Dict[str, Any]:
        """
        Generate two signals for coherence analysis.
        
        Args:
            signal_type: Type of signal relationship
                - 'perfectly_correlated': x and y are identical (coherence ≈ 1)
                - 'highly_correlated': x and y share most frequency components
                - 'partially_correlated': Some frequency bands are correlated
                - 'frequency_dependent': Correlation varies by frequency
                - 'uncorrelated': Independent noise signals (coherence ≈ 0)
                - 'phase_shifted': Same frequencies, different phases
                - 'mixed': Correlated signal + independent noise
            duration: Signal duration in seconds (default: 1.0)
            Fs: Sampling frequency in Hz (default: 1000)
            NFFT: FFT length for coherence calculation (default: 256)
            
        Returns:
            Dictionary with 'x', 'y', 'Fs', and 'NFFT' keys
        """
        import numpy as np
        
        # Generate time array
        t = np.linspace(0, duration, int(Fs * duration))
        
        if signal_type == 'perfectly_correlated':
            # Identical signals - perfect coherence
            freqs = [10, 25, 50, 75]  # Hz
            x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
            y = x.copy()
            
        elif signal_type == 'highly_correlated':
            # Very similar signals with small noise
            freqs = [10, 25, 50, 75]
            x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
            y = x + 0.1 * np.random.randn(len(t))  # Add 10% noise
            
        elif signal_type == 'partially_correlated':
            # Share some frequency components
            common_freqs = [10, 30]
            x_freqs = [10, 30, 60]
            y_freqs = [10, 30, 80]
            
            x = sum(np.sin(2 * np.pi * f * t) for f in x_freqs)
            y = sum(np.sin(2 * np.pi * f * t) for f in y_freqs)
            
        elif signal_type == 'frequency_dependent':
            # Low frequencies correlated, high frequencies not
            # Low frequency component (correlated)
            low_freq = 15
            x_low = np.sin(2 * np.pi * low_freq * t)
            y_low = x_low.copy()
            
            # High frequency components (uncorrelated)
            x_high = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
            y_high = np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 160 * t)
            
            x = x_low + 0.5 * x_high
            y = y_low + 0.5 * y_high
            
        elif signal_type == 'uncorrelated':
            # Pure random noise - no correlation
            x = np.random.randn(len(t))
            y = np.random.randn(len(t))
            
        elif signal_type == 'phase_shifted':
            # Same frequencies, different phases
            freqs = [15, 35, 65]
            x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
            # Phase shift by 90 degrees (π/2)
            y = sum(np.sin(2 * np.pi * f * t + np.pi/2) for f in freqs)
            
        elif signal_type == 'mixed':
            # Correlated signal + independent noise
            freqs = [20, 40, 60]
            signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
            
            noise_level_x = 0.3
            noise_level_y = 0.3
            
            x = signal + noise_level_x * np.random.randn(len(t))
            y = signal + noise_level_y * np.random.randn(len(t))
            
        else:
            # Default to mixed
            freqs = [20, 40, 60]
            signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
            x = signal + 0.3 * np.random.randn(len(t))
            y = signal + 0.3 * np.random.randn(len(t))
        
        # Normalize signals
        x = x / np.max(np.abs(x))
        y = y / np.max(np.abs(y))
        
        return {
            'x': x.tolist(),
            'y': y.tolist(),
            'Fs': Fs,
            'NFFT': NFFT
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
    
    def generate_timeline_data(
        self,
        num_events: int = None,
        date_range: tuple = ('2024-01', '2024-12'),
        event_type: str = 'project',
        use_categories: bool = None
    ) -> Dict[str, Any]:
        """
        Generate data for timeline charts.
        
        Args:
            num_events: Number of events (default: random 4-8)
            date_range: Tuple of (start_date, end_date) as strings
            event_type: Type of events ('project', 'product', 'company', 'custom')
            use_categories: Whether to categorize events (random if None)
            
        Returns:
            Dictionary with 'dates', 'events', and optionally 'categories' keys
        """
        from datetime import datetime, timedelta
        
        count = num_events if num_events else random.randint(4, 8)
        
        # Parse date range
        start_str, end_str = date_range
        start_date = datetime.strptime(start_str, '%Y-%m')
        end_date = datetime.strptime(end_str, '%Y-%m')
        
        # Generate random dates
        date_diff = (end_date - start_date).days
        dates = []
        for _ in range(count):
            random_days = random.randint(0, date_diff)
            event_date = start_date + timedelta(days=random_days)
            dates.append(event_date.strftime('%Y-%m-%d'))
        
        # Sort dates
        dates.sort()
        
        # Generate events based on type
        if event_type == 'project':
            event_templates = [
                'Project Kickoff', 'Design Review', 'Development Phase',
                'Testing Complete', 'Beta Release', 'Production Launch',
                'Milestone Achieved', 'Phase Completion', 'Deadline Extended'
            ]
        elif event_type == 'product':
            event_templates = [
                'Product Launch', 'Feature Release', 'Update v2.0',
                'Bug Fix Release', 'Major Update', 'Beta Testing',
                'Market Expansion', 'Partnership Announced', 'Rebranding'
            ]
        elif event_type == 'company':
            event_templates = [
                'Company Founded', 'Series A Funding', 'New Office Opened',
                'CEO Appointed', 'Acquisition Completed', 'IPO Announced',
                'Team Expansion', 'Award Received', 'Strategic Partnership'
            ]
        else:  # custom
            event_templates = [
                'Event 1', 'Event 2', 'Event 3', 'Event 4',
                'Milestone A', 'Milestone B', 'Milestone C', 'Milestone D'
            ]
        
        events = random.sample(event_templates, min(count, len(event_templates)))
        if len(events) < count:
            events.extend([f'Event {i+1}' for i in range(len(events), count)])
        
        result = {'dates': dates, 'events': events}
        
        # Add categories if requested
        if use_categories is None:
            use_categories = random.choice([True, False])
        
        if use_categories:
            category_types = ['Planning', 'Development', 'Testing', 'Release', 'Marketing']
            categories = [random.choice(category_types) for _ in range(count)]
            result['categories'] = categories
        
        return result
    
    def generate_heatmap_data(
        self,
        num_rows: int = None,
        num_cols: int = None,
        value_range: tuple = (0, 100),
        row_type: str = None,
        col_type: str = None
    ) -> Dict[str, Any]:
        """
        Generate data for annotated heatmap charts.
        
        Args:
            num_rows: Number of rows (auto-determined if None)
            num_cols: Number of columns (auto-determined if None)
            value_range: Range for values (min, max)
            row_type: Type of row labels ('products', 'regions', 'departments', 'months', 'custom')
            col_type: Type of column labels ('quarters', 'months', 'regions', 'products', 'years', 'custom')
            
        Returns:
            Dictionary with 'values', 'row_labels', 'col_labels' keys
        """
        import numpy as np
        
        # Randomize types if not specified
        if row_type is None:
            row_type = random.choice(['products', 'regions', 'departments', 'months'])
        if col_type is None:
            col_type = random.choice(['quarters', 'months', 'regions', 'years'])
        
        # Determine row labels
        if row_type == 'products':
            row_labels = self.PRODUCTS[:num_rows] if num_rows else self.PRODUCTS[:5]
        elif row_type == 'regions':
            row_labels = self.REGIONS[:num_rows] if num_rows else self.REGIONS[:5]
        elif row_type == 'departments':
            row_labels = self.DEPARTMENTS[:num_rows] if num_rows else self.DEPARTMENTS[:5]
        elif row_type == 'months':
            row_labels = self.MONTHS[:num_rows] if num_rows else self.MONTHS[:6]
        else:  # custom
            count = num_rows or 5
            row_labels = [f'Row {i+1}' for i in range(count)]
        
        # Determine column labels
        if col_type == 'quarters':
            col_labels = self.QUARTERS[:num_cols] if num_cols else self.QUARTERS
        elif col_type == 'months':
            col_labels = self.MONTHS[:num_cols] if num_cols else self.MONTHS[:6]
        elif col_type == 'regions':
            col_labels = self.REGIONS[:num_cols] if num_cols else self.REGIONS[:4]
        elif col_type == 'products':
            col_labels = self.PRODUCTS[:num_cols] if num_cols else self.PRODUCTS[:4]
        elif col_type == 'years':
            col_labels = self.YEARS[:num_cols] if num_cols else self.YEARS[-5:]
        else:  # custom
            count = num_cols or 4
            col_labels = [f'Col {i+1}' for i in range(count)]
        
        # Generate values matrix
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        
        values = []
        for i in range(n_rows):
            row = [random.randint(value_range[0], value_range[1]) for _ in range(n_cols)]
            values.append(row)
        
        return {
            'values': values,
            'row_labels': row_labels,
            'col_labels': col_labels
        }
    
    def generate_streamplot_data(
        self,
        flow_type: str = None,
        grid_size: int = 20,
        domain: tuple = (-2, 2, -2, 2)
    ) -> Dict[str, Any]:
        """
        Generate vector field data for streamplot charts.
        
        Args:
            flow_type: Type of flow pattern (None for random choice):
                - 'vortex': Circular rotation around center
                - 'source': Outward radial flow
                - 'sink': Inward radial flow
                - 'saddle': Saddle point (hyperbolic)
                - 'uniform': Parallel flow
                - 'dipole': Source + Sink combination
                - 'shear': Velocity gradient
                - 'wave': Oscillating pattern
            grid_size: Number of grid points per dimension
            domain: (x_min, x_max, y_min, y_max) for the field
            
        Returns:
            Dictionary with 'x', 'y', 'u', 'v' arrays
        """
        import numpy as np
        
        # Randomize flow type if not specified
        if flow_type is None:
            flow_type = random.choice(['vortex', 'source', 'sink', 'saddle', 
                                      'uniform', 'dipole', 'shear', 'wave', 'turbulent'])
        
        # Create grid
        x_min, x_max, y_min, y_max = domain
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Generate flow field based on type
        if flow_type == 'vortex':
            # Circular rotation: u = -y, v = x
            u = -Y
            v = X
            
        elif flow_type == 'source':
            # Radial outward: u = x, v = y
            u = X
            v = Y
            
        elif flow_type == 'sink':
            # Radial inward: u = -x, v = -y
            u = -X
            v = -Y
            
        elif flow_type == 'saddle':
            # Hyperbolic saddle point: u = x, v = -y
            u = X
            v = -Y
            
        elif flow_type == 'uniform':
            # Parallel flow at random angle
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(0.5, 2.0)
            u = np.ones_like(X) * speed * np.cos(angle)
            v = np.ones_like(Y) * speed * np.sin(angle)
            
        elif flow_type == 'turbulent':
            # Create turbulent flow using summed sine waves
            base_speed = 1.0
            u = np.ones_like(X) * base_speed
            v = np.zeros_like(Y)
            
            # Add multiple frequency components (turbulent cascade)
            for scale in [1, 2, 4, 8]:
                amplitude = 0.5 / scale
                u += amplitude * np.sin(X * scale + np.random.uniform(0, 2*np.pi))
                u += amplitude * np.cos(Y * scale + np.random.uniform(0, 2*np.pi))
                v += amplitude * np.sin(Y * scale + np.random.uniform(0, 2*np.pi))
                v += amplitude * np.cos(X * scale + np.random.uniform(0, 2*np.pi))
            
            # Add some rotation/vorticity
            u += -0.3 * Y * np.exp(-(X**2 + Y**2)/10)
            v += 0.3 * X * np.exp(-(X**2 + Y**2)/10)
            
        elif flow_type == 'dipole':
            # Source at (-1, 0) and sink at (1, 0)
            # Source contribution
            r1_sq = (X + 1)**2 + Y**2 + 0.1  # Add small value to avoid division by zero
            u_source = (X + 1) / r1_sq
            v_source = Y / r1_sq
            
            # Sink contribution
            r2_sq = (X - 1)**2 + Y**2 + 0.1
            u_sink = -(X - 1) / r2_sq
            v_sink = -Y / r2_sq
            
            u = u_source + u_sink
            v = v_source + v_sink
            
        elif flow_type == 'shear':
            # Velocity varies with y (like wind shear)
            u = Y + random.uniform(0.5, 1.5)
            v = np.zeros_like(Y) + random.uniform(-0.3, 0.3)
            
        elif flow_type == 'wave':
            # Oscillating pattern
            freq = random.uniform(1, 3)
            u = np.sin(freq * Y) * np.cos(freq * X)
            v = np.cos(freq * Y) * np.sin(freq * X)
        
        else:
            # Default to vortex
            u = -Y
            v = X
        
        # Add small random noise for realism (5% of mean magnitude)
        magnitude = np.sqrt(u**2 + v**2)
        noise_level = 0.05 * np.mean(magnitude)
        u += np.random.normal(0, noise_level, u.shape)
        v += np.random.normal(0, noise_level, v.shape)
        
        return {
            'x': x.tolist(),
            'y': y.tolist(),
            'u': u.tolist(),
            'v': v.tolist(),
            'flow_type': flow_type
        }
    
    def generate_streamplot_variation_data(
        self,
        variation_type: str = None,
        flow_type: str = None,
        grid_size: int = 30
    ) -> Dict[str, Any]:
        """
        Generate streamplot data with specific variations.
        
        Args:
            variation_type: Type of variation:
                - 'varying_density': Different densities
                - 'varying_color': Different colormaps
                - 'varying_linewidth': Different line widths
                - 'starting_points': Custom start points
                - 'masking': With masked region
                - 'unbroken': Dense unbroken streamlines
            flow_type: Base flow pattern
            grid_size: Grid resolution
            
        Returns:
            Dictionary with data and visualization parameters
        """
        import numpy as np
        
        # Generate base flow field
        base_data = self.generate_streamplot_data(
            flow_type=flow_type,
            grid_size=grid_size,
            domain=(-3, 3, -3, 3)
        )
        
        x = np.array(base_data['x'])
        y = np.array(base_data['y'])
        u = np.array(base_data['u'])
        v = np.array(base_data['v'])
        
        result = {
            'x': x.tolist(),
            'y': y.tolist(),
            'u': u.tolist(),
            'v': v.tolist(),
            'flow_type': flow_type
        }
        
        # Apply variation
        if variation_type == 'varying_density':
            # Sparse to dense
            result['density'] = random.choice([0.5, 1.0, 1.5, 2.0, 2.5])
            result['color'] = 'uniform'
            result['linewidth'] = 1.5
            
        elif variation_type == 'varying_color':
            # Color by field with different colormaps
            result['color'] = 'velocity'
            result['cmap'] = random.choice(['viridis', 'plasma', 'RdYlBu', 'coolwarm', 'jet'])
            result['linewidth'] = 1.5
            
        elif variation_type == 'varying_linewidth':
            # Variable line widths
            result['linewidth'] = 'variable'
            result['color'] = 'uniform'
            result['density'] = 1.5
            
        elif variation_type == 'starting_points':
            # Manual starting points
            num_points = random.randint(5, 10)
            start_x = np.random.uniform(-2, 2, num_points)
            start_y = np.random.uniform(-2, 2, num_points)
            result['start_points'] = [[float(sx), float(sy)] for sx, sy in zip(start_x, start_y)]
            result['show_start_points'] = True
            result['color'] = 'velocity'
            result['density'] = 1.0
            
        elif variation_type == 'masking':
            # Create circular or rectangular mask
            X, Y = np.meshgrid(x, y)
            if random.choice([True, False]):
                # Circular mask
                mask_x, mask_y = random.uniform(-1, 1), random.uniform(-1, 1)
                mask_r = random.uniform(0.5, 1.2)
                mask = (X - mask_x)**2 + (Y - mask_y)**2 < mask_r**2
                result['mask_shape'] = 'circle'
                result['mask_params'] = {'x': mask_x, 'y': mask_y, 'r': mask_r}
            else:
                # Rectangular mask
                x_min, x_max = sorted([random.uniform(-2, 0), random.uniform(0, 2)])
                y_min, y_max = sorted([random.uniform(-2, 0), random.uniform(0, 2)])
                mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
                result['mask_shape'] = 'rectangle'
                result['mask_params'] = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
            
            result['mask'] = mask.tolist()
            result['color'] = random.choice(['#FF4444', '#44FF44', '#4444FF'])
            result['density'] = 1.5
            result['show_mask_region'] = True
            result['mask_color'] = 'gray'
            
        elif variation_type == 'unbroken':
            # Very dense unbroken streamlines
            result['density'] = (3.0, 3.0)  # High density
            result['broken_streamlines'] = False
            result['color'] = 'uniform'
            result['linewidth'] = 1.0
        
        return result
    
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
        chart_types = ['line', 'bar', 'horizontal_bar', 'pie', 'scatter', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution', 'cumulative_distribution', 'time_series_histogram', 'treemap', 'surface3d', 'scatter3d', 'bar3d', 'line3d', 'wireframe3d', 'quiver3d', 'stem3d', 'parametric3d', 'contour3d', 'volume3d', 'network3d', 'hist2d', 'cohere', 'signal_pair', 'timeline', 'heatmap', 'streamplot']
        
        if chart_type is None:
            chart_type = random.choice(chart_types)
        
        if filename_root is None:
            random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            filename_root = f"{chart_type}_chart_{random_id}"
        
        if chart_type == 'line':
            num_series = random.choice([1, 1, 2, 3, 4])  # Bias toward single series
            x_type = random.choice(['months', 'quarters', 'numeric'])
            data = self.generate_line_data(
                num_series=num_series,
                x_type=x_type,
                trend=random.choice(['random', 'increasing', 'fluctuating', 'decreasing'])
            )
            return {
                'chart_type': 'line',
                'data': data,
                'filename_root': filename_root,
                'title': f'{random.choice(["Sales", "Revenue", "Performance", "Growth"])} Over Time',
                'xlabel': {'months': 'Month', 'quarters': 'Quarter', 'numeric': 'Time Period'}.get(x_type, 'X-Axis'),
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
        
        elif chart_type == 'box':
            # Generate box plot data
            data = self.generate_box_data(
                num_boxes=random.choice([3, 4, 5]),
                value_range=(20, 100)
            )
            
            # Determine label type from first label
            first_label = data['labels'][0]
            
            if first_label in self.PRODUCTS:
                label_type = 'Product'
                title_suffix = 'Product Performance Distribution'
            elif first_label in self.REGIONS:
                label_type = 'Region'
                title_suffix = 'Regional Performance Distribution'
            elif first_label in self.DEPARTMENTS:
                label_type = 'Department'
                title_suffix = 'Department Performance Distribution'
            elif first_label in self.QUARTERS:
                label_type = 'Quarter'
                title_suffix = 'Quarterly Performance Distribution'
            elif first_label in self.MONTHS:
                label_type = 'Month'
                title_suffix = 'Monthly Performance Distribution'
            else:
                label_type = 'Group'
                title_suffix = 'Performance Distribution'
            
            return {
                'chart_type': 'box',
                'data': data,
                'filename_root': filename_root,
                'title': title_suffix,
                'xlabel': label_type,
                'ylabel': random.choice(['Value', 'Performance Score', 'Sales ($K)', 'Response Time (ms)', 'Rating']),
                'metadata': {
                    'generated': 'random',
                    'num_boxes': len(data['labels']),
                    'samples_per_box': [len(d) for d in data['data']]
                }
            }
        
        elif chart_type == 'area':
            # Generate area chart data - exclude 'single' type
            area_type = random.choice(['range', 'stacked', 'stacked_100', 'overlapping'])
            x_type = random.choice(['months', 'quarters', 'numeric'])
            use_step = random.choice([True, False, False])  # 33% chance of step
            
            if area_type == 'stacked':
                num_series = random.choice([2, 3])
                data = self.generate_area_data(
                    area_type='stacked',
                    x_type=x_type,
                    num_series=num_series,
                    y_range=(20, 80),
                    step=use_step
                )
                title = 'Stacked Performance Over Time'
                ylabel = random.choice(['Total Value', 'Combined Sales ($K)', 'Cumulative Score'])
                
            elif area_type == 'stacked_100':
                num_series = random.choice([2, 3, 4])
                data = self.generate_area_data(
                    area_type='stacked_100',
                    x_type=x_type,
                    num_series=num_series,
                    y_range=(20, 80),
                    step=use_step
                )
                title = 'Percentage Breakdown Over Time'
                ylabel = 'Percentage (%)'
                
            elif area_type == 'overlapping':
                num_series = random.choice([2, 3])
                data = self.generate_area_data(
                    area_type='overlapping',
                    x_type=x_type,
                    num_series=num_series,
                    y_range=(30, 120),
                    step=use_step
                )
                title = random.choice(['Performance Comparison', 'Overlapping Trends', 'Multi-Series Analysis'])
                ylabel = random.choice(['Value', 'Sales ($K)', 'Performance Score'])
                
            elif area_type == 'range':
                data = self.generate_area_data(
                    area_type='range',
                    x_type=x_type,
                    y_range=(30, 100),
                    step=use_step
                )
                title = random.choice(['Performance Range', 'Confidence Interval', 'Value Range Over Time'])
                ylabel = random.choice(['Value', 'Sales ($K)', 'Performance Score', 'Metric'])
            
            # Determine xlabel based on x_type
            if x_type == 'months':
                xlabel = 'Month'
            elif x_type == 'quarters':
                xlabel = 'Quarter'
            elif x_type == 'years':
                xlabel = 'Year'
            else:
                xlabel = 'Time Period'
            
            return {
                'chart_type': 'area',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'metadata': {
                    'generated': 'random',
                    'area_type': area_type,
                    'num_points': len(data.get('x', [])),
                    'step': use_step
                }
            }
        
        elif chart_type == 'discrete_distribution':
            dist_type = random.choice(['binomial', 'poisson', 'uniform', 'rating', 'score', 'dice'])
            
            if dist_type == 'binomial':
                data = self.generate_discrete_distribution_data('binomial', n=random.choice([8, 10, 12, 15]), p=round(random.uniform(0.3, 0.7), 1))
                title = f'Binomial Distribution'
            elif dist_type == 'poisson':
                data = self.generate_discrete_distribution_data('poisson', lambda_rate=round(random.uniform(2.0, 6.0), 1))
                title = f'Poisson Distribution'
            elif dist_type == 'uniform':
                data = self.generate_discrete_distribution_data('uniform', min_val=1, max_val=random.choice([6, 8, 10]))
                title = f'Discrete Uniform Distribution'
            elif dist_type == 'rating':
                data = self.generate_discrete_distribution_data('rating', skew=random.choice(['positive', 'neutral']))
                title = 'Customer Rating Distribution'
            elif dist_type == 'dice':
                data = self.generate_discrete_distribution_data('dice', num_dice=random.choice([1, 2, 3]), sides=random.choice([6, 8, 10]))
                title = 'Dice Roll Distribution'
            else:  # score
                data = self.generate_discrete_distribution_data('score', grade_type=random.choice(['normal', 'easy', 'hard']))
                title = 'Grade Distribution'
            
            return {
                'chart_type': 'discrete_distribution',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': 'Probability',
                'ylabel': random.choice(['Value', 'Outcome', 'Category']),
                'metadata': {
                    'generated': 'random',
                    'distribution_type': dist_type
                }
            }
        
        elif chart_type == 'cumulative_distribution':
            dist_type = random.choice(['binomial', 'poisson', 'uniform', 'rating', 'score'])
            
            if dist_type == 'binomial':
                data = self.generate_cumulative_distribution_data('binomial', n=random.choice([8, 10, 12, 15]), p=round(random.uniform(0.3, 0.7), 1))
                title = 'Binomial CDF'
            elif dist_type == 'poisson':
                data = self.generate_cumulative_distribution_data('poisson', lambda_rate=round(random.uniform(2.0, 6.0), 1))
                title = 'Poisson CDF'
            elif dist_type == 'uniform':
                data = self.generate_cumulative_distribution_data('uniform', min_val=1, max_val=random.choice([6, 8, 10]))
                title = 'Discrete Uniform CDF'
            elif dist_type == 'rating':
                data = self.generate_cumulative_distribution_data('rating', skew=random.choice(['positive', 'neutral']))
                title = 'Customer Rating CDF'
            else:  # score
                data = self.generate_cumulative_distribution_data('score', grade_type=random.choice(['normal', 'easy', 'hard']))
                title = 'Grade CDF'
            
            return {
                'chart_type': 'cumulative_distribution',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': random.choice(['Value', 'Outcome', 'Score']),
                'ylabel': 'Cumulative Probability',
                'metadata': {
                    'generated': 'random',
                    'distribution_type': dist_type
                }
            }
        
        elif chart_type == 'time_series_histogram':
            # Generate time series histogram data
            trend_type = random.choice(['random', 'increasing', 'decreasing', 'cyclical', 
                                       'volatility_increase', 'volatility_decrease'])
            volatility = random.choice(['low', 'medium', 'high'])
            num_times = random.choice([8, 10, 12, 15])
            
            data = self.generate_time_series_histogram_data(
                num_time_points=num_times,
                samples_per_time=random.randint(150, 300),
                value_range=(0, 100),
                trend=trend_type,
                volatility=volatility,
                bins=random.choice([15, 20, 25])
            )
            
            # Create descriptive title based on trend
            trend_titles = {
                'random': 'Random Walk Distribution',
                'increasing': 'Upward Trend Distribution',
                'decreasing': 'Downward Trend Distribution',
                'cyclical': 'Cyclical Distribution Pattern',
                'volatility_increase': 'Increasing Volatility Over Time',
                'volatility_decrease': 'Decreasing Volatility Over Time'
            }
            
            title = trend_titles.get(trend_type, 'Distribution Evolution Over Time')
            
            return {
                'chart_type': 'time_series_histogram',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': 'Time Period',
                'ylabel': random.choice(['Value', 'Measurement', 'Score', 'Performance']),
                'metadata': {
                    'generated': 'random',
                    'trend': trend_type,
                    'volatility': volatility,
                    'num_time_points': num_times
                }
            }
        
        elif chart_type == 'treemap':
            # Generate treemap data (hierarchical by default for better appearance)
            distribution = random.choice(['power_law', 'exponential', 'uniform'])  # Bias toward power_law
            num_categories = random.choice([10, 12, 14, 16])
            
            data = self.generate_treemap_data(
                num_categories=num_categories,
                category_type=None,  # Use hierarchical labels (A-1, A-2, B-1, etc.)
                value_range=(50, 1000),
                distribution=distribution,
                hierarchical=True  # Enable hierarchical grouping
            )
            
            # Use generic title for hierarchical treemaps
            title = random.choice([
                'Hierarchical Data Distribution',
                'Category Breakdown',
                'Market Share Analysis',
                'Resource Allocation',
                'Portfolio Distribution'
            ])
            
            return {
                'chart_type': 'treemap',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': '',
                'ylabel': '',
                'metadata': {
                    'generated': 'random',
                    'hierarchical': True,
                    'distribution': distribution,
                    'num_categories': num_categories
                }
            }
                
        elif chart_type == 'surface3d':
            # Generate 3D surface data
            function_type = random.choice(['gaussian', 'saddle', 'wave', 'peaks', 'ripple'])
            
            data = self.generate_surface3d_data(
                x_range=(-5, 5),
                y_range=(-5, 5),
                resolution=random.choice([40, 50, 60]),
                function_type=function_type
            )
            
            # Create descriptive title
            title_map = {
                'gaussian': 'Gaussian Peak Surface',
                'saddle': 'Hyperbolic Paraboloid (Saddle)',
                'wave': 'Wave Pattern Surface',
                'peaks': 'Complex Peaks Function',
                'ripple': 'Ripple Effect Surface'
            }
            
            title = title_map.get(function_type, '3D Surface Plot')
            
            return {
                'chart_type': 'surface3d',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': 'X Axis',
                'ylabel': 'Y Axis',
                'zlabel': 'Z Value',
                'metadata': {
                    'generated': 'random',
                    'function_type': function_type
                }
            }
        
        elif chart_type == 'scatter3d':
            # Generate 3D scatter data
            distribution = random.choice(['uniform', 'clustered', 'spherical', 'spiral'])
            num_points = random.choice([80, 120, 150, 200])
            
            data = self.generate_scatter3d_data(
                num_points=num_points,
                x_range=(0, 100),
                y_range=(0, 100),
                z_range=(0, 100),
                distribution=distribution,
                num_clusters=random.randint(2, 4)
            )
            
            # Create descriptive title
            title_map = {
                'uniform': '3D Scatter Plot - Uniform Distribution',
                'clustered': '3D Scatter Plot - Clustered Data',
                'spherical': '3D Scatter Plot - Spherical Pattern',
                'spiral': '3D Scatter Plot - Spiral Pattern'
            }
            
            title = title_map.get(distribution, '3D Scatter Plot')
            
            return {
                'chart_type': 'scatter3d',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': random.choice(['Variable X', 'Feature 1', 'Dimension 1']),
                'ylabel': random.choice(['Variable Y', 'Feature 2', 'Dimension 2']),
                'zlabel': random.choice(['Variable Z', 'Feature 3', 'Dimension 3']),
                'metadata': {
                    'generated': 'random',
                    'distribution': distribution,
                    'num_points': num_points
                }
            }
        
        elif chart_type == 'bar3d':
            # Generate 3D bar data
            num_x = random.randint(3, 5)
            num_y = random.randint(3, 5)
            
            data = self.generate_bar3d_data(
                num_x=num_x,
                num_y=num_y,
                value_range=(20, 100)
            )
            
            title = random.choice([
                'Sales by Product and Region',
                'Performance Matrix',
                'Multi-Dimensional Comparison',
                '3D Bar Chart Analysis',
                'Category Distribution Grid'
            ])
            
            return {
                'chart_type': 'bar3d',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': 'Category 1',
                'ylabel': 'Category 2',
                'zlabel': 'Value',
                'metadata': {
                    'generated': 'random',
                    'num_x': num_x,
                    'num_y': num_y
                }
            }
        
        elif chart_type == 'line3d':
            # Generate 3D line/trajectory data
            trajectory_type = random.choice(['spiral', 'helix', 'random_walk', 'lissajous'])
            num_points = random.choice([80, 120, 150])
            
            data = self.generate_line3d_data(
                num_points=num_points,
                num_lines=1,
                trajectory_type=trajectory_type
            )
            
            title_map = {
                'spiral': '3D Spiral Trajectory',
                'helix': '3D Helix Path',
                'random_walk': '3D Random Walk',
                'lissajous': 'Lissajous Curve in 3D'
            }
            
            return {
                'chart_type': 'line3d',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(trajectory_type, '3D Line Plot'),
                'xlabel': 'X',
                'ylabel': 'Y',
                'zlabel': 'Z',
                'metadata': {
                    'generated': 'random',
                    'trajectory_type': trajectory_type,
                    'num_points': num_points
                }
            }
        
        elif chart_type == 'wireframe3d':
            # Generate wireframe data
            function_type = random.choice(['gaussian', 'saddle', 'wave', 'peaks'])
            
            data = self.generate_wireframe3d_data(
                x_range=(-5, 5),
                y_range=(-5, 5),
                resolution=30,
                function_type=function_type
            )
            
            title_map = {
                'gaussian': 'Gaussian Wireframe',
                'saddle': 'Saddle Point Wireframe',
                'wave': 'Wave Wireframe',
                'peaks': 'Peaks Wireframe'
            }
            
            return {
                'chart_type': 'wireframe3d',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(function_type, '3D Wireframe'),
                'xlabel': 'X Axis',
                'ylabel': 'Y Axis',
                'zlabel': 'Z Value',
                'metadata': {
                    'generated': 'random',
                    'function_type': function_type
                }
            }
        
        elif chart_type == 'quiver3d':
            # Generate vector field data
            field_type = random.choice(['radial', 'circular', 'vortex', 'uniform'])
            grid_size = (4, 4, 4)
            
            data = self.generate_quiver3d_data(
                grid_size=grid_size,
                field_type=field_type
            )
            
            title_map = {
                'radial': '3D Radial Vector Field',
                'circular': '3D Circular Flow Field',
                'vortex': '3D Vortex Field',
                'uniform': '3D Uniform Field'
            }
            
            return {
                'chart_type': 'quiver3d',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(field_type, '3D Vector Field'),
                'xlabel': 'X',
                'ylabel': 'Y',
                'zlabel': 'Z',
                'metadata': {
                    'generated': 'random',
                    'field_type': field_type
                }
            }
        
        elif chart_type == 'stem3d':
            # Generate stem plot data
            pattern = random.choice(['random', 'grid', 'decreasing'])
            num_points = random.choice([20, 25, 30])
            
            data = self.generate_stem3d_data(
                num_points=num_points,
                pattern=pattern
            )
            
            title_map = {
                'random': '3D Stem Plot - Random Distribution',
                'grid': '3D Stem Plot - Grid Pattern',
                'decreasing': '3D Stem Plot - Radial Decay'
            }
            
            return {
                'chart_type': 'stem3d',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(pattern, '3D Stem Plot'),
                'xlabel': 'X',
                'ylabel': 'Y',
                'zlabel': 'Value',
                'metadata': {
                    'generated': 'random',
                    'pattern': pattern,
                    'num_points': num_points
                }
            }
        
        elif chart_type == 'parametric3d':
            # Generate parametric surface
            surface_type = random.choice(['torus', 'sphere', 'mobius', 'figure8'])
            
            data = self.generate_parametric3d_data(
                surface_type=surface_type,
                resolution=50
            )
            
            title_map = {
                'torus': 'Parametric Torus',
                'sphere': 'Parametric Sphere',
                'mobius': 'Möbius Strip',
                'figure8': 'Figure-8 Surface'
            }
            
            return {
                'chart_type': 'parametric3d',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(surface_type, 'Parametric Surface'),
                'xlabel': 'X',
                'ylabel': 'Y',
                'zlabel': 'Z',
                'metadata': {
                    'generated': 'random',
                    'surface_type': surface_type
                }
            }
        
        elif chart_type == 'contour3d':
            # Generate 3D contour data
            function_type = random.choice(['gaussian', 'peaks', 'wave'])
            
            data = self.generate_contour3d_data(
                x_range=(-5, 5),
                y_range=(-5, 5),
                resolution=40,
                function_type=function_type
            )
            
            title_map = {
                'gaussian': '3D Contour - Gaussian',
                'peaks': '3D Contour - Peaks Function',
                'wave': '3D Contour - Wave Pattern'
            }
            
            return {
                'chart_type': 'contour3d',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(function_type, '3D Contour Plot'),
                'xlabel': 'X Axis',
                'ylabel': 'Y Axis',
                'zlabel': 'Z Value',
                'metadata': {
                    'generated': 'random',
                    'function_type': function_type
                }
            }
        
        elif chart_type == 'volume3d':
            # Generate volume data
            volume_type = random.choice(['sphere', 'gaussian', 'gradient'])
            grid_size = (8, 8, 8)
            
            data = self.generate_volume3d_data(
                grid_size=grid_size,
                volume_type=volume_type
            )
            
            title_map = {
                'sphere': '3D Volume - Sphere',
                'gaussian': '3D Volume - Gaussian Blob',
                'gradient': '3D Volume - Linear Gradient'
            }
            
            return {
                'chart_type': 'volume3d',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(volume_type, '3D Volume Visualization'),
                'xlabel': 'X',
                'ylabel': 'Y',
                'zlabel': 'Z',
                'metadata': {
                    'generated': 'random',
                    'volume_type': volume_type
                }
            }
        
        elif chart_type == 'network3d':
            # Generate network graph data
            num_nodes = random.randint(12, 20)
            
            data = self.generate_network3d_data(
                num_nodes=num_nodes,
                connection_probability=0.15
            )
            
            return {
                'chart_type': 'network3d',
                'data': data,
                'filename_root': filename_root,
                'title': random.choice([
                    '3D Network Graph',
                    'Network Topology',
                    '3D Node-Edge Diagram',
                    'Spatial Network Structure'
                ]),
                'xlabel': 'X',
                'ylabel': 'Y',
                'zlabel': 'Z',
                'metadata': {
                    'generated': 'random',
                    'num_nodes': num_nodes,
                    'num_edges': len(data['edges'])
                }
            }
          
        elif chart_type == 'hist2d':
            # Generate 2D histogram data
            dist_type = random.choice(['random', 'increasing', 'decreasing', 'cyclical', 'volatility_increase', 'volatility_decrease'])
            num_points = random.choice([150, 200, 250, 300])
            
            data = self.generate_hist2d_data(
                num_points=num_points,
                x_range=(0, 100),
                y_range=(0, 100),
                distribution=dist_type,
                bins=[random.choice([15, 20, 25]), random.choice([15, 20, 25])]
            )
            
            # Create title based on distribution type
            dist_names = {
                'random': 'Uniform Distribution',
                'clustered': 'Clustered Distribution',
                'diagonal': 'Linear Correlation',
                'circular': 'Radial Distribution',
                'normal': 'Bivariate Normal Distribution'
            }
            title = f'2D Histogram: {dist_names.get(dist_type, "Distribution")}'
            
            return {
                'chart_type': 'hist2d',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': random.choice(['Variable X', 'Feature 1', 'Measurement A']),
                'ylabel': random.choice(['Variable Y', 'Feature 2', 'Measurement B']),
                'metadata': {
                    'generated': 'random',
                    'distribution': dist_type,
                    'num_points': num_points,
                    'bins': data['bins']
                }
            }
        
        elif chart_type == 'cohere':
            # Generate coherence data
            signal_type = random.choice([
                'mixed', 'highly_correlated', 'partially_correlated',
                'frequency_dependent', 'phase_shifted'
            ])
            
            Fs = random.choice([500, 1000, 2000])
            duration = random.choice([1.0, 2.0, 3.0])
            
            # Ensure NFFT is valid for signal length
            signal_length = int(Fs * duration)
            # NFFT should be at most half the signal length to allow for proper windowing
            max_nfft = signal_length // 2
            valid_nffts = [n for n in [64, 128, 256, 512] if n <= max_nfft]
            NFFT = random.choice(valid_nffts) if valid_nffts else 64
            
            data = self.generate_cohere_data(
                signal_type=signal_type,
                duration=duration,
                Fs=Fs,
                NFFT=NFFT
            )
            
            # Create descriptive title
            signal_names = {
                'perfectly_correlated': 'Perfect Correlation',
                'highly_correlated': 'High Correlation',
                'partially_correlated': 'Partial Correlation',
                'frequency_dependent': 'Frequency-Dependent Correlation',
                'uncorrelated': 'No Correlation',
                'phase_shifted': 'Phase-Shifted Signals',
                'mixed': 'Correlated Signal with Noise'
            }
            
            return {
                'chart_type': 'cohere',
                'data': data,
                'filename_root': filename_root,
                'title': f'Signal Coherence: {signal_names.get(signal_type, "Analysis")}',
                'xlabel': 'Frequency (Hz)',
                'ylabel': 'Coherence',
                'metadata': {
                    'generated': 'random',
                    'signal_type': signal_type,
                    'sampling_rate': Fs,
                    'duration': duration,
                    'num_samples': len(data['x'])
                }
            }
        
        elif chart_type == 'signal_pair':
            # Generate signal pair data (reuse cohere data generation)
            signal_type = random.choice([
                'mixed', 'highly_correlated', 'partially_correlated',
                'frequency_dependent', 'phase_shifted'
            ])
            
            Fs = random.choice([500, 1000, 2000])
            duration = random.choice([0.5, 1.0, 2.0])
            
            # Ensure NFFT is valid for signal length
            signal_length = int(Fs * duration)
            max_nfft = signal_length // 2
            valid_nffts = [n for n in [64, 128, 256, 512] if n <= max_nfft]
            NFFT = random.choice(valid_nffts) if valid_nffts else 64
            
            data = self.generate_cohere_data(
                signal_type=signal_type,
                duration=duration,
                Fs=Fs,
                NFFT=NFFT
            )
            
            # Add custom labels
            data['labels'] = ['Signal X', 'Signal Y']
            
            # Create descriptive title
            signal_names = {
                'perfectly_correlated': 'Perfectly Correlated',
                'highly_correlated': 'Highly Correlated',
                'partially_correlated': 'Partially Correlated',
                'frequency_dependent': 'Frequency-Dependent Correlation',
                'uncorrelated': 'Uncorrelated',
                'phase_shifted': 'Phase-Shifted',
                'mixed': 'Mixed (Signal + Noise)'
            }
            
            return {
                'chart_type': 'signal_pair',
                'data': data,
                'filename_root': filename_root,
                'title': f'Signal Pair: {signal_names.get(signal_type, "Comparison")}',
                'xlabel': 'Time (seconds)',
                'ylabel': 'Amplitude',
                'metadata': {
                    'generated': 'random',
                    'signal_type': signal_type,
                    'sampling_rate': Fs,
                    'duration': duration,
                    'num_samples': len(data['x'])
                }
            }
        
        elif chart_type == 'timeline':
            event_type = random.choice(['project', 'product', 'company'])
            
            # Generate random date range between 2000-01 and 2024-12
            date_range = self._random_date_range('2000-01', '2024-12')
            
            data = self.generate_timeline_data(
                num_events=random.randint(5, 15),
                date_range=date_range,
                event_type=event_type
            )
            
            title_map = {
                'project': 'Project Timeline',
                'product': 'Product Development Timeline',
                'company': 'Company Milestones'
            }
            
            return {
                'chart_type': 'timeline',
                'data': data,
                'filename_root': filename_root,
                'title': title_map.get(event_type, 'Event Timeline'),
                'xlabel': 'Date',
                'ylabel': 'Events',
                'metadata': {
                    'generated': 'random',
                    'num_events': len(data['dates']),
                    'event_type': event_type,
                    'date_range': date_range
                }
            }
        
        elif chart_type == 'heatmap':
            # Generate heatmap with random row and column types
            data = self.generate_heatmap_data(
                value_range=(10, 100)
            )
            
            # Create descriptive title based on data types
            row_label = data['row_labels'][0]
            col_label = data['col_labels'][0]
            
            # Determine what kind of heatmap this is
            if row_label in self.PRODUCTS:
                row_type = 'Product'
            elif row_label in self.REGIONS:
                row_type = 'Regional'
            elif row_label in self.DEPARTMENTS:
                row_type = 'Department'
            elif row_label in self.MONTHS:
                row_type = 'Monthly'
            elif row_label in self.SCORE_LABELS:
                row_type = 'Score'
            else:
                row_type = 'Category'
            
            if col_label in self.QUARTERS:
                col_type = 'Quarterly'
            elif col_label in self.MONTHS:
                col_type = 'Monthly'
            elif col_label in self.REGIONS:
                col_type = 'Regional'
            elif col_label in self.YEARS:
                col_type = 'Annual'
            elif col_label in self.PRODUCTS:
                col_type = 'Product'
            elif col_label in self.DEPARTMENTS:
                col_type = 'Department'
            else:
                col_type = 'Period'
            
            return {
                'chart_type': 'heatmap',
                'data': data,
                'filename_root': filename_root,
                'title': f'{row_type} Performance by {col_type} Period',
                'xlabel': col_type,
                'ylabel': row_type,
                'metadata': {
                    'generated': 'random',
                    'dimensions': f'{len(data["row_labels"])}x{len(data["col_labels"])}'
                }
            }
        
        elif chart_type == 'streamplot':
            # Generate streamplot with random flow pattern
            use_variation = random.choice([True, True, False])  # 66% chance of variation
            
            if use_variation:
                variation_type = random.choice([
                    'varying_density', 'varying_color', 'varying_linewidth',
                    'starting_points', 'masking', 'unbroken'
                ])
                flow_type = random.choice(['vortex', 'source', 'saddle', 'dipole','shear', 'wave', 'turbulent'])
                data = self.generate_streamplot_variation_data(
                    variation_type=variation_type,
                    flow_type=flow_type,
                    grid_size=random.choice([20, 25, 30])
                )
                
                variation_titles = {
                    'varying_density': f'{flow_type.title()} Flow - Varying Density',
                    'varying_color': f'{flow_type.title()} Flow - Color Mapped',
                    'varying_linewidth': f'{flow_type.title()} Flow - Variable Width',
                    'starting_points': f'{flow_type.title()} Flow - Custom Start Points',
                    'masking': f'{flow_type.title()} Flow - Masked Region',
                    'unbroken': f'{flow_type.title()} Flow - Unbroken Streamlines'
                }
                
                title = variation_titles.get(variation_type, 'Vector Field Flow')
                
            else:
                # Standard streamplot
                flow_type = random.choice(['vortex', 'source', 'sink', 'saddle', 
                                          'uniform', 'dipole', 'shear', 'wave', 'turbulent'])
                data = self.generate_streamplot_data(
                    flow_type=flow_type,
                    grid_size=random.choice([15, 20, 25])
                )
                
                title_map = {
                    'vortex': 'Vortex Flow Pattern',
                    'source': 'Radial Source Flow',
                    'sink': 'Radial Sink Flow',
                    'saddle': 'Saddle Point Flow',
                    'uniform': 'Uniform Flow Field',
                    'dipole': 'Dipole Flow Pattern',
                    'shear': 'Shear Flow',
                    'wave': 'Wave Flow Pattern',
                    'turbulent': 'Turbulent Flow Field'
                }
                title = title_map.get(flow_type, 'Vector Field Flow')
                data['color'] = 'velocity'  # Default color by velocity
            
            application_map = {
                'vortex': 'Rotational Fluid Dynamics',
                'source': 'Expansion/Divergence Field',
                'sink': 'Compression/Convergence Field',
                'saddle': 'Hyperbolic Flow',
                'uniform': 'Laminar Flow',
                'dipole': 'Combined Source-Sink',
                'shear': 'Velocity Gradient',
                'wave': 'Oscillating Field',
                'turbulent': 'Turbulent Flow Simulation'
            }
            
            return {
                'chart_type': 'streamplot',
                'data': data,
                'filename_root': filename_root,
                'title': title,
                'xlabel': 'X Position',
                'ylabel': 'Y Position',
                'metadata': {
                    'generated': 'random',
                    'flow_type': flow_type,
                    'application': application_map.get(flow_type, 'Vector field'),
                    'grid_size': len(data['x']),
                    'variation': use_variation
                }
            }
        
        return {}