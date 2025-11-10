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
        area_type: str = 'single',
        x_type: str = 'months',
        y_range: tuple = (10, 100),
        num_series: int = 2
    ) -> Dict[str, Any]:
        """
        Generate data for area charts (fill_between).
        
        Args:
            num_points: Number of data points (auto-determined if None)
            area_type: 'single', 'range', or 'stacked'
            x_type: Type of x-axis ('months', 'quarters', 'numeric', 'years')
            y_range: Range for y values (min, max)
            num_series: Number of series for stacked areas
            
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
            return {'x': x, 'y': y}
            
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
            
            return {'x': x, 'y1': y1, 'y2': y2}
            
        else:  # stacked
            # Multiple stacked areas
            areas = []
            for i in range(num_series):
                # Generate smaller values for stacking
                adjusted_range = (y_range[0] // num_series, y_range[1] // num_series)
                y = self._generate_series(actual_num_points, adjusted_range, 'fluctuating')
                areas.append({
                    'y': y,
                    'label': random.choice(self.PRODUCTS) if i < len(self.PRODUCTS) else f'Series {i+1}'
                })
            
            return {'x': x, 'areas': areas}
    
    def generate_discrete_distribution_data(
        self,
        distribution_type: str = 'binomial',
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
        chart_types = ['line', 'bar', 'horizontal_bar', 'pie', 'scatter', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution']
        
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
            # Generate area chart data
            area_type = random.choice(['single', 'range', 'stacked'])
            x_type = random.choice(['months', 'quarters', 'numeric'])
            
            if area_type == 'stacked':
                num_series = random.choice([2, 3])
                data = self.generate_area_data(
                    area_type='stacked',
                    x_type=x_type,
                    num_series=num_series,
                    y_range=(20, 80)
                )
                title = 'Stacked Performance Over Time'
                ylabel = random.choice(['Total Value', 'Combined Sales ($K)', 'Cumulative Score'])
            elif area_type == 'range':
                data = self.generate_area_data(
                    area_type='range',
                    x_type=x_type,
                    y_range=(30, 100)
                )
                title = random.choice(['Performance Range', 'Confidence Interval', 'Value Range Over Time'])
                ylabel = random.choice(['Value', 'Sales ($K)', 'Performance Score', 'Metric'])
            else:  # single
                data = self.generate_area_data(
                    area_type='single',
                    x_type=x_type,
                    y_range=(30, 120)
                )
                title = random.choice(['Cumulative Performance', 'Growth Over Time', 'Trend Analysis'])
                ylabel = random.choice(['Value', 'Sales ($K)', 'Revenue', 'Performance'])
            
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
                    'num_points': len(data.get('x', []))
                }
            }
        
        elif chart_type == 'discrete_distribution':
            dist_type = random.choice(['binomial', 'poisson', 'uniform', 'rating', 'score'])
            
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
        
        return {}