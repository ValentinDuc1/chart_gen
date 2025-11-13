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
        chart_types = ['line', 'bar', 'horizontal_bar', 'pie', 'scatter', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution', 'hist2d', 'cohere', 'signal_pair', 'timeline', 'heatmap', 'streamplot']
        
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
        
        elif chart_type == 'hist2d':
            # Generate 2D histogram data
            dist_type = random.choice(['random', 'clustered', 'diagonal', 'circular', 'normal'])
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
            start = f'{random.randint(2000, 2023)}-{random.randint(1, 12):02d}'
            end = f'{random.randint(2023, 2024)}-{random.randint(1, 12):02d}'
            date_range = (start, end)
            data = self.generate_timeline_data(
                num_events=random.randint(5, 20),
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
                    'event_type': event_type
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